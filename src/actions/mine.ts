import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { InputController } from "./input.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3floor } from "../utils/vec3.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { isLogId } from "../world/logIds.js";
import { isAirRuntimeId } from "../world/decoder.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("mine");

/** Break-time estimate in ms. The server does NOT enforce a break timer (see
 * mineBlock), so this is only a pacing value: how long we sustain continue_break
 * before sending predict_break, picked to roughly match real survival break
 * times so the bot mines at a plausible rate. Values >8000 are a sentinel for
 * "can't break this without the right tool" and are skipped by the caller.
 * Real bare-hand reference: oak log ~3.0s, dirt ~0.75s, leaves ~0.2s,
 * stone (pickaxe) ~1.15s. */
function estimateBreakMs(
  blockName: string | undefined,
  heldItem: string | undefined,
  runtimeId?: number,
): number {
  // Names are usually undefined (the decoder gives runtime ids, not names), so
  // when we can positively identify a log by id, use the bare-hand log time and
  // skip the name guessing below.
  if (runtimeId !== undefined && isLogId(runtimeId)) return 3000;
  if (!blockName) return 3000;
  if (blockName.includes("leaves")) return 400;
  if (blockName.includes("flower") || blockName.includes("grass") || blockName.includes("tall_")) return 300;
  if (blockName.includes("dirt") || blockName.includes("sand") || blockName.includes("gravel")) return 900;
  if (blockName.includes("wood") || blockName.includes("log") || blockName.includes("planks")) {
    return heldItem?.includes("axe") ? 1500 : 3000; // bare hand oak log ≈ 3s
  }
  if (blockName.includes("stone") || blockName.includes("cobble")) {
    return heldItem?.includes("pickaxe") ? 1300 : 9999; // bare hand: drops nothing, skip
  }
  if (blockName.includes("ore")) {
    return heldItem?.includes("pickaxe") ? 3000 : 9999;
  }
  if (blockName.includes("obsidian")) return 9999;
  return 3000;
}

/** Pick the block face most directly facing the bot, for the block_action
 * `face` field. Bedrock face ids: 0=-Y,1=+Y,2=-Z,3=+Z,4=-X,5=+X. The server is
 * lenient about face during sustained mining; we approximate from eye->center. */
function faceTowardEye(p: Vec3, eye: Vec3): number {
  const cx = p.x + 0.5, cy = p.y + 0.5, cz = p.z + 0.5;
  const dx = eye.x - cx, dy = eye.y - cy, dz = eye.z - cz;
  const ax = Math.abs(dx), ay = Math.abs(dy), az = Math.abs(dz);
  if (ay >= ax && ay >= az) return dy >= 0 ? 1 : 0;
  if (ax >= az) return dx >= 0 ? 5 : 4;
  return dz >= 0 ? 3 : 2;
}

const TICK_MS = 50; // 20 Hz, matches InputController

export async function mineBlock(
  client: BedrockClient,
  world: World,
  input: InputController,
  position: Vec3,
): Promise<{ broken: boolean; reason?: string }> {
  void client; // mining is now driven through the player_auth_input stream
  const p = v3floor(position);
  const block = world.getBlock(p);
  if (!block) {
    return { broken: false, reason: "unknown_block" };
  }
  if (isAirRuntimeId(block.runtimeId)) {
    return { broken: true }; // already air (server air id is nonzero, e.g. 13080)
  }

  const heldSlot = world.inventory.get(world.selectedHotbarSlot);
  const breakMs = estimateBreakMs(block.name, heldSlot?.name, block.runtimeId);
  if (breakMs > 8000) {
    return { broken: false, reason: "missing_tool" };
  }

  input.lookAt({ x: p.x + 0.5, y: p.y + 0.5, z: p.z + 0.5 });
  setIntent(ActionId.MineFront, breakMs + 500);

  const eye = {
    x: world.self.position.x,
    y: world.self.position.y + 1.62,
    z: world.self.position.z,
  };
  const face = faceTowardEye(p, eye);
  const blockPos = { x: p.x, y: p.y, z: p.z };

  const center = { x: p.x + 0.5, y: p.y + 0.5, z: p.z + 0.5 };

  // STABILIZE before breaking. BDS refuses to complete a break for a player it
  // thinks is falling or mid-position-correction. Every failed run showed the
  // bot airborne / desynced (a correct_player_move_prediction storm) while
  // "mining". Stop all movement, lock the aim on the block, and wait for
  // onGround plus a few ticks of position stability before sending any break.
  input.setMove({ forward: 0, strafe: 0, jump: false, sneak: false, sprint: false });
  {
    let lastPos = { ...world.self.position };
    let stableTicks = 0;
    const stableDeadline = Date.now() + 1500;
    while (Date.now() < stableDeadline) {
      await new Promise((r) => setTimeout(r, TICK_MS));
      input.lookAt(center);
      const pos = world.self.position;
      const moved =
        Math.abs(pos.x - lastPos.x) + Math.abs(pos.y - lastPos.y) + Math.abs(pos.z - lastPos.z);
      lastPos = { x: pos.x, y: pos.y, z: pos.z };
      if (world.self.onGround && moved < 0.02) {
        if (++stableTicks >= 4) break; // ~200ms of grounded stillness
      } else {
        stableTicks = 0;
      }
    }
  }

  // Break via the player_auth_input block_action stream. This BDS has
  // server-authoritative block breaking ON (see server-authoritative-block-breaking-*
  // in server.properties), so the SERVER runs the break timer and decides when the
  // block breaks. Sequence: start_break once, then crack_break every tick to sustain
  // the break; wait for the server to turn the block to air (detected below).
  // TWO things are load-bearing:
  //   - The input ticker MUST set the `block_breaking_delay_enabled` input flag on
  //     every break tick (see input.ts). Without it the server starts the break but
  //     never runs the delayed-completion path, so the block cracks and resets.
  //   - We do NOT send predict_break. Predicting before the server's own timer is
  //     done resets the crack. With a tool the server finishes in ~1.5s so an early
  //     predict was harmless, but bare-handed (~3s) an early predict reset the break
  //     every tick and the block never broke. crack-only is correct for both.
  // continue_break thrashes start/stop here; crack_break is the per-tick action.
  // One clean stream (callers must not overlap).
  input.lookAt(center);
  input.setBlockAction("start_break", blockPos, face);

  const startedAt = Date.now();
  // Generous tail: server-authoritative bare-hand breaks on this BDS take far
  // longer than the vanilla estimate (an axe finishes a log in ~1s, bare hand
  // needs many seconds), so give the server ample time to complete before we
  // give up. We return the instant the block reads air, so a fast break still
  // exits quickly — this only extends how long we wait on slow ones.
  const deadline = startedAt + breakMs + 12_000;
  let sawInfo = false;

  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, TICK_MS));
    input.lookAt(center); // keep the aim locked on the block

    // Sustain the server-side break with crack_break every tick and let the
    // SERVER decide when it's done (it sends the block→air update we detect
    // below). Do NOT send predict_break: predicting before the server's own
    // timer completes resets the crack. With a tool the server finishes fast so
    // an early predict never mattered; bare-handed the break takes ~3s, so an
    // early predict reset it every tick and the block never broke. crack-only is
    // correct for both.
    input.setBlockAction("crack_break", blockPos, face);

    const current = world.getBlock(p);
    if (current) {
      sawInfo = true;
      if (isAirRuntimeId(current.runtimeId)) { // server air id is nonzero (e.g. 13080)
        log.info(`broke ${block.name ?? block.runtimeId} at ${p.x},${p.y},${p.z}`);
        return { broken: true };
      }
    }
  }

  input.setBlockAction("abort_break", blockPos, face);
  if (!sawInfo) return { broken: false, reason: "lost_view" };
  return { broken: false, reason: "timeout" };
}
