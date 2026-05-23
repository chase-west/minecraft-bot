import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { InputController } from "./input.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3floor } from "../utils/vec3.js";
import { safeQueue } from "../connection/version.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("mine");

/** Naive break-time estimate. A proper version would consult bedrock-data block hardness × tool multiplier. */
function estimateBreakMs(blockName: string | undefined, heldItem: string | undefined): number {
  if (!blockName) return 1500;
  if (blockName.includes("leaves") || blockName.includes("flower") || blockName.includes("grass")) return 250;
  if (blockName.includes("dirt") || blockName.includes("sand") || blockName.includes("gravel")) return 600;
  if (blockName.includes("wood") || blockName.includes("log") || blockName.includes("planks")) {
    return heldItem?.includes("axe") ? 600 : 1500;
  }
  if (blockName.includes("stone") || blockName.includes("cobble")) {
    return heldItem?.includes("pickaxe") ? 1000 : 5000;
  }
  if (blockName.includes("ore")) {
    return heldItem?.includes("pickaxe") ? 1500 : 9999;
  }
  if (blockName.includes("obsidian")) return 9999;
  return 1500;
}

export async function mineBlock(
  client: BedrockClient,
  world: World,
  input: InputController,
  position: Vec3,
): Promise<{ broken: boolean; reason?: string }> {
  const p = v3floor(position);
  const block = world.getBlock(p);
  if (!block) {
    return { broken: false, reason: "unknown_block" };
  }
  if (block.runtimeId === 0) {
    return { broken: true }; // already air
  }

  const heldSlot = world.inventory.get(world.selectedHotbarSlot);
  const breakMs = estimateBreakMs(block.name, heldSlot?.name);
  if (breakMs > 8000) {
    return { broken: false, reason: "missing_tool" };
  }

  input.lookAt({ x: p.x + 0.5, y: p.y + 0.5, z: p.z + 0.5 });
  setIntent(ActionId.MineFront, breakMs + 500);

  // Bedrock break sequence: PlayerAction(start_break) → PlayerAction(crack_break)* → PlayerAction(stop_break)
  // The exact field names vary slightly across bedrock-protocol versions. We attempt the standard ones.
  const sendAction = (action: string) => {
    safeQueue(client, "player_action", {
      runtime_entity_id: world.self.runtimeEntityId ?? 0n,
      action,
      position: { x: p.x, y: p.y, z: p.z },
      result_position: { x: p.x, y: p.y, z: p.z },
      face: 1,
    }, `player_action:${action}`);
  };

  sendAction("start_break");
  const startedAt = Date.now();

  // Poll for block to become air (server confirms via update_block).
  // We must distinguish two cases:
  //   current.runtimeId === 0 → server confirmed: block is air (broken).
  //   current === undefined   → no info (evicted from cache); block may still
  //                             stand. Keep polling and return lost_view on
  //                             timeout instead of falsely claiming success.
  let sawInfo = false;
  while (Date.now() - startedAt < breakMs + 2000) {
    await new Promise((r) => setTimeout(r, 50));
    const current = world.getBlock(p);
    if (current) {
      sawInfo = true;
      if (current.runtimeId === 0) {
        sendAction("stop_break");
        log.info(`broke ${block.name ?? block.runtimeId} at ${p.x},${p.y},${p.z}`);
        return { broken: true };
      }
    }
    // current === undefined: cache eviction, keep polling.
  }

  sendAction("abort_break");
  if (!sawInfo) return { broken: false, reason: "lost_view" };
  return { broken: false, reason: "timeout" };
}
