import type { BedrockClient } from "../connection/client.js";
import type { World, EntityInfo } from "./world.js";
import { setDetectedAirId } from "./decoder.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("perception");

const HOSTILE_TYPES = new Set([
  "minecraft:zombie",
  "minecraft:skeleton",
  "minecraft:creeper",
  "minecraft:spider",
  "minecraft:enderman",
  "minecraft:witch",
  "minecraft:husk",
  "minecraft:drowned",
  "minecraft:pillager",
  "minecraft:vindicator",
  "minecraft:ravager",
  "minecraft:phantom",
  "minecraft:zombified_piglin",
  "minecraft:piglin",
  "minecraft:hoglin",
  "minecraft:wither_skeleton",
  "minecraft:blaze",
  "minecraft:ghast",
  "minecraft:magma_cube",
  "minecraft:slime",
]);

const tickNowMs = () => Date.now();

// Tracks per-client teardown hooks so Agent.stop() can cancel the pending
// death-disconnect timer (and similar) instead of letting it fire after the
// agent has already torn down.
const teardownHooks = new WeakMap<BedrockClient, Set<() => void>>();

function registerTeardown(client: BedrockClient, fn: () => void): void {
  let set = teardownHooks.get(client);
  if (!set) { set = new Set(); teardownHooks.set(client, set); }
  set.add(fn);
}

/** Called by Agent.stop() to cancel any timers/intervals owned by perception. */
export function detachPerception(client: BedrockClient): void {
  const set = teardownHooks.get(client);
  if (!set) return;
  for (const fn of set) {
    try { fn(); } catch { /* swallow — teardown is best-effort */ }
  }
  set.clear();
  teardownHooks.delete(client);
}

// Health-attribute NaN warnings are logged once per session to avoid spam
// when BDS sends sparse / partial attribute snapshots.
let warnedHealthNaN = false;

export function attachPerception(client: BedrockClient, world: World): void {
  client.on("start_game", (pkt: any) => {
    // Field names vary between bedrock-protocol versions: entity_id, runtime_entity_id, etc.
    const rid = pkt.runtime_entity_id ?? pkt.entity_id_self ?? pkt.entity_id;
    if (rid !== undefined) world.self.runtimeEntityId = BigInt(rid);

    const pos = pkt.player_position ?? pkt.position ?? pkt.spawn_position;
    if (pos && typeof pos.x === "number") {
      world.self.position = { x: pos.x, y: pos.y ?? 0, z: pos.z ?? 0 };
    }
    if (pkt.rotation) {
      world.self.pitch = pkt.rotation.x ?? 0;
      world.self.yaw = pkt.rotation.z ?? pkt.rotation.y ?? 0;
    }
    if (typeof pkt.player_gamemode === "number") world.self.gameMode = pkt.player_gamemode;
    if (typeof pkt.dimension === "number") world.self.dimension = pkt.dimension;
    log.info(`start_game: pos=${JSON.stringify(world.self.position)} entityId=${world.self.runtimeEntityId} gamemode=${world.self.gameMode} dim=${world.self.dimension}`);

    // CRITICAL post-spawn handshake (bedrock-protocol Discussion #566). Without
    // these three packets, BDS keeps the player in a "still loading / immortal"
    // state and silently drops every player_auth_input — bot appears in the
    // world but never moves and never receives correct_player_move_prediction.
    const c = client as any;
    try {
      c.queue("serverbound_loading_screen", { type: 1 });
      c.queue("serverbound_loading_screen", { type: 2 });
      c.queue("interact", {
        action_id: "mouse_over_entity",
        target_entity_id: 0n,
        position: { x: 0, y: 0, z: 0 },
      });
      log.info("sent post-spawn handshake (loading_screen 1+2, interact)");
    } catch (err) {
      log.warn(`post-spawn handshake failed: ${(err as Error).message}`);
    }
  });

  client.on("play_status", (pkt: any) => {
    log.debug(`play_status: ${pkt.status}`);
  });

  // The first move_player after spawn carries our authoritative spawn position.
  client.on("set_spawn_position", (pkt: any) => {
    if (pkt.player_position) {
      world.self.position = { x: pkt.player_position.x, y: pkt.player_position.y, z: pkt.player_position.z };
      log.info(`set_spawn_position: ${JSON.stringify(world.self.position)}`);
    }
  });

  // Server-driven repositions of the bot itself.
  let movePlayerCount = 0;
  client.on("move_player", (pkt: any) => {
    const targetId = BigInt(pkt.runtime_id ?? pkt.runtime_entity_id ?? pkt.entity_runtime_id ?? 0n);
    const isUs = world.self.runtimeEntityId !== null && targetId === world.self.runtimeEntityId;
    const noIdYet = world.self.runtimeEntityId === null;
    if (!isUs && !noIdYet) return;
    if (pkt.position && typeof pkt.position.x === "number") {
      // Clone — bedrock-protocol may mutate the packet object after dispatch.
      world.self.position = { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z };
    }
    if (pkt.rotation) {
      world.self.pitch = pkt.rotation.x ?? world.self.pitch;
      world.self.yaw = pkt.rotation.z ?? world.self.yaw;
      world.self.headYaw = pkt.rotation.y ?? world.self.headYaw;
    }
    world.self.onGround = !!pkt.on_ground;
    movePlayerCount++;
    if (movePlayerCount <= 3 || movePlayerCount % 50 === 0) {
      log.info(`move_player #${movePlayerCount}: server placed us at ${JSON.stringify(pkt.position)} mode=${pkt.mode}`);
    }
  });

  // Server-authoritative-movement correction: BDS sends this when the client
  // prediction diverges from server simulation. If we see lots of these, our
  // input packets are being ignored / rejected.
  let correctMovePredictionCount = 0;
  client.on("correct_player_move_prediction", (pkt: any) => {
    correctMovePredictionCount++;
    if (correctMovePredictionCount <= 5 || correctMovePredictionCount % 20 === 0) {
      log.warn(`correct_player_move_prediction #${correctMovePredictionCount}: server forcing pos=${JSON.stringify(pkt.position)}`);
    }
    if (pkt.position && typeof pkt.position.x === "number") {
      world.self.position = { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z };
    }
  });

  // Air id detection. BDS uses a NONZERO runtime id for air (e.g. 13080 here),
  // which the chunk decoder cannot know up front. The block at the bot's own
  // body + head is guaranteed air by the server, so we sample those two and,
  // when they agree on a nonzero id, treat that id as air everywhere. Requiring
  // agreement guards against a transient desync placing the bot inside a block.
  let airLogged = false;
  setInterval(() => {
    const p = world.self.position;
    const fx = Math.floor(p.x), fy = Math.floor(p.y), fz = Math.floor(p.z);
    const body = world.getBlock({ x: fx, y: fy, z: fz })?.runtimeId;
    const head = world.getBlock({ x: fx, y: fy + 1, z: fz })?.runtimeId;
    if (body !== undefined && body !== 0 && body === head) {
      setDetectedAirId(body);
      if (!airLogged) { airLogged = true; log.info(`air runtime id detected: ${body}`); }
    }
  }, 2000);

  // Bedrock respawn handshake: server sends respawn(state=0=searching),
  // then respawn(state=1=server_ready). We MUST reply with state=2=client_ready
  // or the server keeps the player in respawn-pending and silently drops every
  // subsequent player_auth_input packet — bot looks frozen even though we're
  // sending movement commands at 20Hz.
  client.on("respawn", (pkt: any) => {
    if (pkt.position) {
      world.self.position = { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z };
    }
    const state = pkt.state;
    log.info(`respawn: pos=${JSON.stringify(world.self.position)} state=${state}`);
    // state values: 0=server_searching, 1=server_ready, 2=client_ready, 3=client_spawn
    // Respond to either with state=2. Initial spawn sends 0,0,1; death-respawn
    // may send just state=0. Acking state=0 immediately is the signal
    // "I'm ready to respawn" — server then sends state=1 and continues the flow.
    if (state === 0 || state === 1) {
      try {
        (client as any).queue("respawn", {
          position: pkt.position ?? world.self.position,
          state: 2,
          runtime_entity_id: world.self.runtimeEntityId ?? 0n,
        });
        log.info(`sent respawn ack (state=2)`);
      } catch (err) {
        log.warn(`respawn ack failed: ${(err as Error).message}`);
      }
    }
    if (state === 1) {
      // Tell the server we've finished re-initializing after death. Same packet
      // bedrock-protocol auto-sends on initial spawn (see client.js:187).
      try {
        (client as any).queue("set_local_player_as_initialized", {
          runtime_entity_id: world.self.runtimeEntityId ?? 0n,
        });
        log.info(`re-sent set_local_player_as_initialized after respawn`);
      } catch (err) {
        log.warn(`post-respawn init failed: ${(err as Error).message}`);
      }
    }
  });

  // Server-corrected position (authority pushes us).
  client.on("set_player_game_type", (pkt: any) => {
    if (typeof pkt.gamemode === "number") world.self.gameMode = pkt.gamemode;
  });

  // Health/food/etc come as attribute updates.
  // Sentinel -1: the first observation never trips the death detector. This
  // prevents a reconnect-storm when BDS sends a stale update_attributes with
  // health=0 before the respawn handshake completes.
  let lastHealth = -1;
  let respawnPending = false;
  let deathTimer: NodeJS.Timeout | null = null;
  const cancelDeathTimer = () => {
    if (deathTimer !== null) {
      clearTimeout(deathTimer);
      deathTimer = null;
    }
  };
  registerTeardown(client, cancelDeathTimer);
  client.on("update_attributes", (pkt: any) => {
    if (world.self.runtimeEntityId === null) return;
    if (BigInt(pkt.runtime_entity_id ?? 0n) !== world.self.runtimeEntityId) return;
    for (const attr of (pkt.attributes ?? [])) {
      const name = String(attr.name);
      const value = Number(attr.value ?? attr.current);
      if (name.endsWith(":health") || name === "health") {
        if (!Number.isFinite(value)) {
          if (!warnedHealthNaN) {
            warnedHealthNaN = true;
            log.warn(`health attribute missing/NaN (value=${attr.value} current=${attr.current}) — skipping (further warnings suppressed)`);
          }
          continue;
        }
        world.self.health = value;
        if (typeof attr.max === "number") world.self.maxHealth = attr.max;
        // A fresh respawn (state=1) arrives via the respawn handler; if a
        // pending death timer is still scheduled, cancel it.
        if (value > 0) cancelDeathTimer();
        if (lastHealth > 0 && value <= 0 && !respawnPending) {
          respawnPending = true;
          log.info("bot died — closing connection to force fresh reconnect (handles respawn cleanly)");
          // Notify the agent so the next trajectory log row is flagged
          // terminal (done=true). bedrock-protocol clients are EventEmitters,
          // so a custom event keeps perception decoupled from agent internals.
          try { (client as any).emit?.("bot_died"); }
          catch (err) { log.warn(`bot_died emit failed: ${(err as Error).message}`); }
          // Give other death packets a moment, then disconnect. Index.ts's
          // session loop will reconnect with a fresh login flow.
          cancelDeathTimer();
          deathTimer = setTimeout(() => {
            deathTimer = null;
            try { (client as any).close?.(); }
            catch (err) { log.warn(`close on death failed: ${(err as Error).message}`); }
          }, 500);
          // Don't pin the event loop on this one timer.
          deathTimer.unref?.();
        } else if (value > 0) {
          respawnPending = false;
        }
        lastHealth = value;
      } else if (name.endsWith(":player.hunger") || name === "player.hunger") {
        if (Number.isFinite(value)) world.self.food = value;
      } else if (name.endsWith(":player.saturation") || name === "player.saturation") {
        if (Number.isFinite(value)) world.self.saturation = value;
      } else if (name.endsWith(":player.level") || name === "player.level") {
        if (Number.isFinite(value)) world.self.experienceLevel = value;
      }
    }
  });

  client.on("add_entity", (pkt: any) => {
    const id = BigInt(pkt.runtime_id ?? pkt.runtime_entity_id ?? 0n);
    if (id === 0n) return;
    const type = String(pkt.entity_type ?? "");
    const e: EntityInfo = {
      runtimeEntityId: id,
      uniqueId: pkt.unique_id !== undefined ? BigInt(pkt.unique_id) : undefined,
      type,
      // Clone — bedrock-protocol may reuse/mutate the packet's vec3 objects.
      position: pkt.position
        ? { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z }
        : { x: 0, y: 0, z: 0 },
      velocity: pkt.velocity
        ? { x: pkt.velocity.x, y: pkt.velocity.y, z: pkt.velocity.z }
        : { x: 0, y: 0, z: 0 },
      yaw: pkt.rotation?.z ?? 0,
      pitch: pkt.rotation?.x ?? 0,
      isHostile: HOSTILE_TYPES.has(type),
      lastSeenTickMs: tickNowMs(),
    };
    world.setEntity(e);
  });

  client.on("add_player", (pkt: any) => {
    const id = BigInt(pkt.runtime_id ?? pkt.runtime_entity_id ?? 0n);
    if (id === 0n) return;
    const e: EntityInfo = {
      runtimeEntityId: id,
      uniqueId: pkt.entity_unique_id !== undefined ? BigInt(pkt.entity_unique_id) : undefined,
      type: "minecraft:player",
      position: pkt.position
        ? { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z }
        : { x: 0, y: 0, z: 0 },
      velocity: pkt.velocity
        ? { x: pkt.velocity.x, y: pkt.velocity.y, z: pkt.velocity.z }
        : { x: 0, y: 0, z: 0 },
      yaw: pkt.yaw ?? 0,
      pitch: pkt.pitch ?? 0,
      isPlayer: true,
      isHostile: false,
      username: pkt.username,
      lastSeenTickMs: tickNowMs(),
    };
    world.setEntity(e);
  });

  client.on("remove_entity", (pkt: any) => {
    // proto.yml @ 1.26.10: packet_remove_entity carries only `entity_id_self`,
    // which is the *unique* entity id (zigzag64). But world.entities is keyed
    // by runtime id. Try the runtime-id fields first (some packet variants
    // include them), then fall back to scanning by uniqueId.
    const runtimeIdRaw = pkt.runtime_id ?? pkt.runtime_entity_id ?? pkt.target_runtime_id;
    if (runtimeIdRaw !== undefined) {
      world.removeEntity(BigInt(runtimeIdRaw));
      return;
    }
    const uniqueRaw = pkt.entity_id_self ?? pkt.unique_entity_id ?? pkt.unique_id;
    if (uniqueRaw === undefined) return;
    const uniqueId = BigInt(uniqueRaw);
    for (const [key, ent] of world.entities) {
      if (ent.uniqueId !== undefined && ent.uniqueId === uniqueId) {
        world.entities.delete(key);
        return;
      }
    }
  });

  client.on("move_entity_absolute", (pkt: any) => {
    const id = BigInt(pkt.runtime_entity_id ?? 0n);
    const ent = world.entities.get(String(id));
    if (!ent) return;
    if (pkt.position) {
      ent.position = { x: pkt.position.x, y: pkt.position.y, z: pkt.position.z };
    }
    if (pkt.rotation) {
      ent.pitch = pkt.rotation.x ?? ent.pitch;
      ent.yaw = pkt.rotation.z ?? ent.yaw;
    }
    ent.lastSeenTickMs = tickNowMs();
  });

  client.on("move_entity_delta", (pkt: any) => {
    const id = BigInt(pkt.runtime_entity_id ?? 0n);
    const ent = world.entities.get(String(id));
    if (!ent) return;
    if (typeof pkt.dx === "number") ent.position.x += pkt.dx;
    if (typeof pkt.dy === "number") ent.position.y += pkt.dy;
    if (typeof pkt.dz === "number") ent.position.z += pkt.dz;
    ent.lastSeenTickMs = tickNowMs();
  });

  // Single-block updates. Chunk decoding is intentionally NOT here — see world/chunk.ts.
  client.on("update_block", (pkt: any) => {
    if (!pkt.position) return;
    const rid = pkt.block_runtime_id ?? 0;
    world.setBlock(pkt.position, { runtimeId: rid });
  });

  // Inventory sync (legacy). item_stack_request responses augment this for slot moves.
  client.on("inventory_content", (pkt: any) => {
    if (pkt.window_id !== 0 && pkt.window_id !== "inventory") return; // only main inv for now
    const items = pkt.input ?? pkt.items ?? [];
    world.inventory.clear();
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      if (!it || it.network_id === 0 || it.count === 0) continue;
      world.inventory.set(i, {
        networkId: it.network_id,
        count: it.count,
        name: it.name,
        nbt: it.extra?.nbt,
      });
    }
  });

  client.on("inventory_slot", (pkt: any) => {
    if (pkt.window_id !== 0 && pkt.window_id !== "inventory") return;
    const it = pkt.item;
    const slot = pkt.slot;
    if (!it || it.network_id === 0 || it.count === 0) {
      world.inventory.delete(slot);
    } else {
      world.inventory.set(slot, {
        networkId: it.network_id,
        count: it.count,
        name: it.name,
        nbt: it.extra?.nbt,
      });
    }
  });

  client.on("mob_equipment", (pkt: any) => {
    if (world.self.runtimeEntityId !== null &&
        BigInt(pkt.runtime_entity_id ?? 0n) === world.self.runtimeEntityId) {
      world.selectedHotbarSlot = pkt.selected_slot ?? world.selectedHotbarSlot;
    }
  });

  client.on("chunk_radius_update", (pkt: any) => {
    log.debug(`chunk radius set to ${pkt.chunk_radius}`);
  });

  client.on("disconnect", (reason: any) => {
    log.warn("disconnected", reason);
  });
}
