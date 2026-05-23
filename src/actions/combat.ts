import type { BedrockClient } from "../connection/client.js";
import type { World, EntityInfo } from "../world/world.js";
import type { InputController } from "./input.js";
import { v3distXZ, v3dist } from "../utils/vec3.js";
import { safeQueue } from "../connection/version.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("combat");

const ATTACK_REACH = 3.0;
const ATTACK_COOLDOWN_MS = 600;

export async function attackEntity(
  client: BedrockClient,
  world: World,
  input: InputController,
  target: EntityInfo,
  opts: { timeoutMs?: number } = {},
): Promise<{ killed: boolean; reason?: string }> {
  const deadline = Date.now() + (opts.timeoutMs ?? 30_000);
  let lastSwing = 0;

  while (Date.now() < deadline) {
    const live = world.entities.get(String(target.runtimeEntityId));
    if (!live) {
      log.info(`target ${target.type} (${target.runtimeEntityId}) is gone`);
      return { killed: true };
    }

    const dist = v3dist(world.self.position, live.position);
    input.lookAt(live.position);

    if (dist > ATTACK_REACH) {
      // close the gap
      input.setMove({ forward: 1, strafe: 0, sprint: true, jump: dist > 5 && world.self.onGround });
      await new Promise((r) => setTimeout(r, 80));
      continue;
    }

    input.setMove({ forward: 0, strafe: 0, sprint: false, jump: false });

    if (Date.now() - lastSwing >= ATTACK_COOLDOWN_MS) {
      setIntent(ActionId.AttackNearest, 400);
      safeQueue(client, "inventory_transaction", {
        transaction: {
          legacy: { legacy_request_id: 0 },
          transaction_type: "item_use_on_entity",
          actions: [],
          transaction_data: {
            entity_runtime_id: live.runtimeEntityId,
            action_type: 1,
            hotbar_slot: world.selectedHotbarSlot,
            held_item: {
              network_id: world.inventory.get(world.selectedHotbarSlot)?.networkId ?? 0,
              count: world.inventory.get(world.selectedHotbarSlot)?.count ?? 0,
              metadata: 0,
              has_stack_id: 0,
              block_runtime_id: 0,
              extra: { has_nbt: 0, can_place_on: [], can_destroy: [] },
            },
            player_pos: world.self.position,
            click_pos: live.position,
          },
        },
      }, "attack");
      lastSwing = Date.now();
    }
    await new Promise((r) => setTimeout(r, 50));
  }

  return { killed: false, reason: "timeout" };
}

/** Sprint away from a hostile until it's `safeDist` away or it loses interest. */
export async function fleeFrom(
  world: World,
  input: InputController,
  threat: EntityInfo,
  safeDist = 16,
  timeoutMs = 15_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const live = world.entities.get(String(threat.runtimeEntityId));
    if (!live) return;
    if (v3distXZ(world.self.position, live.position) > safeDist) return;

    const dx = world.self.position.x - live.position.x;
    const dz = world.self.position.z - live.position.z;
    const yaw = Math.atan2(-dx, dz) * 180 / Math.PI;
    input.desired.lookYaw = yaw;
    input.setMove({ forward: 1, strafe: 0, sprint: true, jump: world.self.onGround });
    await new Promise((r) => setTimeout(r, 100));
  }
}
