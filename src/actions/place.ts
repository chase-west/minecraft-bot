import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { InputController } from "./input.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3floor } from "../utils/vec3.js";
import { safeQueue } from "../connection/version.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("place");

/**
 * Place a block at `target`. Caller must ensure the held hotbar slot contains a placeable.
 *
 * Bedrock places blocks via inventory_transaction with type=item_use and action=click_block.
 * Field layouts vary slightly across protocol versions; we send the conservative form.
 */
export async function placeBlock(
  client: BedrockClient,
  world: World,
  input: InputController,
  target: Vec3,
  against: Vec3,
  face = 1, // 1 = top face by default
): Promise<{ placed: boolean; reason?: string }> {
  const heldSlot = world.inventory.get(world.selectedHotbarSlot);
  if (!heldSlot || heldSlot.count <= 0) {
    return { placed: false, reason: "empty_hand" };
  }

  const t = v3floor(target);
  const a = v3floor(against);

  input.lookAt({ x: t.x + 0.5, y: t.y + 0.5, z: t.z + 0.5 });
  setIntent(ActionId.PlaceFront, 500);

  const ok = safeQueue(client, "inventory_transaction", {
    transaction: {
      legacy: { legacy_request_id: 0 },
      transaction_type: "item_use",
      actions: [],
      transaction_data: {
        action_type: 0,
        block_position: { x: a.x, y: a.y, z: a.z },
        face,
        hotbar_slot: world.selectedHotbarSlot,
        held_item: {
          network_id: heldSlot.networkId,
          count: heldSlot.count,
          metadata: 0,
          has_stack_id: 0,
          block_runtime_id: 0,
          extra: { has_nbt: 0, can_place_on: [], can_destroy: [] },
        },
        player_pos: world.self.position,
        click_pos: { x: t.x + 0.5, y: t.y + 0.5, z: t.z + 0.5 },
        block_runtime_id: 0,
      },
    },
  }, "place");
  if (!ok) return { placed: false, reason: "tx_error" };

  // Wait for update_block confirmation.
  const deadline = Date.now() + 1500;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
    const b = world.getBlock(t);
    if (b && b.runtimeId !== 0) {
      log.info(`placed at ${t.x},${t.y},${t.z}`);
      return { placed: true };
    }
  }
  return { placed: false, reason: "no_confirm" };
}
