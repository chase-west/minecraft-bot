import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import { selectByName } from "./inventory.js";
import { safeQueue } from "../connection/version.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("eat");

const FOOD_NAMES = [
  "bread", "apple", "cooked_beef", "beef", "cooked_porkchop", "porkchop",
  "cooked_chicken", "chicken", "cooked_mutton", "mutton", "cooked_rabbit",
  "carrot", "potato", "baked_potato", "melon", "cooked_cod", "cod",
  "cooked_salmon", "salmon", "golden_apple", "golden_carrot",
];

export function findFood(world: World): string | null {
  for (const food of FOOD_NAMES) {
    if (world.itemCount(food) > 0) return food;
  }
  return null;
}

export async function eat(client: BedrockClient, world: World): Promise<{ ate: boolean; reason?: string }> {
  const food = findFood(world);
  if (!food) return { ate: false, reason: "no_food" };

  if (!selectByName(client, world, food)) return { ate: false, reason: "select_failed" };

  const startFood = world.self.food;
  setIntent(ActionId.Eat, 2500);

  const ok = safeQueue(client, "inventory_transaction", {
    transaction: {
      legacy: { legacy_request_id: 0 },
      transaction_type: "item_use",
      actions: [],
      transaction_data: {
        action_type: 1,
        block_position: { x: 0, y: 0, z: 0 },
        face: 0,
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
        click_pos: { x: 0, y: 0, z: 0 },
        block_runtime_id: 0,
      },
    },
  }, "eat");
  if (!ok) return { ate: false, reason: "tx_error" };

  // Eating takes ~1.6s. Poll for hunger increase.
  const deadline = Date.now() + 2500;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 100));
    if (world.self.food > startFood) {
      log.info(`ate ${food}: hunger ${startFood} → ${world.self.food}`);
      return { ate: true };
    }
  }
  return { ate: false, reason: "no_increase" };
}
