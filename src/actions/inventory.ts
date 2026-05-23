import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import { safeQueue } from "../connection/version.js";

/** Select hotbar slot 0..8. Updates local state optimistically; server confirms via mob_equipment. */
export function selectHotbar(client: BedrockClient, world: World, slot: number): void {
  if (slot < 0 || slot > 8) throw new Error(`hotbar slot out of range: ${slot}`);
  const item = world.inventory.get(slot);
  safeQueue(client, "mob_equipment", {
    runtime_entity_id: world.self.runtimeEntityId ?? 0n,
    item: item
      ? { network_id: item.networkId, count: item.count, metadata: 0, has_stack_id: 0, block_runtime_id: 0, extra: { has_nbt: 0, can_place_on: [], can_destroy: [] } }
      : { network_id: 0, count: 0, metadata: 0, has_stack_id: 0, block_runtime_id: 0, extra: { has_nbt: 0, can_place_on: [], can_destroy: [] } },
    slot,
    selected_slot: slot,
    window_id: 0,
  }, "select_hotbar");
  world.selectedHotbarSlot = slot;
}

/** Convenience: select the first hotbar slot containing an item matching `nameIncludes`. */
export function selectByName(client: BedrockClient, world: World, nameIncludes: string): boolean {
  for (let slot = 0; slot < 9; slot++) {
    const item = world.inventory.get(slot);
    if (item?.name?.includes(nameIncludes)) {
      selectHotbar(client, world, slot);
      return true;
    }
  }
  return false;
}
