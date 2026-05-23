import type { World } from "../world/world.js";
import type { BotState } from "./types.js";
import { DEFAULT_STATE } from "./types.js";
import { findNearbyTree, findNearbyStone } from "../world/semantic.js";

/** Project the voxel/entity world model into the planner's symbolic state. */
export function sense(world: World): BotState {
  const inv = world.inventory;
  const counts = (substr: string) => {
    let total = 0;
    for (const slot of inv.values()) if (slot.name?.includes(substr)) total += slot.count;
    return total;
  };

  const wood = counts("log");
  const planks = counts("planks");
  const sticks = counts("stick");
  const cobblestone = counts("cobblestone");
  const coal = counts("coal");
  const iron = counts("iron_ingot");

  const hasWoodenPickaxe = !!world.findInventorySlot((s) => !!s.name?.includes("wooden_pickaxe"));
  const hasStonePickaxe = !!world.findInventorySlot((s) => !!s.name?.includes("stone_pickaxe"));
  const hasWoodenAxe = !!world.findInventorySlot((s) => !!s.name?.includes("wooden_axe"));
  const hasWoodenSword = !!world.findInventorySlot((s) => !!s.name?.includes("wooden_sword"));
  const hasCraftingTable = !!world.findInventorySlot((s) => !!s.name?.includes("crafting_table"));
  const hasFurnace = !!world.findInventorySlot((s) => !!s.name?.includes("furnace"));

  const FOOD_KEYS = ["bread", "apple", "beef", "porkchop", "chicken", "mutton", "carrot", "potato", "melon", "cod", "salmon"];
  const hasFood = FOOD_KEYS.some((k) => counts(k) > 0);

  const threat = world.nearestHostile(20);
  const threatNearby = !!threat;

  // Use geometric semantics: trees = vertical pillars, stone = dominant non-air id near surface.
  const nearTree = findNearbyTree(world) !== null;
  const nearStone = findNearbyStone(world) !== null;

  return {
    ...DEFAULT_STATE,
    wood, planks, sticks, cobblestone, coal, iron,
    hasWoodenPickaxe, hasStonePickaxe, hasWoodenAxe, hasWoodenSword,
    hasCraftingTable, hasFurnace,
    hunger: world.self.food,
    health: world.self.health,
    hasFood,
    nearTree, nearStone, threatNearby,
    isDay: true, // TODO: derive from time-of-day packet
    hasShelter: false, // TODO: derive from a placed-blocks tracker
  };
}
