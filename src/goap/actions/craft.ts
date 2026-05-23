import type { Action, BotState } from "../types.js";

export const craftPlanksAction: Action = {
  name: "CraftPlanks",
  preconditions: (s: BotState) => s.wood >= 1,
  effects: (s: BotState) => ({ wood: s.wood - 1, planks: s.planks + 4 }),
  cost: 1,
};

export const craftSticksAction: Action = {
  name: "CraftSticks",
  preconditions: (s: BotState) => s.planks >= 2,
  effects: (s: BotState) => ({ planks: s.planks - 2, sticks: s.sticks + 4 }),
  cost: 1,
};

export const craftCraftingTableAction: Action = {
  name: "CraftCraftingTable",
  preconditions: (s: BotState) => s.planks >= 4 && !s.hasCraftingTable,
  effects: (s: BotState) => ({ planks: s.planks - 4, hasCraftingTable: true }),
  cost: 1,
};

export const craftWoodenPickaxeAction: Action = {
  name: "CraftWoodenPickaxe",
  preconditions: (s: BotState) => s.planks >= 3 && s.sticks >= 2 && s.hasCraftingTable && !s.hasWoodenPickaxe,
  effects: (s: BotState) => ({ planks: s.planks - 3, sticks: s.sticks - 2, hasWoodenPickaxe: true }),
  cost: 2,
};

export const craftWoodenAxeAction: Action = {
  name: "CraftWoodenAxe",
  preconditions: (s: BotState) => s.planks >= 3 && s.sticks >= 2 && s.hasCraftingTable && !s.hasWoodenAxe,
  effects: (s: BotState) => ({ planks: s.planks - 3, sticks: s.sticks - 2, hasWoodenAxe: true }),
  cost: 2,
};

export const craftWoodenSwordAction: Action = {
  name: "CraftWoodenSword",
  preconditions: (s: BotState) => s.planks >= 2 && s.sticks >= 1 && s.hasCraftingTable && !s.hasWoodenSword,
  effects: (s: BotState) => ({ planks: s.planks - 2, sticks: s.sticks - 1, hasWoodenSword: true }),
  cost: 2,
};

export const craftStonePickaxeAction: Action = {
  name: "CraftStonePickaxe",
  preconditions: (s: BotState) =>
    s.cobblestone >= 3 && s.sticks >= 2 && s.hasCraftingTable && !s.hasStonePickaxe,
  effects: (s: BotState) => ({ cobblestone: s.cobblestone - 3, sticks: s.sticks - 2, hasStonePickaxe: true }),
  cost: 2,
};

export const craftFurnaceAction: Action = {
  name: "CraftFurnace",
  preconditions: (s: BotState) => s.cobblestone >= 8 && s.hasCraftingTable && !s.hasFurnace,
  effects: (s: BotState) => ({ cobblestone: s.cobblestone - 8, hasFurnace: true }),
  cost: 2,
};
