import type { Action, BotState } from "../types.js";

export const findTreeAction: Action = {
  name: "FindTree",
  preconditions: (s: BotState) => !s.nearTree,
  effects: { nearTree: true },
  cost: 5,
};

export const chopTreeAction: Action = {
  name: "ChopTree",
  preconditions: (s: BotState) => s.nearTree,
  effects: (s: BotState) => ({ wood: s.wood + 4 }),
  cost: 4,
};

export const findStoneAction: Action = {
  name: "FindStone",
  preconditions: (s: BotState) => !s.nearStone && s.hasWoodenPickaxe,
  effects: { nearStone: true },
  cost: 6,
};

export const mineStoneAction: Action = {
  name: "MineStone",
  preconditions: (s: BotState) => s.nearStone && s.hasWoodenPickaxe,
  effects: (s: BotState) => ({ cobblestone: s.cobblestone + 4 }),
  cost: 5,
};

export const huntFoodAction: Action = {
  name: "HuntFood",
  preconditions: (s: BotState) => !s.hasFood,
  effects: { hasFood: true },
  cost: 8,
};
