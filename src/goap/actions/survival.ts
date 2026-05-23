import type { Action, BotState } from "../types.js";

export const eatFoodAction: Action = {
  name: "EatFood",
  preconditions: (s: BotState) => s.hasFood && s.hunger < 20,
  effects: (s: BotState) => ({ hunger: Math.min(20, s.hunger + 6) }),
  cost: 1,
};

export const placeShelterAction: Action = {
  name: "PlaceShelter",
  preconditions: (s: BotState) => s.cobblestone >= 16 || s.planks >= 16,
  effects: (s: BotState) => ({
    hasShelter: true,
    cobblestone: Math.max(0, s.cobblestone - 16),
    planks: s.cobblestone >= 16 ? s.planks : Math.max(0, s.planks - 16),
  }),
  cost: 6,
};

export const fleeAction: Action = {
  name: "Flee",
  preconditions: (s: BotState) => s.threatNearby,
  effects: { threatNearby: false },
  cost: 3,
};

export const fightAction: Action = {
  name: "FightHostile",
  preconditions: (s: BotState) => s.threatNearby && (s.hasWoodenSword || s.hasStonePickaxe),
  effects: { threatNearby: false },
  cost: 4,
};
