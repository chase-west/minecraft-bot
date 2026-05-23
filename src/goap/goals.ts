import type { Goal, BotState } from "./types.js";

export const surviveGoal: Goal = {
  name: "Survive",
  priority: (s: BotState) => {
    // Very high if hungry/low health/threatened
    let p = 0;
    if (s.hunger < 8) p += 100 - s.hunger * 5;
    if (s.health < 12) p += 100 - s.health * 5;
    if (s.threatNearby) p += 80;
    if (!s.isDay && !s.hasShelter) p += 40;
    return p;
  },
  satisfied: (s: BotState) => s.hunger >= 16 && s.health >= 18 && !s.threatNearby,
  heuristic: (s: BotState) => Math.max(0, 16 - s.hunger) + Math.max(0, 18 - s.health) + (s.threatNearby ? 5 : 0),
};

export const earlyToolingGoal: Goal = {
  name: "EarlyTooling",
  priority: (s: BotState) => {
    let p = 20;
    if (!s.hasWoodenPickaxe) p += 30;
    if (!s.hasWoodenSword) p += 10;
    return p;
  },
  satisfied: (s: BotState) => s.hasWoodenPickaxe && s.hasWoodenSword && s.hasWoodenAxe,
  heuristic: (s: BotState) => {
    let h = 0;
    if (!s.hasWoodenPickaxe) h += 6;
    if (!s.hasWoodenSword) h += 4;
    if (!s.hasWoodenAxe) h += 4;
    return h;
  },
};

export const upgradeToolingGoal: Goal = {
  name: "UpgradeTooling",
  priority: (s: BotState) => (s.hasWoodenPickaxe && !s.hasStonePickaxe ? 25 : 0),
  satisfied: (s: BotState) => s.hasStonePickaxe && s.hasFurnace,
  heuristic: (s: BotState) => {
    let h = 0;
    if (!s.hasStonePickaxe) h += 5;
    if (!s.hasFurnace) h += 8;
    return h;
  },
};

export const shelterGoal: Goal = {
  name: "Shelter",
  priority: (s: BotState) => (s.hasShelter ? 0 : s.isDay ? 15 : 60),
  satisfied: (s: BotState) => s.hasShelter,
  heuristic: (s: BotState) => (s.hasShelter ? 0 : 6),
};

export const ALL_GOALS: Goal[] = [surviveGoal, earlyToolingGoal, upgradeToolingGoal, shelterGoal];
