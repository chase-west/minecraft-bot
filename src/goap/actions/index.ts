import type { Action } from "../types.js";
import { eatFoodAction, placeShelterAction, fleeAction, fightAction } from "./survival.js";
import {
  findTreeAction, chopTreeAction, findStoneAction, mineStoneAction, huntFoodAction,
} from "./gather.js";
import {
  craftPlanksAction, craftSticksAction, craftCraftingTableAction,
  craftWoodenPickaxeAction, craftWoodenAxeAction, craftWoodenSwordAction,
  craftStonePickaxeAction, craftFurnaceAction,
} from "./craft.js";

export const ALL_ACTIONS: Action[] = [
  eatFoodAction, placeShelterAction, fleeAction, fightAction,
  findTreeAction, chopTreeAction, findStoneAction, mineStoneAction, huntFoodAction,
  craftPlanksAction, craftSticksAction, craftCraftingTableAction,
  craftWoodenPickaxeAction, craftWoodenAxeAction, craftWoodenSwordAction,
  craftStonePickaxeAction, craftFurnaceAction,
];
