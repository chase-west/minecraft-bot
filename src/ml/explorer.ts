import { ActionId } from "./actions.js";

/**
 * Exploration policy for bootstrap data collection. Drives the bot with a
 * "sticky random walk" — pick an action, hold it for a short duration, then
 * pick another. No goal-specific logic: the bot wanders, occasionally tries
 * mining/placing/jumping, and the trajectory logger captures whatever happens
 * with reward signals from the world. Useful both for:
 *   - generating diverse training data that ISN'T 99% Noop
 *   - making the bot visibly move while we collect data
 *
 * Once a learned policy exists, this should be mixed in via epsilon-greedy
 * (policy.act with epsilon=0.1) rather than replacing the policy outright.
 */
const MOVE_ACTIONS = [ActionId.MoveN, ActionId.MoveS, ActionId.MoveE, ActionId.MoveW] as const;

interface ActionWeight {
  action: ActionId;
  weight: number;
  minMs: number;
  maxMs: number;
}

/** Inventory snapshot used to bias exploration toward feasible crafts. */
export interface CraftHint {
  logs: number;
  planks: number;
  sticks: number;
  hasTable: boolean;
}

// Heavily biased toward movement so the bot actually goes places. Mining/placing
// included so they appear in the training distribution.
const WEIGHTS: ActionWeight[] = [
  { action: ActionId.MoveN, weight: 25, minMs: 800, maxMs: 2500 },
  { action: ActionId.MoveS, weight: 25, minMs: 800, maxMs: 2500 },
  { action: ActionId.MoveE, weight: 25, minMs: 800, maxMs: 2500 },
  { action: ActionId.MoveW, weight: 25, minMs: 800, maxMs: 2500 },
  { action: ActionId.Jump, weight: 8, minMs: 100, maxMs: 300 },
  { action: ActionId.ToggleSprint, weight: 4, minMs: 100, maxMs: 200 },
  { action: ActionId.MineFront, weight: 12, minMs: 1500, maxMs: 3000 },
  { action: ActionId.PlaceFront, weight: 3, minMs: 200, maxMs: 500 },
  { action: ActionId.AttackNearest, weight: 4, minMs: 600, maxMs: 1500 },
  { action: ActionId.Eat, weight: 2, minMs: 1500, maxMs: 2500 },
  { action: ActionId.MoveForwardJump, weight: 10, minMs: 400, maxMs: 1200 },
  { action: ActionId.Noop, weight: 2, minMs: 100, maxMs: 300 },
  // Craft macros. Low base weight (tried occasionally even with no materials);
  // boosted hard by applyHint() when the bot actually has the ingredients, so
  // exploration generates real crafting transitions for the DQN to learn from.
  { action: ActionId.CraftPlanks, weight: 4, minMs: 1500, maxMs: 3000 },
  { action: ActionId.CraftSticks, weight: 3, minMs: 1500, maxMs: 3000 },
  { action: ActionId.CraftCraftingTable, weight: 2, minMs: 1500, maxMs: 3000 },
  { action: ActionId.CraftWoodenPickaxe, weight: 2, minMs: 1500, maxMs: 3000 },
];

const TOTAL_WEIGHT = WEIGHTS.reduce((s, w) => s + w.weight, 0);

// When the bot holds the right materials, sharply raise the weight of the
// craft that becomes feasible so exploration actually triggers it (and the
// tech-tree reward fires). Everything else keeps its base weight. This only
// biases exploration; the DQN still has to learn the action values.
function applyHint(hint: CraftHint): ActionWeight[] {
  return WEIGHTS.map((w) => {
    let weight = w.weight;
    if (w.action === ActionId.CraftPlanks && hint.logs >= 1) weight = 40;
    else if (w.action === ActionId.CraftSticks && hint.planks >= 2) weight = 30;
    else if (w.action === ActionId.CraftCraftingTable && hint.planks >= 4 && !hint.hasTable) weight = 25;
    else if (w.action === ActionId.CraftWoodenPickaxe && hint.planks >= 3 && hint.sticks >= 2 && hint.hasTable) weight = 35;
    return { ...w, weight };
  });
}

export class Explorer {
  private currentAction: ActionId = ActionId.Noop;
  private untilTs = 0;
  private actionsExecuted = 0;
  private lastChosenAt = 0;

  /** Returns the action to execute at this tick. If we're still inside the
   * current sticky window, returns the same action; otherwise picks fresh. */
  nextAction(now = Date.now(), hint?: CraftHint): ActionId {
    if (now < this.untilTs) return this.currentAction;
    const choice = this.pickWeighted(hint);
    this.currentAction = choice.action;
    this.untilTs = now + this.randIn(choice.minMs, choice.maxMs);
    this.actionsExecuted++;
    this.lastChosenAt = now;
    return this.currentAction;
  }

  private pickWeighted(hint?: CraftHint): ActionWeight {
    const weights = hint ? applyHint(hint) : WEIGHTS;
    const total = hint ? weights.reduce((s, w) => s + w.weight, 0) : TOTAL_WEIGHT;
    let r = Math.random() * total;
    for (const w of weights) {
      r -= w.weight;
      if (r <= 0) return w;
    }
    return weights[0]!;
  }

  private randIn(lo: number, hi: number): number {
    return lo + Math.random() * (hi - lo);
  }

  stats(): { actionsExecuted: number; currentAction: ActionId; remainingMs: number } {
    return {
      actionsExecuted: this.actionsExecuted,
      currentAction: this.currentAction,
      remainingMs: Math.max(0, this.untilTs - Date.now()),
    };
  }
}
