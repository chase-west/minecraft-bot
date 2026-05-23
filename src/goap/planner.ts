import type { Action, BotState, Goal, Plan, StatePatch } from "./types.js";
import { MinHeap } from "../utils/heap.js";

function applyPatch(state: BotState, patch: StatePatch): BotState {
  return { ...state, ...patch };
}

function preconditionsMet(action: Action, state: BotState): boolean {
  const p = action.preconditions;
  if (typeof p === "function") return p(state);
  for (const k of Object.keys(p) as (keyof BotState)[]) {
    const expected = p[k];
    if (expected === undefined) continue;
    if (typeof expected === "boolean") {
      if ((state as any)[k] !== expected) return false;
    } else if (typeof expected === "number") {
      if (((state as any)[k] as number) < expected) return false; // numeric preconditions are "at least"
    }
  }
  return true;
}

function getEffects(action: Action, state: BotState): StatePatch {
  return typeof action.effects === "function" ? action.effects(state) : action.effects;
}

function getCost(action: Action, state: BotState): number {
  return typeof action.cost === "function" ? action.cost(state) : action.cost;
}

function stateHash(s: BotState): string {
  // Quantize floats to keep search space finite.
  return [
    s.wood, s.planks, s.sticks, s.cobblestone, s.coal, s.iron,
    s.hasWoodenPickaxe ? 1 : 0,
    s.hasStonePickaxe ? 1 : 0,
    s.hasWoodenAxe ? 1 : 0,
    s.hasWoodenSword ? 1 : 0,
    s.hasCraftingTable ? 1 : 0,
    s.hasFurnace ? 1 : 0,
    Math.floor(s.hunger), Math.floor(s.health),
    s.hasFood ? 1 : 0, s.hasShelter ? 1 : 0,
    s.nearTree ? 1 : 0, s.nearStone ? 1 : 0, s.threatNearby ? 1 : 0, s.isDay ? 1 : 0,
  ].join("|");
}

interface Node {
  state: BotState;
  g: number;
  f: number;
  parent: Node | null;
  action: Action | null;
}

export function plan(start: BotState, goal: Goal, actions: Action[], opts: { maxNodes?: number } = {}): Plan | null {
  const maxNodes = opts.maxNodes ?? 5000;
  const open = new MinHeap<Node>((a, b) => a.f - b.f);
  const closed = new Set<string>();
  const best = new Map<string, number>();

  const startNode: Node = { state: start, g: 0, f: goal.heuristic(start), parent: null, action: null };
  open.push(startNode);
  best.set(stateHash(start), 0);

  let expanded = 0;
  while (open.size > 0) {
    const cur = open.pop()!;
    const key = stateHash(cur.state);
    if (closed.has(key)) continue;
    closed.add(key);
    expanded++;
    if (expanded > maxNodes) break;

    if (goal.satisfied(cur.state)) {
      const steps: Action[] = [];
      let n: Node | null = cur;
      while (n && n.action) {
        steps.unshift(n.action);
        n = n.parent;
      }
      return { steps, totalCost: cur.g };
    }

    for (const a of actions) {
      if (!preconditionsMet(a, cur.state)) continue;
      const next = applyPatch(cur.state, getEffects(a, cur.state));
      const g = cur.g + getCost(a, cur.state);
      const nk = stateHash(next);
      const prev = best.get(nk);
      if (prev !== undefined && g >= prev) continue;
      best.set(nk, g);
      open.push({ state: next, g, f: g + goal.heuristic(next), parent: cur, action: a });
    }
  }

  return null;
}

export function selectGoal(state: BotState, goals: Goal[]): Goal | null {
  let best: Goal | null = null;
  let bestScore = -Infinity;
  for (const g of goals) {
    if (g.satisfied(state)) continue;
    const score = g.priority(state);
    if (score > bestScore) { best = g; bestScore = score; }
  }
  return best;
}
