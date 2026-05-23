/** Bot-domain world state (planning-level, not voxel-level). */
export interface BotState {
  // Resources
  wood: number;
  planks: number;
  sticks: number;
  cobblestone: number;
  coal: number;
  iron: number;

  // Tools
  hasWoodenPickaxe: boolean;
  hasStonePickaxe: boolean;
  hasWoodenAxe: boolean;
  hasWoodenSword: boolean;
  hasCraftingTable: boolean;
  hasFurnace: boolean;

  // Survival
  hunger: number;     // 0..20
  health: number;     // 0..20
  hasFood: boolean;
  hasShelter: boolean;

  // Awareness
  nearTree: boolean;
  nearStone: boolean;
  threatNearby: boolean;
  isDay: boolean;
}

export type StatePatch = Partial<BotState>;

export type Predicate = (s: BotState) => boolean;

export interface Action {
  name: string;
  /** Either a literal patch to check, or a free-form predicate. */
  preconditions: StatePatch | Predicate;
  /** Either a literal patch or a function returning a patch (lets effects depend on state). */
  effects: StatePatch | ((s: BotState) => StatePatch);
  cost: number | ((s: BotState) => number);
  /** Runtime implementation hook — bound separately so planner stays pure. */
  exec?: (ctx: ExecContext) => Promise<{ ok: boolean; reason?: string }>;
}

export interface Goal {
  name: string;
  priority(s: BotState): number; // higher = more urgent
  satisfied(s: BotState): boolean;
  heuristic(s: BotState): number; // admissible underestimate of cost-to-goal
}

export interface Plan {
  steps: Action[];
  totalCost: number;
}

export interface ExecContext {
  // Filled in by the bot runtime; keeps planner agnostic.
  [k: string]: unknown;
}

export const DEFAULT_STATE: BotState = {
  wood: 0,
  planks: 0,
  sticks: 0,
  cobblestone: 0,
  coal: 0,
  iron: 0,
  hasWoodenPickaxe: false,
  hasStonePickaxe: false,
  hasWoodenAxe: false,
  hasWoodenSword: false,
  hasCraftingTable: false,
  hasFurnace: false,
  hunger: 20,
  health: 20,
  hasFood: false,
  hasShelter: false,
  nearTree: false,
  nearStone: false,
  threatNearby: false,
  isDay: true,
};
