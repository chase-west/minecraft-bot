import type { Action, BotState, ExecContext, Plan } from "./types.js";
import type { World } from "../world/world.js";
import type { InputController } from "../actions/input.js";
import type { BedrockClient } from "../connection/client.js";
import { eat, findFood } from "../actions/eat.js";
import { mineBlock } from "../actions/mine.js";
import { placeBlock } from "../actions/place.js";
import { attackEntity, fleeFrom } from "../actions/combat.js";
import { selectByName } from "../actions/inventory.js";
import { navigateTo } from "../pathfinding/executor.js";
import { v3floor } from "../utils/vec3.js";
import { makeLogger } from "../utils/logger.js";
import { RecipeRegistry } from "../crafting/registry.js";
import { craftByOutputName } from "../crafting/craft.js";
import { openCraftingTable, closeContainer, isContainerOpen } from "../crafting/container.js";
import { findNearbyTree, findNearbyStone } from "../world/semantic.js";

const log = makeLogger("goap-exec");

export interface RuntimeCtx extends ExecContext {
  client: BedrockClient;
  world: World;
  input: InputController;
  recipes: RecipeRegistry;
}

async function ensureTableOpen(ctx: RuntimeCtx, requireTable: boolean): Promise<{ ok: boolean; reason?: string }> {
  if (!requireTable) return { ok: true };
  if (isContainerOpen()) return { ok: true };
  const tablePos = findNearbyBlockByNameContains(ctx.world, "crafting_table");
  if (!tablePos) return { ok: false, reason: "no_crafting_table_nearby" };
  const r = await openCraftingTable(ctx.client, ctx.world, tablePos);
  return r.opened ? { ok: true } : { ok: false, reason: r.reason };
}

async function doCraft(ctx: RuntimeCtx, outputSubstr: string, requireTable: boolean): Promise<{ ok: boolean; reason?: string }> {
  if (!ctx.recipes.isReady()) return { ok: false, reason: "recipes_not_loaded" };
  const opened = await ensureTableOpen(ctx, requireTable);
  if (!opened.ok) return opened;
  const r = await craftByOutputName(ctx.client, ctx.world, ctx.recipes, outputSubstr, 1, !requireTable);
  return { ok: r.crafted, reason: r.reason };
}

type Impl = (ctx: RuntimeCtx) => Promise<{ ok: boolean; reason?: string }>;

/**
 * Maps each GOAP action name to its concrete implementation.
 * Actions can fail at runtime even if their preconditions held — the executor
 * surfaces failures so the agent re-plans.
 */
export const IMPLEMENTATIONS: Record<string, Impl> = {
  EatFood: async ({ client, world }) => {
    if (!findFood(world)) return { ok: false, reason: "no_food_runtime" };
    const r = await eat(client, world);
    return { ok: r.ate, reason: r.reason };
  },

  Flee: async ({ world, input }) => {
    const threat = world.nearestHostile(24);
    if (!threat) return { ok: true };
    await fleeFrom(world, input, threat, 20, 15_000);
    const stillThere = world.nearestHostile(20);
    return { ok: !stillThere, reason: stillThere ? "still_pursued" : undefined };
  },

  FightHostile: async ({ client, world, input }) => {
    const threat = world.nearestHostile(20);
    if (!threat) return { ok: true };
    if (!selectByName(client, world, "sword")) selectByName(client, world, "axe");
    const r = await attackEntity(client, world, input, threat, { timeoutMs: 20_000 });
    return { ok: r.killed, reason: r.reason };
  },

  FindTree: async ({ world, input }) => {
    // Geometric tree detection (works without block-name registry — see world/semantic.ts).
    const tree = findNearbyTree(world);
    if (!tree) {
      // Fall back to name-substring if we ever do have names (legacy/registry-backed path).
      const fallback = findNearbyBlockByNameContains(world, "log");
      if (!fallback) return { ok: false, reason: "no_tree_visible" };
      const r = await navigateTo(world, input, fallback, { timeoutMs: 60_000 });
      return { ok: r.arrived, reason: r.reason };
    }
    const r = await navigateTo(world, input, tree, { timeoutMs: 60_000 });
    return { ok: r.arrived, reason: r.reason };
  },

  ChopTree: async (ctx) => {
    const { client, world, input } = ctx;
    if (selectByName(client, world, "axe")) { /* prefer axe */ }
    const tree = findNearbyTree(world) ?? findNearbyBlockByNameContains(world, "log");
    if (!tree) return { ok: false, reason: "no_tree" };
    log.info(`tree found at (${tree.x},${tree.y},${tree.z}); bot at (${world.self.position.x.toFixed(1)},${world.self.position.y.toFixed(1)},${world.self.position.z.toFixed(1)}); world has ${world.blocks.size} blocks`);
    // Try four cardinal stand-spots; first reachable wins. Each candidate sits at trunk.y
    // (the trunk's base) so we're at eye-height with the bottom log.
    const candidates = [
      { x: tree.x + 1, y: tree.y, z: tree.z },
      { x: tree.x - 1, y: tree.y, z: tree.z },
      { x: tree.x, y: tree.y, z: tree.z + 1 },
      { x: tree.x, y: tree.y, z: tree.z - 1 },
    ];
    let arrived = false;
    let lastReason: string | undefined;
    for (const spot of candidates) {
      const nav = await navigateTo(world, input, spot, { timeoutMs: 30_000, maxNodes: 30_000 });
      if (nav.arrived) { arrived = true; break; }
      lastReason = nav.reason;
    }
    if (!arrived) return { ok: false, reason: `nav_to_tree:${lastReason ?? "unknown"}` };
    const trunkBlock = world.getBlock(tree);
    const trunkId = trunkBlock?.runtimeId;
    let chopped = 0;
    for (let dy = 0; dy < 7; dy++) {
      const block = { x: tree.x, y: tree.y + dy, z: tree.z };
      const here = world.getBlock(block);
      if (!here) break;
      const matches = (trunkId !== undefined && here.runtimeId === trunkId) || !!here.name?.includes("log");
      if (!matches) break;
      const r = await mineBlock(client, world, input, block);
      if (r.broken) chopped++;
      else break;
    }
    return { ok: chopped > 0, reason: chopped === 0 ? "no_progress" : undefined };
  },

  FindStone: async ({ world, input }) => {
    const stone = findNearbyStone(world) ?? findNearbyBlockByNameContains(world, "stone");
    if (!stone) return { ok: false, reason: "no_stone_visible" };
    const r = await navigateTo(world, input, stone, { timeoutMs: 90_000 });
    return { ok: r.arrived, reason: r.reason };
  },

  MineStone: async ({ client, world, input }) => {
    selectByName(client, world, "pickaxe");
    let mined = 0;
    for (let i = 0; i < 4; i++) {
      const candidate = findNearbyStone(world) ?? findNearbyBlockByNameContains(world, "stone");
      if (!candidate) break;
      const r = await mineBlock(client, world, input, candidate);
      if (r.broken) mined++;
      else break;
    }
    return { ok: mined > 0 };
  },

  HuntFood: async () => ({ ok: false, reason: "not_implemented_yet" }),

  // Crafting: 2x2 grid for planks/sticks/table; 3x3 (table) for tools.
  // Each impl auto-opens a nearby crafting table when required, then closes after.
  CraftPlanks: async (ctx) => doCraft(ctx, "planks", false),
  CraftSticks: async (ctx) => doCraft(ctx, "stick", false),
  CraftCraftingTable: async (ctx) => {
    const r = await doCraft(ctx, "crafting_table", false);
    if (r.ok) {
      // Place the table next to us so subsequent crafts can open it.
      const here = v3floor(ctx.world.self.position);
      const target = { x: here.x + 1, y: here.y, z: here.z };
      const against = { x: here.x, y: here.y - 1, z: here.z };
      selectByName(ctx.client, ctx.world, "crafting_table");
      await placeBlock(ctx.client, ctx.world, ctx.input, target, against);
    }
    return r;
  },
  CraftWoodenPickaxe: async (ctx) => {
    const r = await doCraft(ctx, "wooden_pickaxe", true);
    await closeContainer(ctx.client);
    return r;
  },
  CraftWoodenAxe: async (ctx) => {
    const r = await doCraft(ctx, "wooden_axe", true);
    await closeContainer(ctx.client);
    return r;
  },
  CraftWoodenSword: async (ctx) => {
    const r = await doCraft(ctx, "wooden_sword", true);
    await closeContainer(ctx.client);
    return r;
  },
  CraftStonePickaxe: async (ctx) => {
    const r = await doCraft(ctx, "stone_pickaxe", true);
    await closeContainer(ctx.client);
    return r;
  },
  CraftFurnace: async (ctx) => {
    const r = await doCraft(ctx, "furnace", true);
    await closeContainer(ctx.client);
    return r;
  },

  PlaceShelter: async ({ client, world, input }) => {
    // Place a 3-block tall wall around current position using whatever placeable we have.
    const placeable = selectByName(client, world, "cobblestone") || selectByName(client, world, "planks")
      || selectByName(client, world, "dirt");
    if (!placeable) return { ok: false, reason: "no_block" };
    const here = v3floor(world.self.position);
    const offsets = [
      { dx: 1, dz: 0 }, { dx: -1, dz: 0 }, { dx: 0, dz: 1 }, { dx: 0, dz: -1 },
    ];
    let placed = 0;
    for (const off of offsets) {
      for (let dy = 0; dy < 3; dy++) {
        const target = { x: here.x + off.dx, y: here.y + dy, z: here.z + off.dz };
        const against = { x: here.x, y: here.y + dy, z: here.z };
        const r = await placeBlock(client, world, input, target, against);
        if (r.placed) placed++;
      }
    }
    return { ok: placed > 0, reason: placed === 0 ? "no_placements" : undefined };
  },
};

function findNearbyBlockByNameContains(world: World, needle: string): { x: number; y: number; z: number } | null {
  let best: { x: number; y: number; z: number } | null = null;
  let bestD = Infinity;
  for (const [key, block] of world.blocks.entries()) {
    if (!block.name || !block.name.includes(needle)) continue;
    const [xs, ys, zs] = key.split(",");
    const x = Number(xs), y = Number(ys), z = Number(zs);
    const dx = x - world.self.position.x;
    const dy = y - world.self.position.y;
    const dz = z - world.self.position.z;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) { bestD = d; best = { x, y, z }; }
  }
  return best;
}

export async function executePlan(plan: Plan, ctx: RuntimeCtx): Promise<{ completed: number; failedAt: number | null; reason?: string }> {
  for (let i = 0; i < plan.steps.length; i++) {
    const a = plan.steps[i]!;
    const impl = IMPLEMENTATIONS[a.name];
    if (!impl) {
      log.warn(`no runtime impl for action ${a.name}`);
      return { completed: i, failedAt: i, reason: "no_impl" };
    }
    log.info(`exec[${i}] ${a.name}`);
    const r = await impl(ctx);
    if (!r.ok) {
      log.warn(`exec[${i}] ${a.name} FAILED (${r.reason})`);
      return { completed: i, failedAt: i, reason: r.reason };
    }
  }
  return { completed: plan.steps.length, failedAt: null };
}
