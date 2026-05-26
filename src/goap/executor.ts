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
import { isStandable, landingY } from "../pathfinding/safety.js";
import { v3floor } from "../utils/vec3.js";
import type { Vec3 } from "../utils/vec3.js";
import { makeLogger } from "../utils/logger.js";
import { RecipeRegistry } from "../crafting/registry.js";
import { craftByOutputName } from "../crafting/craft.js";
import { openCraftingTable, closeContainer, isContainerOpen } from "../crafting/container.js";
import { findNearbyTree, findNearbyStone, blacklistTree } from "../world/semantic.js";

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
    // Navigate to a standable spot ADJACENT to the trunk, not the log itself
    // (a solid block A* can't route into). If we can't find footing, fall back
    // to the base so at least we get close.
    const goal = findTreeStandSpot(world, tree) ?? tree;
    const r = await navigateTo(world, input, goal, { timeoutMs: 60_000 });
    return { ok: r.arrived, reason: r.reason };
  },

  ChopTree: async (ctx) => {
    const { client, world, input } = ctx;
    if (selectByName(client, world, "axe")) { /* prefer axe */ }
    const tree = findNearbyTree(world) ?? findNearbyBlockByNameContains(world, "log");
    if (!tree) return { ok: false, reason: "no_tree" };
    const trunkBlock = world.getBlock(tree);
    const trunkId = trunkBlock?.runtimeId;
    // A tree is a VERTICAL COLUMN of logs from tree.y upward. We can chop ANY log in
    // the column, so build the trunk column and target the log nearest to our own
    // height — the base may be on a ledge we can't path down to.
    const isLog = (b: { runtimeId: number; name?: string } | undefined): boolean =>
      !!b && ((trunkId !== undefined && b.runtimeId === trunkId) || !!b.name?.includes("log"));
    const column: Vec3[] = [];
    for (let dy = 0; dy <= 6; dy++) {
      const p = { x: tree.x, y: tree.y + dy, z: tree.z };
      if (!isLog(world.getBlock(p))) break;
      column.push(p);
    }
    if (column.length === 0) column.push({ x: tree.x, y: tree.y, z: tree.z });

    // Stand on solid footing ADJACENT to the trunk before mining. Never mine
    // from on top of / inside the trunk (that desyncs the bot and the server
    // rejects the break). If we can't find footing from here, the tree is too
    // far to perceive well — navigate toward the trunk first (A* stops at the
    // nearest reachable cell, since the log itself is solid), then re-evaluate.
    let standSpot = findTreeStandSpot(world, tree);
    if (!standSpot) {
      log.info(`tree at (${tree.x},${tree.y},${tree.z}): no footing from (${world.self.position.x.toFixed(1)},${world.self.position.y.toFixed(1)},${world.self.position.z.toFixed(1)}); navigating closer`);
      await navigateTo(world, input, tree, { timeoutMs: 12_000, maxNodes: 30_000 });
      standSpot = findTreeStandSpot(world, tree);
    }
    log.info(`tree at (${tree.x},${tree.y},${tree.z}) col=${column.length}; standSpot=${standSpot ? `(${standSpot.x},${standSpot.y},${standSpot.z})` : "none"}; bot at (${world.self.position.x.toFixed(1)},${world.self.position.y.toFixed(1)},${world.self.position.z.toFixed(1)})`);
    if (!standSpot) { blacklistTree(tree); return { ok: false, reason: "no_stand_spot" }; }
    const nav = await navigateTo(world, input, standSpot, { timeoutMs: 12_000, maxNodes: 30_000 });
    if (!nav.arrived) { blacklistTree(tree); return { ok: false, reason: `nav_to_tree:${nav.reason ?? "unknown"}` }; }

    // Mine the column bottom-up, skipping logs out of reach from where we
    // actually stand (the base may be buried in a hillside). mineBlock()
    // stabilizes (stops + waits for onGround) before each break. Tolerate a
    // couple of misses rather than bailing on the first.
    let chopped = 0;
    let fails = 0;
    for (const block of column) {
      if (!isLog(world.getBlock(block))) continue;
      const ddx = block.x + 0.5 - world.self.position.x;
      const ddy = block.y + 0.5 - (world.self.position.y + 1.62);
      const ddz = block.z + 0.5 - world.self.position.z;
      if (Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) > 5.0) continue; // out of reach
      const idBefore = world.getBlock(block)?.runtimeId;
      const r = await mineBlock(client, world, input, block);
      log.info(`chop (${block.x},${block.y},${block.z}) id=${idBefore} broken=${r.broken}`);
      if (r.broken) chopped++;
      else if (++fails >= 2) break;
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

/**
 * Pick a block the bot can actually STAND on near the trunk — never the trunk
 * column itself. Navigating to a log position (a solid block) makes A* fail to
 * route ("no path"), and the straight-line fallback then walks the bot INTO the
 * trunk, desyncing it (the server thinks it's inside a log) so every break is
 * rejected. The four immediate neighbours are often blocked by the canopy
 * (isSolid counts leaves as solid), so we scan a radius-2 area at the local
 * floor (via landingY) for a genuinely clear, ground-supported spot from which
 * at least one trunk log is within break reach, and return the one closest to
 * the bot. Returns null only if nothing nearby is standable.
 */
function findTreeStandSpot(world: World, base: Vec3): Vec3 | null {
  const REACH = 4.8; // Bedrock survival break reach is ~5-6 blocks; stay conservative.
  const trunkId = world.getBlock(base)?.runtimeId;
  const isLog = (b: { runtimeId: number; name?: string } | undefined): boolean =>
    !!b && ((trunkId !== undefined && b.runtimeId === trunkId) || !!b.name?.includes("log"));
  // Build the trunk column so we can test reach against any of its logs.
  const column: Vec3[] = [];
  for (let dy = 0; dy <= 8; dy++) {
    const p = { x: base.x, y: base.y + dy, z: base.z };
    if (!isLog(world.getBlock(p))) break;
    column.push(p);
  }
  if (column.length === 0) column.push(base);

  const bp = world.self.position;
  let best: Vec3 | null = null;
  let bestScore = Infinity;
  for (let dx = -2; dx <= 2; dx++) {
    for (let dz = -2; dz <= 2; dz++) {
      if (dx === 0 && dz === 0) continue; // never the trunk column itself
      // Find the floor in this column, scanning down from above the canopy base.
      const floorY = landingY(world, { x: base.x + dx, y: base.y + 4, z: base.z + dz }, 10);
      if (floorY === null) continue;
      const spot = { x: base.x + dx, y: floorY, z: base.z + dz };
      if (!isStandable(world, spot)) continue;
      // Need at least one trunk log within break reach from the eye.
      const eyeY = spot.y + 1.62;
      const reachable = column.some((lp) => {
        const ddx = lp.x + 0.5 - (spot.x + 0.5);
        const ddy = lp.y + 0.5 - eyeY;
        const ddz = lp.z + 0.5 - (spot.z + 0.5);
        return Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) <= REACH;
      });
      if (!reachable) continue;
      const score =
        Math.abs(spot.x + 0.5 - bp.x) + Math.abs(spot.z + 0.5 - bp.z) + Math.abs(spot.y - bp.y);
      if (score < bestScore) { bestScore = score; best = spot; }
    }
  }
  return best;
}

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
