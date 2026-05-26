import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { InputController } from "../actions/input.js";
import type { Vec3 } from "../utils/vec3.js";
import { mineBlock } from "../actions/mine.js";
import { placeBlock } from "../actions/place.js";
import { attackEntity } from "../actions/combat.js";
import { eat } from "../actions/eat.js";
import { selectByName } from "../actions/inventory.js";
import { v3floor } from "../utils/vec3.js";
import type { RecipeRegistry } from "../crafting/registry.js";
import { craftByOutputName } from "../crafting/craft.js";
import { openCraftingTable, closeContainer, isContainerOpen } from "../crafting/container.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("ml-act");

export enum ActionId {
  Noop = 0,
  MoveN = 1,
  MoveS = 2,
  MoveE = 3,
  MoveW = 4,
  Jump = 5,
  ToggleSprint = 6,
  MineFront = 7,
  PlaceFront = 8,
  AttackNearest = 9,
  Eat = 10,
  // Combined "walk into a hill" primitive — moves forward AND holds jump so
  // the bot can climb 1-block obstacles. Direction = whichever movement was
  // last issued (or N if none yet).
  MoveForwardJump = 11,
  // --- Craft macro-actions. These reuse the crafting primitives so the
  // learned policy can progress the early tech tree instead of only moving
  // and mining. Each runs an async craft across several ticks; the craftBusy
  // guard serialises overlapping ticks (see executeAction).
  CraftPlanks = 12,        // logs -> planks (2x2, no table)
  CraftSticks = 13,        // planks -> sticks (2x2, no table)
  CraftCraftingTable = 14, // planks -> crafting table (2x2), then placed adjacent
  CraftWoodenPickaxe = 15, // planks + sticks -> wooden pickaxe (3x3, opens a nearby table)
}

export const ACTION_COUNT = 16;

export interface ActionContext {
  client: BedrockClient;
  world: World;
  input: InputController;
  recipes: RecipeRegistry;
}

// Module-private state held across calls.
const state = {
  lastSprint: false,
  // Bedrock yaw 180 = north (matches the "or N if none yet" comment on
  // MoveForwardJump). Previously 0 = south, which contradicted that contract.
  lastMoveYaw: 180 as number, // remembers facing for MoveForwardJump
};

// Bedrock yaw convention: 0 = south (+Z), 90 = west (-X), 180 = north (-Z),
// 270 = east (+X). To move in a direction we face that direction and apply
// forward = 1, then the input controller's prediction step does the rest.
const DIR_YAW: Readonly<Record<ActionId, number | null>> = {
  [ActionId.Noop]: null,
  [ActionId.MoveN]: 180,
  [ActionId.MoveS]: 0,
  [ActionId.MoveE]: 270,
  [ActionId.MoveW]: 90,
  [ActionId.Jump]: null,
  [ActionId.ToggleSprint]: null,
  [ActionId.MineFront]: null,
  [ActionId.PlaceFront]: null,
  [ActionId.AttackNearest]: null,
  [ActionId.Eat]: null,
  [ActionId.MoveForwardJump]: null,
  [ActionId.CraftPlanks]: null,
  [ActionId.CraftSticks]: null,
  [ActionId.CraftCraftingTable]: null,
  [ActionId.CraftWoodenPickaxe]: null,
};

// Guards against overlapping mineBlock calls (the online tick fires MineFront
// fire-and-forget every 100ms; mineBlock runs for seconds). See MineFront case.
let mineBusy = false;

function frontBlock(world: World, distance = 1.5): Vec3 {
  const yawRad = (world.self.yaw * Math.PI) / 180;
  const px = world.self.position.x + Math.sin(-yawRad) * distance;
  const py = world.self.position.y;
  const pz = world.self.position.z + Math.cos(yawRad) * distance;
  return { x: Math.floor(px), y: Math.floor(py), z: Math.floor(pz) };
}

// Guards against overlapping craft macros. A craft runs async (open table,
// wait for the server's item_stack_response) across several 100ms ticks; the
// online tick fires fire-and-forget, so without this the same craft is sent
// repeatedly and the server thrashes. Mirrors mineBusy.
let craftBusy = false;

function isCraftAction(a: ActionId): boolean {
  return (
    a === ActionId.CraftPlanks ||
    a === ActionId.CraftSticks ||
    a === ActionId.CraftCraftingTable ||
    a === ActionId.CraftWoodenPickaxe
  );
}

/** Nearest known crafting_table block, or null. Mirrors the GOAP executor scan. */
function findNearbyCraftingTable(world: World): Vec3 | null {
  let best: Vec3 | null = null;
  let bestD = Infinity;
  for (const [key, block] of world.blocks.entries()) {
    if (!block.name || !block.name.includes("crafting_table")) continue;
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

/**
 * Run one craft via the existing crafting primitives. 2x2 recipes craft from
 * the inventory grid (no table); 3x3 recipes need a crafting table open first.
 * Returns silently on any failure (no materials, no table, recipes not loaded)
 * — the policy just learns those states yield no reward.
 */
async function doMlCraft(ctx: ActionContext, outputSubstr: string, requireTable: boolean): Promise<void> {
  if (!ctx.recipes.isReady()) return;
  if (requireTable && !isContainerOpen()) {
    const table = findNearbyCraftingTable(ctx.world);
    if (!table) return;
    const opened = await openCraftingTable(ctx.client, ctx.world, table);
    if (!opened.opened) return;
  }
  await craftByOutputName(ctx.client, ctx.world, ctx.recipes, outputSubstr, 1, !requireTable);
  if (requireTable) await closeContainer(ctx.client);
}

export async function executeAction(action: ActionId, ctx: ActionContext): Promise<void> {
  const { client, world, input } = ctx;

  // A craft macro is mid-flight (opening a table / awaiting the server). Don't
  // let movement or mining actions fight it; the craft case freezes movement.
  if (craftBusy && !isCraftAction(action)) return;

  switch (action) {
    case ActionId.Noop: {
      input.setMove({ forward: 0, strafe: 0, jump: false, sneak: false, sprint: state.lastSprint });
      return;
    }
    case ActionId.MoveN:
    case ActionId.MoveS:
    case ActionId.MoveE:
    case ActionId.MoveW: {
      const yaw = DIR_YAW[action];
      if (yaw !== null) state.lastMoveYaw = yaw;
      input.setMove({
        forward: 1,
        strafe: 0,
        jump: false,
        sneak: false,
        sprint: state.lastSprint,
        ...(yaw !== null ? { lookYaw: yaw } : {}),
      });
      return;
    }
    case ActionId.MoveForwardJump: {
      // Move in the direction we last faced AND hold jump — lets the bot
      // climb 1-block obstacles. Server only consumes jump when onGround,
      // so spamming this is safe.
      input.setMove({
        forward: 1,
        strafe: 0,
        jump: true,
        sneak: false,
        sprint: state.lastSprint,
        lookYaw: state.lastMoveYaw,
      });
      return;
    }
    case ActionId.Jump: {
      input.setMove({ forward: 0, strafe: 0, jump: true, sneak: false, sprint: state.lastSprint });
      return;
    }
    case ActionId.ToggleSprint: {
      state.lastSprint = !state.lastSprint;
      input.setMove({ sprint: state.lastSprint });
      return;
    }
    case ActionId.MineFront: {
      // One mine at a time. The online tick fires MineFront fire-and-forget
      // every 100ms, and mineBlock runs for seconds; without this guard the
      // overlapping calls all send start_break/predict_break at the same block
      // and the server thrashes start<->stop, never completing a break.
      if (mineBusy) return;
      mineBusy = true;
      const target = frontBlock(world);
      try {
        await mineBlock(client, world, input, target);
      } catch (err) {
        log.warn("mineFront failed", (err as Error).message);
      } finally {
        mineBusy = false;
      }
      return;
    }
    case ActionId.PlaceFront: {
      // Equip a placeable first — placeBlock no-ops on an empty / non-placeable
      // hand. Selection order mirrors the GOAP PlaceShelter action.
      const havePlaceable =
        selectByName(client, world, "cobblestone") ||
        selectByName(client, world, "planks") ||
        selectByName(client, world, "dirt") ||
        selectByName(client, world, "log");
      if (!havePlaceable) return;
      const target = frontBlock(world);
      const against: Vec3 = { x: target.x, y: target.y - 1, z: target.z };
      try {
        await placeBlock(client, world, input, target, against, 1);
      } catch (err) {
        log.warn("placeFront failed", (err as Error).message);
      }
      return;
    }
    case ActionId.AttackNearest: {
      const target = world.nearestHostile(16);
      if (!target) return;
      try {
        await attackEntity(client, world, input, target, { timeoutMs: 1500 });
      } catch (err) {
        log.warn("attackNearest failed", (err as Error).message);
      }
      return;
    }
    case ActionId.Eat: {
      try {
        await eat(client, world);
      } catch (err) {
        log.warn("eat failed", (err as Error).message);
      }
      return;
    }
    case ActionId.CraftPlanks:
    case ActionId.CraftSticks:
    case ActionId.CraftCraftingTable:
    case ActionId.CraftWoodenPickaxe: {
      if (craftBusy) return;
      craftBusy = true;
      // Stand still while crafting so movement packets don't desync the
      // table-open / item_stack_request handshake.
      input.setMove({ forward: 0, strafe: 0, jump: false, sneak: false, sprint: state.lastSprint });
      try {
        if (action === ActionId.CraftPlanks) {
          await doMlCraft(ctx, "planks", false);
        } else if (action === ActionId.CraftSticks) {
          await doMlCraft(ctx, "stick", false);
        } else if (action === ActionId.CraftCraftingTable) {
          await doMlCraft(ctx, "crafting_table", false);
          // Place the table next to us so CraftWoodenPickaxe can open it later.
          const here = v3floor(world.self.position);
          const target: Vec3 = { x: here.x + 1, y: here.y, z: here.z };
          const against: Vec3 = { x: here.x, y: here.y - 1, z: here.z };
          if (selectByName(client, world, "crafting_table")) {
            await placeBlock(client, world, input, target, against);
          }
        } else {
          await doMlCraft(ctx, "wooden_pickaxe", true);
        }
      } catch (err) {
        log.warn(`craft action ${action} failed`, (err as Error).message);
      } finally {
        craftBusy = false;
      }
      return;
    }
    default: {
      // Exhaustiveness: any unrecognised int is treated as noop.
      log.warn(`unknown action id ${action as number}`);
      return;
    }
  }
}
