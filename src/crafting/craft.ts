import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { RecipeRegistry, RecipeEntry, RecipeIngredient } from "./registry.js";
import { nextRequestId } from "./request_id.js";
import { safeQueue } from "../connection/version.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("craft");

const CONTAINER_INVENTORY = "inventory";
const CONTAINER_HOTBAR = "hotbar";
const CONTAINER_CREATIVE_OUTPUT = "creative_output";
const CONTAINER_CRAFTING_INPUT = "crafting_input";

/**
 * Pending crafts indexed by request_id so we can resolve promises when the server
 * answers via item_stack_response.
 */
const pending = new Map<number, { resolve: (ok: boolean) => void; recipeId: string }>();

export function attachCraftResponses(client: BedrockClient): void {
  client.on("item_stack_response", (pkt: any) => {
    const responses = pkt.responses ?? [];
    for (const r of responses) {
      const id = r.request_id;
      const entry = pending.get(id);
      if (!entry) continue;
      pending.delete(id);
      const status = r.status ?? r.result;
      const ok = status === "ok" || status === 0 || status === undefined;
      if (!ok) log.warn(`craft rejected (${entry.recipeId}): status=${status}`);
      entry.resolve(ok);
    }
  });
}

interface CraftOptions {
  destHotbarSlot?: number;
  times?: number;
  timeoutMs?: number;
}

/**
 * Find a free hotbar slot to receive the crafted output (any slot whose contents
 * we don't currently track or have count=0).
 */
function findFreeHotbar(world: World): number {
  for (let s = 0; s < 9; s++) {
    const item = world.inventory.get(s);
    if (!item || item.count === 0) return s;
  }
  return 0; // overwrite slot 0 if nothing free
}

/**
 * True when an inventory item satisfies a recipe ingredient. Three descriptor
 * kinds: exact item network id, name substring, or item tag. Tags arrive like
 * "logs"/"planks"/"wooden_tool_materials"; we approximate with substring checks
 * against the singular form ("logs" matches "oak_log").
 */
function matchesIngredient(item: { networkId: number; name?: string }, ing: RecipeIngredient): boolean {
  if (typeof ing.networkId === "number" && ing.networkId === item.networkId) return true;
  if (!item.name) return false;
  if (ing.name && item.name.includes(ing.name)) return true;
  if (ing.tag) {
    const tag = ing.tag.endsWith("s") ? ing.tag.slice(0, -1) : ing.tag;
    if (item.name.includes(tag)) return true;
    // "logs_that_burn" / "wooden_*" style tags: match on the first segment.
    const head = tag.split("_")[0];
    if (head && head.length >= 3 && item.name.includes(head)) return true;
  }
  return false;
}

function findIngredient(world: World, ingredient: RecipeIngredient): { slot: number; stackId: number; available: number } | null {
  for (const [slot, item] of world.inventory.entries()) {
    if (item.count > 0 && matchesIngredient(item, ingredient)) {
      // stackId would be tracked from inventory_slot packets; default 0 = "any/ignore" works on most servers.
      return { slot, stackId: 0, available: item.count };
    }
  }
  return null;
}

/** item_stack_request addresses hotbar slots (0-8) and main inventory (9-35)
 * as separate containers, both with absolute slot indices. */
function containerForSlot(slot: number): string {
  return slot < 9 ? CONTAINER_HOTBAR : CONTAINER_INVENTORY;
}

export async function craft(
  client: BedrockClient,
  world: World,
  registry: RecipeRegistry,
  recipe: RecipeEntry,
  opts: CraftOptions = {},
): Promise<{ crafted: boolean; reason?: string }> {
  const requestId = nextRequestId();
  const destSlot = opts.destHotbarSlot ?? findFreeHotbar(world);
  const timesCrafted = opts.times ?? 1;

  // Map ingredients to their current inventory slots, merging ingredients that
  // resolve to the same slot (e.g. planks appearing in several grid cells).
  const bySlot = new Map<number, { slot: number; stackId: number; count: number; available: number }>();
  for (const ing of recipe.inputs) {
    const found = findIngredient(world, ing);
    if (!found) {
      return { crafted: false, reason: `missing_ingredient:${ing.name ?? ing.tag ?? ing.networkId}` };
    }
    const need = (ing.count ?? 1) * timesCrafted;
    const existing = bySlot.get(found.slot);
    if (existing) existing.count += need;
    else bySlot.set(found.slot, { ...found, count: need });
  }
  const consumes = Array.from(bySlot.values());
  for (const c of consumes) {
    if (c.count > c.available) {
      return { crafted: false, reason: `not_enough:slot${c.slot}:need${c.count}:have${c.available}` };
    }
  }

  // Build the action list.
  const actions: any[] = [
    {
      type_id: "craft_recipe",
      recipe_network_id: recipe.networkId,
      times_crafted: timesCrafted,
    },
  ];
  for (const c of consumes) {
    actions.push({
      type_id: "consume",
      count: c.count,
      source: {
        slot_type: { container_id: containerForSlot(c.slot), dynamic_container_id: undefined },
        slot: c.slot,
        stack_id: c.stackId,
      },
    });
  }
  actions.push({
    type_id: "take",
    count: (recipe.outputCount ?? 1) * timesCrafted,
    source: {
      slot_type: { container_id: CONTAINER_CREATIVE_OUTPUT, dynamic_container_id: undefined },
      slot: 50,
      stack_id: 0,
    },
    destination: {
      slot_type: { container_id: CONTAINER_HOTBAR, dynamic_container_id: undefined },
      slot: destSlot,
      stack_id: 0,
    },
  });

  log.info(`craft ${recipe.recipeId} (netId=${recipe.networkId} times=${timesCrafted} reqId=${requestId})`);

  const sent = safeQueue(client, "item_stack_request", {
    requests: [{
      request_id: requestId,
      actions,
      custom_names: [],
      cause: 0,
    }],
  }, recipe.recipeId);

  if (!sent) {
    return { crafted: false, reason: "queue_failed" };
  }

  // Wait for response.
  const timeoutMs = opts.timeoutMs ?? 4000;
  return await new Promise<{ crafted: boolean; reason?: string }>((resolve) => {
    const timer = setTimeout(() => {
      pending.delete(requestId);
      resolve({ crafted: false, reason: "timeout" });
    }, timeoutMs);
    pending.set(requestId, {
      recipeId: recipe.recipeId,
      resolve: (ok) => {
        clearTimeout(timer);
        resolve({ crafted: ok, reason: ok ? undefined : "rejected" });
      },
    });
  });
}

/** High-level: craft N copies of an output whose name contains `nameSubstr`. */
export async function craftByOutputName(
  client: BedrockClient,
  world: World,
  registry: RecipeRegistry,
  nameSubstr: string,
  count = 1,
  preferNoTable = false,
): Promise<{ crafted: boolean; reason?: string }> {
  if (!registry.isReady()) {
    return { crafted: false, reason: "recipes_not_loaded" };
  }
  const recipe = registry.findByOutputName(nameSubstr, preferNoTable);
  if (!recipe) {
    return { crafted: false, reason: `no_recipe:${nameSubstr}` };
  }
  return craft(client, world, registry, recipe, { times: count });
}
