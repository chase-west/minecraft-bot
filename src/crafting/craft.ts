import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { RecipeRegistry, RecipeEntry } from "./registry.js";
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

function findIngredient(world: World, ingredient: { name?: string }): { slot: number; stackId: number } | null {
  if (!ingredient.name) return null;
  for (const [slot, item] of world.inventory.entries()) {
    if (item.name?.includes(ingredient.name) && item.count > 0) {
      // stackId would be tracked from inventory_slot packets; default 0 = "any/ignore" works on most servers.
      return { slot, stackId: 0 };
    }
  }
  return null;
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

  // Map ingredients to their current inventory slots.
  const consumes: Array<{ slot: number; stackId: number; count: number }> = [];
  for (const ing of recipe.inputs) {
    const found = findIngredient(world, ing);
    if (!found) {
      return { crafted: false, reason: `missing_ingredient:${ing.name}` };
    }
    consumes.push({ ...found, count: (ing.count ?? 1) * timesCrafted });
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
        slot_type: { container_id: CONTAINER_INVENTORY, dynamic_container_id: 0 },
        slot: c.slot,
        stack_id: c.stackId,
      },
    });
  }
  actions.push({
    type_id: "take",
    count: (recipe.outputCount ?? 1) * timesCrafted,
    source: {
      slot_type: { container_id: CONTAINER_CREATIVE_OUTPUT, dynamic_container_id: 0 },
      slot: 50,
      stack_id: 0,
    },
    destination: {
      slot_type: { container_id: CONTAINER_HOTBAR, dynamic_container_id: 0 },
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
