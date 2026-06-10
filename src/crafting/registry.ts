import type { BedrockClient } from "../connection/client.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("recipes");

export interface RecipeIngredient {
  name?: string;             // e.g. "oak_planks" (string_id_meta descriptors)
  tag?: string;              // e.g. "planks" (item_tag descriptors — any matching item works)
  networkId?: number;        // item network id (int_id_meta descriptors)
  count: number;
  metadata?: number;
}

export interface RecipeEntry {
  recipeId: string;          // e.g. "minecraft:oak_planks"
  networkId: number;         // server-assigned, used in craft_recipe action
  outputName?: string;       // e.g. "oak_planks"
  outputCount?: number;
  inputs: RecipeIngredient[];
  width?: number;            // 1..3
  height?: number;           // 1..3
  needsTable: boolean;       // true if pattern won't fit 2x2
}

/**
 * Listens for `crafting_data` packets (sent on join) and builds a recipe table
 * mapping recipe_id and output_name → network_id.
 *
 * The server assigns network_ids per session — they are NOT static. We MUST
 * receive this packet before crafting anything.
 */
export class RecipeRegistry {
  private readonly byRecipeId = new Map<string, RecipeEntry>();
  private readonly byOutput = new Map<string, RecipeEntry[]>();
  private ready = false;

  isReady(): boolean { return this.ready; }
  size(): number { return this.byRecipeId.size; }

  attach(client: BedrockClient): void {
    client.on("crafting_data", (pkt: any) => {
      const recipes = pkt.recipes ?? pkt.crafting_data ?? [];
      for (const r of recipes) {
        try {
          const entry = parseRecipe(r);
          if (!entry) continue;
          this.byRecipeId.set(entry.recipeId, entry);
          if (entry.outputName) {
            const arr = this.byOutput.get(entry.outputName) ?? [];
            arr.push(entry);
            this.byOutput.set(entry.outputName, arr);
          }
        } catch (err) {
          log.debug("skipped recipe entry", (err as Error).message);
        }
      }
      this.ready = true;
      log.info(`recipes loaded: ${this.byRecipeId.size}`);
    });
  }

  /** Find the simplest recipe that produces an item whose name contains `nameSubstr`. */
  findByOutputName(nameSubstr: string, preferNoTable = false): RecipeEntry | null {
    let best: RecipeEntry | null = null;
    let bestScore = -Infinity;
    for (const arr of this.byOutput.values()) {
      for (const r of arr) {
        if (!r.outputName?.includes(nameSubstr)) continue;
        let s = (r.outputCount ?? 1) - r.inputs.length;
        if (preferNoTable && !r.needsTable) s += 100;
        if (s > bestScore) { bestScore = s; best = r; }
      }
    }
    return best;
  }

  getByRecipeId(id: string): RecipeEntry | null {
    return this.byRecipeId.get(id) ?? null;
  }
}

function parseRecipe(r: any): RecipeEntry | null {
  // bedrock-protocol exposes recipes with shape: { type: 'shaped'|'shapeless'|..., recipe: {...} }
  const type = r.type ?? r.recipe_type;
  const data = r.recipe ?? r;
  if (!data) return null;
  const networkId = data.network_id ?? data.recipe_network_id;
  if (typeof networkId !== "number") return null;

  const recipeId = String(data.recipe_id ?? data.identifier ?? data.uuid ?? `unknown_${networkId}`);
  const width = data.width;
  const height = data.height;

  // Outputs
  const outputs = data.output ?? data.result ?? data.outputs ?? [];
  const firstOut = Array.isArray(outputs) ? outputs[0] : outputs;
  const outputName = stripMc(firstOut?.name ?? firstOut?.item_name);
  const outputCount = firstOut?.count ?? firstOut?.stack_size ?? 1;

  // Inputs. Shaped recipes deliver a width×height NESTED array of ingredients
  // (empty cells included); shapeless recipes deliver a flat array. Flatten
  // first, then aggregate identical ingredients so a 3-plank row becomes one
  // {planks, count:3} entry instead of three count:1 entries.
  const inputArr = data.input ?? data.ingredients ?? data.inputs ?? [];
  const flat: any[] = [];
  for (const i of (Array.isArray(inputArr) ? inputArr : [inputArr])) {
    if (Array.isArray(i)) flat.push(...i);
    else flat.push(i);
  }
  const byKey = new Map<string, RecipeIngredient>();
  for (const i of flat) {
    const ing = parseIngredient(i);
    if (!ing) continue;
    const key = ing.name ?? ing.tag ?? `id:${ing.networkId}`;
    const existing = byKey.get(key);
    if (existing) existing.count += ing.count;
    else byKey.set(key, ing);
  }
  const inputs = Array.from(byKey.values());

  // 2x2 capable when shaped pattern fits 2x2, or shapeless with ≤4 distinct inputs.
  const needsTable = type === "shaped" || type === "shaped_chemistry"
    ? ((width ?? 0) > 2 || (height ?? 0) > 2)
    : inputs.length > 4;

  return {
    recipeId,
    networkId,
    outputName,
    outputCount,
    inputs,
    width,
    height,
    needsTable,
  };
}

/**
 * Parse one RecipeIngredient from the wire. bedrock-protocol flattens the
 * descriptor variant into the ingredient object, so the shape depends on type:
 *   string_id_meta → { type, name, metadata, count }
 *   item_tag       → { type, tag, count }          (any item with the tag works)
 *   int_id_meta    → { type, network_id, metadata, count }
 *   invalid        → empty grid cell, skip
 * Older bedrock-protocol versions used { descriptor: {...} } nesting; keep
 * those fallbacks so the parser survives lib upgrades.
 */
function parseIngredient(i: any): RecipeIngredient | null {
  if (!i) return null;
  const kind = i.type ?? i.descriptor_type;
  if (kind === "invalid" || kind === "molang") return null;
  const count = typeof i.count === "number" && i.count > 0 ? i.count : 1;
  const name = stripMc(i.name ?? i.item_name ?? i.descriptor?.identifier ?? i.descriptor?.name);
  const tag = stripMc(i.tag ?? i.descriptor?.tag);
  const networkId = typeof i.network_id === "number" ? i.network_id : i.descriptor?.network_id;
  if (networkId === 0) return null; // empty slot in older formats
  if (!name && !tag && typeof networkId !== "number") return null;
  return { name, tag, networkId, count };
}

function stripMc(name?: string): string | undefined {
  if (!name) return undefined;
  return name.startsWith("minecraft:") ? name.slice("minecraft:".length) : name;
}
