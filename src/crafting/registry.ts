import type { BedrockClient } from "../connection/client.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("recipes");

export interface RecipeEntry {
  recipeId: string;          // e.g. "minecraft:oak_planks"
  networkId: number;         // server-assigned, used in craft_recipe action
  outputName?: string;       // e.g. "oak_planks"
  outputCount?: number;
  inputs: Array<{ name?: string; count?: number; metadata?: number }>;
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

  // Inputs
  const inputArr = data.input ?? data.ingredients ?? data.inputs ?? [];
  const inputs: RecipeEntry["inputs"] = [];
  for (const i of (Array.isArray(inputArr) ? inputArr : [inputArr])) {
    if (!i || i.network_id === 0) continue;
    inputs.push({
      name: stripMc(i.name ?? i.item_name ?? i.descriptor?.identifier),
      count: i.count ?? 1,
      metadata: i.metadata,
    });
  }

  // 2x2 capable when width≤2 AND height≤2 AND type is shapeless OR shaped 2x2
  const needsTable = type === "shaped" ? ((width ?? 0) > 2 || (height ?? 0) > 2) : false;

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

function stripMc(name?: string): string | undefined {
  if (!name) return undefined;
  return name.startsWith("minecraft:") ? name.slice("minecraft:".length) : name;
}
