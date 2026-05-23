import type { BedrockClient } from "./client.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("ver");

/**
 * Resilient feature flags derived once at spawn so packet writes can branch
 * on version without re-parsing version strings every tick. Refactor pattern
 * from PrismarineJS recommendation: check `client.versionGreaterThanOrEqualTo`
 * during init, then read these booleans hot-path.
 */
export interface VersionFlags {
  version: string;
  /** Item stack uses dynamic_container_id field. Added pre-1.20. */
  hasDynamicContainerId: boolean;
  /** SubChunk request mode (level_chunk sub_chunk_count < 0). 1.18+. */
  supportsSubChunkRequest: boolean;
  /** Player auth input includes analogue_move_vector. ~1.20+. */
  hasAnalogueMoveVector: boolean;
  /** Newer recipe format in crafting_data with recipe_unlocking_requirement. ~1.20+. */
  hasRecipeUnlocking: boolean;
}

export function detectVersionFlags(client: BedrockClient): VersionFlags {
  const c = client as any;
  const version = String(c.options?.version ?? c.version ?? "unknown");

  const gte = (v: string): boolean => {
    try {
      return typeof c.versionGreaterThanOrEqualTo === "function" && c.versionGreaterThanOrEqualTo(v);
    } catch {
      return true; // assume modern
    }
  };

  const flags: VersionFlags = {
    version,
    hasDynamicContainerId: true,
    supportsSubChunkRequest: gte("1.18.0"),
    hasAnalogueMoveVector: gte("1.20.0"),
    hasRecipeUnlocking: gte("1.20.10"),
  };
  log.info(`version flags`, flags);
  return flags;
}

/** Best-effort packet write that swallows schema-shape errors so the bot keeps running. */
export function safeQueue(client: BedrockClient, name: string, payload: unknown, tag = "?"): boolean {
  try {
    client.queue(name as any, payload as any);
    return true;
  } catch (err) {
    log.warn(`safeQueue(${name}, ${tag}) failed`, (err as Error).message);
    return false;
  }
}
