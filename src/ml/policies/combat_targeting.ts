import type { World, EntityInfo } from "../../world/world.js";
import type { PolicyHandle } from "../runtime.js";
import { runPolicy } from "../runtime.js";

/**
 * Combat targeting policy: given the bot's stats and up to K hostile candidates,
 * scores each candidate. Highest score wins.
 *
 * Feature layout (per candidate, K=4):
 *   [self_health/20, self_hunger/20, has_sword, has_pickaxe,
 *    cand_distance/24, cand_dx/24, cand_dy/12, cand_dz/24,
 *    cand_is_creeper, cand_is_skeleton, cand_is_zombie, cand_is_other]
 *
 * Output: 4-vector of scores.
 *
 * When no ONNX model is loaded, falls back to a hand-tuned heuristic.
 */
const K = 4;
const FEATURES_PER = 12;

function encode(world: World, hostiles: EntityInfo[]): Float32Array {
  const feat = new Float32Array(K * FEATURES_PER);
  const hasSword = !!world.findInventorySlot((s) => !!s.name?.includes("sword"));
  const hasPick = !!world.findInventorySlot((s) => !!s.name?.includes("pickaxe"));
  for (let i = 0; i < K; i++) {
    const base = i * FEATURES_PER;
    const c = hostiles[i];
    feat[base + 0] = world.self.health / 20;
    feat[base + 1] = world.self.food / 20;
    feat[base + 2] = hasSword ? 1 : 0;
    feat[base + 3] = hasPick ? 1 : 0;
    if (!c) continue;
    const dx = c.position.x - world.self.position.x;
    const dy = c.position.y - world.self.position.y;
    const dz = c.position.z - world.self.position.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    feat[base + 4] = Math.min(1, dist / 24);
    feat[base + 5] = Math.max(-1, Math.min(1, dx / 24));
    feat[base + 6] = Math.max(-1, Math.min(1, dy / 12));
    feat[base + 7] = Math.max(-1, Math.min(1, dz / 24));
    feat[base + 8] = c.type.includes("creeper") ? 1 : 0;
    feat[base + 9] = c.type.includes("skeleton") ? 1 : 0;
    feat[base + 10] = c.type.includes("zombie") ? 1 : 0;
    feat[base + 11] = (feat[base + 8] === 0 && feat[base + 9] === 0 && feat[base + 10] === 0) ? 1 : 0;
  }
  return feat;
}

function heuristicScore(world: World, c: EntityInfo): number {
  const dx = c.position.x - world.self.position.x;
  const dy = c.position.y - world.self.position.y;
  const dz = c.position.z - world.self.position.z;
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
  let score = -dist; // prefer close targets
  if (c.type.includes("creeper")) score -= 8; // avoid creepers unless very close
  if (c.type.includes("skeleton")) score += 2;
  return score;
}

export async function pickTarget(
  world: World,
  policy: PolicyHandle | null,
  candidates: EntityInfo[],
): Promise<EntityInfo | null> {
  if (candidates.length === 0) return null;
  if (!policy) {
    let best: EntityInfo | null = null;
    let bestScore = -Infinity;
    for (const c of candidates) {
      const s = heuristicScore(world, c);
      if (s > bestScore) { bestScore = s; best = c; }
    }
    return best;
  }
  const top = candidates.slice(0, K);
  const features = encode(world, top);
  const scores = await runPolicy(policy, features, [1, K, FEATURES_PER]);
  let bestI = 0; let bestS = -Infinity;
  for (let i = 0; i < top.length; i++) {
    const s = scores[i] ?? -Infinity;
    if (s > bestS) { bestS = s; bestI = i; }
  }
  return top[bestI] ?? null;
}
