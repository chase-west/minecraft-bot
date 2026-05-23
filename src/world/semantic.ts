/**
 * Semantic block inference *without* a name table.
 *
 * Background: prismarine-registry ships no `blocksByRuntimeId` data for any Bedrock
 * version, and we can't statically map runtime IDs → block names. So we infer block
 * semantics from the WORLD GEOMETRY rather than the block dictionary.
 *
 * Key observations:
 *  - Air is always runtime_id 0.
 *  - The single most-common non-air block in any chunk slab IS the ground (stone, dirt,
 *    or sand depending on biome). Call its id `groundId`.
 *  - A "tree" is a vertical pillar of 4+ identical non-ground non-air blocks (the trunk),
 *    capped above by a contiguous blob of a DIFFERENT non-ground non-air id (leaves).
 *  - "Stone" / "common mining target": runs of the dominant ground id below the surface.
 *
 * We use these heuristics to identify trees and stone without ever knowing the names.
 *
 * Caveat: this is approximate. False positives can include cactus pillars (no leaves
 * cap → we won't flag those) and tall flowers (too short → filtered by height ≥ 4).
 */

import type { Vec3 } from "../utils/vec3.js";
import type { World } from "./world.js";

const AIR = 0;

/** Returns the most-common non-zero runtime ID seen in the world map (the "ground" block). */
export function inferGroundId(world: World): number | null {
  const counts = new Map<number, number>();
  let scanned = 0;
  for (const block of world.blocks.values()) {
    if (block.runtimeId === AIR) continue;
    counts.set(block.runtimeId, (counts.get(block.runtimeId) ?? 0) + 1);
    if (++scanned > 50_000) break;
  }
  let bestId = -1;
  let bestN = 0;
  for (const [id, n] of counts) {
    if (n > bestN) { bestN = n; bestId = id; }
  }
  return bestId === -1 ? null : bestId;
}

/**
 * Find the nearest tree trunk: a vertical column of ≥4 identical non-air non-ground
 * blocks. Returns the (x, y, z) of the lowest trunk block.
 *
 * Trees only exist near the surface, so we restrict the search to a vertical window
 * around the bot's y coordinate. Without this constraint, vertical pillars of stone
 * underground (cave walls) trigger false positives.
 */
export function findNearbyTree(world: World, maxRange = 32, maxYDelta = 16): Vec3 | null {
  const groundId = inferGroundId(world);
  const px = Math.floor(world.self.position.x);
  const py = Math.floor(world.self.position.y);
  const pz = Math.floor(world.self.position.z);

  // Index blocks by (x,z) → sorted list of y values per (x,z) column.
  const columns = new Map<string, Array<{ y: number; runtimeId: number }>>();
  for (const [key, block] of world.blocks.entries()) {
    if (block.runtimeId === AIR) continue;
    if (block.runtimeId === groundId) continue;
    const [xs, ys, zs] = key.split(",");
    const x = Number(xs), y = Number(ys), z = Number(zs);
    if (Math.abs(x - px) > maxRange || Math.abs(z - pz) > maxRange) continue;
    if (Math.abs(y - py) > maxYDelta) continue;
    const k = `${x},${z}`;
    let arr = columns.get(k);
    if (!arr) { arr = []; columns.set(k, arr); }
    arr.push({ y, runtimeId: block.runtimeId });
  }

  let best: Vec3 | null = null;
  let bestD = Infinity;
  for (const [k, arr] of columns) {
    if (arr.length < 4) continue;
    arr.sort((a, b) => a.y - b.y);
    let runStart = 0;
    for (let i = 1; i <= arr.length; i++) {
      const same = i < arr.length && arr[i]!.y === arr[i - 1]!.y + 1 && arr[i]!.runtimeId === arr[i - 1]!.runtimeId;
      if (!same) {
        const runLen = i - runStart;
        if (runLen >= 4) {
          const [xs, zs] = k.split(",");
          const x = Number(xs), z = Number(zs);
          const trunkBottom = arr[runStart]!.y;
          const trunkTop = arr[i - 1]!.y;
          const trunkId = arr[runStart]!.runtimeId;
          // Real trees have a canopy: a DIFFERENT non-ground non-trunk block within
          // the 3×3×3 cube immediately above the trunk top. Pillars/spawn structures
          // don't pass this filter.
          if (!hasCanopy(world, x, trunkTop, z, trunkId, groundId)) {
            runStart = i;
            continue;
          }
          const dx = x - px, dz = z - pz, dy = trunkBottom - py;
          const d = dx * dx + dz * dz + dy * dy;
          if (d < bestD) { bestD = d; best = { x, y: trunkBottom, z }; }
        }
        runStart = i;
      }
    }
  }
  return best;
}

function hasCanopy(world: World, x: number, trunkTopY: number, z: number, trunkId: number, groundId: number | null): boolean {
  // Check the 3×3 horizontal slabs at trunkTopY+1 and trunkTopY+2 for ≥3 leaf blocks
  // (different from trunk + ground).
  for (let dy = 1; dy <= 3; dy++) {
    let leafCount = 0;
    for (let ox = -2; ox <= 2; ox++) {
      for (let oz = -2; oz <= 2; oz++) {
        const b = world.blocks.get(`${x + ox},${trunkTopY + dy},${z + oz}`);
        if (!b || b.runtimeId === AIR) continue;
        if (b.runtimeId === trunkId) continue;
        if (groundId !== null && b.runtimeId === groundId) continue;
        leafCount++;
      }
    }
    if (leafCount >= 3) return true;
  }
  return false;
}

/**
 * Find the nearest "stone" — interpreted as the most-common ground block ID.
 * Returns a block position near surface level so the bot can navigate + mine it.
 */
export function findNearbyStone(world: World, maxRange = 32): Vec3 | null {
  const groundId = inferGroundId(world);
  if (groundId === null) return null;
  const px = Math.floor(world.self.position.x);
  const py = Math.floor(world.self.position.y);
  const pz = Math.floor(world.self.position.z);

  let best: Vec3 | null = null;
  let bestD = Infinity;
  for (const [key, block] of world.blocks.entries()) {
    if (block.runtimeId !== groundId) continue;
    const [xs, ys, zs] = key.split(",");
    const x = Number(xs), y = Number(ys), z = Number(zs);
    if (Math.abs(x - px) > maxRange || Math.abs(z - pz) > maxRange) continue;
    // Prefer blocks near the player's y level (avoid bedrock floor / sky).
    if (Math.abs(y - py) > 6) continue;
    const dx = x - px, dz = z - pz, dy = y - py;
    const d = dx * dx + dz * dz + dy * dy;
    if (d < bestD) { bestD = d; best = { x, y, z }; }
  }
  return best;
}
