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
 *  - A "tree" is a vertical pillar of 4+ identical blocks whose runtime id is a
 *    KNOWN LOG (see logIds.ts). Logs are positively identified by id, so we no
 *    longer rely on a leaf-canopy heuristic that false-positived on stone pillars.
 *  - "Stone" / "common mining target": runs of the dominant ground id below the surface.
 *
 * We use these heuristics to identify trees and stone without ever knowing the names.
 *
 * Caveat: this is approximate. False positives can include cactus pillars (no leaves
 * cap → we won't flag those) and tall flowers (too short → filtered by height ≥ 4).
 */

import type { Vec3 } from "../utils/vec3.js";
import type { World } from "./world.js";
import { isAirRuntimeId } from "./decoder.js";
import { isLogId, addLogId } from "./logIds.js";

// Trees the bot tried and failed to reach, suppressed for a while so it stops
// fixating on an unreachable one (e.g. a tree down a pit) and looks elsewhere.
const treeBlacklist = new Map<string, number>(); // "x,y,z" -> expiry ms
export function blacklistTree(pos: Vec3, ms = 90_000): void {
  treeBlacklist.set(`${pos.x},${pos.y},${pos.z}`, Date.now() + ms);
}
function isTreeBlacklisted(x: number, y: number, z: number): boolean {
  const exp = treeBlacklist.get(`${x},${y},${z}`);
  if (exp === undefined) return false;
  if (Date.now() > exp) { treeBlacklist.delete(`${x},${y},${z}`); return false; }
  return true;
}

/** Returns the most-common non-zero runtime ID seen in the world map (the "ground" block). */
export function inferGroundId(world: World): number | null {
  const counts = new Map<number, number>();
  let scanned = 0;
  for (const block of world.blocks.values()) {
    if (isAirRuntimeId(block.runtimeId)) continue;
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
    if (isAirRuntimeId(block.runtimeId)) continue;
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
          // A real tree stands in open air; a buried cluster (ore/dirt pocket,
          // structure) is encased in solid blocks. Require ≥1 trunk block to
          // have an air neighbour, which rejects underground false positives.
          if (!trunkHasAirNeighbor(world, x, trunkBottom, trunkTop, z)) {
            runStart = i;
            continue;
          }
          // Identify the trunk as a log. Fast path: a known log runtime id.
          // Otherwise fall back to a species-agnostic shape test — a real tree
          // has a leaf canopy above the trunk, which stone pillars and spawn
          // structures lack. This lets the bot recognize any wood species
          // without a hardcoded id table; we learn the id on first sighting so
          // later detections (and the chopper) take the fast path.
          if (!isLogId(trunkId)) {
            // Unknown id: accept only if it looks like a real trunk — a leaf
            // canopy above, solid ground beneath the base, AND a THIN column.
            // The thin-column test is what stops us learning a leaf id: a log
            // trunk is one block wide, so its sides are air/leaves/ground, never
            // another trunk block; a leaf inside a canopy is flanked by same-id
            // leaves. Without it, a vertical run of leaves whose base rests on
            // another block (e.g. the canopy left behind after the trunk is
            // chopped, or leaves drooping onto terrain) passes the canopy +
            // grounded gates, gets addLogId'd, and the bot then strips canopies
            // forever instead of trunks.
            if (
              !hasLeafCanopy(world, x, trunkTop, z, groundId) ||
              !isGrounded(world, x, trunkBottom, z) ||
              !isThinColumn(world, x, trunkBottom, trunkTop, z, trunkId)
            ) {
              runStart = i;
              continue;
            }
            addLogId(trunkId);
          }
          if (isTreeBlacklisted(x, trunkBottom, z)) {
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

/** True if any block in the trunk column [bottomY,topY] has an air block on one of
 * its 4 horizontal sides — i.e. the trunk stands in open air (a surface tree), not
 * buried in solid terrain. world.getBlock returning undefined (unknown) also counts
 * as "not solid" so partially-perceived surface trees still pass. */
function trunkHasAirNeighbor(world: World, x: number, bottomY: number, topY: number, z: number): boolean {
  const dirs = [{ dx: 1, dz: 0 }, { dx: -1, dz: 0 }, { dx: 0, dz: 1 }, { dx: 0, dz: -1 }];
  for (let y = bottomY; y <= topY; y++) {
    for (const d of dirs) {
      const b = world.getBlock({ x: x + d.dx, y, z: z + d.dz });
      if (!b || isAirRuntimeId(b.runtimeId)) return true;
    }
  }
  return false;
}

/** True if the vertical run [bottomY,topY] at (x,z) is a THIN column: none of
 * the 4 horizontal neighbours, sampled at every height of the run, share the
 * run's runtime id. A log trunk is one block wide, so its sides are air, leaves
 * (a different id), or ground. A leaf sitting inside a canopy is surrounded by
 * same-id leaves, so a vertical leaf run fails this test. This is the check that
 * keeps addLogId from ever learning a leaf id as a log. Wide trunks (2x2 jungle/
 * dark oak) also fail it, so they aren't auto-learned — an acceptable trade since
 * the alternative is stripping canopies, and known log ids skip this path. */
function isThinColumn(world: World, x: number, bottomY: number, topY: number, z: number, id: number): boolean {
  const dirs = [{ dx: 1, dz: 0 }, { dx: -1, dz: 0 }, { dx: 0, dz: 1 }, { dx: 0, dz: -1 }];
  for (let y = bottomY; y <= topY; y++) {
    for (const d of dirs) {
      const b = world.getBlock({ x: x + d.dx, y, z: z + d.dz });
      if (b && b.runtimeId === id) return false;
    }
  }
  return true;
}

/** True if the column base sits on a solid (non-air) block. Real tree trunks are
 * rooted in the ground; a floating run of leaves has air beneath it. Treats an
 * unperceived (undefined) block below as NOT grounded so we don't accept leaf
 * blobs whose support simply hasn't been decoded. */
function isGrounded(world: World, x: number, baseY: number, z: number): boolean {
  const below = world.getBlock({ x, y: baseY - 1, z });
  return !!below && !isAirRuntimeId(below.runtimeId);
}

/** True if there's a leaf-like canopy around the trunk top: several non-air,
 * non-ground blocks in the box above (x, trunkTop, z). Real trees have a bushy
 * crown; bare stone/dirt pillars and most spawn structures don't. Species-
 * agnostic, so it identifies logs of any wood type without a runtime-id table. */
function hasLeafCanopy(world: World, x: number, trunkTop: number, z: number, groundId: number | null): boolean {
  let leaves = 0;
  for (let dy = 0; dy <= 3; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      for (let dz = -2; dz <= 2; dz++) {
        if (dx === 0 && dz === 0) continue; // skip the trunk column itself
        const b = world.getBlock({ x: x + dx, y: trunkTop + dy, z: z + dz });
        if (!b || isAirRuntimeId(b.runtimeId)) continue;
        if (groundId !== null && b.runtimeId === groundId) continue;
        if (++leaves >= 6) return true;
      }
    }
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
