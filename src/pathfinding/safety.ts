import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";

const HAZARD_NAMES = ["lava", "fire", "magma", "cactus", "wither_rose", "sweet_berry", "campfire", "cobweb"];

export function isHazard(world: World, pos: Vec3): boolean {
  const b = world.getBlock(pos);
  if (!b?.name) return false;
  return HAZARD_NAMES.some((h) => b.name!.includes(h));
}

export function isWater(world: World, pos: Vec3): boolean {
  const b = world.getBlock(pos);
  return !!b?.name?.includes("water");
}

export function isSolid(world: World, pos: Vec3): boolean {
  const b = world.getBlock(pos);
  if (!b) return false; // unknown ⇒ treat as non-solid for pathfinding (caller decides)
  if (b.runtimeId === 0) return false;
  if (!b.name) return true;
  if (b.name.includes("water") || b.name.includes("lava")) return false;
  if (b.name.includes("air")) return false;
  if (b.name.includes("flower") || b.name.includes("grass") || b.name.includes("tall_")) return false;
  if (b.name.includes("torch") || b.name.includes("lever") || b.name.includes("button")) return false;
  return true;
}

export function isStandable(world: World, pos: Vec3): boolean {
  // The bot occupies (pos.y) and (pos.y+1); requires solid at (pos.y-1).
  const below = { x: pos.x, y: pos.y - 1, z: pos.z };
  const here = pos;
  const head = { x: pos.x, y: pos.y + 1, z: pos.z };
  if (!isSolid(world, below)) return false;
  if (isSolid(world, here)) return false;
  if (isSolid(world, head)) return false;
  if (isHazard(world, below)) return false;
  return true;
}

/** Scan down from `top` until solid or maxDrop exceeded; returns the floor y (with bot standing on it) or null. */
export function landingY(world: World, top: Vec3, maxDrop = 4): number | null {
  for (let dy = 1; dy <= maxDrop + 1; dy++) {
    const p = { x: top.x, y: top.y - dy, z: top.z };
    if (isSolid(world, p)) {
      return p.y + 1;
    }
    if (isHazard(world, p)) return null;
  }
  return null;
}
