import type { World, EntityInfo, InventorySlot } from "../world/world.js";
import type { BlockIdRegistry } from "./blockIdRegistry.js";
import { findNearbyTree } from "../world/semantic.js";

export const OBS_DIM = 605;

// Layout offsets
const SELF_OFFSET = 0;       // 8 floats
const SELF_LEN = 8;
const GRID_OFFSET = 8;       // 9*5*9 = 405 floats
const GRID_X = 9;
const GRID_Y = 5;
const GRID_Z = 9;
const GRID_LEN = GRID_X * GRID_Y * GRID_Z; // 405
const ENTITY_OFFSET = GRID_OFFSET + GRID_LEN; // 413
const ENTITY_FLOATS_EACH = 8;
const ENTITY_COUNT = 4;
const ENTITY_LEN = ENTITY_FLOATS_EACH * ENTITY_COUNT; // 32
const INV_OFFSET = ENTITY_OFFSET + ENTITY_LEN;        // 445
const INV_SLOT_COUNT = 32;
const INV_SLOT_FLOATS = 2;
const INV_SLOTS_LEN = INV_SLOT_COUNT * INV_SLOT_FLOATS; // 64
const INV_BAG_LEN = 92;
const INV_LEN = INV_SLOTS_LEN + INV_BAG_LEN; // 156
const TREE_OFFSET = INV_OFFSET + INV_LEN;             // 601
const TREE_LEN = 4;
const TREE_RANGE = 32;
const TREE_RECOMPUTE_MS = 500;

const ENTITY_RANGE = 16;
const ENTITY_RANGE_SQ = ENTITY_RANGE * ENTITY_RANGE;
const AGE_NORM_MS = 1200;

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

interface EntityCandidate {
  e: EntityInfo;
  d2: number;
  dx: number;
  dy: number;
  dz: number;
  dist: number;
}

export class Encoder {
  // findNearbyTree scans world.blocks, which is too heavy to run every 100ms
  // tick. Cache the last result and only rescan every TREE_RECOMPUTE_MS.
  private lastTree: { x: number; y: number; z: number } | null = null;
  private lastTreeAt = 0;

  /**
   * Encode the current world snapshot into a 605-float vector.
   *
   * The grid samples block runtime IDs and translates each to its dense
   * registry index. Entity and inventory blocks emit dense integer indices
   * as plain floats; the trainer's network is expected to apply embedding
   * lookups so we deliberately do NOT one-hot anything here.
   */
  encode(world: World, registry: BlockIdRegistry, scratch?: Float32Array): Float32Array {
    const out = scratch && scratch.length === OBS_DIM ? scratch : new Float32Array(OBS_DIM);
    if (scratch) out.fill(0);

    this.writeSelf(world, out);
    this.writeGrid(world, registry, out);
    this.writeEntities(world, registry, out);
    this.writeInventory(world, registry, out);
    this.writeTree(world, out);

    return out;
  }

  private writeSelf(world: World, out: Float32Array): void {
    const s = world.self;
    const base = SELF_OFFSET;
    out[base + 0] = clamp(s.health / 20, 0, 1);
    out[base + 1] = clamp(s.food / 20, 0, 1);
    out[base + 2] = clamp(s.saturation / 20, 0, 1);
    out[base + 3] = s.yaw / Math.PI;
    out[base + 4] = s.pitch / Math.PI;
    out[base + 5] = clamp(s.velocity.y, -2, 2);
    out[base + 6] = s.position.y / 256;
    out[base + 7] = s.onGround ? 1 : 0;
    // (suppress unused-len warning) SELF_LEN documents layout, not loop bound
    void SELF_LEN;
  }

  private writeGrid(world: World, registry: BlockIdRegistry, out: Float32Array): void {
    const px = Math.floor(world.self.position.x);
    const py = Math.floor(world.self.position.y);
    const pz = Math.floor(world.self.position.z);

    let idx = GRID_OFFSET;
    for (let dx = -4; dx <= 4; dx++) {
      for (let dy = -1; dy <= 3; dy++) {
        for (let dz = -4; dz <= 4; dz++) {
          const block = world.getBlock({ x: px + dx, y: py + dy, z: pz + dz });
          if (!block || block.runtimeId === 0) {
            out[idx] = 0; // air
          } else {
            out[idx] = registry.denseIndex(block.runtimeId);
          }
          idx++;
        }
      }
    }
  }

  private writeEntities(world: World, registry: BlockIdRegistry, out: Float32Array): void {
    const now = Date.now();
    const sp = world.self.position;
    const candidates: EntityCandidate[] = [];

    for (const e of world.entities.values()) {
      const dx = e.position.x - sp.x;
      const dy = e.position.y - sp.y;
      const dz = e.position.z - sp.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > ENTITY_RANGE_SQ) continue;
      candidates.push({ e, d2, dx, dy, dz, dist: Math.sqrt(d2) });
    }

    candidates.sort((a, b) => a.d2 - b.d2);
    const take = Math.min(ENTITY_COUNT, candidates.length);

    for (let i = 0; i < take; i++) {
      const c = candidates[i]!;
      const base = ENTITY_OFFSET + i * ENTITY_FLOATS_EACH;
      out[base + 0] = c.dx / ENTITY_RANGE;
      out[base + 1] = c.dy / ENTITY_RANGE;
      out[base + 2] = c.dz / ENTITY_RANGE;
      out[base + 3] = registry.entityTypeIndex(c.e.type);
      out[base + 4] = clamp((c.e.health ?? 0) / 20, 0, 1);
      out[base + 5] = c.e.isHostile ? 1 : 0;
      out[base + 6] = c.dist / ENTITY_RANGE;
      const ageMs = Math.max(0, now - c.e.lastSeenTickMs);
      out[base + 7] = clamp(ageMs / AGE_NORM_MS, 0, 1);
    }
    // Remaining slots already zeroed by Float32Array default or fill(0).
  }

  private writeInventory(world: World, registry: BlockIdRegistry, out: Float32Array): void {
    // Per-slot block: first INV_SLOT_COUNT slots ordered by slot index ascending.
    const sortedSlots: Array<[number, InventorySlot]> = Array.from(world.inventory.entries())
      .sort((a, b) => a[0] - b[0]);

    const slotsToEmit = Math.min(INV_SLOT_COUNT, sortedSlots.length);
    for (let i = 0; i < slotsToEmit; i++) {
      const entry = sortedSlots[i];
      if (!entry) continue;
      const slot = entry[1];
      const base = INV_OFFSET + i * INV_SLOT_FLOATS;
      out[base + 0] = registry.itemIdIndex(slot.networkId);
      out[base + 1] = clamp(slot.count / 64, 0, 1);
    }

    // Aggregate count-bag for dense item ids 0..INV_BAG_LEN-1.
    const bagBase = INV_OFFSET + INV_SLOTS_LEN;
    for (const slot of world.inventory.values()) {
      const di = registry.itemIdIndex(slot.networkId);
      if (di < 0 || di >= INV_BAG_LEN) continue;
      const cur = out[bagBase + di] ?? 0;
      out[bagBase + di] = cur + slot.count / 64;
    }
    // Clip the bag to [0,1] so the network sees a stable range.
    for (let i = 0; i < INV_BAG_LEN; i++) {
      const v = out[bagBase + i] ?? 0;
      out[bagBase + i] = clamp(v, 0, 1);
    }
  }

  private writeTree(world: World, out: Float32Array): void {
    // findNearbyTree walks world.blocks, so reuse the cached hit between
    // refreshes; rescan at most every TREE_RECOMPUTE_MS.
    const now = Date.now();
    if (now - this.lastTreeAt >= TREE_RECOMPUTE_MS) {
      this.lastTree = findNearbyTree(world);
      this.lastTreeAt = now;
    }

    const tree = this.lastTree;
    const base = TREE_OFFSET;
    const sp = world.self.position;
    if (tree) {
      const dx = tree.x - sp.x;
      const dz = tree.z - sp.z;
      const horizontalDist = Math.sqrt(dx * dx + dz * dz);
      out[base + 0] = 1;                              // tree exists
      out[base + 1] = clamp(dx / TREE_RANGE, -1, 1);  // east/west direction
      out[base + 2] = clamp(dz / TREE_RANGE, -1, 1);  // north/south direction
      out[base + 3] = clamp(horizontalDist / TREE_RANGE, 0, 1); // normalized distance
    } else {
      out[base + 0] = 0;
      out[base + 1] = 0;
      out[base + 2] = 0;
      out[base + 3] = 1; // no tree -> max (far) distance
    }
    // TREE_LEN documents the layout, not a loop bound.
    void TREE_LEN;
  }
}
