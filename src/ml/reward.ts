import type { World } from "../world/world.js";
import { findNearbyTree } from "../world/semantic.js";
import { isLogId } from "../world/logIds.js";
import { ActionId } from "./actions.js";

interface Snapshot {
  ts: number;
  health: number;
  food: number;
  x: number;
  z: number;
  totalCount: number;
  populatedItemIds: Set<number>;
}

/** Clamp a value into [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** True when a block's name marks it as wood/log material. Robust to undefined. */
function isLogBlock(name: string | undefined): boolean {
  if (!name) return false;
  return name.includes("log");
}

/** True when an inventory slot holds a wood resource. Robust to undefined names. */
function isWoodItem(name: string | undefined): boolean {
  if (!name) return false;
  return name.includes("log") || name.includes("wood") || name.includes("planks");
}

/**
 * Generic shaped reward. Keeps the original generic signals (movement, health,
 * food, death, item discovery) and layers on dense wood-mining shaping so the
 * policy is guided toward finding, facing, and chopping trees.
 *
 * Signals are tuned for ~10–20 Hz call rate.
 */
export class RewardCalculator {
  private prev: Snapshot | null = null;
  private dead = false;
  // Horizontal distance to the nearest tree on the previous tick. null when no
  // tree was visible last tick, so we don't emit a spurious approach delta the
  // first time one appears.
  private prevTreeDist: number | null = null;
  // Count of wood items (logs/wood/planks) held on the previous tick.
  private prevWoodCount = 0;

  step(world: World, ts: number, lastAction?: ActionId): number {
    const snap = this.snapshot(world, ts);
    let reward = 0;

    // No alive bonus — previously +0.01/tick caused a local-minimum where the
    // policy learned to spam Jump forever (no risk, free reward). Movement now
    // earns the only baseline positive signal.

    if (this.prev) {
      // Movement reward: positive for actual horizontal displacement. Caps at
      // 0.5 blocks/tick (sprint speed ≈ 0.28) so a teleport correction doesn't
      // dump a giant reward. Encourages exploration without dwarfing item
      // discovery.
      const dx = snap.x - this.prev.x;
      const dz = snap.z - this.prev.z;
      const dist = Math.min(Math.sqrt(dx * dx + dz * dz), 0.5);
      reward += dist * 0.05;

      // Health/hunger deltas.
      const dHealth = snap.health - this.prev.health;
      const dFood = snap.food - this.prev.food;
      if (dHealth < 0) reward += dHealth * 0.1;       // -0.1 per HP lost
      if (dFood < 0) reward += dFood * 0.5;           // -0.5 per food point lost

      // Death detection (transition from alive to zero HP).
      if (this.prev.health > 0 && snap.health <= 0 && !this.dead) {
        reward -= 50;
        this.dead = true;
      } else if (snap.health > 0 && this.dead) {
        // Respawned; clear the latch so we can detect future deaths.
        this.dead = false;
      }

      // Total count delta — small positive signal per item gained.
      const dTotal = snap.totalCount - this.prev.totalCount;
      if (dTotal > 0) reward += dTotal * 0.05;

      // New dense-item-id slots becoming populated. Generic stand-in for
      // "discovered a new resource"; the trainer reads this as a curiosity
      // bonus without hard-coding e.g. "log".
      for (const id of snap.populatedItemIds) {
        if (!this.prev.populatedItemIds.has(id)) {
          reward += 0.5;
        }
      }
    }

    // --- Dense wood-mining shaping ---------------------------------------

    // APPROACH: reward closing horizontal distance to the nearest tree. Only
    // pays out when we have a previous distance to compare against, so the
    // first tick a tree appears contributes nothing.
    const tree = findNearbyTree(world);
    if (tree) {
      const tdx = tree.x - snap.x;
      const tdz = tree.z - snap.z;
      const d = Math.sqrt(tdx * tdx + tdz * tdz);
      if (this.prevTreeDist !== null) {
        reward += clamp((this.prevTreeDist - d) * 0.3, -0.5, 0.5);
      }
      this.prevTreeDist = d;
    } else {
      this.prevTreeDist = null;
    }

    // FACING A LOG: small bonus when a log sits within ~3 blocks straight ahead
    // at eye level. Cheap signal that nudges the bot to line up on a trunk.
    if (this.logAhead(world, 3)) {
      reward += 0.1;
    }

    // MINE-AIMED: reward actually swinging at a log we are pointed at. The mine
    // primitive only connects when a trunk is close in front, so this credits
    // the right behaviour even before any item lands in the inventory.
    if (lastAction === ActionId.MineFront && this.logAhead(world, 4.5)) {
      reward += 0.3;
    }

    // WOOD GAIN: the big payoff. Count logs/wood/planks in the inventory and
    // reward any increase. Decreases (crafting, dropping) are ignored.
    const woodCount = this.woodCount(world);
    const dWood = woodCount - this.prevWoodCount;
    if (dWood > 0) reward += dWood * 3.0;
    this.prevWoodCount = woodCount;

    this.prev = snap;
    return reward;
  }

  /**
   * True when a log block lies along the bot's facing direction within
   * `maxDist` blocks at roughly eye level. Steps forward in 0.5-block
   * increments and checks each cell. Robust to undefined blocks/names.
   *
   * Uses the same Bedrock yaw convention as actions.ts/frontBlock: forward is
   * (sin(-yawRad), cos(yawRad)) with yaw in degrees.
   */
  private logAhead(world: World, maxDist: number): boolean {
    const yawRad = (world.self.yaw * Math.PI) / 180;
    const fx = Math.sin(-yawRad);
    const fz = Math.cos(yawRad);
    const ox = world.self.position.x;
    const oy = world.self.position.y;
    const oz = world.self.position.z;
    for (let d = 1.0; d <= maxDist; d += 0.5) {
      const bx = Math.floor(ox + fx * d);
      const bz = Math.floor(oz + fz * d);
      // Check eye level and the block just below it so a slightly-off pitch
      // still registers a trunk. Match on runtime id first (the decoder gives
      // ids, not names, so the name check below is usually a no-op) and fall
      // back to the name substring for any registry-backed blocks.
      for (const by of [Math.floor(oy + 1), Math.floor(oy)]) {
        const b = world.getBlock({ x: bx, y: by, z: bz });
        if (b && (isLogId(b.runtimeId) || isLogBlock(b.name))) return true;
      }
    }
    return false;
  }

  /** Total count of wood items (logs/wood/planks) in the inventory. */
  private woodCount(world: World): number {
    let total = 0;
    for (const slot of world.inventory.values()) {
      if (slot.count > 0 && isWoodItem(slot.name)) total += slot.count;
    }
    return total;
  }

  private snapshot(world: World, ts: number): Snapshot {
    let totalCount = 0;
    const populated: Set<number> = new Set();
    for (const slot of world.inventory.values()) {
      if (slot.count > 0) {
        totalCount += slot.count;
        populated.add(slot.networkId);
      }
    }
    return {
      ts,
      health: world.self.health,
      food: world.self.food,
      x: world.self.position.x,
      z: world.self.position.z,
      totalCount,
      populatedItemIds: populated,
    };
  }

  /** Reset internal memory — useful when a new episode begins. */
  reset(): void {
    this.prev = null;
    this.dead = false;
    this.prevTreeDist = null;
    this.prevWoodCount = 0;
  }
}
