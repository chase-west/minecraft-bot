import type { World } from "../world/world.js";

interface Snapshot {
  ts: number;
  health: number;
  food: number;
  x: number;
  z: number;
  totalCount: number;
  populatedItemIds: Set<number>;
}

/**
 * Generic shaped reward. Avoids hardcoded item names so the same calculator
 * works across BC and online RL. Signals are tuned for ~10–20 Hz call rate.
 */
export class RewardCalculator {
  private prev: Snapshot | null = null;
  private dead = false;

  step(world: World, ts: number): number {
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

    this.prev = snap;
    return reward;
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
  }
}
