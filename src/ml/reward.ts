import type { World } from "../world/world.js";

interface Snapshot {
  ts: number;
  health: number;
  food: number;
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

    // Idle/urgency budget: +0.01 alive minus 0.001 mild urgency each tick.
    reward += 0.01;
    reward -= 0.001;

    if (this.prev) {
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
