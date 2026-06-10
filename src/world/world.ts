import type { Vec3 } from "../utils/vec3.js";
import { v3key, v3floor } from "../utils/vec3.js";

export interface BlockInfo {
  runtimeId: number;
  name?: string;
}

export interface EntityInfo {
  runtimeEntityId: bigint;
  uniqueId?: bigint;
  type: string;
  position: Vec3;
  velocity: Vec3;
  yaw: number;
  pitch: number;
  health?: number;
  isHostile?: boolean;
  isPlayer?: boolean;
  username?: string;
  lastSeenTickMs: number;
}

export interface InventorySlot {
  networkId: number;
  count: number;
  name?: string;
  nbt?: unknown;
}

export interface SelfState {
  runtimeEntityId: bigint | null;
  position: Vec3;
  velocity: Vec3;
  yaw: number;
  pitch: number;
  headYaw: number;
  health: number;
  maxHealth: number;
  food: number;
  saturation: number;
  experienceLevel: number;
  gameMode: number;
  onGround: boolean;
  inWater: boolean;
  inLava: boolean;
  dimension: number;
}

export type BlockMap = Map<string, BlockInfo>;

export const AIR_RUNTIME_ID = 0;

export class World {
  readonly blocks: BlockMap = new Map();
  readonly entities: Map<string, EntityInfo> = new Map(); // key = runtimeEntityId as string
  readonly inventory: Map<number, InventorySlot> = new Map(); // slot index → contents
  // Item palette from start_game.itemstates (<=1.21.50) or the item_registry
  // packet (1.21.60+): item network id → name ("oak_log", no "minecraft:"
  // prefix). Wire-format item stacks carry only numeric ids, so every
  // name-based lookup (crafting, rewards, selectByName) depends on this map.
  readonly itemNames: Map<number, string> = new Map();
  readonly self: SelfState = {
    runtimeEntityId: null,
    position: { x: 0, y: 0, z: 0 },
    velocity: { x: 0, y: 0, z: 0 },
    yaw: 0,
    pitch: 0,
    headYaw: 0,
    health: 20,
    maxHealth: 20,
    food: 20,
    saturation: 5,
    experienceLevel: 0,
    gameMode: 0,
    onGround: true,
    inWater: false,
    inLava: false,
    dimension: 0,
  };

  selectedHotbarSlot = 0;

  setBlock(pos: Vec3, info: BlockInfo): void {
    this.blocks.set(v3key(v3floor(pos)), info);
  }

  getBlock(pos: Vec3): BlockInfo | undefined {
    return this.blocks.get(v3key(v3floor(pos)));
  }

  isBlockKnown(pos: Vec3): boolean {
    return this.blocks.has(v3key(v3floor(pos)));
  }

  isAir(pos: Vec3): boolean {
    const b = this.getBlock(pos);
    return !b || b.runtimeId === AIR_RUNTIME_ID;
  }

  setEntity(e: EntityInfo): void {
    this.entities.set(String(e.runtimeEntityId), e);
  }

  removeEntity(runtimeId: bigint): void {
    this.entities.delete(String(runtimeId));
  }

  nearestHostile(maxDist = 32): EntityInfo | null {
    let best: EntityInfo | null = null;
    let bestD = maxDist * maxDist;
    const p = this.self.position;
    for (const e of this.entities.values()) {
      if (!e.isHostile) continue;
      const dx = e.position.x - p.x, dy = e.position.y - p.y, dz = e.position.z - p.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bestD) { bestD = d2; best = e; }
    }
    return best;
  }

  /** Record the item palette. Entries look like {name: "minecraft:oak_log", runtime_id: n}. */
  registerItemStates(states: Array<{ name?: string; runtime_id?: number }>): void {
    for (const s of states) {
      if (!s || typeof s.runtime_id !== "number" || !s.name) continue;
      const name = s.name.startsWith("minecraft:") ? s.name.slice("minecraft:".length) : s.name;
      this.itemNames.set(s.runtime_id, name);
    }
  }

  /** Name for an item network id, or undefined if the palette hasn't loaded. */
  itemName(networkId: number): string | undefined {
    return this.itemNames.get(networkId);
  }

  itemCount(name: string): number {
    let total = 0;
    for (const slot of this.inventory.values()) {
      if (slot.name === name) total += slot.count;
    }
    return total;
  }

  findInventorySlot(predicate: (s: InventorySlot) => boolean): number | undefined {
    for (const [idx, slot] of this.inventory.entries()) {
      if (predicate(slot)) return idx;
    }
    return undefined;
  }
}
