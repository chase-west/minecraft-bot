import * as fs from "node:fs";
import * as path from "node:path";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("idreg");

const BLOCK_CAP = 1024; // 0..1022 = real, 1023 = overflow
const ENTITY_CAP = 64;  // 0..62 = real, 63 = overflow
const ITEM_CAP = 256;   // 0..254 = real, 255 = overflow

const BLOCK_OVERFLOW = BLOCK_CAP - 1;
const ENTITY_OVERFLOW = ENTITY_CAP - 1;
const ITEM_OVERFLOW = ITEM_CAP - 1;

interface PersistedTable<K> {
  entries: Array<[K, number]>;
}

/**
 * Maps unbounded runtime/network IDs and entity type strings into small
 * dense integer slots so an embedding layer can consume them. Assignments
 * are insertion-ordered and stable once persisted: a runtime ID that ever
 * received slot N keeps slot N forever (until the JSON is deleted).
 */
export class BlockIdRegistry {
  private blocks: Map<number, number> = new Map();
  private entities: Map<string, number> = new Map();
  private items: Map<number, number> = new Map();

  private blockNextIdx = 1; // reserve 0 for air
  private entityNextIdx = 0;
  private itemNextIdx = 1;  // reserve 0 for "empty slot"

  private dirty = false;

  constructor(private readonly dataDir: string = "data") {
    this.load();
  }

  denseIndex(runtimeId: number): number {
    if (runtimeId === 0) return 0; // air reserved
    const existing = this.blocks.get(runtimeId);
    if (existing !== undefined) return existing;
    if (this.blockNextIdx >= BLOCK_OVERFLOW) return BLOCK_OVERFLOW;
    const idx = this.blockNextIdx++;
    this.blocks.set(runtimeId, idx);
    this.dirty = true;
    return idx;
  }

  entityTypeIndex(type: string): number {
    const existing = this.entities.get(type);
    if (existing !== undefined) return existing;
    if (this.entityNextIdx >= ENTITY_OVERFLOW) return ENTITY_OVERFLOW;
    const idx = this.entityNextIdx++;
    this.entities.set(type, idx);
    this.dirty = true;
    return idx;
  }

  itemIdIndex(networkId: number): number {
    if (networkId === 0) return 0; // empty
    const existing = this.items.get(networkId);
    if (existing !== undefined) return existing;
    if (this.itemNextIdx >= ITEM_OVERFLOW) return ITEM_OVERFLOW;
    const idx = this.itemNextIdx++;
    this.items.set(networkId, idx);
    this.dirty = true;
    return idx;
  }

  /** Flush in-memory tables to JSON. Idempotent. */
  save(): void {
    if (!this.dirty) return;
    try {
      fs.mkdirSync(this.dataDir, { recursive: true });
      this.writeTable<number>("block_ids.json", this.blocks);
      this.writeTable<string>("entity_types.json", this.entities);
      this.writeTable<number>("item_ids.json", this.items);
      this.dirty = false;
    } catch (err) {
      log.warn("save failed", (err as Error).message);
    }
  }

  private writeTable<K>(file: string, map: Map<K, number>): void {
    const payload: PersistedTable<K> = { entries: Array.from(map.entries()) };
    fs.writeFileSync(path.join(this.dataDir, file), JSON.stringify(payload));
  }

  private load(): void {
    this.readTable<number>("block_ids.json", this.blocks);
    this.readTable<string>("entity_types.json", this.entities);
    this.readTable<number>("item_ids.json", this.items);

    // Recompute next indices as max(existing) + 1, respecting the reserved 0/air slot.
    let blockMax = 0;
    for (const v of this.blocks.values()) if (v > blockMax) blockMax = v;
    this.blockNextIdx = Math.max(1, blockMax + 1);

    let entityMax = -1;
    for (const v of this.entities.values()) if (v > entityMax) entityMax = v;
    this.entityNextIdx = entityMax + 1;

    let itemMax = 0;
    for (const v of this.items.values()) if (v > itemMax) itemMax = v;
    this.itemNextIdx = Math.max(1, itemMax + 1);
  }

  private readTable<K>(file: string, into: Map<K, number>): void {
    const p = path.join(this.dataDir, file);
    if (!fs.existsSync(p)) return;
    try {
      const raw = fs.readFileSync(p, "utf8");
      const parsed = JSON.parse(raw) as PersistedTable<K>;
      for (const [k, v] of parsed.entries) into.set(k, v);
    } catch (err) {
      log.warn(`failed to read ${file}: ${(err as Error).message}`);
    }
  }
}
