import type { BedrockClient } from "../connection/client.js";
import type { World } from "./world.js";
import { safeQueue } from "../connection/version.js";
import { decodeSubChunk } from "./decoder.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("chunk");

/**
 * Subchunk cache by (cx, cz) — we keep the raw decoded blocks per subchunk so we
 * can replay them into the World map after the player moves to a new vertical band.
 */
interface CachedSubChunk {
  cx: number;
  cz: number;
  yIndex: number;
  blocks: Array<{ x: number; y: number; z: number; runtimeId: number }>;
}

// Cap how many chunk columns we cache. As the bot wanders, it streams new
// chunks indefinitely — without a cap the heap balloons until OOM.
// At 4 KB per subchunk * ~16 subs per column = ~64 KB/column, 800 cols ~= 50 MB.
const MAX_CACHED_COLUMNS = 800;

export class ChunkCache {
  // Map preserves insertion order; we use that as a poor-man's LRU.
  private readonly chunks = new Map<string, CachedSubChunk[]>();
  size(): number { return this.chunks.size; }
  add(cx: number, cz: number, sub: CachedSubChunk): void {
    const key = `${cx},${cz}`;
    const arr = this.chunks.get(key) ?? [];
    const existing = arr.findIndex((s) => s.yIndex === sub.yIndex);
    if (existing >= 0) arr[existing] = sub;
    else arr.push(sub);
    // Refresh recency: re-insert at the end of the iteration order.
    this.chunks.delete(key);
    this.chunks.set(key, arr);
    // Evict oldest column(s) when over cap.
    while (this.chunks.size > MAX_CACHED_COLUMNS) {
      const first = this.chunks.keys().next().value;
      if (first === undefined) break;
      this.chunks.delete(first);
    }
  }
  get(cx: number, cz: number): CachedSubChunk[] {
    return this.chunks.get(`${cx},${cz}`) ?? [];
  }
}

function blockKey(x: number, y: number, z: number): string {
  return `${x | 0},${y | 0},${z | 0}`;
}

// Maximum horizontal/vertical distance from the bot at which a decoded block
// is materialized into the flat world map. Anything farther stays only in the
// SubChunkCache (still queryable via getCachedBlock). This keeps the flat map
// at O(few × 10k) instead of O(millions) so per-tick reads and the GC are fast.
const PROJECTION_RADIUS_XZ = 48;
const PROJECTION_RADIUS_Y = 32;

/** Project a decoded subchunk into the World's flat block map — but only blocks
 * within PROJECTION_RADIUS of the bot. Far-away blocks are skipped (they live
 * in the cache for later access if the bot moves there). */
function projectSubChunk(world: World, cx: number, cz: number, sub: CachedSubChunk): number {
  const baseX = cx * 16;
  const baseZ = cz * 16;
  const baseY = sub.yIndex * 16;
  const bx = world.self.position.x;
  const by = world.self.position.y;
  const bz = world.self.position.z;

  // Quick reject: if the entire subchunk is outside projection radius, skip.
  const subMinX = baseX, subMaxX = baseX + 15;
  const subMinZ = baseZ, subMaxZ = baseZ + 15;
  const subMinY = baseY, subMaxY = baseY + 15;
  const xFar = Math.max(0, Math.max(subMinX - bx, bx - subMaxX));
  const zFar = Math.max(0, Math.max(subMinZ - bz, bz - subMaxZ));
  const yFar = Math.max(0, Math.max(subMinY - by, by - subMaxY));
  if (xFar > PROJECTION_RADIUS_XZ || zFar > PROJECTION_RADIUS_XZ || yFar > PROJECTION_RADIUS_Y) {
    return 0;
  }

  let added = 0;
  for (const b of sub.blocks) {
    const wx = baseX + b.x;
    const wy = baseY + b.y;
    const wz = baseZ + b.z;
    if (Math.abs(wx - bx) > PROJECTION_RADIUS_XZ) continue;
    if (Math.abs(wz - bz) > PROJECTION_RADIUS_XZ) continue;
    if (Math.abs(wy - by) > PROJECTION_RADIUS_Y) continue;
    const key = blockKey(wx, wy, wz);
    if (!world.blocks.has(key)) added++;
    world.blocks.set(key, { runtimeId: b.runtimeId });
  }
  return added;
}

/** Re-project blocks from the cache around the bot. Called periodically as
 * the bot moves so we don't have stale far-away blocks dominating the flat map. */
export function reprojectAroundBot(world: World, cache: ChunkCache): number {
  // Cap the flat map at a soft limit; if we're well over, evict outliers.
  const SOFT_LIMIT = 200_000;
  if (world.blocks.size > SOFT_LIMIT) {
    const bx = world.self.position.x;
    const by = world.self.position.y;
    const bz = world.self.position.z;
    let evicted = 0;
    for (const [key, _] of world.blocks) {
      const [xs, ys, zs] = key.split(",");
      const x = Number(xs), y = Number(ys), z = Number(zs);
      if (Math.abs(x - bx) > PROJECTION_RADIUS_XZ || Math.abs(z - bz) > PROJECTION_RADIUS_XZ || Math.abs(y - by) > PROJECTION_RADIUS_Y) {
        world.blocks.delete(key);
        evicted++;
      }
    }
    return evicted;
  }
  return 0;
}

export function attachChunkStream(client: BedrockClient, world: World, cache: ChunkCache): void {
  let firstChunkLogged = false;
  let firstSubLogged = false;
  let subDecodeOk = 0;
  let subDecodeFail = 0;

  client.on("level_chunk", async (pkt: any) => {
    if (pkt.x === undefined || pkt.z === undefined) return;
    const subChunkCount = pkt.sub_chunk_count ?? 0;

    if (!firstChunkLogged) {
      firstChunkLogged = true;
      log.info(`receiving chunks (first: x=${pkt.x} z=${pkt.z} subChunks=${subChunkCount})`);
    }

    // Subchunk-request mode: server signals it with a negative count and an empty/short
    // payload. We must request each subchunk individually via subchunk_request.
    if (subChunkCount < 0) {
      const requests: Array<{ dx: number; dy: number; dz: number }> = [];
      // Request a vertical column of 12 subchunks centered roughly around sea level (y=64).
      // y_index range -4..7 covers world y -64..127 — overworld surface + a bit above and below.
      for (let dy = -4; dy <= 7; dy++) {
        requests.push({ dx: 0, dy, dz: 0 });
      }
      safeQueue(client, "subchunk_request", {
        dimension: world.self.dimension,
        origin: { x: pkt.x, y: 0, z: pkt.z },
        requests,
      }, "subchunk_req");
      return;
    }

    // Full-chunk payload (older mode): decode all subchunks sequentially.
    const payload = Buffer.isBuffer(pkt.payload) ? pkt.payload : Buffer.from(pkt.payload ?? []);
    let off = 0;
    for (let s = 0; s < subChunkCount; s++) {
      try {
        // Each subchunk consumes from the same payload. We need a per-subchunk decoder
        // that returns bytes consumed; for simplicity, slice and rely on decoder's
        // internal cursor (it won't read past what it needs because format is self-describing).
        const sub = decodeSubChunk(payload.subarray(off), s);
        cache.add(pkt.x, pkt.z, { cx: pkt.x, cz: pkt.z, yIndex: sub.yIndex, blocks: sub.blocks });
        projectSubChunk(world, pkt.x, pkt.z, { cx: pkt.x, cz: pkt.z, yIndex: sub.yIndex, blocks: sub.blocks });
        subDecodeOk++;
        // Conservatively advance — for legacy mode we can't easily compute the exact size.
        // Most servers use subchunk-request mode now, so this branch is rarely hit.
        break;
      } catch (err) {
        subDecodeFail++;
        if (subDecodeFail <= 3) log.warn(`level_chunk subchunk decode failed: ${(err as Error).message}`);
        break;
      }
    }
  });

  let entriesSinceYield = 0;
  client.on("subchunk", async (pkt: any) => {
    const origin = pkt.origin ?? { x: 0, y: 0, z: 0 };
    const entries = pkt.entries ?? pkt.sub_chunk_entries ?? [];
    for (const entry of entries) {
      // Yield every 4 entries so setInterval (shadow logger, input ticker) can fire.
      if (++entriesSinceYield >= 4) {
        entriesSinceYield = 0;
        await new Promise<void>((r) => setImmediate(r));
      }
      const status = entry.result ?? entry.status;
      // status 0 = success in newer protocol; absent in older — we accept both.
      if (status !== undefined && status !== 0 && status !== "success") continue;

      const dx = entry.dx ?? entry.offset?.x ?? 0;
      const dy = entry.dy ?? entry.offset?.y ?? 0;
      const dz = entry.dz ?? entry.offset?.z ?? 0;
      const cx = origin.x + dx;
      const cz = origin.z + dz;
      const payload = Buffer.isBuffer(entry.payload) ? entry.payload : Buffer.from(entry.payload ?? []);
      if (payload.length === 0) continue;

      try {
        const sub = decodeSubChunk(payload, dy);
        cache.add(cx, cz, { cx, cz, yIndex: sub.yIndex, blocks: sub.blocks });
        const added = projectSubChunk(world, cx, cz, { cx, cz, yIndex: sub.yIndex, blocks: sub.blocks });
        subDecodeOk++;
        if (!firstSubLogged && added > 0) {
          firstSubLogged = true;
          log.info(`subchunk decoder ALIVE — first decode cx=${cx} cz=${cz} dy=${dy} blocks=${sub.blocks.length} added=${added} palette=${sub.palette.length}`);
        }
        if (subDecodeOk === 100 || subDecodeOk === 500 || subDecodeOk % 2000 === 0) {
          log.info(`subchunk decode stats: ok=${subDecodeOk} fail=${subDecodeFail} worldBlocks=${world.blocks.size}`);
        }
      } catch (err) {
        subDecodeFail++;
        if (subDecodeFail <= 3) log.warn(`subchunk decode failed cx=${cx} cz=${cz} dy=${dy}: ${(err as Error).message}`);
      }
    }
  });

  // Perception must FOLLOW the bot. The level_chunk-driven requests above only
  // fire for the chunks the server pushes near spawn, so without this the bot is
  // blind anywhere it walks far or is teleported (world.blocks stays anchored at
  // spawn). Every 2s, if the bot has moved >=2 chunks since the last request,
  // request a 7x7 grid of subchunk columns around its current chunk and reproject
  // cached blocks around it.
  let lastReqCx = Infinity, lastReqCz = Infinity;
  setInterval(() => {
    if (world.self.runtimeEntityId === null) return; // not spawned yet
    const cx = Math.floor(world.self.position.x / 16);
    const cz = Math.floor(world.self.position.z / 16);
    if (Math.abs(cx - lastReqCx) < 2 && Math.abs(cz - lastReqCz) < 2) return;
    lastReqCx = cx; lastReqCz = cz;
    const requests: Array<{ dx: number; dy: number; dz: number }> = [];
    for (let dx = -3; dx <= 3; dx++) {
      for (let dz = -3; dz <= 3; dz++) {
        for (let dy = -4; dy <= 7; dy++) requests.push({ dx, dy, dz });
      }
    }
    safeQueue(client, "subchunk_request", {
      dimension: world.self.dimension,
      origin: { x: cx, y: 0, z: cz },
      requests,
    }, "subchunk_req_follow");
    reprojectAroundBot(world, cache);
    log.info(`chunk-follow: requested ${requests.length} subchunks around chunk (${cx},${cz})`);
  }, 2000);
}
