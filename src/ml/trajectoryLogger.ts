import * as fs from "node:fs";
import * as path from "node:path";
import type { ActionId } from "./actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("traj");

/**
 * Append-only JSONL trajectory logger. One file per session keyed by
 * Date.now() at construction. Each call emits one line:
 *   {"obs_b64": "<base64 of Float32Array.buffer>", "action": int, "reward": float, "t": ms, "meta"?}
 *
 * Why base64 of the raw float buffer instead of an expanded number array?
 *   - JSON serialization of 601 floats is the throughput bottleneck (per-tick
 *     budget at 10 Hz is 100 ms; expanding+stringifying takes 10-50 ms on a
 *     typical machine). Base64 + binary buffer = <1 ms.
 *   - File size shrinks ~4× (601 × ~7 chars/float ≈ 4.2 KB → 601 × 4 B = 2.4 KB
 *     raw → ~3.2 KB base64). The Python trainer round-trips via base64 → np.frombuffer.
 *
 * Writes go through an in-process queue that drains via async appendFile,
 * so the calling tick is never blocked by disk I/O.
 */
export class TrajectoryLogger {
  readonly sessionId: number;
  readonly filePath: string;
  private readonly startedAt: number;
  private queue: string[] = [];
  private flushing = false;
  private closed = false;

  constructor(dir: string) {
    this.sessionId = Date.now();
    this.startedAt = this.sessionId;
    try {
      fs.mkdirSync(dir, { recursive: true });
    } catch (err) {
      log.warn(`mkdir ${dir} failed: ${(err as Error).message}`);
    }
    this.filePath = path.join(dir, `${this.sessionId}.jsonl`);
    log.info(`trajectory log → ${this.filePath}`);
  }

  log(obs: Float32Array, action: ActionId, reward: number, terminal = false, meta?: object): void {
    if (this.closed) return;
    // Float32Array.buffer is the underlying ArrayBuffer; wrap it in a Buffer view
    // (no copy) and base64 it. Length is implicit (obs.length × 4 bytes).
    const view = Buffer.from(obs.buffer, obs.byteOffset, obs.byteLength);
    const obs_b64 = view.toString("base64");
    const record: Record<string, unknown> = {
      obs_b64,
      obs_len: obs.length,
      action,
      reward,
      t: Date.now() - this.startedAt,
    };
    // The Python DQN trainer (training/train_dqn.py) checks row.get("done") to
    // mark a transition as terminal (done=True self-loop, no bootstrap). Without
    // this, the -50 death reward gets blended through gamma*Q(next) and the
    // bot never learns to stop dying.
    if (terminal) record.done = true;
    if (meta !== undefined) record.meta = meta;
    this.queue.push(JSON.stringify(record) + "\n");
    this.kickFlush();
  }

  private kickFlush(): void {
    if (this.flushing || this.queue.length === 0) return;
    this.flushing = true;
    const batch = this.queue.join("");
    this.queue = [];
    fs.appendFile(this.filePath, batch, (err) => {
      this.flushing = false;
      if (err) log.warn(`append failed: ${err.message}`);
      // If new items arrived while we were flushing, drain them too.
      if (this.queue.length > 0) this.kickFlush();
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    // Sync flush whatever remains so process exit doesn't lose data.
    if (this.queue.length > 0) {
      try { fs.appendFileSync(this.filePath, this.queue.join("")); }
      catch (err) { log.warn(`final flush failed: ${(err as Error).message}`); }
      this.queue = [];
    }
    log.info(`trajectory closed: ${this.filePath}`);
  }
}
