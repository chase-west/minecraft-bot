import * as fs from "node:fs/promises";
import * as path from "node:path";
import * as ort from "onnxruntime-node";
import { OBS_DIM } from "./encoder.js";
import { makeLogger } from "../utils/logger.js";
// IMPORTANT: do NOT import from "./actions.js" here.  `actions.ts` transitively
// pulls in the full action runtime (mine/place/combat/eat → intent.ts → back
// to actions.ts), a cycle that breaks when `policy.ts` is the entrypoint
// (e.g. from the unit test).  Use the bare numeric action ids directly; we
// import the *type* below.
import type { ActionId } from "./actions.js";

const ACTION_COUNT = 12;
const NOOP: ActionId = 0 as ActionId;

const log = makeLogger("policy");

export interface ActOptions {
  /**
   * If > 0, sample from softmax(logits / temperature) instead of taking the
   * argmax. Smaller values approach argmax; larger values approach uniform.
   */
  temperature?: number;
  /**
   * If > 0, with this probability return a uniformly random ActionId instead
   * of running inference (epsilon-greedy exploration).
   */
  epsilon?: number;
}

interface PolicySession {
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
  inputBuffer: Float32Array;
  inputTensor: ort.Tensor;
  // Number of in-flight act() calls that captured this session as their local
  // ref. watchForReload() must NOT call session.release() while this is >0 or
  // the native ONNX handle gets freed mid-inference (possible crash/segfault).
  inflight: number;
}

/**
 * Wraps an ONNX policy that maps a 605-dim observation to logits/Q-values
 * over the 12-action discrete space defined in `actions.ts`.
 *
 * Design notes:
 *   - load() never throws on a missing file: it returns false so the caller
 *     can degrade gracefully.
 *   - act() reuses one Float32Array + one ort.Tensor per session, so steady-
 *     state inference does no per-tick allocation.
 *   - watchForReload() polls the ONNX file's mtime and atomically swaps in a
 *     new InferenceSession when a fresh model is written. The Python DQN
 *     trainer writes via os.replace so the swap-in always sees a complete file.
 *   - Each act() snapshots the current session into a local — a swap mid-call
 *     is harmless because the new session takes effect on the NEXT act().
 *   - When the policy is not loaded, act() returns ActionId.Noop.
 */
export class LearnedPolicy {
  private current: PolicySession | null = null;
  private resolvedPath = "";
  private lastMtimeMs = 0;
  private watchHandle: NodeJS.Timeout | null = null;
  private reloading = false;
  private reloadCount = 0;

  constructor(private readonly modelPath: string = "models/policy.onnx") {}

  async load(): Promise<boolean> {
    this.resolvedPath = path.isAbsolute(this.modelPath)
      ? this.modelPath
      : path.join(process.cwd(), this.modelPath);
    let stat: { mtimeMs: number } | null = null;
    try {
      stat = await fs.stat(this.resolvedPath);
    } catch {
      log.info(`policy file not found at ${this.resolvedPath} — running without learned policy`);
      return false;
    }
    const built = await this.buildSession(this.resolvedPath);
    if (!built) return false;
    this.current = built;
    this.lastMtimeMs = stat.mtimeMs;
    log.info(
      `policy loaded from ${this.resolvedPath} (input=${built.inputName} output=${built.outputName})`,
    );
    return true;
  }

  private async buildSession(p: string): Promise<PolicySession | null> {
    try {
      const session = await ort.InferenceSession.create(p, { executionProviders: ["cpu"] });
      const inputs = session.inputNames;
      const outputs = session.outputNames;
      const inputName = inputs.includes("obs") ? "obs" : (inputs[0] ?? "obs");
      const outputName = outputs.includes("logits") ? "logits" : (outputs[0] ?? "logits");
      const inputBuffer = new Float32Array(OBS_DIM);
      const inputTensor = new ort.Tensor("float32", inputBuffer, [1, OBS_DIM]);
      return { session, inputName, outputName, inputBuffer, inputTensor, inflight: 0 };
    } catch (err) {
      log.warn(`failed to load policy at ${p}: ${(err as Error).message}`);
      return null;
    }
  }

  /**
   * Begin polling the ONNX file every `intervalMs` (default 5s). When the
   * mtime changes AND the file size has stabilized for one poll, rebuild the
   * session and atomically swap it in. Call stopWatching() before process exit.
   */
  watchForReload(intervalMs = 5000): void {
    if (this.watchHandle !== null) return;
    let lastSize = -1;
    let pendingMtimeMs = 0;
    this.watchHandle = setInterval(async () => {
      if (this.reloading) return;
      try {
        const stat = await fs.stat(this.resolvedPath);
        if (stat.mtimeMs === this.lastMtimeMs) return;
        // Wait for file size to stabilize across two polls — protects against
        // reading mid-write even though the trainer writes atomically.
        if (stat.size !== lastSize || stat.mtimeMs !== pendingMtimeMs) {
          lastSize = stat.size;
          pendingMtimeMs = stat.mtimeMs;
          return;
        }
        this.reloading = true;
        const fresh = await this.buildSession(this.resolvedPath);
        if (fresh) {
          const prev = this.current;
          this.current = fresh;
          this.lastMtimeMs = stat.mtimeMs;
          this.reloadCount++;
          log.info(`policy hot-reloaded (#${this.reloadCount}) from ${this.resolvedPath}`);
          // Release the old session asynchronously, but only after any
          // in-flight act() call using it has resolved. Releasing while
          // session.run() is mid-await frees the native ONNX handle from
          // under the running inference — possible crash/segfault. Poll the
          // inflight counter every 50ms up to 2s, then release anyway as a
          // safety valve so we don't leak sessions if a call wedges.
          if (prev) {
            const deadline = Date.now() + 2000;
            while (prev.inflight > 0 && Date.now() < deadline) {
              await new Promise((r) => setTimeout(r, 50));
            }
            if (prev.inflight > 0) {
              log.warn(`releasing previous session with ${prev.inflight} act() call(s) still in flight after 2s`);
            }
            try { await prev.session.release(); }
            catch (_) { /* old session already dead — ignore */ }
          }
        }
      } catch (err) {
        // File may have been temporarily missing during atomic replace — just
        // try again next tick.
        if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
          log.warn(`watch poll failed: ${(err as Error).message}`);
        }
      } finally {
        this.reloading = false;
      }
    }, intervalMs);
    // Don't pin the event loop on the watcher — agent.stop() also calls
    // stopWatching(), but this is belt-and-suspenders for clean shutdown.
    this.watchHandle.unref?.();
  }

  stopWatching(): void {
    if (this.watchHandle !== null) {
      clearInterval(this.watchHandle);
      this.watchHandle = null;
    }
  }

  isLoaded(): boolean {
    return this.current !== null;
  }

  async act(obs: Float32Array, opts: ActOptions = {}): Promise<ActionId> {
    // Snapshot the current session: hot-reload may swap this.current mid-call,
    // but the call below must use one consistent set of references.
    const sess = this.current;
    if (!sess) return NOOP;
    if (obs.length !== OBS_DIM) {
      throw new Error(`act(): expected obs length ${OBS_DIM}, got ${obs.length}`);
    }

    const epsilon = opts.epsilon ?? 0;
    if (epsilon > 0 && Math.random() < epsilon) {
      return randomAction();
    }

    sess.inputBuffer.set(obs);
    const feeds: Record<string, ort.Tensor> = { [sess.inputName]: sess.inputTensor };
    // Mark this session in-flight so a concurrent hot-reload waits for us
    // before calling session.release() — releasing the native ONNX handle
    // while session.run() is mid-await can crash the process.
    sess.inflight++;
    let logits: Float32Array;
    try {
      const out = await sess.session.run(feeds);
      const result = out[sess.outputName];
      if (!result) {
        throw new Error(`policy returned no output for name ${sess.outputName}`);
      }
      logits = result.data as Float32Array;
      if (logits.length < ACTION_COUNT) {
        throw new Error(`policy returned ${logits.length} logits, expected ${ACTION_COUNT}`);
      }
    } finally {
      sess.inflight--;
    }

    const temperature = opts.temperature ?? 0;
    if (temperature > 0) {
      return sampleSoftmax(logits, temperature);
    }
    return argmax(logits);
  }
}

function argmax(logits: Float32Array): ActionId {
  let bestIdx = 0;
  let bestVal = logits[0] ?? -Infinity;
  for (let i = 1; i < ACTION_COUNT; i++) {
    const v = logits[i] ?? -Infinity;
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx as ActionId;
}

function sampleSoftmax(logits: Float32Array, temperature: number): ActionId {
  // Numerically stable softmax: subtract max before exp.
  let maxLogit = -Infinity;
  for (let i = 0; i < ACTION_COUNT; i++) {
    const v = logits[i] ?? -Infinity;
    if (v > maxLogit) maxLogit = v;
  }
  const probs = new Float32Array(ACTION_COUNT);
  let sum = 0;
  for (let i = 0; i < ACTION_COUNT; i++) {
    const v = ((logits[i] ?? 0) - maxLogit) / temperature;
    const e = Math.exp(v);
    probs[i] = e;
    sum += e;
  }
  if (sum <= 0 || !Number.isFinite(sum)) return argmax(logits);

  const r = Math.random() * sum;
  let acc = 0;
  for (let i = 0; i < ACTION_COUNT; i++) {
    acc += probs[i] ?? 0;
    if (r < acc) return i as ActionId;
  }
  return (ACTION_COUNT - 1) as ActionId;
}

function randomAction(): ActionId {
  return Math.floor(Math.random() * ACTION_COUNT) as ActionId;
}
