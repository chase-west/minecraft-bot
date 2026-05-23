import { test } from "node:test";
import assert from "node:assert/strict";
import { LearnedPolicy } from "../ml/policy.js";
import { OBS_DIM } from "../ml/encoder.js";

// NOTE: This file deliberately does NOT import from "../ml/actions.js".
// `actions.ts` pulls in the action runtime (mine/place/combat/eat), some of
// which import "../ml/intent.js", which in turn imports "../ml/actions.js"
// — a benign-at-runtime cycle that nonetheless reads `ActionId.Noop` at
// module-init time and explodes if the test entrypoint kicks off the chain
// from a different starting node than the agent does.  We use the raw enum
// value 0 (= ActionId.Noop) instead so the test stays isolated.
const NOOP = 0;

test("LearnedPolicy: load() returns false when model file is missing", async () => {
  const policy = new LearnedPolicy("models/__definitely_does_not_exist__.onnx");
  const ok = await policy.load();
  assert.equal(ok, false);
  assert.equal(policy.isLoaded(), false);
});

test("LearnedPolicy: load() is idempotently safe on a missing path", async () => {
  const policy = new LearnedPolicy("models/__definitely_does_not_exist__.onnx");
  // Call twice — must not throw, must consistently report not-loaded.
  assert.equal(await policy.load(), false);
  assert.equal(await policy.load(), false);
  assert.equal(policy.isLoaded(), false);
});

test("LearnedPolicy: act() returns Noop when not loaded (documented graceful path)", async () => {
  const policy = new LearnedPolicy("models/__definitely_does_not_exist__.onnx");
  await policy.load();
  const obs = new Float32Array(OBS_DIM);
  const action = await policy.act(obs);
  // We chose to return Noop rather than throw so the agent's 10 Hz tick
  // loop doesn't hard-fail when no model is present.
  assert.equal(action, NOOP);
});

test("LearnedPolicy: act() returns Noop with epsilon-greedy even when not loaded", async () => {
  // The not-loaded short-circuit takes precedence over the epsilon-greedy
  // branch (we early-out before the random check).
  const policy = new LearnedPolicy("models/__definitely_does_not_exist__.onnx");
  await policy.load();
  const obs = new Float32Array(OBS_DIM);
  const action = await policy.act(obs, { epsilon: 1.0 });
  assert.equal(action, NOOP);
});
