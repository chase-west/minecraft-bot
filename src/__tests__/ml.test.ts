import { test } from "node:test";
import assert from "node:assert/strict";
import { BlockIdRegistry } from "../ml/blockIdRegistry.js";
import { Encoder } from "../ml/encoder.js";
import { RewardCalculator } from "../ml/reward.js";
import { World } from "../world/world.js";

test("BlockIdRegistry: same runtime ID gets same dense index across calls", () => {
  const reg = new BlockIdRegistry("data/test-tmp");
  const a = reg.denseIndex(13791);
  const b = reg.denseIndex(13791);
  const c = reg.denseIndex(42);
  assert.equal(a, b);
  assert.notEqual(a, c);
});

test("BlockIdRegistry: air (runtime 0) always maps to dense 0", () => {
  const reg = new BlockIdRegistry("data/test-tmp");
  assert.equal(reg.denseIndex(0), 0);
  reg.denseIndex(100);
  reg.denseIndex(200);
  assert.equal(reg.denseIndex(0), 0);
});

test("Encoder: produces a 605-long Float32Array", () => {
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  world.self.health = 20; world.self.food = 18;
  const reg = new BlockIdRegistry("data/test-tmp");
  const enc = new Encoder();
  const obs = enc.encode(world, reg);
  assert.equal(obs.length, 605);
  // Self stats slice
  assert.equal(obs[0], 1.0); // health/20
  assert.ok(Math.abs((obs[1] ?? 0) - 18 / 20) < 1e-6); // food/20
});

test("Encoder: writes block grid dense indices around the bot", () => {
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  // Put a known block right in front of the bot at (0, 63, 0)
  world.setBlock({ x: 0, y: 63, z: 0 }, { runtimeId: 1234 });
  const reg = new BlockIdRegistry("data/test-tmp");
  const enc = new Encoder();
  const obs = enc.encode(world, reg);
  // The grid is positions 8..412 (405 cells). The cell for (dx=0, dy=-1, dz=0)
  // — the block immediately below the bot — should now contain a non-zero dense index.
  // We don't need to compute the exact index; just verify ≥ 1 grid cell is non-zero.
  let nonZeroCells = 0;
  for (let i = 8; i < 413; i++) if (obs[i] !== 0) nonZeroCells++;
  assert.ok(nonZeroCells >= 1, "expected at least one grid cell populated");
});

test("RewardCalculator: rewards alive ticks, penalizes hunger loss", () => {
  const world = new World();
  world.self.health = 20;
  world.self.food = 20;
  const rc = new RewardCalculator();
  const r0 = rc.step(world, 0);
  // First call: no prior snapshot → reward is mostly the alive + urgency baseline.
  assert.ok(r0 > -1 && r0 < 1, `r0=${r0} should be in (-1, 1)`);
  world.self.food = 18; // lost 2 hunger
  const r1 = rc.step(world, 100);
  // -0.5 * 2 hunger lost + alive baseline = ~ -1.0 + 0.009
  assert.ok(r1 < -0.5, `r1=${r1} should reflect hunger loss`);
});
