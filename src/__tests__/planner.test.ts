import { test } from "node:test";
import assert from "node:assert/strict";
import { plan } from "../goap/planner.js";
import { ALL_ACTIONS } from "../goap/actions/index.js";
import { earlyToolingGoal } from "../goap/goals.js";
import { DEFAULT_STATE, type BotState } from "../goap/types.js";
import { findPath } from "../pathfinding/astar.js";
import { World } from "../world/world.js";

test("GOAP can plan a wooden pickaxe from scratch (nearTree available)", () => {
  const state: BotState = { ...DEFAULT_STATE, nearTree: true };
  const result = plan(state, earlyToolingGoal, ALL_ACTIONS, { maxNodes: 20_000 });
  assert.ok(result, "expected a plan");
  const names = result!.steps.map((s) => s.name);
  // We expect at least chop, craft planks, craft sticks, craft table, craft pickaxe
  assert.ok(names.includes("ChopTree"), `plan missing ChopTree: ${names.join(",")}`);
  assert.ok(names.includes("CraftPlanks"), `plan missing CraftPlanks: ${names.join(",")}`);
  assert.ok(names.includes("CraftSticks"), `plan missing CraftSticks: ${names.join(",")}`);
  assert.ok(names.includes("CraftCraftingTable"), `plan missing CraftCraftingTable: ${names.join(",")}`);
  assert.ok(names.includes("CraftWoodenPickaxe"), `plan missing CraftWoodenPickaxe: ${names.join(",")}`);
});

test("3D A* finds a straight-line path across a flat floor", () => {
  const world = new World();
  // Lay down a 8x1x8 stone floor at y=63
  for (let x = 0; x < 8; x++) {
    for (let z = 0; z < 8; z++) {
      world.setBlock({ x, y: 63, z }, { runtimeId: 1, name: "minecraft:stone" });
    }
  }
  const result = findPath(world, { x: 0, y: 64, z: 0 }, { x: 7, y: 64, z: 7 });
  assert.ok(result.reached, "expected to reach goal on flat floor");
  assert.ok(result.path.length >= 7, `path too short: ${result.path.length}`);
});

test("3D A* rejects when no walkable surface", () => {
  const world = new World();
  // Empty world — no floor; isStandable will fail.
  const result = findPath(world, { x: 0, y: 64, z: 0 }, { x: 5, y: 64, z: 5 });
  assert.equal(result.reached, false);
});

test("3D A* navigates a single-step-up obstacle", () => {
  const world = new World();
  // Floor at y=63 from x=0..7, plus a single step at x=3 raised to y=64.
  for (let x = 0; x < 8; x++) {
    for (let z = 0; z < 3; z++) {
      world.setBlock({ x, y: 63, z }, { runtimeId: 1, name: "minecraft:stone" });
    }
  }
  world.setBlock({ x: 3, y: 64, z: 1 }, { runtimeId: 1, name: "minecraft:stone" });
  const result = findPath(world, { x: 0, y: 64, z: 1 }, { x: 7, y: 64, z: 1 });
  assert.ok(result.reached, "expected to navigate around the step");
});
