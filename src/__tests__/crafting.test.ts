import { test } from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { RecipeRegistry } from "../crafting/registry.js";
import { nextRequestId, resetRequestIds } from "../crafting/request_id.js";

test("RecipeRegistry parses shaped and shapeless recipes from a crafting_data packet", () => {
  const fakeClient = new EventEmitter() as any;
  fakeClient.queue = () => {};
  const reg = new RecipeRegistry();
  reg.attach(fakeClient);

  fakeClient.emit("crafting_data", {
    recipes: [
      {
        type: "shapeless",
        recipe: {
          recipe_id: "minecraft:oak_planks",
          network_id: 101,
          input: [{ name: "minecraft:oak_log", count: 1 }],
          output: [{ name: "minecraft:oak_planks", count: 4 }],
        },
      },
      {
        type: "shaped",
        recipe: {
          recipe_id: "minecraft:wooden_pickaxe",
          network_id: 102,
          width: 3,
          height: 3,
          input: [
            { name: "minecraft:oak_planks", count: 1 },
            { name: "minecraft:oak_planks", count: 1 },
            { name: "minecraft:oak_planks", count: 1 },
            { name: "minecraft:stick", count: 1 },
            { name: "minecraft:stick", count: 1 },
          ],
          output: [{ name: "minecraft:wooden_pickaxe", count: 1 }],
        },
      },
    ],
  });

  assert.ok(reg.isReady(), "registry should be ready after crafting_data");
  assert.equal(reg.size(), 2);

  const planks = reg.findByOutputName("planks", true);
  assert.ok(planks);
  assert.equal(planks!.networkId, 101);
  assert.equal(planks!.needsTable, false, "oak_planks (1-input shapeless) must not need a table");

  const pick = reg.findByOutputName("pickaxe", false);
  assert.ok(pick);
  assert.equal(pick!.networkId, 102);
  assert.equal(pick!.needsTable, true, "wooden_pickaxe (3x3 shaped) must need a table");
});

test("RecipeRegistry parses real wire shapes: nested shaped grids and descriptor variants", () => {
  const fakeClient = new EventEmitter() as any;
  fakeClient.queue = () => {};
  const reg = new RecipeRegistry();
  reg.attach(fakeClient);

  // Mirrors what bedrock-protocol actually emits: shaped input is a
  // width×height NESTED array; ingredients are flattened descriptor variants
  // (string_id_meta → name, item_tag → tag, invalid → empty cell).
  fakeClient.emit("crafting_data", {
    recipes: [
      {
        type: "shapeless",
        recipe: {
          recipe_id: "minecraft:stick",
          network_id: 201,
          input: [{ type: "item_tag", tag: "minecraft:planks", count: 2 }],
          output: [{ name: "minecraft:stick", count: 4 }],
        },
      },
      {
        type: "shaped",
        recipe: {
          recipe_id: "minecraft:wooden_pickaxe",
          network_id: 202,
          width: 3,
          height: 3,
          input: [
            [
              { type: "item_tag", tag: "minecraft:planks", count: 1 },
              { type: "item_tag", tag: "minecraft:planks", count: 1 },
              { type: "item_tag", tag: "minecraft:planks", count: 1 },
            ],
            [
              { type: "invalid", count: 0 },
              { type: "string_id_meta", name: "minecraft:stick", metadata: 0, count: 1 },
              { type: "invalid", count: 0 },
            ],
            [
              { type: "invalid", count: 0 },
              { type: "string_id_meta", name: "minecraft:stick", metadata: 0, count: 1 },
              { type: "invalid", count: 0 },
            ],
          ],
          output: [{ name: "minecraft:wooden_pickaxe", count: 1 }],
        },
      },
    ],
  });

  const sticks = reg.findByOutputName("stick", true);
  assert.ok(sticks);
  assert.equal(sticks!.inputs.length, 1);
  assert.equal(sticks!.inputs[0]!.tag, "planks", "tag should be captured with minecraft: stripped");
  assert.equal(sticks!.inputs[0]!.count, 2);
  assert.equal(sticks!.needsTable, false);

  const pick = reg.findByOutputName("wooden_pickaxe", false);
  assert.ok(pick);
  assert.equal(pick!.needsTable, true);
  // 9 grid cells → 3 planks (aggregated, by tag) + 2 sticks (aggregated, by name).
  assert.equal(pick!.inputs.length, 2);
  const planksIng = pick!.inputs.find((i) => i.tag === "planks");
  const stickIng = pick!.inputs.find((i) => i.name === "stick");
  assert.ok(planksIng && planksIng.count === 3, "planks cells should aggregate to count 3");
  assert.ok(stickIng && stickIng.count === 2, "stick cells should aggregate to count 2");
});

test("request IDs are unique, odd, and negative", () => {
  resetRequestIds();
  const ids = new Set<number>();
  for (let i = 0; i < 10; i++) {
    const id = nextRequestId();
    assert.ok(id < 0, `id should be negative, got ${id}`);
    assert.ok(id % 2 !== 0, `id should be odd, got ${id}`);
    assert.ok(!ids.has(id), `id should be unique, got duplicate ${id}`);
    ids.add(id);
  }
});
