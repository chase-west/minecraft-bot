import { test } from "node:test";
import assert from "node:assert/strict";
import { decodeSubChunk } from "../world/decoder.js";
import { World } from "../world/world.js";
import { findNearbyTree, findNearbyStone, inferGroundId } from "../world/semantic.js";
import { resetLearnedLogIds } from "../world/logIds.js";

function writeVarInt(buf: number[], value: number): void {
  while (value & ~0x7f) {
    buf.push((value & 0x7f) | 0x80);
    value >>>= 7;
  }
  buf.push(value & 0x7f);
}

function writeZigZag(buf: number[], value: number): void {
  writeVarInt(buf, (value << 1) ^ (value >> 31));
}

test("decoder: uniform (all-air) subchunk yields zero blocks", () => {
  // version=8, storageCount=1, header=0 (bpb=0 uniform), single varint id=0 (air)
  const payload = Buffer.from([8, 1, 0, 0]);
  const sub = decodeSubChunk(payload);
  assert.equal(sub.blocks.length, 0);
});

test("decoder: uniform stone subchunk yields 4096 blocks", () => {
  // header=0 (bpb=0 uniform); single varint encoding zigzag id=1 → varint=2
  // Pre-decode: idRaw=2, runtimeId = 2 >>> 1 = 1
  const payload = Buffer.from([8, 1, 0, 2]);
  const sub = decodeSubChunk(payload);
  assert.equal(sub.blocks.length, 4096);
  assert.equal(sub.blocks[0]!.runtimeId, 1);
});

test("decoder: bits_per_block=1 with two-entry palette decodes correctly", () => {
  // version 9, storageCount=1, yIndex=0
  // header: bpb=1, netflag=1 → (1<<1)|1 = 3
  // 4096 blocks @ 1 bit = 4096 bits = 128 u32 words (blocksPerWord=32, wordsCount=128)
  // All bits 0 except index 0 = 1. Pattern: word[0] = 0x00000001
  const bytes: number[] = [9, 1, 0, 3];
  // 128 words of 0, but with word[0] bit 0 = 1
  for (let i = 0; i < 128; i++) {
    if (i === 0) bytes.push(0x01, 0x00, 0x00, 0x00);
    else bytes.push(0, 0, 0, 0);
  }
  // palette length zigzag = 2 → encoded as varint 4
  writeZigZag(bytes, 2);
  // palette[0] = id 5 (zigzag 10), palette[1] = id 7 (zigzag 14)
  writeZigZag(bytes, 5);
  writeZigZag(bytes, 7);
  const payload = Buffer.from(bytes);
  const sub = decodeSubChunk(payload);
  // index 0 → palette[1] = 7; all others → palette[0] = 5
  // Index 0 is (x=0,z=0,y=0).
  const block000 = sub.blocks.find((b) => b.x === 0 && b.y === 0 && b.z === 0);
  assert.ok(block000, "should have block at 0,0,0");
  assert.equal(block000!.runtimeId, 7);
  // 4096 non-air blocks since neither palette entry is 0.
  assert.equal(sub.blocks.length, 4096);
  // Spot-check another block — index 1 is (x=0,z=0,y=1) → palette[0] = 5
  const block001 = sub.blocks.find((b) => b.x === 0 && b.y === 1 && b.z === 0);
  assert.equal(block001!.runtimeId, 5);
});

test("semantic: findNearbyTree detects a 4-block log pillar WITH canopy", () => {
  resetLearnedLogIds(); // auto-learned ids are process-global; isolate this test
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  // Ground (lots of id 1 stone)
  for (let x = -8; x < 8; x++) for (let z = -8; z < 8; z++) {
    world.setBlock({ x, y: 63, z }, { runtimeId: 1 });
  }
  // Trunk (id 42)
  for (let dy = 0; dy < 5; dy++) {
    world.setBlock({ x: 3, y: 64 + dy, z: 3 }, { runtimeId: 42 });
  }
  // Leaf canopy (id 77) — 3x3 cluster above the trunk top
  for (let ox = -1; ox <= 1; ox++) for (let oz = -1; oz <= 1; oz++) {
    world.setBlock({ x: 3 + ox, y: 69, z: 3 + oz }, { runtimeId: 77 });
  }
  const groundId = inferGroundId(world);
  assert.equal(groundId, 1);
  const tree = findNearbyTree(world);
  assert.ok(tree, "expected to find tree");
  assert.equal(tree!.x, 3);
  assert.equal(tree!.z, 3);
  assert.equal(tree!.y, 64);
});

test("semantic: findNearbyTree REJECTS a pillar with no canopy", () => {
  resetLearnedLogIds(); // don't inherit an id learned by an earlier test
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  for (let x = -8; x < 8; x++) for (let z = -8; z < 8; z++) {
    world.setBlock({ x, y: 63, z }, { runtimeId: 1 });
  }
  // Naked pillar with no leaves — should be ignored (false positive in v1).
  for (let dy = 0; dy < 5; dy++) {
    world.setBlock({ x: 3, y: 64 + dy, z: 3 }, { runtimeId: 42 });
  }
  const tree = findNearbyTree(world);
  assert.equal(tree, null, "naked pillar should not be reported as a tree");
});

test("semantic: findNearbyTree REJECTS a grounded leaf blob (no canopy-stripping)", () => {
  // Regression: a dense block of identical "leaf" blocks resting on the ground
  // passes the canopy + grounded gates (each interior column has a block below
  // it and leaves all around). The old gate addLogId'd that id and the bot then
  // stripped canopies instead of trunks. A leaf column is wide — flanked by
  // same-id neighbours — so the thin-column test must reject it.
  resetLearnedLogIds();
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  for (let x = -8; x < 8; x++) for (let z = -8; z < 8; z++) {
    world.setBlock({ x, y: 63, z }, { runtimeId: 1 }); // ground
  }
  // 3x3x5 leaf cuboid (id 88) sitting on the ground at (3,3): vertical runs of 5,
  // every column flanked by same-id leaves, surrounded by a leaf "canopy".
  for (let ox = -1; ox <= 1; ox++) for (let oz = -1; oz <= 1; oz++) {
    for (let dy = 0; dy < 5; dy++) {
      world.setBlock({ x: 3 + ox, y: 64 + dy, z: 3 + oz }, { runtimeId: 88 });
    }
  }
  const tree = findNearbyTree(world);
  assert.equal(tree, null, "a solid leaf blob must not be learned/reported as a tree");
});

test("semantic: findNearbyStone returns the dominant ground block near player", () => {
  const world = new World();
  world.self.position = { x: 0, y: 64, z: 0 };
  for (let x = -4; x < 4; x++) for (let z = -4; z < 4; z++) {
    world.setBlock({ x, y: 63, z }, { runtimeId: 1 });
  }
  const stone = findNearbyStone(world);
  assert.ok(stone);
  assert.equal(world.getBlock(stone!)!.runtimeId, 1);
});
