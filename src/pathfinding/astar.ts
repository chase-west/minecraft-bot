import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3key, v3floor } from "../utils/vec3.js";
import { MinHeap } from "../utils/heap.js";
import { isStandable, landingY } from "./safety.js";

export type MoveKind = "walk" | "jump" | "drop" | "diagonal";

export interface PathStep {
  to: Vec3;
  kind: MoveKind;
  cost: number;
}

interface Node {
  pos: Vec3;
  g: number;
  f: number;
  parent: Node | null;
  step: PathStep | null;
}

const SQRT2 = Math.SQRT2;

function octile(a: Vec3, b: Vec3): number {
  const dx = Math.abs(a.x - b.x);
  const dz = Math.abs(a.z - b.z);
  return Math.abs(dx - dz) + Math.min(dx, dz) * SQRT2 + Math.abs(a.y - b.y);
}

interface NeighborCandidate {
  pos: Vec3;
  kind: MoveKind;
  cost: number;
}

function neighbors(world: World, p: Vec3): NeighborCandidate[] {
  const out: NeighborCandidate[] = [];
  const dirs4 = [
    { dx: 1, dz: 0 }, { dx: -1, dz: 0 }, { dx: 0, dz: 1 }, { dx: 0, dz: -1 },
  ];
  const diag = [
    { dx: 1, dz: 1 }, { dx: 1, dz: -1 }, { dx: -1, dz: 1 }, { dx: -1, dz: -1 },
  ];

  for (const d of dirs4) {
    // walk forward
    const flat = { x: p.x + d.dx, y: p.y, z: p.z + d.dz };
    if (isStandable(world, flat)) out.push({ pos: flat, kind: "walk", cost: 1 });

    // jump up
    const up = { x: p.x + d.dx, y: p.y + 1, z: p.z + d.dz };
    if (isStandable(world, up)) out.push({ pos: up, kind: "jump", cost: 2 });

    // drop down
    const dropTop = { x: p.x + d.dx, y: p.y, z: p.z + d.dz };
    const ly = landingY(world, dropTop, 4);
    if (ly !== null && ly < p.y) {
      const dropTarget = { x: dropTop.x, y: ly, z: dropTop.z };
      if (isStandable(world, dropTarget)) {
        const fall = p.y - ly;
        out.push({ pos: dropTarget, kind: "drop", cost: 1 + fall * 0.2 });
      }
    }
  }

  for (const d of diag) {
    const target = { x: p.x + d.dx, y: p.y, z: p.z + d.dz };
    if (!isStandable(world, target)) continue;
    // require both perpendicular cardinals to be walkable to avoid shoulder-clip
    const sideA = { x: p.x + d.dx, y: p.y, z: p.z };
    const sideB = { x: p.x, y: p.y, z: p.z + d.dz };
    if (!isStandable(world, sideA)) continue;
    if (!isStandable(world, sideB)) continue;
    out.push({ pos: target, kind: "diagonal", cost: SQRT2 });
  }

  return out;
}

export interface PathResult {
  path: PathStep[];
  reached: boolean;
}

export function findPath(world: World, start: Vec3, goal: Vec3, opts: { maxNodes?: number } = {}): PathResult {
  const startF = v3floor(start);
  const goalF = v3floor(goal);
  const maxNodes = opts.maxNodes ?? 8000;

  const open = new MinHeap<Node>((a, b) => a.f - b.f);
  const closed = new Set<string>();
  const best = new Map<string, number>();

  const startNode: Node = { pos: startF, g: 0, f: octile(startF, goalF), parent: null, step: null };
  open.push(startNode);
  best.set(v3key(startF), 0);

  let expanded = 0;
  let bestFallback: Node = startNode;
  let bestFallbackH = octile(startF, goalF);

  while (open.size > 0) {
    const cur = open.pop()!;
    const key = v3key(cur.pos);
    if (closed.has(key)) continue;
    closed.add(key);
    expanded += 1;

    const h = octile(cur.pos, goalF);
    if (h < bestFallbackH) { bestFallback = cur; bestFallbackH = h; }

    if (cur.pos.x === goalF.x && cur.pos.y === goalF.y && cur.pos.z === goalF.z) {
      return { path: reconstruct(cur), reached: true };
    }
    if (expanded > maxNodes) break;

    for (const n of neighbors(world, cur.pos)) {
      const nk = v3key(n.pos);
      if (closed.has(nk)) continue;
      const g = cur.g + n.cost;
      const prev = best.get(nk);
      if (prev !== undefined && g >= prev) continue;
      best.set(nk, g);
      open.push({
        pos: n.pos,
        g,
        f: g + octile(n.pos, goalF),
        parent: cur,
        step: { to: n.pos, kind: n.kind, cost: n.cost },
      });
    }
  }

  return { path: reconstruct(bestFallback), reached: false };
}

function reconstruct(node: Node): PathStep[] {
  const out: PathStep[] = [];
  let cur: Node | null = node;
  while (cur && cur.step) {
    out.push(cur.step);
    cur = cur.parent;
  }
  return out.reverse();
}
