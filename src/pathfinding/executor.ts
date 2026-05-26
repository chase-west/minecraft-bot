import type { World } from "../world/world.js";
import type { InputController } from "../actions/input.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3floor, v3distXZ } from "../utils/vec3.js";
import { findPath, type PathStep } from "./astar.js";
import { walkTo } from "../actions/move.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("path-exec");

export interface NavOptions {
  maxNodes?: number;
  replanAfterSteps?: number; // re-plan after this many executed steps (world may change)
  timeoutMs?: number;
}

export async function navigateTo(
  world: World,
  input: InputController,
  goal: Vec3,
  opts: NavOptions = {},
): Promise<{ arrived: boolean; reason?: string }> {
  const overallDeadline = Date.now() + (opts.timeoutMs ?? 120_000);
  const replanEvery = opts.replanAfterSteps ?? 8;

  while (Date.now() < overallDeadline) {
    const start = v3floor(world.self.position);
    // Require Y proximity too: when the goal is below (e.g. a tree base down a
    // slope), being XZ-over it while still up on the ledge is NOT arrival.
    if (v3distXZ(start, goal) < 1 && Math.abs(start.y - goal.y) <= 1.5) {
      return { arrived: true };
    }

    const plan = findPath(world, start, goal, { maxNodes: opts.maxNodes ?? 8000 });
    if (plan.path.length === 0) {
      if (plan.reached) return { arrived: false, reason: "already_there" };
      // A* couldn't generate any neighbors from start (bot on an unrecognized
      // surface, edge case terrain, etc.). Fall back to walking in a straight
      // line toward the goal for a few seconds. Often gets the bot off the
      // weird block and into terrain A* understands. Won't progress if there's
      // a real wall, but it won't busy-loop either.
      log.info(`a* no path from ${start.x},${start.y},${start.z} to ${goal.x},${goal.y},${goal.z}; trying direct walkTo fallback`);
      const target = { x: goal.x + 0.5, y: goal.y, z: goal.z + 0.5 };
      const fb = await walkTo(input, world, target, { sprint: true, allowJump: true, timeoutMs: 6000 });
      if (fb.arrived) return { arrived: true };
      return { arrived: false, reason: `no_path_fallback:${fb.reason ?? "stuck"}` };
    }
    log.debug(`plan: ${plan.path.length} steps, reached=${plan.reached}`);

    let executed = 0;
    for (const step of plan.path) {
      if (executed >= replanEvery) break; // re-plan with fresh world data
      const ok = await executeStep(world, input, step);
      if (!ok.success) {
        log.debug(`step failed (${ok.reason}); re-planning`);
        break;
      }
      executed += 1;
    }

    if (executed === 0) {
      // A* produced a path but we couldn't execute the first step (perched on a
      // ledge, a descent the step-planner balks at, etc.). Walk straight at the
      // goal for a few seconds — this lets the bot drop off ledges and cross
      // awkward terrain. If it makes progress, re-plan from the new spot.
      const beforeD = v3distXZ(v3floor(world.self.position), goal);
      const target = { x: goal.x + 0.5, y: goal.y, z: goal.z + 0.5 };
      const fb = await walkTo(input, world, target, { sprint: true, allowJump: true, timeoutMs: 5000 });
      if (fb.arrived || v3distXZ(v3floor(world.self.position), goal) < 1) return { arrived: true };
      const afterD = v3distXZ(v3floor(world.self.position), goal);
      if (afterD >= beforeD - 0.5) return { arrived: false, reason: `step_blocked:${fb.reason ?? "stuck"}` };
      // else: made progress, loop re-plans from the new position
    }
  }

  return { arrived: false, reason: "timeout" };
}

async function executeStep(world: World, input: InputController, step: PathStep): Promise<{ success: boolean; reason?: string }> {
  const target = { x: step.to.x + 0.5, y: step.to.y, z: step.to.z + 0.5 };
  const isDrop = step.kind === "drop";
  const allowJump = step.kind === "jump" || step.kind === "walk";
  // Drops need the bot to actually lose altitude (server-driven fall), so require
  // Y arrival and allow extra time for the fall to complete.
  const res = await walkTo(input, world, target, {
    sprint: step.kind !== "diagonal",
    allowJump,
    requireArriveY: isDrop,
    timeoutMs: isDrop ? 7000 : 4000,
  });
  if (!res.arrived) return { success: false, reason: res.reason };
  return { success: true };
}
