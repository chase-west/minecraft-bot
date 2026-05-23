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
    if (v3distXZ(start, goal) < 1) {
      return { arrived: true };
    }

    const plan = findPath(world, start, goal, { maxNodes: opts.maxNodes ?? 8000 });
    if (plan.path.length === 0) {
      return { arrived: false, reason: plan.reached ? "already_there" : "no_path" };
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
      return { arrived: false, reason: "step_blocked" };
    }
  }

  return { arrived: false, reason: "timeout" };
}

async function executeStep(world: World, input: InputController, step: PathStep): Promise<{ success: boolean; reason?: string }> {
  const target = { x: step.to.x + 0.5, y: step.to.y, z: step.to.z + 0.5 };
  const allowJump = step.kind === "jump" || step.kind === "walk";
  const res = await walkTo(input, world, target, { sprint: step.kind !== "diagonal", allowJump, timeoutMs: 4000 });
  if (!res.arrived) return { success: false, reason: res.reason };
  return { success: true };
}
