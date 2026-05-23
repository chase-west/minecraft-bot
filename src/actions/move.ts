import type { InputController } from "./input.js";
import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3distXZ } from "../utils/vec3.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("move");

const ARRIVE_DIST = 0.4;
const STUCK_TIMEOUT_MS = 4000;
const POLL_MS = 50;

export interface MoveOptions {
  sprint?: boolean;
  allowJump?: boolean;
  timeoutMs?: number;
}

/** Walk in a straight line toward a target block-position. Returns when within ARRIVE_DIST or on timeout. */
export async function walkTo(
  input: InputController,
  world: World,
  target: Vec3,
  opts: MoveOptions = {},
): Promise<{ arrived: boolean; reason?: string }> {
  const deadline = Date.now() + (opts.timeoutMs ?? 30_000);
  let lastDist = Infinity;
  let lastProgressAt = Date.now();

  while (Date.now() < deadline) {
    const p = world.self.position;
    const dist = v3distXZ(p, target);
    if (dist < ARRIVE_DIST) {
      input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
      log.debug(`arrived at ${target.x.toFixed(2)},${target.z.toFixed(2)}`);
      return { arrived: true };
    }

    if (dist < lastDist - 0.05) {
      lastDist = dist;
      lastProgressAt = Date.now();
    } else if (Date.now() - lastProgressAt > STUCK_TIMEOUT_MS) {
      input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
      return { arrived: false, reason: "stuck" };
    }

    input.lookAt({ x: target.x, y: p.y + 1.62, z: target.z });
    const needJump = (opts.allowJump ?? true) && target.y - p.y > 0.5 && world.self.onGround;
    input.setMove({
      forward: 1,
      strafe: 0,
      jump: needJump,
      sneak: false,
      sprint: opts.sprint ?? true,
    });
    // Publish a move intent for the shadow trainer: classify by current heading.
    const dx = target.x - p.x, dz = target.z - p.z;
    const action = needJump ? ActionId.Jump : Math.abs(dx) > Math.abs(dz)
      ? (dx >= 0 ? ActionId.MoveE : ActionId.MoveW)
      : (dz >= 0 ? ActionId.MoveS : ActionId.MoveN);
    setIntent(action, 200);

    await new Promise((r) => setTimeout(r, POLL_MS));
  }

  input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
  return { arrived: false, reason: "timeout" };
}

/** Stop all motion. */
export function halt(input: InputController): void {
  input.setMove({ forward: 0, strafe: 0, jump: false, sneak: false, sprint: false });
}
