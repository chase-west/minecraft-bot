import type { InputController } from "./input.js";
import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";
import { v3distXZ } from "../utils/vec3.js";
import { setIntent } from "../ml/intent.js";
import { ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("move");

const ARRIVE_DIST = 0.4;
const ARRIVE_Y = 1.0;
const STUCK_TIMEOUT_MS = 4000;
const POLL_MS = 50;

export interface MoveOptions {
  sprint?: boolean;
  allowJump?: boolean;
  timeoutMs?: number;
  /** Also require vertical proximity to count as arrived. Set for drop steps so
   * the bot waits to actually fall onto the lower block instead of reporting
   * "arrived" the instant it's XZ-over the target while still high on the ledge.
   * The bot has no client-side gravity (server is authoritative for Y), so a
   * descent only happens by standing XZ-over the landing cell and letting the
   * server pull us down. Without this, drop steps silently no-op. */
  requireArriveY?: boolean;
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
    const yClose = !opts.requireArriveY || Math.abs(p.y - target.y) <= ARRIVE_Y;
    if (dist < ARRIVE_DIST && yClose) {
      input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
      log.debug(`arrived at ${target.x.toFixed(2)},${target.z.toFixed(2)}`);
      return { arrived: true };
    }

    // Drop step: XZ-over the landing cell but still too high. Stop pushing
    // forward (so we don't overshoot) and wait for the server's gravity to drop
    // us onto the lower block — our stale high-Y prediction over an empty column
    // makes the server pull us down. This is vertical progress, not a stall, so
    // keep the stuck timer fresh.
    if (dist < ARRIVE_DIST && !yClose) {
      input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
      input.lookAt({ x: target.x, y: target.y + 1.62, z: target.z });
      lastProgressAt = Date.now();
      await new Promise((r) => setTimeout(r, POLL_MS));
      continue;
    }

    if (dist < lastDist - 0.05) {
      lastDist = dist;
      lastProgressAt = Date.now();
    } else if (Date.now() - lastProgressAt > STUCK_TIMEOUT_MS) {
      input.setMove({ forward: 0, strafe: 0, jump: false, sprint: false });
      return { arrived: false, reason: "stuck" };
    }

    input.lookAt({ x: target.x, y: p.y + 1.62, z: target.z });
    // Jump when the planned step rises, OR when we've stalled against an
    // obstacle: A* sometimes classifies a 1-block lip as a flat "walk" on the
    // uneven ground around tree bases, so the bot pushes forward into it and the
    // server clamps us in place (a stream of identical correct_player_move_prediction).
    // The stall case intentionally drops the onGround guard — correct_player_move_prediction
    // never refreshes onGround, so requiring it would suppress the very jump that
    // frees us. predictPosition still gates the +0.42 rise on onGround, so an
    // airborne jump bit is a harmless no-op.
    const stalled = Date.now() - lastProgressAt > 600;
    const needJump = (opts.allowJump ?? true) && (target.y - p.y > 0.5 || stalled);
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
