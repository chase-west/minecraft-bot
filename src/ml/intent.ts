import type { ActionId } from "./actions.js";

/**
 * Shared "what is the bot currently trying to do" channel. The GOAP layer
 * publishes its current primitive intent here (via setIntent) so the shadow
 * trajectory logger can record (obs, action) pairs without instrumenting every
 * call site. The learned policy bypasses this entirely.
 *
 * NOTE: we use a type-only import for ActionId and the literal value 0 (= Noop)
 * to avoid a circular module-load dependency through actions/{mine,move,...}.ts
 * which in turn import this file.
 */
const NOOP = 0 as ActionId;
let current: ActionId = NOOP;
let untilTs = 0;

const DECAY_MS = 250;

export function setIntent(action: ActionId, holdMs = 200): void {
  current = action;
  untilTs = Date.now() + holdMs;
}

export function readIntent(): ActionId {
  if (Date.now() > untilTs + DECAY_MS) return NOOP;
  return current;
}

export function clearIntent(): void {
  current = NOOP;
  untilTs = 0;
}
