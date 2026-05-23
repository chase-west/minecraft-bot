import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { InputController } from "../actions/input.js";
import type { Vec3 } from "../utils/vec3.js";
import { mineBlock } from "../actions/mine.js";
import { placeBlock } from "../actions/place.js";
import { attackEntity } from "../actions/combat.js";
import { eat } from "../actions/eat.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("ml-act");

export enum ActionId {
  Noop = 0,
  MoveN = 1,
  MoveS = 2,
  MoveE = 3,
  MoveW = 4,
  Jump = 5,
  ToggleSprint = 6,
  MineFront = 7,
  PlaceFront = 8,
  AttackNearest = 9,
  Eat = 10,
  // Combined "walk into a hill" primitive — moves forward AND holds jump so
  // the bot can climb 1-block obstacles. Direction = whichever movement was
  // last issued (or N if none yet).
  MoveForwardJump = 11,
}

export const ACTION_COUNT = 12;

export interface ActionContext {
  client: BedrockClient;
  world: World;
  input: InputController;
}

// Module-private state held across calls.
const state = {
  lastSprint: false,
  // Bedrock yaw 180 = north (matches the "or N if none yet" comment on
  // MoveForwardJump). Previously 0 = south, which contradicted that contract.
  lastMoveYaw: 180 as number, // remembers facing for MoveForwardJump
};

// Bedrock yaw convention: 0 = south (+Z), 90 = west (-X), 180 = north (-Z),
// 270 = east (+X). To move in a direction we face that direction and apply
// forward = 1, then the input controller's prediction step does the rest.
const DIR_YAW: Readonly<Record<ActionId, number | null>> = {
  [ActionId.Noop]: null,
  [ActionId.MoveN]: 180,
  [ActionId.MoveS]: 0,
  [ActionId.MoveE]: 270,
  [ActionId.MoveW]: 90,
  [ActionId.Jump]: null,
  [ActionId.ToggleSprint]: null,
  [ActionId.MineFront]: null,
  [ActionId.PlaceFront]: null,
  [ActionId.AttackNearest]: null,
  [ActionId.Eat]: null,
  [ActionId.MoveForwardJump]: null,
};

function frontBlock(world: World, distance = 1.5): Vec3 {
  const yawRad = (world.self.yaw * Math.PI) / 180;
  const px = world.self.position.x + Math.sin(-yawRad) * distance;
  const py = world.self.position.y;
  const pz = world.self.position.z + Math.cos(yawRad) * distance;
  return { x: Math.floor(px), y: Math.floor(py), z: Math.floor(pz) };
}

export async function executeAction(action: ActionId, ctx: ActionContext): Promise<void> {
  const { client, world, input } = ctx;

  switch (action) {
    case ActionId.Noop: {
      input.setMove({ forward: 0, strafe: 0, jump: false, sneak: false, sprint: state.lastSprint });
      return;
    }
    case ActionId.MoveN:
    case ActionId.MoveS:
    case ActionId.MoveE:
    case ActionId.MoveW: {
      const yaw = DIR_YAW[action];
      if (yaw !== null) state.lastMoveYaw = yaw;
      input.setMove({
        forward: 1,
        strafe: 0,
        jump: false,
        sneak: false,
        sprint: state.lastSprint,
        ...(yaw !== null ? { lookYaw: yaw } : {}),
      });
      return;
    }
    case ActionId.MoveForwardJump: {
      // Move in the direction we last faced AND hold jump — lets the bot
      // climb 1-block obstacles. Server only consumes jump when onGround,
      // so spamming this is safe.
      input.setMove({
        forward: 1,
        strafe: 0,
        jump: true,
        sneak: false,
        sprint: state.lastSprint,
        lookYaw: state.lastMoveYaw,
      });
      return;
    }
    case ActionId.Jump: {
      input.setMove({ forward: 0, strafe: 0, jump: true, sneak: false, sprint: state.lastSprint });
      return;
    }
    case ActionId.ToggleSprint: {
      state.lastSprint = !state.lastSprint;
      input.setMove({ sprint: state.lastSprint });
      return;
    }
    case ActionId.MineFront: {
      const target = frontBlock(world);
      try {
        await mineBlock(client, world, input, target);
      } catch (err) {
        log.warn("mineFront failed", (err as Error).message);
      }
      return;
    }
    case ActionId.PlaceFront: {
      const target = frontBlock(world);
      const against: Vec3 = { x: target.x, y: target.y - 1, z: target.z };
      try {
        await placeBlock(client, world, input, target, against, 1);
      } catch (err) {
        log.warn("placeFront failed", (err as Error).message);
      }
      return;
    }
    case ActionId.AttackNearest: {
      const target = world.nearestHostile(16);
      if (!target) return;
      try {
        await attackEntity(client, world, input, target, { timeoutMs: 1500 });
      } catch (err) {
        log.warn("attackNearest failed", (err as Error).message);
      }
      return;
    }
    case ActionId.Eat: {
      try {
        await eat(client, world);
      } catch (err) {
        log.warn("eat failed", (err as Error).message);
      }
      return;
    }
    default: {
      // Exhaustiveness: any unrecognised int is treated as noop.
      log.warn(`unknown action id ${action as number}`);
      return;
    }
  }
}
