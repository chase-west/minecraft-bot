import type { BedrockClient } from "../connection/client.js";
import type { World } from "../world/world.js";
import type { Vec3 } from "../utils/vec3.js";
import { detectVersionFlags, type VersionFlags } from "../connection/version.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("input");

// Bedrock 1.20+ input bit flags (subset we care about).
// Reference: bedrock-protocol packet schema for PlayerAuthInput.
export const InputFlag = {
  Ascend:      1 << 0,
  Descend:     1 << 1,
  NorthJump:   1 << 2,
  JumpDown:    1 << 3,
  SprintDown:  1 << 4,
  ChangeHeight:1 << 5,
  Jumping:     1 << 6,
  AutoJumpingInWater: 1 << 7,
  Sneaking:    1 << 8,
  SneakDown:   1 << 9,
  Up:          1 << 10,
  Down:        1 << 11,
  Left:        1 << 12,
  Right:       1 << 13,
  UpLeft:      1 << 14,
  UpRight:     1 << 15,
  WantUp:      1 << 16,
  WantDown:    1 << 17,
  WantDownSlow:1 << 18,
  WantUpSlow:  1 << 19,
  Sprinting:   1 << 20,
  AscendBlock: 1 << 21,
  DescendBlock:1 << 22,
  SneakToggleDown: 1 << 23,
  PersistSneak:1 << 24,
  StartSprinting: 1 << 25,
  StopSprinting: 1 << 26,
  StartSneaking: 1 << 27,
  StopSneaking: 1 << 28,
  StartSwimming: 1 << 29,
  StopSwimming: 1 << 30,
} as const;

export interface DesiredMove {
  forward: number; // -1..1
  strafe: number;  // -1..1
  jump: boolean;
  sneak: boolean;
  sprint: boolean;
  lookYaw?: number;   // degrees
  lookPitch?: number; // degrees
}

const TICK_MS = 50; // 20 Hz

export class InputController {
  private tickHandle: NodeJS.Timeout | null = null;
  private tickCount = 0n;
  private lastSent = { x: 0, y: 0, z: 0 };
  private flags: VersionFlags | null = null;
  desired: DesiredMove = { forward: 0, strafe: 0, jump: false, sneak: false, sprint: false };

  constructor(private readonly client: BedrockClient, private readonly world: World) {}

  start(): void {
    if (this.tickHandle !== null) return;
    this.flags = detectVersionFlags(this.client);
    this.tickHandle = setInterval(() => this.tick(), TICK_MS);
    log.info("input ticker started @ 20Hz");
  }

  stop(): void {
    if (this.tickHandle !== null) {
      clearInterval(this.tickHandle);
      this.tickHandle = null;
    }
  }

  setMove(desired: Partial<DesiredMove>): void {
    this.desired = { ...this.desired, ...desired };
  }

  /** Aim at a world position. */
  lookAt(target: Vec3): void {
    const p = this.world.self.position;
    const dx = target.x - p.x;
    const dy = target.y - (p.y + 1.62); // eye height
    const dz = target.z - p.z;
    const horiz = Math.sqrt(dx * dx + dz * dz);
    const yaw = Math.atan2(-dx, dz) * 180 / Math.PI;
    const pitch = -Math.atan2(dy, horiz) * 180 / Math.PI;
    this.desired.lookYaw = yaw;
    this.desired.lookPitch = pitch;
  }

  /** Build a flag object for the protocol's `bitflags` container.
   * The serializer expects { flag_name: boolean } pairs, not a packed bigint. */
  private buildInputData(d: DesiredMove): Record<string, boolean> {
    const flags: Record<string, boolean> = {};
    if (d.forward > 0.1) flags.up = true;
    if (d.forward < -0.1) flags.down = true;
    if (d.strafe < -0.1) flags.left = true;
    if (d.strafe > 0.1) flags.right = true;
    if (d.jump) { flags.jumping = true; flags.jump_down = true; }
    if (d.sneak) { flags.sneak_down = true; flags.sneaking = true; }
    if (d.sprint) { flags.sprint_down = true; flags.sprinting = true; }
    // Always ack any pending teleport. BDS sends a respawn-teleport at login
    // and won't apply client predictions until we set this bit at least once.
    // Cheap to always send; server only consumes it when one is outstanding.
    flags.handled_teleport = true;
    return flags;
  }

  /** Predict the next position from desired horizontal motion. Lets the server validate cheaply. */
  private predictPosition(d: DesiredMove): Vec3 {
    const yawRad = ((this.desired.lookYaw ?? this.world.self.yaw) * Math.PI) / 180;
    const speed = d.sprint ? 0.28 : 0.21; // approx blocks/tick on land
    const fx = Math.sin(-yawRad) * d.forward + Math.cos(-yawRad) * d.strafe;
    const fz = Math.cos(yawRad) * d.forward + Math.sin(yawRad) * d.strafe;
    return {
      x: this.world.self.position.x + fx * speed,
      y: this.world.self.position.y + (d.jump && this.world.self.onGround ? 0.42 : 0),
      z: this.world.self.position.z + fz * speed,
    };
  }

  private tick(): void {
    this.tickCount += 1n;
    const d = this.desired;
    const yaw = d.lookYaw ?? this.world.self.yaw;
    const pitch = d.lookPitch ?? this.world.self.pitch;
    const inputData = this.buildInputData(d);
    const predicted = this.predictPosition(d);
    this.lastSent = predicted;

    // We optimistically update our position; the server will correct via move_player if wrong.
    this.world.self.position = predicted;
    this.world.self.yaw = yaw;
    this.world.self.headYaw = yaw;
    this.world.self.pitch = pitch;

    // Schema (bedrock-protocol 1.26 — see minecraft-data 1.26.10 protocol.json):
    // vec2f keys are {x, z} not {x, y} — even when the semantic axes are
    // pitch/yaw (interact_rotation). Three trailing fields (analogue_move_vector,
    // camera_orientation, raw_move_vector) are REQUIRED unconditionally; if
    // omitted, the serializer reads `undefined.x` and silently emits 0s,
    // leading to the server applying zero motion (bot looks frozen).
    const yawRad = (yaw * Math.PI) / 180;
    const pitchRad = (pitch * Math.PI) / 180;
    const camDir = {
      x: -Math.sin(yawRad) * Math.cos(pitchRad),
      y: -Math.sin(pitchRad),
      z: Math.cos(yawRad) * Math.cos(pitchRad),
    };
    const payload: any = {
      pitch,
      yaw,
      position: predicted,
      move_vector: { x: d.strafe, z: d.forward },
      head_yaw: yaw,
      input_data: inputData,
      input_mode: "mouse",
      play_mode: "normal",
      interaction_model: "crosshair",
      interact_rotation: { x: pitch, z: yaw },
      tick: this.tickCount,
      delta: { x: 0, y: 0, z: 0 },
      analogue_move_vector: { x: d.strafe, z: d.forward },
      camera_orientation: camDir,
      raw_move_vector: { x: d.strafe, z: d.forward },
    };
    try {
      this.client.queue("player_auth_input" as any, payload);
    } catch (err) {
      if (this.tickCount % 40n === 0n) log.warn("player_auth_input write failed", (err as Error).message);
    }
  }
}
