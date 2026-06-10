import type { BedrockClient } from "../connection/client.js";
import { World } from "../world/world.js";
import { attachPerception, detachPerception } from "../world/perception.js";
import { attachChunkStream, ChunkCache } from "../world/chunk.js";
import { InputController } from "../actions/input.js";
import { walkTo } from "../actions/move.js";
import { ALL_ACTIONS } from "../goap/actions/index.js";
import { ALL_GOALS } from "../goap/goals.js";
import { plan, selectGoal } from "../goap/planner.js";
import { executePlan, type RuntimeCtx } from "../goap/executor.js";
import { sense } from "../goap/sensors.js";
import { LLMPlanner } from "../llm/planner.js";
import { loadPolicy, type PolicyHandle } from "../ml/runtime.js";
import { RecipeRegistry } from "../crafting/registry.js";
import { attachCraftResponses } from "../crafting/craft.js";
import { attachContainerEvents } from "../crafting/container.js";
import { BlockIdRegistry } from "../ml/blockIdRegistry.js";
import { Encoder } from "../ml/encoder.js";
import { RewardCalculator } from "../ml/reward.js";
import { TrajectoryLogger } from "../ml/trajectoryLogger.js";
import { readIntent } from "../ml/intent.js";
import { LearnedPolicy } from "../ml/policy.js";
import { Explorer, type CraftHint } from "../ml/explorer.js";
import { executeAction, type ActionContext, ActionId } from "../ml/actions.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("agent");

export class Agent {
  private readonly world = new World();
  private readonly chunks = new ChunkCache();
  private readonly input: InputController;
  private readonly llm = new LLMPlanner();
  private readonly recipes = new RecipeRegistry();
  private combatPolicy: PolicyHandle | null = null;
  private running = false;

  // POLICY_MODE options:
  //   explore — sticky random-walk policy drives the bot (default). Generates
  //             diverse trajectories with reward signal for bootstrap training.
  //   shadow  — GOAP drives the bot; we log GOAP's intents as supervision.
  //   learned — ONNX policy drives the bot. Falls back to explore if model
  //             file missing.
  //   online  — like learned, but ε-greedy with the Explorer for fresh data,
  //             and hot-reloads the ONNX as the DQN trainer rewrites it.
  //             Falls back to explore until the first model appears.
  private readonly mlMode = (process.env.POLICY_MODE ?? "explore").toLowerCase();
  private readonly onlineEpsilon = Number(process.env.ONLINE_EPSILON ?? "0.15");
  // Hold each policy-chosen action for this many ms before re-querying. Matches
  // how the explorer sticks on actions: without this the policy flickers
  // through 10 different actions per second and the bot looks frozen.
  private readonly policyStickyMs = Number(process.env.POLICY_STICKY_MS ?? "400");
  private stickyAction: ActionId = ActionId.Noop;
  private stickyUntil = 0;
  // Reward accrued since the last logged trajectory row. The reward calculator
  // runs every 100ms tick, but in learned/online mode rows are only written at
  // decision boundaries (~every POLICY_STICKY_MS). Without accumulation the
  // reward earned on sticky ticks (e.g. a log landing in the inventory mid-
  // window) was silently dropped, so the DQN never saw most of the payoff.
  private rewardAccum = 0;
  private readonly blockIdRegistry = new BlockIdRegistry();
  private readonly encoder = new Encoder();
  private readonly reward = new RewardCalculator();
  private readonly policy = new LearnedPolicy(process.env.POLICY_PATH ?? "models/policy.onnx");
  private readonly explorer = new Explorer();
  private trajLogger: TrajectoryLogger | null = null;
  private shadowHandle: NodeJS.Timeout | null = null;
  private shadowSamples = 0;
  private learnedActive = false;
  private exploreActive = false;
  private actInFlight = false;
  private goalFailStreak = 0;
  // Set by the bot_died event listener (perception emits this when health
  // crosses to ≤0). The next trajectory log call consumes the flag and writes
  // done=true so the DQN trainer treats that transition as terminal instead of
  // bootstrap-blending the -50 death reward through gamma*Q(next).
  private markTerminal = false;

  constructor(private readonly client: BedrockClient) {
    this.input = new InputController(client, this.world);
  }

  /** Bind every packet listener. MUST be called BEFORE waiting for spawn so
   * start_game / crafting_data / level_chunk events fire into our handlers. */
  async attachHandlers(): Promise<void> {
    attachPerception(this.client, this.world);
    attachChunkStream(this.client, this.world, this.chunks);
    attachContainerEvents(this.client);
    attachCraftResponses(this.client);
    this.recipes.attach(this.client);
    // Perception emits "bot_died" when health crosses to ≤0. Latch a flag so
    // the next trajectory log marks the transition terminal — critical for
    // DQN learning to attribute the -50 death reward without bootstrap-bleed.
    (this.client as any).on?.("bot_died", () => {
      this.markTerminal = true;
    });
  }

  /** Called once the spawn packet has been received. Safe to start sending
   * input now, load ML policies, etc. */
  async afterSpawn(): Promise<void> {
    const rid = this.world.self.runtimeEntityId ?? 0n;
    log.info(`runtime_entity_id=${rid}`);
    this.input.start();
    this.combatPolicy = await loadPolicy("combat_targeting");

    // Try loading the learned policy when we're in learned/online mode. In
    // other modes we don't bother — the policy file may not exist yet.
    let policyLoaded = false;
    if (this.mlMode === "learned") {
      policyLoaded = await this.policy.load();
      this.learnedActive = policyLoaded;
      if (!policyLoaded) {
        log.warn("POLICY_MODE=learned but model failed to load — falling back to explore behavior");
        this.exploreActive = true;
      }
    } else if (this.mlMode === "online") {
      // Online RL: drive with the learned policy if it exists, otherwise
      // explore until the DQN trainer publishes one. Either way, watch the
      // file so we hot-swap as soon as a fresh model lands.
      policyLoaded = await this.policy.load();
      this.policy.watchForReload();
      this.learnedActive = policyLoaded;
      this.exploreActive = !policyLoaded;
      log.info(
        `online mode: policy_loaded=${policyLoaded} epsilon=${this.onlineEpsilon} ` +
        `${policyLoaded ? "(learned drives, explorer for ε-greedy)" : "(no model yet — exploring while we wait)"}`,
      );
    } else if (this.mlMode === "explore") {
      this.exploreActive = true;
    }
    log.info(`policy loaded=${policyLoaded}, explore_active=${this.exploreActive}, online=${this.mlMode === "online"}`);

    // ML tick loop: encode obs → pick action → execute → compute reward → log.
    // The action source depends on mode:
    //   learned → ONNX policy (argmax)
    //   online  → ONNX policy with ε-explorer for fresh data + hot-reload.
    //             Falls through to explorer until the first model lands.
    //   explore → sticky random-walk Explorer
    //   shadow  → record GOAP's intent (GOAP main loop drives the bot)
    this.trajLogger = new TrajectoryLogger("data/online");
    const isOnline = this.mlMode === "online";
    const scratch = new Float32Array(605);
    this.shadowHandle = setInterval(() => {
      try {
        const obs = this.encoder.encode(this.world, this.blockIdRegistry, scratch);
        this.rewardAccum += this.reward.step(this.world, Date.now(), this.stickyAction);
        const r = this.rewardAccum;
        const ctx: ActionContext = { client: this.client, world: this.world, input: this.input, recipes: this.recipes };

        // In online mode, the policy may become loaded later via hot-reload;
        // promote ourselves from explore → learned the moment it shows up.
        if (isOnline && !this.learnedActive && this.policy.isLoaded()) {
          this.learnedActive = true;
          this.exploreActive = false;
          log.info("online: learned policy is now ready — switching from explorer to ε-greedy");
        }

        const now = Date.now();
        // Sticky re-execution only needs a policy to repeat; it does NOT need
        // !actInFlight (an in-flight policy.act has already settled stickyAction
        // for the current decision). Keeping these decoupled means a slow
        // inference doesn't blackhole the tick — sticky keeps firing at 10Hz.
        const inStickyWindow = now < this.stickyUntil;
        const useLearned = this.learnedActive && !this.actInFlight;
        const epsilonRoll = isOnline && useLearned && !inStickyWindow && Math.random() < this.onlineEpsilon;

        if (this.learnedActive && inStickyWindow) {
          // Still inside the sticky window of the most recent policy decision.
          // Re-fire executeAction so the input ticker holds the movement bit,
          // but do NOT log a new trajectory row: the (obs, action) pair was
          // already canonical at dispatch time. Logging a fresh obs paired with
          // the stale action would teach the DQN Q(s_{t+k}, a_t), which is wrong.
          executeAction(this.stickyAction, ctx).catch((err) => {
            if (this.shadowSamples < 5) log.warn(`sticky action ${this.stickyAction} failed: ${(err as Error).message}`);
          });
        } else if (useLearned && !epsilonRoll) {
          // Fire-and-forget inference. Snapshot obs before handing off. The
          // reward accumulator resets NOW (not in .then) so ticks that elapse
          // during inference accrue to the next row, not this one.
          const obsCopy = new Float32Array(obs);
          const terminal = this.markTerminal; this.markTerminal = false;
          this.rewardAccum = 0;
          this.actInFlight = true;
          this.policy.act(obsCopy)
            .then(async (actionId) => {
              this.stickyAction = actionId;
              this.stickyUntil = Date.now() + this.policyStickyMs;
              this.trajLogger?.log(obsCopy, actionId, r, terminal);
              try { await executeAction(actionId, ctx); }
              catch (err) { log.warn(`executeAction(${actionId}) failed: ${(err as Error).message}`); }
            })
            .catch((err) => {
              log.warn(`policy.act failed: ${(err as Error).message}`);
              this.trajLogger?.log(obsCopy, ActionId.Noop, r, terminal);
            })
            .finally(() => { this.actInFlight = false; });
        } else if (this.exploreActive || epsilonRoll) {
          // Sticky random-walk — pure explore mode, or ε-greedy fresh-data
          // sampling inside online mode. The craft hint biases exploration
          // toward crafts the bot currently has materials for.
          const actionId = this.explorer.nextAction(now, this.craftHint());
          const terminal = this.markTerminal; this.markTerminal = false;
          // Track the chosen action so the reward calculator's action-aware
          // shaping (e.g. the MineFront aim bonus) sees it next tick.
          this.stickyAction = actionId;
          this.trajLogger!.log(obs, actionId, r, terminal);
          this.rewardAccum = 0;
          executeAction(actionId, ctx).catch((err) => {
            if (this.shadowSamples < 5) log.warn(`explore action ${actionId} failed: ${(err as Error).message}`);
          });
        } else if (!this.learnedActive) {
          // Shadow path: GOAP drives the bot from run(); we only log intent.
          const intent = readIntent();
          const terminal = this.markTerminal; this.markTerminal = false;
          this.trajLogger!.log(obs, intent, r, terminal);
          this.rewardAccum = 0;
        }

        this.shadowSamples++;
        if (this.shadowSamples % 600 === 0) {
          const tag = isOnline ? "online" : this.learnedActive ? "learned" : this.exploreActive ? "explore" : "shadow";
          const p = this.world.self.position;
          log.info(
            `ml ${tag}: ${this.shadowSamples} samples ` +
            `pos=(${p.x.toFixed(1)},${p.y.toFixed(1)},${p.z.toFixed(1)}) ` +
            `yaw=${this.world.self.yaw.toFixed(0)} onGround=${this.world.self.onGround}`,
          );
          this.blockIdRegistry.save();
        }
      } catch (err) {
        if (this.shadowSamples < 5) log.warn("ml tick failed", (err as Error).message);
      }
    }, 100);
    const modeTag = isOnline ? "online" : this.learnedActive ? "learned" : this.exploreActive ? "explore" : "shadow";
    log.info(`ml ${modeTag} mode active: logging to data/online/*.jsonl @ 10Hz`);
    log.info(`agent initialized (llm=${this.llm.isEnabled() ? "on" : "off"}, combat_policy=${this.combatPolicy ? "loaded" : "fallback"}, ml_mode=${this.mlMode}, policy_loaded=${policyLoaded}, pos=${JSON.stringify(this.world.self.position)})`);
  }

  /**
   * Cheap inventory snapshot used to bias the Explorer toward feasible crafts.
   * Counts by name-substring (slot names look like "oak_planks"/"stick"). The
   * crafting-table block scan only runs when a pickaxe is otherwise craftable,
   * to keep this off the hot path.
   */
  private craftHint(): CraftHint {
    let logs = 0, planks = 0, sticks = 0;
    for (const slot of this.world.inventory.values()) {
      const n = slot.name;
      if (!n || slot.count <= 0) continue;
      if (n.includes("log")) logs += slot.count;
      else if (n.includes("planks")) planks += slot.count;
      else if (n.includes("stick")) sticks += slot.count;
    }
    let hasTable = false;
    if (planks >= 3 && sticks >= 2) {
      for (const b of this.world.blocks.values()) {
        if (b.name && b.name.includes("crafting_table")) { hasTable = true; break; }
      }
    }
    return { logs, planks, sticks, hasTable };
  }

  /** Backwards-compatible single-call init (if no perception ordering needed). */
  async init(): Promise<void> {
    await this.attachHandlers();
    await this.afterSpawn();
  }

  async run(): Promise<void> {
    this.running = true;
    let lastPlanAt = 0;
    let lastGoalName = "";

    // When a non-GOAP controller (learned policy OR explorer) is driving,
    // the 100ms setInterval in afterSpawn() is the control loop. We idle here
    // so GOAP doesn't fight it for input.setMove() calls.
    if (this.learnedActive || this.exploreActive) {
      const tag = this.learnedActive ? "learned policy" : "explorer";
      log.info(`${tag} active — GOAP main loop disabled`);
      while (this.running) {
        await sleep(1000);
      }
      return;
    }

    while (this.running) {
      const state = sense(this.world);

      // Pick a goal — LLM (if enabled) augments GOAP's utility scorer every few seconds.
      let goal = selectGoal(state, ALL_GOALS);
      if (this.llm.isEnabled() && Date.now() - lastPlanAt > 300_000) {
        const suggestion = await this.llm.suggestGoal(
          state,
          ALL_GOALS.map((g) => g.name),
        );
        if (suggestion?.kind === "goap_goal") {
          const llmGoal = ALL_GOALS.find((g) => g.name === suggestion.goalName);
          if (llmGoal && !llmGoal.satisfied(state)) goal = llmGoal;
        }
        lastPlanAt = Date.now();
      }

      if (!goal) {
        log.debug("no goal pending — idling 2s");
        await sleep(2000);
        continue;
      }

      if (goal.name !== lastGoalName) {
        log.info(`pursuing goal: ${goal.name}`);
        lastGoalName = goal.name;
      }

      const planResult = plan(state, goal, ALL_ACTIONS, { maxNodes: 4000 });
      if (!planResult || planResult.steps.length === 0) {
        log.warn(`no plan for goal ${goal.name}; backing off`);
        await sleep(3000);
        continue;
      }
      log.info(`plan(${goal.name}): [${planResult.steps.map((a) => a.name).join(" → ")}] cost=${planResult.totalCost}`);

      const ctx: RuntimeCtx = { client: this.client, world: this.world, input: this.input, recipes: this.recipes };
      const result = await executePlan(planResult, ctx);
      log.info(`plan result: completed=${result.completed}/${planResult.steps.length} failedAt=${result.failedAt} reason=${result.reason ?? "ok"}`);

      // Back off when a goal makes zero progress, so we don't hammer the same
      // failing plan (e.g. unreachable tree) hundreds of times per minute. Each
      // consecutive zero-progress failure adds 2s of cooldown, capped at 15s,
      // during which the LLM gets a chance to pick a different goal.
      if (result.completed === 0) {
        this.goalFailStreak++;
        const backoff = Math.min(2000 * this.goalFailStreak, 15000);
        log.warn(`goal ${goal.name} made no progress (streak=${this.goalFailStreak}); backing off ${backoff}ms`);
        // Force the next iteration to re-query the LLM for a fresh goal.
        lastPlanAt = 0;
        // Only relocate after SEVERAL consecutive failures. Wandering on the
        // first failure yanks the bot away before perception and navigation can
        // settle — e.g. right after a teleport (chunk-follow briefly drops the
        // target while it reprojects) or while A* is still routing to a tree the
        // bot can already see. Give it a couple of cycles to stay put and make
        // progress; only treat it as genuinely stuck (and wander to fresh
        // ground) once it has failed repeatedly from the same place.
        if (this.goalFailStreak >= 3) {
          await this.wander();
        } else {
          await sleep(800);
        }
      } else {
        this.goalFailStreak = 0;
        await sleep(500);
      }
    }
  }

  /** Walk in a random horizontal direction to escape a stuck goal. Keeps the bot
   * visibly moving and relocates it so it can perceive (and reach) new targets. */
  private async wander(): Promise<void> {
    const ang = Math.random() * Math.PI * 2;
    const dist = 12;
    const p = this.world.self.position;
    const target = { x: p.x + Math.cos(ang) * dist, y: p.y, z: p.z + Math.sin(ang) * dist };
    log.info(`wandering toward (${target.x.toFixed(0)},${target.z.toFixed(0)}) to escape stuck goal`);
    await walkTo(this.input, this.world, target, { sprint: true, allowJump: true, timeoutMs: 5000 });
  }

  stop(): void {
    this.running = false;
    this.input.stop();
    if (this.shadowHandle !== null) { clearInterval(this.shadowHandle); this.shadowHandle = null; }
    if (this.trajLogger) { this.trajLogger.close(); this.trajLogger = null; }
    this.policy.stopWatching();
    detachPerception(this.client);
    this.blockIdRegistry.save();
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
