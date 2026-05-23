import type { BedrockClient } from "../connection/client.js";
import { World } from "../world/world.js";
import { attachPerception } from "../world/perception.js";
import { attachChunkStream, ChunkCache } from "../world/chunk.js";
import { InputController } from "../actions/input.js";
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
import { Explorer } from "../ml/explorer.js";
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
    const scratch = new Float32Array(601);
    this.shadowHandle = setInterval(() => {
      try {
        const obs = this.encoder.encode(this.world, this.blockIdRegistry, scratch);
        const r = this.reward.step(this.world, Date.now());
        const ctx: ActionContext = { client: this.client, world: this.world, input: this.input };

        // In online mode, the policy may become loaded later via hot-reload;
        // promote ourselves from explore → learned the moment it shows up.
        if (isOnline && !this.learnedActive && this.policy.isLoaded()) {
          this.learnedActive = true;
          this.exploreActive = false;
          log.info("online: learned policy is now ready — switching from explorer to ε-greedy");
        }

        const now = Date.now();
        const useLearned = this.learnedActive && !this.actInFlight;
        const inStickyWindow = now < this.stickyUntil;
        const epsilonRoll = isOnline && useLearned && !inStickyWindow && Math.random() < this.onlineEpsilon;

        if (useLearned && inStickyWindow) {
          // Still holding the previous policy action — re-execute and log it
          // so the input ticker keeps the movement bit set, but skip inference.
          this.trajLogger!.log(obs, this.stickyAction, r);
          executeAction(this.stickyAction, ctx).catch((err) => {
            if (this.shadowSamples < 5) log.warn(`sticky action ${this.stickyAction} failed: ${(err as Error).message}`);
          });
        } else if (useLearned && !epsilonRoll) {
          // Fire-and-forget inference. Snapshot obs before handing off.
          const obsCopy = new Float32Array(obs);
          this.actInFlight = true;
          this.policy.act(obsCopy)
            .then(async (actionId) => {
              this.stickyAction = actionId;
              this.stickyUntil = Date.now() + this.policyStickyMs;
              this.trajLogger?.log(obsCopy, actionId, r);
              try { await executeAction(actionId, ctx); }
              catch (err) { log.warn(`executeAction(${actionId}) failed: ${(err as Error).message}`); }
            })
            .catch((err) => {
              log.warn(`policy.act failed: ${(err as Error).message}`);
              this.trajLogger?.log(obsCopy, ActionId.Noop, r);
            })
            .finally(() => { this.actInFlight = false; });
        } else if (this.exploreActive || epsilonRoll) {
          // Sticky random-walk — pure explore mode, or ε-greedy fresh-data
          // sampling inside online mode.
          const actionId = this.explorer.nextAction();
          this.trajLogger!.log(obs, actionId, r);
          executeAction(actionId, ctx).catch((err) => {
            if (this.shadowSamples < 5) log.warn(`explore action ${actionId} failed: ${(err as Error).message}`);
          });
        } else if (!this.learnedActive) {
          // Shadow path: GOAP drives the bot from run(); we only log intent.
          const intent = readIntent();
          this.trajLogger!.log(obs, intent, r);
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
      if (this.llm.isEnabled() && Date.now() - lastPlanAt > 8_000) {
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

      await sleep(500);
    }
  }

  stop(): void {
    this.running = false;
    this.input.stop();
    if (this.shadowHandle !== null) { clearInterval(this.shadowHandle); this.shadowHandle = null; }
    if (this.trajLogger) { this.trajLogger.close(); this.trajLogger = null; }
    this.blockIdRegistry.save();
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}
