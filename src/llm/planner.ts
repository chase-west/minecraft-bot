import * as fs from "node:fs/promises";
import * as path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { BotState } from "../goap/types.js";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("llm");

export interface SkillSpec {
  name: string;
  description: string;
  /** Pre-flight check on world state; cheap and synchronous. */
  applicable: (s: BotState) => boolean;
  /** Optional cooldown to prevent thrashing. */
  cooldownMs?: number;
}

export type GoalSuggestion =
  | { kind: "goap_goal"; goalName: string }
  | { kind: "skill"; skillName: string };

type Provider = "anthropic" | "gemini" | "none";

const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";
const DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001";
const GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models";

/** Static instructions shared by every call — kept separate so Claude can cache it. */
const SYSTEM_PROMPT = [
  "You are the high-level planner for a Minecraft Bedrock survival bot.",
  "Pick the single best goal to pursue right now from the goals offered.",
  "",
  "Priority order:",
  "1. Survival: if health<6 or hunger<6, eat or flee.",
  "2. Tool tier progression: no pickaxe -> get wood -> craft pickaxe -> mine stone.",
  "3. Resource gathering for the next tier.",
  "4. Exploration / shelter / night safety once survival needs are met.",
].join("\n");

interface GeminiResponse {
  candidates?: Array<{
    content?: { parts?: Array<{ text?: string }> };
    finishReason?: string;
  }>;
  promptFeedback?: { blockReason?: string };
  error?: { message?: string; code?: number; status?: string };
}

/**
 * High-level planner — called sparingly (every few seconds or on goal change),
 * not per tick. Its job is to pick what to pursue; GOAP and the executor handle how.
 *
 * Provider-agnostic. Selection priority at construction:
 *   1. ANTHROPIC_API_KEY set -> Anthropic Claude (model from ANTHROPIC_MODEL).
 *   2. else GEMINI_API_KEY set -> Google Gemini via REST (model from LLM_MODEL).
 *   3. else -> disabled; GOAP utility scorer drives goals.
 */
export class LLMPlanner {
  private readonly provider: Provider;
  private readonly apiKey: string | null;
  private readonly model: string;
  private readonly skillsDir: string;
  private readonly anthropic: Anthropic | null = null;
  private lastCallAt = 0;
  private readonly minIntervalMs = 5_000;
  /** When rate-limited (HTTP 429 / overloaded), we suppress all calls until this timestamp. */
  private disabledUntilMs = 0;

  constructor(opts: { apiKey?: string; model?: string; skillsDir?: string } = {}) {
    const anthropicKey = process.env.ANTHROPIC_API_KEY ?? null;
    const geminiKey = opts.apiKey ?? process.env.GEMINI_API_KEY ?? null;

    if (anthropicKey) {
      this.provider = "anthropic";
      this.apiKey = anthropicKey;
      this.model = opts.model ?? process.env.ANTHROPIC_MODEL ?? DEFAULT_ANTHROPIC_MODEL;
      this.anthropic = new Anthropic({ apiKey: anthropicKey });
      log.info(`LLM planner active: provider=anthropic model=${this.model}`);
    } else if (geminiKey) {
      this.provider = "gemini";
      this.apiKey = geminiKey;
      this.model = opts.model ?? process.env.LLM_MODEL ?? DEFAULT_GEMINI_MODEL;
      log.info(`LLM planner active: provider=gemini model=${this.model}`);
    } else {
      this.provider = "none";
      this.apiKey = null;
      this.model = DEFAULT_ANTHROPIC_MODEL;
      log.info(
        "Neither ANTHROPIC_API_KEY nor GEMINI_API_KEY set — LLM planner disabled, falling back to GOAP-only",
      );
    }

    this.skillsDir = opts.skillsDir ?? path.join(process.cwd(), "src/llm/skills");
  }

  isEnabled(): boolean {
    return this.provider !== "none";
  }

  /**
   * Lightweight startup probe: confirm the active provider's key works.
   * Logs success/failure. Returns true on a parseable response.
   */
  async healthCheck(): Promise<boolean> {
    if (this.provider === "none") {
      log.info("healthCheck skipped — no LLM provider configured");
      return false;
    }
    try {
      if (this.provider === "anthropic") {
        const resp = await this.anthropic!.beta.messages.create({
          model: this.model,
          max_tokens: 8,
          messages: [{ role: "user", content: "Reply with the single word: ok" }],
        });
        const ok = textFromAnthropic(resp).trim().length > 0;
        if (ok) log.info(`Anthropic healthCheck ok (model=${this.model})`);
        else log.warn("Anthropic healthCheck returned empty text", resp);
        return ok;
      }
      const resp = await this.callGemini(
        {
          contents: [{ role: "user", parts: [{ text: "Reply with the single word: ok" }] }],
          generationConfig: { temperature: 0, maxOutputTokens: 8 },
        },
        5_000,
      );
      const text = extractText(resp) ?? "";
      const ok = text.trim().length > 0;
      if (ok) log.info(`Gemini healthCheck ok (model=${this.model})`);
      else log.warn("Gemini healthCheck returned empty text", resp);
      return ok;
    } catch (err) {
      log.warn(`${this.provider} healthCheck failed`, err);
      return false;
    }
  }

  async suggestGoal(state: BotState, availableGoals: string[]): Promise<GoalSuggestion | null> {
    if (this.provider === "none") return null;
    if (availableGoals.length === 0) return null;

    const now = Date.now();
    // Rate-limited earlier: stay silent and skip the API call entirely.
    if (now < this.disabledUntilMs) return null;
    if (now - this.lastCallAt < this.minIntervalMs) return null;
    this.lastCallAt = now;

    let goalName: string | undefined;
    try {
      goalName =
        this.provider === "anthropic"
          ? await this.suggestViaAnthropic(state, availableGoals)
          : await this.suggestViaGemini(state, availableGoals);
    } catch (err) {
      this.handleError(err);
      return null;
    }

    if (typeof goalName !== "string" || !availableGoals.includes(goalName)) {
      log.warn(`${this.provider} returned invalid goalName: ${JSON.stringify(goalName)}`);
      return null;
    }

    log.info(`LLM picked goal: ${goalName}`);
    await this.recordDecision(state, goalName);
    return { kind: "goap_goal", goalName };
  }

  /**
   * Anthropic path: forced tool-use. We register a single `select_goal` tool whose
   * `goalName` parameter is constrained to an enum of the available goals, and force
   * the model to call it. This guarantees a structured, schema-valid answer instead of
   * relying on the model to emit well-formed JSON in free text.
   */
  private async suggestViaAnthropic(
    state: BotState,
    availableGoals: string[],
  ): Promise<string | undefined> {
    const resp = await this.anthropic!.beta.messages.create({
      model: this.model,
      max_tokens: 256,
      // Enable prompt caching (SDK 0.30.x exposes cache_control only on the beta route).
      betas: ["prompt-caching-2024-07-31"],
      // Static instructions live in a cached system block; only the dynamic per-call
      // state goes in the user message. cache_control marks the cacheable breakpoint.
      system: [
        {
          type: "text",
          text: SYSTEM_PROMPT,
          cache_control: { type: "ephemeral" },
        },
      ],
      tools: [
        {
          name: "select_goal",
          description: "Pick the single best goal for the bot right now",
          input_schema: {
            type: "object",
            properties: { goalName: { type: "string", enum: [...availableGoals] } },
            required: ["goalName"],
          },
        },
      ],
      tool_choice: { type: "tool", name: "select_goal" },
      messages: [{ role: "user", content: buildUserContext(state, availableGoals) }],
    });

    for (const block of resp.content) {
      if (block.type === "tool_use" && block.name === "select_goal") {
        const input = block.input as { goalName?: unknown };
        if (typeof input?.goalName === "string") return input.goalName;
      }
    }
    log.warn("Anthropic returned no select_goal tool_use block", resp.content);
    return undefined;
  }

  /** Gemini path: REST generateContent with a JSON response schema (unchanged behavior). */
  private async suggestViaGemini(
    state: BotState,
    availableGoals: string[],
  ): Promise<string | undefined> {
    const prompt = `${SYSTEM_PROMPT}\n\n${buildUserContext(state, availableGoals)}\n\nReply with JSON only: {"goalName": "<one of the available goals>"}`;
    const resp = await this.callGemini(
      {
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              goalName: { type: "string", enum: [...availableGoals] },
            },
            required: ["goalName"],
          },
          temperature: 0.3,
          maxOutputTokens: 200,
        },
      },
      10_000,
    );

    const text = extractText(resp);
    if (!text) {
      log.warn("Gemini returned no text", resp);
      return undefined;
    }
    try {
      const parsed = JSON.parse(text) as { goalName?: unknown };
      return typeof parsed?.goalName === "string" ? parsed.goalName : undefined;
    } catch (parseErr) {
      log.warn(`Gemini JSON parse failed: ${(parseErr as Error).message} — raw: ${text}`);
      return undefined;
    }
  }

  /** Shared error handling: detect rate-limit / overload and arm the cooldown. */
  private handleError(err: unknown): void {
    const msg = (err as Error)?.message ?? String(err);
    const status = (err as { status?: number })?.status;
    const isRateLimited =
      status === 429 ||
      status === 529 ||
      /\b429\b|\b529\b|rate limit|quota|RESOURCE_EXHAUSTED|overloaded/i.test(msg);
    if (isRateLimited) {
      // Honor a "retry in Ns" hint if present, else default to 1 hour.
      const m = msg.match(/retry(?:\s*in)?\s*(\d+(?:\.\d+)?)\s*s/i);
      const retryMs = m ? Math.ceil(parseFloat(m[1]!) * 1000) : 3_600_000;
      this.disabledUntilMs = Date.now() + retryMs;
      const mins = Math.ceil(retryMs / 60_000);
      log.warn(
        `${this.provider} rate-limited/overloaded; disabling planner for ${mins}min — GOAP scorer drives goals meanwhile`,
      );
      return;
    }
    log.warn(`${this.provider} call failed`, err);
  }

  private async callGemini(body: unknown, timeoutMs: number): Promise<GeminiResponse> {
    const url = `${GEMINI_ENDPOINT}/${encodeURIComponent(this.model)}:generateContent?key=${encodeURIComponent(
      this.apiKey ?? "",
    )}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const data = (await res.json().catch(() => ({}))) as GeminiResponse;
      if (!res.ok) {
        const msg = data?.error?.message ?? `HTTP ${res.status} ${res.statusText}`;
        throw new Error(`Gemini ${res.status}: ${msg}`);
      }
      if (data.promptFeedback?.blockReason) {
        throw new Error(`Gemini blocked: ${data.promptFeedback.blockReason}`);
      }
      return data;
    } finally {
      clearTimeout(timer);
    }
  }

  private async recordDecision(state: BotState, goal: string): Promise<void> {
    try {
      await fs.mkdir(this.skillsDir, { recursive: true });
      const logPath = path.join(this.skillsDir, "decisions.jsonl");
      const row = JSON.stringify({ ts: Date.now(), goal, state: { ...state } });
      await fs.appendFile(logPath, row + "\n", "utf8");
    } catch {
      // non-fatal
    }
  }
}

function extractText(resp: GeminiResponse): string | null {
  const parts = resp.candidates?.[0]?.content?.parts;
  if (!parts || parts.length === 0) return null;
  const joined = parts.map((p) => p.text ?? "").join("");
  return joined.length > 0 ? joined : null;
}

function textFromAnthropic(resp: Anthropic.Beta.Messages.BetaMessage): string {
  return resp.content
    .map((b) => (b.type === "text" ? b.text : ""))
    .join("");
}

/** Dynamic, per-call state. Static instructions live in SYSTEM_PROMPT (cacheable). */
function buildUserContext(s: BotState, availableGoals: string[]): string {
  const tools: string[] = [];
  if (s.hasWoodenPickaxe) tools.push("wood_pick");
  if (s.hasStonePickaxe) tools.push("stone_pick");
  if (s.hasWoodenAxe) tools.push("wood_axe");
  if (s.hasWoodenSword) tools.push("wood_sword");
  if (s.hasCraftingTable) tools.push("crafting_table");
  if (s.hasFurnace) tools.push("furnace");
  const toolsStr = tools.length ? tools.join(",") : "none";

  return [
    "Current state:",
    `- Health: ${s.health}/20, Hunger: ${s.hunger}/20, HasFood: ${s.hasFood}`,
    `- Inventory: wood=${s.wood} planks=${s.planks} sticks=${s.sticks} cobble=${s.cobblestone} coal=${s.coal} iron=${s.iron}`,
    `- Tools: ${toolsStr}`,
    `- Awareness: nearTree=${s.nearTree} nearStone=${s.nearStone} threatNearby=${s.threatNearby} isDay=${s.isDay} hasShelter=${s.hasShelter}`,
    "",
    `Available goals: ${availableGoals.join(", ")}`,
  ].join("\n");
}
