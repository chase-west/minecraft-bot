import Anthropic from "@anthropic-ai/sdk";
import * as fs from "node:fs/promises";
import * as path from "node:path";
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

/**
 * High-level planner — called sparingly (every few seconds or on goal change),
 * not per tick. Its job is to pick what to pursue; GOAP and the executor handle how.
 *
 * Voyager's contribution we're keeping: an evolving skill library. The LLM proposes
 * structured intents (not raw JS), and we cache its decisions in skills/ so future
 * sessions can replay without an API call.
 */
export class LLMPlanner {
  private client: Anthropic | null = null;
  private readonly model: string;
  private readonly skillsDir: string;
  private lastCallAt = 0;
  private readonly minIntervalMs = 5_000;

  constructor(opts: { apiKey?: string; model?: string; skillsDir?: string } = {}) {
    const key = opts.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (key) {
      this.client = new Anthropic({ apiKey: key });
    } else {
      log.info("ANTHROPIC_API_KEY not set — LLM planner disabled, falling back to GOAP-only");
    }
    this.model = opts.model ?? process.env.LLM_MODEL ?? "claude-haiku-4-5-20251001";
    this.skillsDir = opts.skillsDir ?? path.join(process.cwd(), "src/llm/skills");
  }

  isEnabled(): boolean {
    return this.client !== null;
  }

  async suggestGoal(state: BotState, availableGoals: string[]): Promise<GoalSuggestion | null> {
    if (!this.client) return null;
    const now = Date.now();
    if (now - this.lastCallAt < this.minIntervalMs) return null;
    this.lastCallAt = now;

    const stateSummary = summarizeState(state);
    const prompt = [
      "You are the high-level planner for a Minecraft Bedrock survival bot.",
      "Pick exactly one GOAP goal name for the bot to pursue right now.",
      "",
      `Available goals: ${availableGoals.join(", ")}`,
      "",
      "Bot state:",
      stateSummary,
      "",
      "Respond with JSON only: {\"goal\": \"<one of the available goals>\", \"why\": \"<brief>\"}",
    ].join("\n");

    try {
      const resp = await this.client.messages.create({
        model: this.model,
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }],
      });
      const text = resp.content
        .filter((b) => b.type === "text")
        .map((b: any) => b.text as string)
        .join("");
      const m = text.match(/\{[\s\S]*\}/);
      if (!m) return null;
      const parsed = JSON.parse(m[0]);
      if (typeof parsed.goal === "string" && availableGoals.includes(parsed.goal)) {
        log.info(`LLM picked goal: ${parsed.goal} — ${parsed.why ?? ""}`);
        await this.recordDecision(state, parsed.goal, parsed.why ?? "");
        return { kind: "goap_goal", goalName: parsed.goal };
      }
      return null;
    } catch (err) {
      log.warn("LLM call failed", err);
      return null;
    }
  }

  private async recordDecision(state: BotState, goal: string, why: string): Promise<void> {
    try {
      await fs.mkdir(this.skillsDir, { recursive: true });
      const log = path.join(this.skillsDir, "decisions.jsonl");
      const row = JSON.stringify({ ts: Date.now(), goal, why, state: { ...state } });
      await fs.appendFile(log, row + "\n", "utf8");
    } catch {
      // non-fatal
    }
  }
}

function summarizeState(s: BotState): string {
  return [
    `- hunger=${s.hunger}/20  health=${s.health}/20`,
    `- inventory: wood=${s.wood} planks=${s.planks} sticks=${s.sticks} cobble=${s.cobblestone} iron=${s.iron}`,
    `- tools: pickaxe(w/s)=${s.hasWoodenPickaxe}/${s.hasStonePickaxe} axe=${s.hasWoodenAxe} sword=${s.hasWoodenSword} table=${s.hasCraftingTable} furnace=${s.hasFurnace}`,
    `- env: nearTree=${s.nearTree} nearStone=${s.nearStone} threat=${s.threatNearby} day=${s.isDay} shelter=${s.hasShelter}`,
  ].join("\n");
}
