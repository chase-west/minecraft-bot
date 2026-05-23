import { createClient, waitForSpawn, configFromEnv } from "./connection/client.js";
import { Agent } from "./bot/agent.js";
import { makeLogger } from "./utils/logger.js";

const log = makeLogger("main");

async function runOneSession(): Promise<"reconnect" | "exit"> {
  const cfg = configFromEnv();
  log.info(`booting minecraft-bedrock-ai session`);
  const client = createClient(cfg);
  const agent = new Agent(client);
  await agent.attachHandlers();

  // Reconnect on any disconnect: death-respawn, kick, network drop. Doing a
  // full fresh login is more reliable than fighting the respawn(state=2)
  // handshake which BDS handles inconsistently.
  let reconnectRequested = false;
  const triggerReconnect = (cause: string) => {
    if (reconnectRequested) return;
    reconnectRequested = true;
    log.warn(`reconnect triggered by ${cause}`);
    agent.stop();
    try { (client as any).close?.(); } catch { /* ignore */ }
  };
  (client as any).on?.("kick", () => triggerReconnect("kick"));
  (client as any).on?.("close", () => triggerReconnect("close"));
  (client as any).on?.("disconnect", () => triggerReconnect("disconnect"));

  await waitForSpawn(client, cfg);
  await agent.afterSpawn();

  const shutdown = (sig: string) => {
    log.warn(`shutdown via ${sig}`);
    agent.stop();
    try { (client as any).close?.(); } catch { /* ignore */ }
    setTimeout(() => process.exit(0), 200);
  };
  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  // run() returns when the agent is stopped (e.g., on disconnect).
  await agent.run();
  return reconnectRequested ? "reconnect" : "exit";
}

async function main(): Promise<void> {
  // Single session per process. On disconnect/death/crash, exit and let the
  // supervisor (scripts/run-bot.ps1) restart us with a fresh heap. In-process
  // reconnects leak bedrock-protocol internals and OOM after ~30 min.
  log.info(`session start`);
  try {
    const result = await runOneSession();
    log.info(`session ended (result=${result}) — exiting for supervisor restart`);
    process.exit(result === "reconnect" ? 10 : 0);
  } catch (err) {
    log.warn(`session crashed: ${(err as Error).message} — exiting for supervisor restart`);
    process.exit(10);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
