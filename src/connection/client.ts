import bedrock from "bedrock-protocol";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("conn");

export interface ConnectionConfig {
  host: string;
  port: number;
  username: string;
  offline: boolean;
  version?: string;
}

export type BedrockClient = ReturnType<typeof bedrock.createClient>;

/** Build the client synchronously so the caller can attach perception
 * handlers BEFORE the server's start_game/spawn packets arrive. */
export function createClient(cfg: ConnectionConfig): BedrockClient {
  const client = bedrock.createClient({
    host: cfg.host,
    port: cfg.port,
    username: cfg.username,
    offline: cfg.offline,
    ...(cfg.version ? { version: cfg.version as any } : {}),
  } as any);
  client.on("error", (err) => log.error("client error", err));
  client.on("kick", (reason) => log.warn("kicked", reason));
  client.on("close", () => log.warn("connection closed"));
  log.info(`connecting to ${cfg.host}:${cfg.port} as ${cfg.username} (offline=${cfg.offline})`);
  return client;
}

/** Resolve when the bot has fully spawned. Attach perception/etc handlers
 * BEFORE calling this — start_game fires during login, well before spawn. */
export function waitForSpawn(client: BedrockClient, cfg: ConnectionConfig): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const onError = (err: unknown) => {
      if (settled) return;
      settled = true;
      reject(err);
    };
    client.on("error", onError);
    client.on("kick", (reason) => onError(new Error(`kicked: ${JSON.stringify(reason)}`)));
    client.once("spawn", () => {
      if (settled) return;
      settled = true;
      log.info(`spawned into world as ${cfg.username} on ${cfg.host}:${cfg.port}`);
      resolve();
    });
  });
}

/** Convenience: create, attach error handlers, wait for spawn. Prefer the
 * two-step API when you need to bind perception handlers first. */
export async function connect(cfg: ConnectionConfig): Promise<BedrockClient> {
  const client = createClient(cfg);
  await waitForSpawn(client, cfg);
  return client;
}

export function configFromEnv(): ConnectionConfig {
  return {
    host: process.env.BEDROCK_HOST ?? "localhost",
    port: Number(process.env.BEDROCK_PORT ?? 19132),
    username: process.env.BEDROCK_USERNAME ?? "AIBot",
    offline: (process.env.BEDROCK_OFFLINE ?? "true").toLowerCase() === "true",
    version: process.env.BEDROCK_VERSION,
  };
}
