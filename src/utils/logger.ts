type Level = "debug" | "info" | "warn" | "error";

const LEVELS: Record<Level, number> = { debug: 10, info: 20, warn: 30, error: 40 };

const envLevel = (process.env.LOG_LEVEL ?? "info").toLowerCase() as Level;
const threshold = LEVELS[envLevel] ?? LEVELS.info;

function ts(): string {
  return new Date().toISOString();
}

function log(level: Level, tag: string, msg: string, extra?: unknown): void {
  if (LEVELS[level] < threshold) return;
  const line = `[${ts()}] ${level.toUpperCase().padEnd(5)} [${tag}] ${msg}`;
  if (extra !== undefined) {
    console.log(line, extra);
  } else {
    console.log(line);
  }
}

export function makeLogger(tag: string) {
  return {
    debug: (msg: string, extra?: unknown) => log("debug", tag, msg, extra),
    info: (msg: string, extra?: unknown) => log("info", tag, msg, extra),
    warn: (msg: string, extra?: unknown) => log("warn", tag, msg, extra),
    error: (msg: string, extra?: unknown) => log("error", tag, msg, extra),
  };
}

export type Logger = ReturnType<typeof makeLogger>;
