import * as path from "node:path";
import * as fs from "node:fs/promises";
import * as ort from "onnxruntime-node";
import { makeLogger } from "../utils/logger.js";

const log = makeLogger("ml");

export interface PolicyHandle {
  name: string;
  session: ort.InferenceSession;
  inputName: string;
  outputName: string;
}

export async function loadPolicy(name: string, modelPath?: string): Promise<PolicyHandle | null> {
  const p = modelPath ?? path.join(process.cwd(), "models", `${name}.onnx`);
  try {
    await fs.access(p);
  } catch {
    log.info(`policy ${name} not found at ${p} — skipping (will run on rules)`);
    return null;
  }
  const session = await ort.InferenceSession.create(p);
  const inputName = session.inputNames[0]!;
  const outputName = session.outputNames[0]!;
  log.info(`loaded policy ${name} (input=${inputName} output=${outputName})`);
  return { name, session, inputName, outputName };
}

export async function runPolicy(policy: PolicyHandle, features: Float32Array, shape: number[]): Promise<Float32Array> {
  const tensor = new ort.Tensor("float32", features, shape);
  const feeds: Record<string, ort.Tensor> = { [policy.inputName]: tensor };
  const out = await policy.session.run(feeds);
  const result = out[policy.outputName];
  if (!result) throw new Error(`policy ${policy.name} returned no output for ${policy.outputName}`);
  return result.data as Float32Array;
}
