# Behavior cloning trainer

Trains a discrete-action policy from JSONL session logs the running bot writes
to `data/online/*.jsonl`. Each line is one `(obs, action)` sample; the obs is a
base64-packed length-601 `float32` vector, the action is an int in `[0, 12)`.

## Install

```
pip install -r training/requirements.txt
```

## Train

After the bot has accumulated some data, from the project root:

```
python training/train_bc.py
```

Useful flags:

- `--data-dir data/online` (default)
- `--out models/policy.onnx` (default)
- `--epochs 20`, `--batch-size 256`, `--lr 3e-4`
- `--device auto|cpu|cuda`
- `--limit N` to cap the number of samples for a fast smoke run

The trainer rebalances classes via `1/sqrt(count)` weights so the dominant
Noop action does not swamp the rest. A `.pt` checkpoint is written next to the
ONNX for later fine-tuning / PPO.

## Use the model

The ONNX lands at `models/policy.onnx`. Point the bot at it by setting
`POLICY_MODE=learned` in `.env` and restart the bot. The Node side loads the
file via `src/ml/runtime.ts` and runs it at ~10 Hz with input name `obs`
(shape `[batch, 601]`) and output name `logits` (shape `[batch, 12]`).
