# minecraft-bedrock-ai

A bot that plays Minecraft Bedrock Edition. It logs in over the regular client protocol, reads the world from packets, and learns to play by training a DQN on its own rollouts. A symbolic GOAP planner and an optional LLM goal-picker are also in the repo, used as teachers and for bootstrapping.

The goal was to see how far you can get without doing pixel-based RL. Turns out: pretty far.

## How it picks moves

One controller is in charge of the bot at a time. Which one is picked by `POLICY_MODE` in `.env`:

- **`online`** (default): the DQN policy drives. Every 100ms it encodes the world into a 601-float observation, runs ONNX inference, and emits one of 12 discrete actions (move NSEW, jump, sprint toggle, mine front, place front, attack nearest, eat, move+jump, noop). Action choices are held for 400ms so the bot doesn't flicker. With probability `ONLINE_EPSILON` (default 0.15) it samples from the bootstrap explorer instead, to keep generating fresh data. The trainer in `training/train_dqn.py` watches `data/online/*.jsonl` rollouts and rewrites `models/policy.onnx` as it learns. The bot hot-reloads the file in place.
- **`explore`**: no neural net at all. A sticky random-walk Explorer picks actions weighted toward movement so the bot generates diverse rollouts. This is the bootstrap mode used to seed training before any model exists.
- **`shadow`**: GOAP drives the bot using A* over a symbolic state. The ML side runs in parallel and logs GOAP's chosen action as a supervision label, which is what behaviour cloning trains on.
- **`learned`**: pure DQN argmax, no exploration. Eval mode.

The LLM (Claude Haiku) only fires in shadow mode, where it nudges GOAP toward a high level goal every 8 seconds or so. It is optional and most of the time it is off.

Underneath whichever controller is driving sits a 20 Hz `PlayerAuthInput` ticker that talks to the server, plus a 3D A* pathfinder that GOAP uses for navigation.

```
        POLICY_MODE picks one driver
        +----------------------------+
online: |    DQN policy (ONNX)       |  argmax, ε-greedy, hot-reload
        +----------------------------+
explore:|    sticky random walk      |  bootstrap data
        +----------------------------+
shadow: |  LLM -> GOAP -> A* path    |  GOAP drives, ML side just logs labels
        +----------------------------+
                       |
                       v
              12 discrete actions
                       |
                       v
        20 Hz PlayerAuthInput ticker
                       |
                       v
                bedrock-protocol  ->  BDS / Realm
```

## Why this shape

Doing end-to-end RL on Minecraft pixels is a research project (see VPT, ~100K GPU hours). We don't have pixels, we have packets, which means we already get structured state for free. So the encoder is short, the action space is small, and a tiny DQN trained on a few hours of rollouts is competitive with the hand-written GOAP.

- DQN: small (601 -> hidden -> 12 logits), fast to train, cheap to run
- GOAP: deterministic baseline, used as a teacher in shadow mode
- A*: navigation has been solved for years (mineflayer-pathfinder did it for Java)
- LLM: open-ended goal picking when you want it, ignored otherwise

## Layout

```
src/
  connection/   bedrock-protocol wrapper, login, reconnect
  world/        block + entity + inventory model, packet decoders
  actions/      primitive actions (mine, place, walk, eat, combat)
  pathfinding/  3D A* with jumps, drops, diagonals
  goap/         planner, actions, goals, sensors, executor
  ml/           ONNX runtime, encoder, explorer, reward, trajectory logger
  llm/          high level planner + skill library
  bot/          top-level Agent that wires everything together
  utils/        logger, vec3, heap
training/       Python: behaviour cloning + DQN trainers, ONNX export
scripts/        supervisor scripts (PowerShell)
models/         *.onnx files (gitignored, regenerate from training/)
data/           reference tables (block ids, entity types); online/ is gitignored rollouts
```

## Quickstart

You need:

- Node 20+ (22 LTS works)
- A Bedrock dedicated server running on `localhost:19132` with `online-mode=false`
- Optional: an Anthropic API key if you want the LLM planner

```powershell
npm install
copy .env.example .env
# edit .env if you want
npm run dev
```

The bot will connect, wait for spawn, then start its 100ms perception/action loop. The repo ships without any `.onnx` files, so on first launch `online` mode finds no model and runs the bootstrap explorer instead. It will switch over to the learned policy the moment a model file appears in `models/`.

## Training your own policy

In one terminal, run the bot in `online` mode (the default). It writes rollouts to `data/online/*.jsonl` as it plays.

In another terminal, run the trainer:

```powershell
cd training
pip install -r requirements.txt
python train_dqn.py
```

`train_dqn.py` tails `data/online/`, trains a DQN, and rewrites `models/policy.onnx` periodically. The bot watches that file and hot-swaps weights on the fly. No restart needed.

If you have recorded demos and want behaviour cloning as a warm start instead:

```powershell
python train_bc.py --data data/combat.jsonl --out ../models/policy.onnx --epochs 20
```

## What works

- **Crafting** via `item_stack_request` + `craft_recipe`. Recipe network IDs are pulled from the `crafting_data` packet the server sends on join. 2x2 inventory crafts (planks, sticks, tables) skip the container open dance. 3x3 crafts walk to a nearby table and open it.
- **Chunk decoding** via `prismarine-chunk`. Handles the subchunk-request mode (negative `sub_chunk_count`) by replying with explicit `subchunk_request` packets. Blocks land in a flat world map the pathfinder consumes.
- **Schema drift handling**. All packet writes go through a `safeQueue` that logs and continues instead of crashing. Version-conditional fields (`analogue_move_vector` and friends) are gated by feature flags detected from the client version.
- **Reconnect**. On kick, close, or disconnect the process exits with code 10 and the PowerShell supervisor restarts it from scratch. Trying to reconnect in-process leaked memory after ~30 minutes, so we don't.

## Things to know

- `prismarine-chunk` defaults to Bedrock 1.21.0 palette tables. If you run a much newer server, override `client.options.version` to match or the runtime IDs will be wrong.
- Some servers don't accept the legacy `inventory_transaction` shape for placement. We send it anyway because most servers handle it, but newer servers may want `item_stack_request` instead.
- The LLM planner is optional. Leave `ANTHROPIC_API_KEY` empty and it just won't fire.
- BDS Realms are not supported (no auth wired up). Local dedicated servers and LAN games work.

## Tested with

- Node 20, 22 LTS
- bedrock-protocol 3.56.x
- BDS 1.21 through 1.26

## License

MIT, see `LICENSE`.
