"""Behavior-cloning trainer for the Minecraft bot policy.

Reads JSONL session files written by the running bot (one sample per line):

    {"obs_b64": "<base64 float32[601]>", "obs_len": 601, "action": int,
     "reward": float, "t": ms_since_session_start}

Trains a split-path embedding+MLP that maps a length-601 observation vector
to logits over 12 discrete actions, then exports an ONNX model that the
Node side loads via src/ml/runtime.ts at 10 Hz.

Usage:
    python training/train_bc.py
    python training/train_bc.py --epochs 30 --batch-size 512
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---- observation layout constants ----------------------------------------
OBS_DIM = 601
N_ACTIONS = 12

SELF_LO, SELF_HI = 0, 8                 # 8 continuous self stats

BLOCK_LO, BLOCK_HI = 8, 413             # 405 block-grid dense IDs (int 0..1023)
N_BLOCKS = BLOCK_HI - BLOCK_LO          # 405
BLOCK_VOCAB = 1024
BLOCK_EMB = 16

ENT_LO, ENT_HI = 413, 445               # 4 entities * 8 floats = 32
N_ENT = 4
ENT_STRIDE = 8
ENT_TYPE_OFFSET = 3                     # type id is the 4th of the 8 (positions 416, 424, 432, 440)
ENT_TYPE_POSITIONS = [ENT_LO + i * ENT_STRIDE + ENT_TYPE_OFFSET for i in range(N_ENT)]
# -> [416, 424, 432, 440]
ENT_VOCAB = 64
ENT_EMB = 8

INV_LO, INV_HI = 445, 509               # 32 slots * 2 = 64
N_INV = 32
INV_STRIDE = 2
INV_ID_POSITIONS = [INV_LO + i * INV_STRIDE for i in range(N_INV)]
# -> [445, 447, ..., 507]
INV_VOCAB = 256
INV_EMB = 8

BAG_LO, BAG_HI = 509, 601               # 92 aggregate item count floats

# Indices of the 160 continuous floats in the input vector (everything that
# is NOT an embedded int slot).
_ENT_TYPE_SET = set(ENT_TYPE_POSITIONS)
_INV_ID_SET = set(INV_ID_POSITIONS)
CONT_INDICES: List[int] = (
    list(range(SELF_LO, SELF_HI))
    + [i for i in range(ENT_LO, ENT_HI) if i not in _ENT_TYPE_SET]
    + [i for i in range(INV_LO, INV_HI) if i not in _INV_ID_SET]
    + list(range(BAG_LO, BAG_HI))
)
assert len(CONT_INDICES) == 160, f"continuous index count mismatch: {len(CONT_INDICES)}"

# Feature dims fed to the MLP trunk.
_FEAT_DIM = (
    N_BLOCKS * BLOCK_EMB        # 6480
    + N_ENT * ENT_EMB           #   32
    + N_INV * INV_EMB           #  256
    + len(CONT_INDICES)         #  160
)
# 6480 + 32 + 256 + 160 = 6928


# ---- model ---------------------------------------------------------------
class BCPolicy(nn.Module):
    """601-vec -> 12 action logits with split embedding + MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.block_emb = nn.Embedding(BLOCK_VOCAB, BLOCK_EMB)
        self.ent_emb = nn.Embedding(ENT_VOCAB, ENT_EMB)
        self.inv_emb = nn.Embedding(INV_VOCAB, INV_EMB)

        # Register continuous-index gather tensor as a buffer so it moves
        # with .to(device) and is excluded from parameters.
        self.register_buffer(
            "cont_idx", torch.tensor(CONT_INDICES, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "ent_type_idx",
            torch.tensor(ENT_TYPE_POSITIONS, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "inv_id_idx",
            torch.tensor(INV_ID_POSITIONS, dtype=torch.long),
            persistent=False,
        )

        self.trunk = nn.Sequential(
            nn.Linear(_FEAT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (B, 601) float32
        B = obs.size(0)

        block_ids = obs[:, BLOCK_LO:BLOCK_HI].long().clamp_(0, BLOCK_VOCAB - 1)
        ent_ids = obs.index_select(1, self.ent_type_idx).long().clamp_(0, ENT_VOCAB - 1)
        inv_ids = obs.index_select(1, self.inv_id_idx).long().clamp_(0, INV_VOCAB - 1)

        block_feat = self.block_emb(block_ids).reshape(B, N_BLOCKS * BLOCK_EMB)
        ent_feat = self.ent_emb(ent_ids).reshape(B, N_ENT * ENT_EMB)
        inv_feat = self.inv_emb(inv_ids).reshape(B, N_INV * INV_EMB)

        cont_feat = obs.index_select(1, self.cont_idx)  # (B, 160)

        x = torch.cat([block_feat, ent_feat, inv_feat, cont_feat], dim=1)
        return self.trunk(x)


def build_model() -> BCPolicy:
    """Public helper used by smoke tests."""
    return BCPolicy()


# ---- dataset -------------------------------------------------------------
def _load_jsonl_files(data_dir: Path, limit: int | None) -> Tuple[np.ndarray, np.ndarray, int]:
    paths = sorted(glob.glob(str(data_dir / "*.jsonl")))
    if not paths:
        raise RuntimeError(f"no *.jsonl files under {data_dir}")

    obs_list: List[np.ndarray] = []
    act_list: List[int] = []

    for p in paths:
        # Read fully into memory (the bot may still be appending to the
        # latest file; we deliberately do not tail).
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Likely a partial trailing write while the bot is running.
                continue
            b64 = row.get("obs_b64")
            act = row.get("action")
            if b64 is None or act is None:
                continue
            buf = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
            if buf.shape[0] != OBS_DIM:
                continue
            obs_list.append(buf)
            act_list.append(int(act))
            if limit is not None and len(obs_list) >= limit:
                break
        if limit is not None and len(obs_list) >= limit:
            break

    if not obs_list:
        raise RuntimeError(f"no usable samples found in {data_dir}")

    obs_arr = np.stack(obs_list, axis=0)        # (N, 601) float32
    act_arr = np.asarray(act_list, dtype=np.int64)
    return obs_arr, act_arr, len(paths)


class BCDataset(Dataset):
    def __init__(self, obs: np.ndarray, actions: np.ndarray) -> None:
        self.obs = torch.from_numpy(obs)            # float32
        self.actions = torch.from_numpy(actions)    # int64

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, i: int):
        return self.obs[i], self.actions[i]


# ---- training ------------------------------------------------------------
def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _class_weights(actions: np.ndarray) -> torch.Tensor:
    counts = np.bincount(actions, minlength=N_ACTIONS).astype(np.float64)
    # 1/sqrt(count); clamp counts to 1 to avoid div-by-zero for unseen actions.
    safe = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(safe)
    # Normalize so weights average to 1 (keeps loss magnitude sensible).
    w = w * (N_ACTIONS / w.sum())
    return torch.tensor(w, dtype=torch.float32)


def train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    obs_arr, act_arr, n_files = _load_jsonl_files(data_dir, args.limit)
    n_samples = obs_arr.shape[0]

    counts = np.bincount(act_arr, minlength=N_ACTIONS)
    weights = _class_weights(act_arr)
    freq_str = ", ".join(
        f"{a}:{counts[a]}({weights[a].item():.2f})" for a in range(N_ACTIONS)
    )
    print(
        f"trained on {n_samples} samples from {n_files} files, "
        f"balanced action freq: {freq_str}"
    )

    device = _resolve_device(args.device)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    ds = BCDataset(obs_arr, act_arr)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    for ep in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_correct = 0
        n_seen = 0
        for obs_b, act_b in loader:
            obs_b = obs_b.to(device, non_blocking=True)
            act_b = act_b.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(obs_b)
            loss = loss_fn(logits, act_b)
            loss.backward()
            opt.step()

            total_loss += loss.item() * obs_b.size(0)
            n_correct += (logits.argmax(dim=1) == act_b).sum().item()
            n_seen += obs_b.size(0)

        avg_loss = total_loss / max(n_seen, 1)
        acc = n_correct / max(n_seen, 1)
        print(f"epoch {ep:03d}  loss={avg_loss:.4f}  top1={acc * 100:.2f}%")

    # ---- export ----------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_path.with_suffix(".pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
            "class_counts": counts.tolist(),
        },
        ckpt_path,
    )
    print(f"saved checkpoint {ckpt_path}")

    model.eval()
    # Export on CPU for portability; the Node onnxruntime side typically
    # runs CPU EP unless explicitly configured otherwise.
    export_model = model.cpu()
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy,
        str(out_path),
        input_names=["obs"],
        output_names=["logits"],
        dynamic_axes={"obs": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior cloning for the Minecraft bot")
    p.add_argument("--data-dir", default="data/online")
    p.add_argument("--out", default="models/policy.onnx")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:0 ...",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional cap on total number of samples loaded",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
