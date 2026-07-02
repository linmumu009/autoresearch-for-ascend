"""
Fixed data, dataloader, and validation utilities for Ascend autoresearch.

This file is intentionally kept out of the experiment surface. Agents should
edit train.py only, so scores stay comparable across experiments.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


MODEL_PATH = os.environ.get("ASCEND_AUTORESEARCH_MODEL", "/models/Qwen3-0.6B")
WORKSPACE = Path(os.environ.get("ASCEND_AUTORESEARCH_WORKSPACE", "/workspace"))
DATA_DIR = WORKSPACE / "inputs" / "raw"
CACHE_DIR = WORKSPACE / "cache" / "ascend_autoresearch"
TRAIN_JSON = DATA_DIR / "deepscaler.json"
VAL_JSONL = DATA_DIR / "ceval_val.jsonl"

MAX_SEQ_LEN = 512
TIME_BUDGET = 300
EVAL_TOKENS = 16 * 2048
SEED = 42


@dataclass(frozen=True)
class TokenData:
    train_tokens: torch.Tensor
    val_tokens: torch.Tensor


def _fallback_texts() -> tuple[list[str], list[str]]:
    train = [
        "Question: What is autoresearch?\nAnswer: A loop where an agent changes code, trains, evaluates, and keeps improvements.",
        "Question: What is Ascend?\nAnswer: A Huawei AI accelerator platform used for model training and inference.",
        "Question: What is validation loss?\nAnswer: A fixed metric used to compare training experiments fairly.",
    ] * 64
    val = [
        "Question: Why keep evaluation fixed?\nAnswer: So experiments cannot improve by changing the test.",
        "Question: What should the agent edit?\nAnswer: Only train.py.",
    ] * 32
    return train, val


def _record_to_text(record: dict) -> str:
    if "problem" in record:
        solution = record.get("solution") or ""
        answer = record.get("answer") or ""
        return f"Problem:\n{record['problem']}\n\nSolution:\n{solution}\n\nAnswer:\n{answer}"

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        parts = []
        system = record.get("system")
        if system:
            parts.append(f"System:\n{system}")
        for turn in conversations:
            role = turn.get("from") or turn.get("role") or "unknown"
            value = turn.get("value") or turn.get("content") or ""
            parts.append(f"{role}:\n{value}")
        return "\n\n".join(parts)

    return json.dumps(record, ensure_ascii=False)


def _read_json_or_jsonl(path: Path, limit: int | None = None) -> list[str]:
    if not path.exists():
        return []

    texts: list[str] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                texts.append(_record_to_text(json.loads(line)))
                if limit and len(texts) >= limit:
                    break
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("data", [data])
        for record in data:
            texts.append(_record_to_text(record))
            if limit and len(texts) >= limit:
                break
    return texts


def load_texts() -> tuple[list[str], list[str]]:
    train = _read_json_or_jsonl(TRAIN_JSON, limit=4096)
    val = _read_json_or_jsonl(VAL_JSONL, limit=512)
    if not train or not val:
        return _fallback_texts()
    return train, val


def tokenize_texts(tokenizer, texts: Iterable[str]) -> torch.Tensor:
    eos = tokenizer.eos_token_id
    ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(encoded)
        if eos is not None:
            ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def load_or_build_token_data(tokenizer) -> TokenData:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_path = CACHE_DIR / "train_tokens.pt"
    val_path = CACHE_DIR / "val_tokens.pt"

    if train_path.exists() and val_path.exists():
        return TokenData(torch.load(train_path), torch.load(val_path))

    train_texts, val_texts = load_texts()
    train_tokens = tokenize_texts(tokenizer, train_texts)
    val_tokens = tokenize_texts(tokenizer, val_texts)
    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)
    return TokenData(train_tokens, val_tokens)


def iter_batches(tokens: torch.Tensor, batch_size: int, seq_len: int, device: str, *, seed: int = SEED):
    assert tokens.numel() > seq_len + 1, "not enough tokens for requested seq_len"
    rng = random.Random(seed)
    max_start = tokens.numel() - seq_len - 1
    while True:
        starts = [rng.randint(0, max_start) for _ in range(batch_size)]
        x = torch.stack([tokens[s:s + seq_len] for s in starts])
        y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts])
        yield x.to(device), y.to(device)


@torch.no_grad()
def evaluate_loss(model, val_tokens: torch.Tensor, batch_size: int, seq_len: int, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loader = iter_batches(val_tokens, batch_size, seq_len, device, seed=SEED + 1)
    steps = max(1, EVAL_TOKENS // (batch_size * seq_len))
    for _ in range(steps):
        x, y = next(loader)
        out = model(input_ids=x, labels=y)
        total_loss += float(out.loss.detach().cpu()) * y.numel()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens


def perplexity(loss: float) -> float:
    return float(math.exp(min(loss, 20.0)))
