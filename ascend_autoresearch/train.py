"""
Editable experiment file for Ascend autoresearch.

Agents may change this file. Keep prepare.py fixed.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import MAX_SEQ_LEN, MODEL_PATH, TIME_BUDGET, evaluate_loss, iter_batches, load_or_build_token_data, perplexity


# Experiment knobs. Agents are expected to edit these.
SEQ_LEN = MAX_SEQ_LEN
MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 2.0e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.05
MIN_LR_RATIO = 0.1
FREEZE_EMBEDDINGS = True
GRADIENT_CHECKPOINTING = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=int, default=TIME_BUDGET)
    parser.add_argument("--model-path", default=MODEL_PATH)
    return parser.parse_args()


def lr_multiplier(progress: float) -> float:
    if progress < WARMUP_RATIO:
        return progress / max(WARMUP_RATIO, 1e-8)
    decay_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-8)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(decay_progress * torch.pi))).item()
    return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    device = "npu"

    t_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    data = load_or_build_token_data(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)

    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if FREEZE_EMBEDDINGS and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = iter_batches(data.train_tokens, MICRO_BATCH_SIZE, SEQ_LEN, device)

    total_training_time = 0.0
    step = 0
    smooth_loss = 0.0
    model.train()

    while total_training_time < args.time_budget:
        torch.npu.synchronize()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_for_log = None

        for _ in range(GRAD_ACCUM_STEPS):
            x, y = next(loader)
            out = model(input_ids=x, labels=y)
            loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()
            loss_for_log = out.loss.detach()

        progress = min(total_training_time / args.time_budget, 1.0)
        mult = lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE * mult
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        torch.npu.synchronize()
        dt = time.time() - t0
        if step > 2:
            total_training_time += dt

        loss_float = float(loss_for_log.cpu()) if loss_for_log is not None else 0.0
        smooth_loss = 0.9 * smooth_loss + 0.1 * loss_float
        debiased = smooth_loss / (1.0 - 0.9 ** (step + 1))
        remaining = max(0, args.time_budget - total_training_time)
        print(
            f"\rstep {step:05d} | loss: {debiased:.6f} | lr_mult: {mult:.3f} | "
            f"dt: {dt:.2f}s | remaining: {remaining:.0f}s",
            end="",
            flush=True,
        )
        step += 1

    print()
    val_loss = evaluate_loss(model, data.val_tokens, MICRO_BATCH_SIZE, SEQ_LEN, device)
    total_seconds = time.time() - t_start
    peak_mem = torch.npu.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_ppl:          {perplexity(val_loss):.3f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_npu_mem_mb:  {peak_mem:.1f}")
    print(f"num_steps:        {step}")
    print(f"seq_len:          {SEQ_LEN}")
    print(f"micro_batch_size: {MICRO_BATCH_SIZE}")
    print(f"grad_accum_steps: {GRAD_ACCUM_STEPS}")


if __name__ == "__main__":
    main()
