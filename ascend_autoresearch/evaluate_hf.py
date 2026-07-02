"""
Evaluate a Hugging Face causal LM with the fixed Ascend autoresearch data.

This script is intentionally eval-only. It reuses prepare.py so converted
MindSpeed checkpoints can be measured on the same raw next-token validation
surface as the thin HF training loop.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import MAX_SEQ_LEN, MODEL_PATH, evaluate_loss, load_or_build_token_data, perplexity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer_path or args.model_path
    device = "npu"

    torch.manual_seed(42)
    torch.npu.manual_seed(42)
    t_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    data = load_or_build_token_data(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)

    val_loss = evaluate_loss(model, data.val_tokens, args.batch_size, args.seq_len, device)
    total_seconds = time.time() - t_start
    peak_mem = torch.npu.max_memory_allocated() / 1024 / 1024

    print("---")
    print(f"model_path:       {args.model_path}")
    print(f"tokenizer_path:   {tokenizer_path}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"val_ppl:          {perplexity(val_loss):.3f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_npu_mem_mb:  {peak_mem:.1f}")
    print(f"seq_len:          {args.seq_len}")
    print(f"batch_size:       {args.batch_size}")


if __name__ == "__main__":
    main()
