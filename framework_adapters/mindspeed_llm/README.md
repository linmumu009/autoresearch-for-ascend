# MindSpeed-LLM Adapter

This folder contains our thin integration layer for evaluating MindSpeed-LLM
against the existing Ascend autoresearch prototype.

The external MindSpeed-LLM and MindSpeed repositories are kept under
`reference/` and are not committed here.

## Smoke Script

First prepare the deepscaler-derived SFT data:

```bash
bash /workspace/framework_adapters/mindspeed_llm/prepare_deepscaler_alpaca.sh
```

Run from inside the `llin-autoresearch` container:

```bash
bash /workspace/framework_adapters/mindspeed_llm/smoke_qwen3_0p6_full_sft.sh
```

Useful overrides:

```bash
TRAIN_ITERS=3 SEQ_LENGTH=2048 GLOBAL_BATCH_SIZE=1 \
  bash /workspace/framework_adapters/mindspeed_llm/smoke_qwen3_0p6_full_sft.sh
```

For a held-out validation smoke:

```bash
TRAIN_ITERS=6 DATA_SPLIT=90,10,0 EVAL_INTERVAL=3 EVAL_ITERS=2 \
  bash /workspace/framework_adapters/mindspeed_llm/smoke_qwen3_0p6_full_sft.sh
```

## Default Paths

- MindSpeed-LLM: `/workspace/reference/MindSpeed-LLM`
- Matched MindSpeed: `/workspace/reference/MindSpeed-rjx`
- MCore checkpoint: `/models/Qwen3-0p6`
- Tokenizer: `/models/Qwen3-0.6B`
- Dataset prefix: `/workspace/datasets/deepscaler_alpaca/processed/deepscaler`

## Evaluation Rule

The first target is not a final benchmark. It is a minimal evidence check:

1. import MindSpeed-LLM successfully,
2. launch Qwen3-0.6B SFT on one visible NPU,
3. record loss/throughput/memory,
4. compare against the HF thin-loop baseline.

## Current Smoke Result

The deepscaler-derived 512-sample smoke completed 3 full-SFT iterations:

- train loss: `1.0280 -> 0.9771 -> 0.6329`
- steady step time after warmup: about `0.24-0.25s`
- allocated HBM: about `10.3 GB`

This proves the path can run. It is not yet a strict effect comparison because
the HF thin loop reports validation loss while this MindSpeed smoke currently
reports training loss.

The first validation smoke used `DATA_SPLIT=90,10,0` and reached final
validation-set loss `0.611867` after 6 train iterations. This is useful for
MindSpeed-local comparisons, but still not directly comparable to the HF
thin-loop validation loss.
