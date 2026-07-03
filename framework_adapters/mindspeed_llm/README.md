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

## HF Conversion

Convert a saved MindSpeed/MCore checkpoint back to Hugging Face format:

```bash
bash /workspace/framework_adapters/mindspeed_llm/convert_mcore_to_hf.sh
```

Then evaluate it with the fixed raw next-token HF validation surface:

```bash
cd /workspace/ascend_autoresearch
python evaluate_hf.py --model-path /workspace/outputs/mindspeed_qwen3_0p6_deepscaler_eval_smoke_hf
```

First converted-checkpoint result:

- base Qwen3-0.6B raw HF val_loss: `14.977717`
- converted MindSpeed 6-step checkpoint raw HF val_loss: `14.963873`

## Autoresearch Runner

Run one full candidate loop:

```bash
CANDIDATE_ENV=/workspace/framework_adapters/mindspeed_llm/candidates/baseline_6step.env \
  bash /workspace/framework_adapters/mindspeed_llm/run_autoresearch_candidate.sh
```

The runner does four things in order:

1. train a MindSpeed-LLM candidate,
2. convert the MCore checkpoint to Hugging Face format,
3. evaluate the converted checkpoint with `ascend_autoresearch/evaluate_hf.py`,
4. append metrics to `/workspace/runs/mindspeed_llm/results.tsv`.

Each candidate gets its own directory under `/workspace/runs/mindspeed_llm/`.

Initial search candidates:

| Candidate | LR | Purpose |
| --- | ---: | --- |
| `baseline_6step.env` | `1.25e-6` | Runner baseline. |
| `lr_low_6step.env` | `6.25e-7` | Test lower LR under the same 6-step budget. |
| `lr_high_6step.env` | `2.5e-6` | Test higher LR under the same 6-step budget. |
| `lr_higher_6step.env` | `5.0e-6` | Check whether the first high-LR gain continues or over-shoots. |
| `lr_7p5em6_6step.env` | `7.5e-6` | Probe whether the higher-LR gain continues. |
| `lr_1em5_6step.env` | `1.0e-5` | Upper LR probe before moving to longer budgets. |
| `lr_1p5em5_6step.env` | `1.5e-5` | Probe the over-shoot boundary above the current best. |
| `lr_2em5_6step.env` | `2.0e-5` | Higher over-shoot boundary probe. |
| `lr_3em5_6step.env` | `3.0e-5` | Continue probing the 6-step over-shoot boundary. |
| `lr_2em5_12step.env` | `2.0e-5` | Verify the current best LR on a longer 12-step budget. |
| `lr_3em5_12step.env` | `3.0e-5` | Compare 3.0e-5 and 2.0e-5 under the same 12-step budget. |
| `lr_4em5_6step.env` | `4.0e-5` | Continue probing the 6-step over-shoot boundary. |
| `lr_4em5_12step.env` | `4.0e-5` | Verify 4.0e-5 under the same 12-step budget. |
| `lr_5em5_6step.env` | `5.0e-5` | Continue probing the 6-step over-shoot boundary. |
| `lr_5em5_12step.env` | `5.0e-5` | Verify 5.0e-5 under the same 12-step budget. |
| `lr_6em5_6step.env` | `6.0e-5` | Continue probing the 6-step over-shoot boundary. |
| `lr_4p5em5_12step.env` | `4.5e-5` | Refine the 12-step boundary between 4.0e-5 and 5.0e-5. |
| `lr_7em5_6step.env` | `7.0e-5` | Continue probing the 6-step over-shoot boundary. |
| `lr_8em5_6step.env` | `8.0e-5` | Probe whether the 6-step boundary starts to over-shoot. |
| `lr_9em5_6step.env` | `9.0e-5` | Probe whether the 6-step boundary over-shoots above 8.0e-5. |
| `lr_1em4_6step.env` | `1.0e-4` | Probe whether the 6-step boundary over-shoots above 9.0e-5. |
| `lr_1p1em4_6step.env` | `1.1e-4` | Probe whether the 6-step boundary over-shoots above 1.0e-4. |
| `lr_1p2em4_6step.env` | `1.2e-4` | Probe whether the 6-step boundary over-shoots above 1.1e-4. |
| `lr_1p3em4_6step.env` | `1.3e-4` | Probe whether the 6-step boundary over-shoots above 1.2e-4. |
| `lr_1p4em4_6step.env` | `1.4e-4` | Probe whether the 6-step boundary over-shoots above 1.3e-4. |
| `lr_1p5em4_6step.env` | `1.5e-4` | Probe whether the 6-step boundary over-shoots above 1.4e-4. |
| `lr_1p6em4_6step.env` | `1.6e-4` | Probe whether the 6-step boundary over-shoots above 1.5e-4. |

LR search results:

| Candidate | Raw HF Val Loss | MindSpeed Valid Loss |
| --- | ---: | ---: |
| `baseline_6step.env` | 14.962966 | 0.612093 |
| `lr_low_6step.env` | 14.975369 | 0.615938 |
| `lr_high_6step.env` | 14.927566 | 0.590943 |
| `lr_higher_6step.env` | 14.819130 | 0.532507 |
| `lr_7p5em6_6step.env` | 14.622040 | 0.448289 |
| `lr_1em5_6step.env` | 14.503275 | 0.408763 |
| `lr_1p5em5_6step.env` | 14.061219 | 0.329362 |
| `lr_2em5_6step.env` | 13.792945 | 0.310141 |
| `lr_3em5_6step.env` | 13.120849 | 0.305617 |
| `lr_2em5_12step.env` | 13.097093 | 0.293768 |
| `lr_3em5_12step.env` | 12.725430 | 0.307299 |
| `lr_4em5_6step.env` | 12.786428 | 0.315171 |
| `lr_4em5_12step.env` | 12.668455 | 0.324619 |
| `lr_5em5_6step.env` | 12.643408 | 0.326887 |
| `lr_5em5_12step.env` | 12.686586 | 0.349943 |
| `lr_6em5_6step.env` | 12.540532 | 0.347458 |
| `lr_4p5em5_12step.env` | 12.678998 | 0.334108 |
| `lr_7em5_6step.env` | 12.459524 | 0.365993 |
| `lr_8em5_6step.env` | 12.434883 | 0.372704 |
| `lr_9em5_6step.env` | 12.420206 | 0.381243 |
| `lr_1em4_6step.env` | 12.381389 | 0.394805 |
| `lr_1p1em4_6step.env` | 12.271653 | 0.411056 |
| `lr_1p2em4_6step.env` | 12.152827 | 0.431454 |
| `lr_1p3em4_6step.env` | 12.037202 | 0.446334 |
| `lr_1p4em4_6step.env` | 12.018144 | 0.460213 |
| `lr_1p5em4_6step.env` | 11.958964 | 0.471812 |
| `lr_1p6em4_6step.env` | 11.906736 | 0.489890 |

Current observed best: `lr_1p6em4_6step.env`.
