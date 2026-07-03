# Framework Evaluation

This document tracks framework choices for adapting the `autoresearch` idea to
our Ascend 910C environment. The scoring lens is intentionally practical:

1. can it run in our isolated environment,
2. is it efficient enough to justify using it,
3. does it improve or preserve the research signal.

## Current Task

- Model: Qwen3-0.6B
- Hardware: Ascend 910C server, one visible NPU for smoke tests
- Container: `llin-autoresearch`
- Network assumption: no non-China network dependency
- Training type: SFT / next-token supervised fine-tuning, not PPO or DPO

## Results So Far

| Framework | Status | Efficiency Signal | Effect Signal | Notes |
| --- | --- | --- | --- | --- |
| HF + `torch_npu` thin loop | Runs | 5-minute budget completes on one NPU; about 4.6 GB HBM in observed runs. | Best val_loss `6.127654`. | This is the working baseline and the simplest autoresearch loop. |
| MindSpeed-LLM | Runs | Deepscaler smoke steady steps around 0.18-0.25 s after warmup; about 10.3 GB allocated HBM. | Best observed converted checkpoint raw HF val_loss `12.381389` vs base `14.977717`. | Best fit for Qwen3-0.6B text SFT among the Ascend frameworks inspected so far. |
| MindSpeed-MM | Deferred | Not measured | Not measured | Useful Ascend reference, but less direct for pure text Qwen3-0.6B SFT. |

## MindSpeed-LLM Notes

The first MindSpeed-LLM check found a version mismatch: the local MindSpeed-LLM
code expected `mindspeed.core.optimizer.fix_duplicate_allgather`, which was
missing from one MindSpeed checkout. A paired MindSpeed checkout on the server
contained the module, so the adapter uses that matched MindSpeed code path for
the smoke run.

The container-only dependency additions are ordinary Python packages needed by
MindSpeed-LLM import paths. They were installed inside `llin-autoresearch`; the
physical machine and other containers were not changed.

The server had an existing preprocessed deepscaler dataset whose `labels`
records were shorter than `input_ids`, for example `input_ids` length 170 and
`labels` length 7. That format reaches the training forward pass but fails at
token-level cross entropy. The working path is to rebuild the data with
`AlpacaStyleInstructionHandler`, which produces equal-length `input_ids`,
`attention_mask`, and `labels`.

## MindSpeed-LLM Smoke Result

Run settings:

- checkpoint: `/models/Qwen3-0p6`
- tokenizer: `/models/Qwen3-0.6B`
- data: 512 deepscaler records converted to Alpaca-style SFT
- NPU scope: one visible NPU in `llin-autoresearch`
- steps: 3
- global batch size: 1
- sequence length argument: 2048, with no-pad instruction dataset mode

Observed output:

| Iteration | Step Time | Train Loss | Grad Norm |
| ---: | ---: | ---: | ---: |
| 1 | 1729.6 ms | 1.028000 | 27.809 |
| 2 | 249.9 ms | 0.977111 | 33.739 |
| 3 | 238.9 ms | 0.632855 | 17.502 |

Memory reported after the first iteration:

- allocated: about 10232.9 MB
- max allocated: about 10248.9 MB
- reserved: about 10308 MB

Interpretation: MindSpeed-LLM now clears the practical "can run" bar. The
efficiency signal is mixed but promising: it uses more memory than the thin HF
loop because it builds full Megatron optimizer/checkpoint machinery, but the
steady step time is low after startup. The effect signal is not yet comparable
to the HF validation loss because this run logs training loss only.

## MindSpeed-LLM Validation Smoke

Run settings:

- data: same 512 deepscaler-derived SFT records
- split: `90,10,0`
- steps: 6
- eval interval: 3
- eval iters: 2
- global batch size: 1

Observed train/eval output:

| Iteration | Train Step Time | Train Loss | Validation Loss |
| ---: | ---: | ---: | ---: |
| 1 | 1703.7 ms | 0.939238 | - |
| 2 | 237.8 ms | 1.102373 | - |
| 3 | 240.9 ms | 0.616439 | 1.254728 |
| 4 | 184.7 ms | 0.589018 | - |
| 5 | 176.9 ms | 0.681547 | - |
| 6 | 177.4 ms | 0.844050 | 0.691403 |

The final validation-set pass after checkpoint save reported:

- validation loss: `0.611867`
- validation PPL: `1.843871`

Interpretation: the MindSpeed path now has a repeatable held-out validation
signal. It is still not a direct apples-to-apples comparison with the HF
thin-loop `val_loss=6.127654`, because the two metrics are computed over
different label surfaces. The next strict comparison should either convert the
same validation records into both formats with equivalent labels, or convert
MindSpeed checkpoints back to HF and run the existing `prepare.py` evaluator.

## Converted Checkpoint HF Evaluation

The first strict raw-validation bridge is now in place:

1. Train Qwen3-0.6B with MindSpeed-LLM.
2. Convert the MindSpeed/MCore checkpoint back to Hugging Face format.
3. Evaluate the converted HF checkpoint with `ascend_autoresearch/evaluate_hf.py`.

Run settings:

- converted checkpoint:
  `/workspace/outputs/mindspeed_qwen3_0p6_deepscaler_eval_smoke_hf`
- validation surface: existing `prepare.py` raw next-token validation tokens
- seq length: 512
- batch size: 1

Observed results:

| Model | Raw HF Val Loss | Raw HF Val PPL | Eval Time | Peak HBM |
| --- | ---: | ---: | ---: | ---: |
| Base `/models/Qwen3-0.6B` | 14.977717 | 3196979.536 | 4.4 s | 2243.6 MB |
| Converted MindSpeed 6-step checkpoint | 14.963873 | 3153024.804 | 4.3 s | 2243.6 MB |

Interpretation: the effect is tiny but directionally positive on the raw HF
validation surface. That is exactly the scale we should expect from a 6-step
smoke. The important engineering result is the closed loop: MindSpeed training
can now be evaluated by the same raw validation code used by the HF thin-loop
prototype.

## Autoresearch Runner

The v0.3.0 runner turns the manual sequence into a repeatable candidate loop:

1. train a MindSpeed-LLM candidate,
2. convert its MCore checkpoint to Hugging Face format,
3. evaluate the converted checkpoint with the fixed raw HF validation script,
4. append metrics to a TSV file.

Default output layout:

```text
/workspace/runs/mindspeed_llm/
  results.tsv
  <run_name>/
    mcore/
    hf/
    logs/
```

This is the first practical bridge to Karpathy-style autoresearch on Ascend:
agents can now vary a small candidate env file, run the same loop, and compare
raw HF validation loss before deciding whether to keep the change.

Validated baseline candidate:

| Run | Raw HF Val Loss | Raw HF Val PPL | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: |
| `mindspeed_qwen3_0p6_baseline_6step` | 14.962966 | 3150167.515 | 0.612093 | 0.844955 |

The same runner invocation also evaluated the base model at raw HF val_loss
`14.977717`, so the baseline candidate is directionally positive on the fixed
HF validation surface.

## First LR Search

The first actual MindSpeed autoresearch search pass varied only learning rate
while keeping the runner loop fixed:

- training: 6 MindSpeed-LLM full-SFT iterations
- data: 512 deepscaler-derived Alpaca-style SFT records
- conversion: MindSpeed/MCore to Hugging Face
- evaluation: fixed raw HF validation via `evaluate_hf.py`

| Candidate | LR | Raw HF Val Loss | Raw HF Val PPL | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_6step` | `1.25e-6` | 14.962966 | 3150167.515 | 0.612093 | 0.844955 |
| `lr_low_6step` | `6.25e-7` | 14.975369 | 3189481.816 | 0.615938 | 0.847379 |
| `lr_high_6step` | `2.5e-6` | 14.927566 | 3040603.000 | 0.590943 | 0.826132 |
| `lr_higher_6step` | `5.0e-6` | 14.819130 | 2728138.084 | 0.532507 | 0.772222 |
| `lr_7p5em6_6step` | `7.5e-6` | 14.622040 | 2240120.089 | 0.448289 | 0.693946 |
| `lr_1em5_6step` | `1.0e-5` | 14.503275 | 1989263.400 | 0.408763 | 0.643258 |
| `lr_1p5em5_6step` | `1.5e-5` | 14.061219 | 1278526.649 | 0.329362 | 0.547393 |
| `lr_2em5_6step` | `2.0e-5` | 13.792945 | 977686.773 | 0.310141 | 0.513032 |
| `lr_3em5_6step` | `3.0e-5` | 13.120849 | 499243.503 | 0.305617 | 0.491179 |
| `lr_2em5_12step` | `2.0e-5` | 13.097093 | 487523.225 | 0.293768 | 0.477166 |
| `lr_3em5_12step` | `3.0e-5` | 12.725430 | 336189.379 | 0.307299 | 0.481535 |
| `lr_4em5_6step` | `4.0e-5` | 12.786428 | 357334.816 | 0.315171 | 0.509350 |
| `lr_4em5_12step` | `4.0e-5` | 12.668455 | 317570.340 | 0.324619 | 0.492905 |
| `lr_5em5_6step` | `5.0e-5` | 12.643408 | 309714.910 | 0.326887 | 0.528050 |
| `lr_5em5_12step` | `5.0e-5` | 12.686586 | 323380.772 | 0.349943 | 0.516511 |
| `lr_6em5_6step` | `6.0e-5` | 12.540532 | 279436.825 | 0.347458 | 0.555464 |
| `lr_4p5em5_12step` | `4.5e-5` | 12.678998 | 320936.202 | 0.334108 | 0.503089 |
| `lr_7em5_6step` | `7.0e-5` | 12.459524 | 257692.972 | 0.365993 | 0.577371 |
| `lr_8em5_6step` | `8.0e-5` | 12.434883 | 251420.643 | 0.372704 | 0.590512 |
| `lr_9em5_6step` | `9.0e-5` | 12.420206 | 247757.501 | 0.381243 | 0.601568 |
| `lr_1em4_6step` | `1.0e-4` | 12.381389 | 238324.589 | 0.394805 | 0.612619 |

Interpretation: lower LR was worse, and higher LR helped under this tiny 6-step
budget. The 6-step gain continued through `1.0e-4`, but the improvement from
`8.0e-5` to `1.0e-4` is much smaller than earlier jumps and MindSpeed-local
valid/train loss is rising. Under the same 12-step budget, both `4.5e-5` and
`5.0e-5` are worse than `4.0e-5`, so the longer-budget local best among tested
candidates remains `4.0e-5`. The next choice is a small `1.1e-4` 6-step probe
to bracket the short-budget over-shoot boundary, then a longer-budget
confirmation around the best bracket.

## 5-Minute Budget

Karpathy's original 5-minute loop is still valuable as the outer research
discipline: every attempt should have a tight time cap, a single measurable
result, and a keep/revert decision.

For Ascend/MindSpeed, the first run may need a separate smoke budget because
framework startup, extension loading, and dataset checks can dominate the first
few minutes. After a stable launch path exists, the comparable budget should be:

- 60 seconds for environment smoke,
- 300 seconds for autoresearch candidate evaluation,
- same fixed validation data and metric per framework when possible.
