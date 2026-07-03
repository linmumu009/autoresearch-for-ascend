# Changelog

## v0.4.0 - 2026-07-04

Added the first ms-swift DPO LoRA/FSDP2 learning-rate search runner for
Qwen3.6-27B on the existing `llin-rl-dpo` Ascend runtime.

### Added

- Added `framework_adapters/ms_swift_dpo/run_lr_candidate.sh`.
- Added configurable `SPLIT_DATASET_RATIO`, `EVAL_STRATEGY`, and `EVAL_STEPS`
  controls for short-budget DPO holdout checks.

### Results

All runs used Qwen3.6-27B, DPO, LoRA rank 8 / alpha 32, FSDP2 full shard,
8 NPUs, `max_length=512`, cosine LR schedule, no warmup, and 10% holdout for
the 40-step confirmation runs.

| LR | Steps | Eval Loss | Eval Margin | Eval Chosen | Eval Rejected | Train Steps/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2.0e-4` | 40 | `1.2e-7` | `17.98` | `5.09` | `-12.88` | `0.236` |
| `3.0e-4` | 40 | `4.1e-7` | `17.29` | `4.69` | `-12.58` | `0.231` |
| `2.5e-4` | 40 | `4.2e-7` | `17.16` | `4.58` | `-12.57` | `0.230` |
| `1.5e-4` | 40 | `5.6e-7` | `16.24` | `6.25` | `-10.01` | `0.229` |
| `4.0e-4` | 40 | `8.2e-7` | `16.75` | `3.86` | `-12.92` | `0.233` |
| `1.0e-4` | 40 | `2.84e-6` | `14.71` | `6.80` | `-7.92` | `0.239` |

### Notes

- The current best LR is `2.0e-4` on the 40-step holdout surface.
- Higher rates still run stably, but `4.0e-4` starts to lower chosen rewards
  while not improving holdout loss, so it looks like the first over-aggressive
  region for this short-budget setup.
- This is a DPO preference-training search, not SFT/PPO/DPO-vs-PPO model
  quality proof. A longer confirmation run and downstream generation checks are
  still needed before calling it final.

## v0.3.18 - 2026-07-03

Used a wider 6-step LR bracket probe at `2.0e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_2em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_2em4_6step.env` | `2.0e-4` | 6 | `11.678953` | `0.588330` | `0.789087` |

### Notes

- `LR=2.0e-4` improved raw HF val_loss by `0.227783` over `1.6e-4`, so the
  wider bracket did not over-shoot.
- MindSpeed-local validation and train loss are now much higher, so the 6-step
  optimum remains a short-budget signal only.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `3.284013`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `3.298764`.

## v0.3.17 - 2026-07-03

Continued the 6-step LR boundary search at `1.6e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p6em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p6em4_6step.env` | `1.6e-4` | 6 | `11.906736` | `0.489890` | `0.703591` |

### Notes

- `LR=1.6e-4` improved raw HF val_loss by `0.052228` over `1.5e-4`.
- The short-budget over-shoot point is still not bracketed, but the
  MindSpeed-local validation loss is now close to `0.49`, so the next probe
  should bracket faster rather than keep inching upward.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `3.056230`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `3.070981`.

## v0.3.16 - 2026-07-03

Continued the 6-step LR boundary search at `1.5e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p5em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p5em4_6step.env` | `1.5e-4` | 6 | `11.958964` | `0.471812` | `0.688767` |

### Notes

- `LR=1.5e-4` improved raw HF val_loss by `0.059180` over `1.4e-4`, so the
  short-budget over-shoot point is still not bracketed.
- MindSpeed-local validation and train loss continue rising, preserving the
  need for longer-budget confirmation after the boundary search.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `3.004002`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `3.018753`.

## v0.3.15 - 2026-07-03

Continued the 6-step LR boundary search at `1.4e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p4em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p4em4_6step.env` | `1.4e-4` | 6 | `12.018144` | `0.460213` | `0.667446` |

### Notes

- `LR=1.4e-4` is still the best 6-step raw HF result, but the improvement over
  `1.3e-4` is only `0.019058`.
- The short-budget boundary is likely close.
- MindSpeed-local validation and train loss continue rising, so longer-budget
  confirmation should wait until the 6-step over-shoot point is bracketed.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.944822`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.959573`.

## v0.3.14 - 2026-07-03

Continued the 6-step LR boundary search at `1.3e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p3em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p3em4_6step.env` | `1.3e-4` | 6 | `12.037202` | `0.446334` | `0.656360` |

### Notes

- `LR=1.3e-4` still improves the 6-step raw HF validation surface.
- The raw-HF improvement over `1.2e-4` is `0.115625`, so the short-budget
  boundary still has not been bracketed.
- MindSpeed-local validation and train loss continue rising, so high-LR
  candidates remain short-budget probes until a longer confirmation passes.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.925764`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.940515`.

## v0.3.13 - 2026-07-03

Continued the 6-step LR boundary search at `1.2e-4` and tightened the runner's
Ascend runtime library path handling.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p2em4_6step.env`.

### Changed

- `run_autoresearch_candidate.sh` now uses a minimal verified Ascend runtime
  `LD_LIBRARY_PATH` when the CANN aarch64 runtime directory exists. This avoids
  inheriting incompatible `ascend-toolkit/latest` library entries during plain
  `docker exec` launches.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p2em4_6step.env` | `1.2e-4` | 6 | `12.152827` | `0.431454` | `0.644243` |

### Notes

- `LR=1.2e-4` still improves the 6-step raw HF validation surface.
- MindSpeed-local validation and train loss keep rising, so the short-budget
  search should continue only as a boundary bracket before longer confirmation.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.810139`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.824890`.

## v0.3.12 - 2026-07-03

Continued the 6-step LR boundary search at `1.1e-4` and hardened the runner's
Ascend runtime library selection.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1p1em4_6step.env`.

### Changed

- `run_autoresearch_candidate.sh` now prepends
  `/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64` to `LD_LIBRARY_PATH` when
  that directory exists. This avoids loading the wrong `libruntime_common.so`
  when launching the runner through a plain `docker exec`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1p1em4_6step.env` | `1.1e-4` | 6 | `12.271653` | `0.411056` | `0.626973` |

### Notes

- `LR=1.1e-4` still improves the 6-step raw HF validation surface and gives the
  largest single-step gain since `7.0e-5`.
- MindSpeed-local validation and train loss rose again, so this high LR remains
  a short-budget candidate that needs bracketing before longer training.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.691313`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.706064`.

## v0.3.11 - 2026-07-03

Continued the 6-step LR boundary search at `1.0e-4`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_1em4_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_1em4_6step.env` | `1.0e-4` | 6 | `12.381389` | `0.394805` | `0.612619` |

### Notes

- `LR=1.0e-4` still improves the 6-step raw HF validation surface.
- MindSpeed-local valid loss and last train loss both continue rising, so this
  is a short-budget gain rather than evidence that the LR is safe for longer
  training.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.581577`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.596328`.

## v0.3.10 - 2026-07-03

Continued the 6-step LR boundary search at `9.0e-5`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_9em5_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_9em5_6step.env` | `9.0e-5` | 6 | `12.420206` | `0.381243` | `0.601568` |

### Notes

- `LR=9.0e-5` still improves the 6-step raw HF validation surface.
- The gain over `8.0e-5` is small, suggesting the short-budget boundary is
  close.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.542760`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.557511`.

## v0.3.9 - 2026-07-03

Continued the 6-step LR boundary search at `8.0e-5`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_8em5_6step.env`.

### Results

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_8em5_6step.env` | `8.0e-5` | 6 | `12.434883` | `0.372704` | `0.590512` |

### Notes

- `LR=8.0e-5` still improves the 6-step raw HF validation surface, but the gain
  over `7.0e-5` is now small.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.528083`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.542834`.

## v0.3.8 - 2026-07-03

Refined the 12-step LR boundary and continued the 6-step LR boundary search at
`7.0e-5`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_4p5em5_12step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_7em5_6step.env`.

### Results

Both candidates used the same train -> MCore-to-HF conversion -> fixed raw HF
validation runner.

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_4p5em5_12step.env` | `4.5e-5` | 12 | `12.678998` | `0.334108` | `0.503089` |
| `lr_7em5_6step.env` | `7.0e-5` | 6 | `12.459524` | `0.365993` | `0.577371` |

### Notes

- `LR=4.5e-5` is worse than `4.0e-5` under the 12-step budget, so the 12-step
  local best among tested candidates remains `4.0e-5`.
- `LR=7.0e-5` still improves the 6-step raw HF validation surface and is the
  best observed candidate so far.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.503442`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.518193`.

## v0.3.7 - 2026-07-03

Verified `5.0e-5` under the 12-step budget and continued the 6-step LR boundary
search at `6.0e-5`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_5em5_12step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_6em5_6step.env`.

### Results

Both candidates used the same train -> MCore-to-HF conversion -> fixed raw HF
validation runner.

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_5em5_12step.env` | `5.0e-5` | 12 | `12.686586` | `0.349943` | `0.516511` |
| `lr_6em5_6step.env` | `6.0e-5` | 6 | `12.540532` | `0.347458` | `0.555464` |

### Notes

- `LR=5.0e-5` is worse than `4.0e-5` under the 12-step budget, so the 12-step
  over-shoot boundary likely sits between `4.0e-5` and `5.0e-5`.
- `LR=6.0e-5` still improves the 6-step raw HF validation surface and is the
  best observed candidate so far.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.422434`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.437185`.

## v0.3.6 - 2026-07-03

Verified `4.0e-5` under the 12-step budget and continued the 6-step LR boundary
search at `5.0e-5`.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_4em5_12step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_5em5_6step.env`.

### Results

Both candidates used the same train -> MCore-to-HF conversion -> fixed raw HF
validation runner.

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_4em5_12step.env` | `4.0e-5` | 12 | `12.668455` | `0.324619` | `0.492905` |
| `lr_5em5_6step.env` | `5.0e-5` | 6 | `12.643408` | `0.326887` | `0.528050` |

### Notes

- `LR=4.0e-5` beats `LR=3.0e-5` when both use the 12-step budget.
- `LR=5.0e-5` still improves the 6-step raw HF validation surface and is the
  best observed candidate so far.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.319558`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.334309`.

## v0.3.5 - 2026-07-03

Compared the higher LR under the same 12-step budget and continued probing the
6-step LR boundary.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_3em5_12step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_4em5_6step.env`.

### Results

Both candidates used the same train -> MCore-to-HF conversion -> fixed raw HF
validation runner.

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_3em5_12step.env` | `3.0e-5` | 12 | `12.725430` | `0.307299` | `0.481535` |
| `lr_4em5_6step.env` | `4.0e-5` | 6 | `12.786428` | `0.315171` | `0.509350` |

### Notes

- `LR=3.0e-5` beats `LR=2.0e-5` when both use the 12-step budget.
- `LR=4.0e-5` still improves the 6-step raw HF validation surface, so the
  6-step over-shoot boundary is still above `4.0e-5`.
- Compared with runner baseline raw HF val_loss `14.962966`, the current best
  candidate improved by `2.237536`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the current best
  candidate improved by `2.252287`.

## v0.3.4 - 2026-07-03

Probed a higher 6-step learning rate and verified the current best LR on a
longer 12-step budget.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_3em5_6step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_2em5_12step.env`.

### Results

Both candidates used the same train -> MCore-to-HF conversion -> fixed raw HF
validation runner. The 12-step candidate uses the same data and LR as the
previous best 6-step candidate, but doubles `TRAIN_ITERS`.

| Candidate | LR | Steps | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lr_3em5_6step.env` | `3.0e-5` | 6 | `13.120849` | `0.305617` | `0.491179` |
| `lr_2em5_12step.env` | `2.0e-5` | 12 | `13.097093` | `0.293768` | `0.477166` |

### Notes

- `LR=3.0e-5` still improved the 6-step LR boundary search, so the 6-step
  over-shoot boundary is still above `3.0e-5`.
- `LR=2.0e-5` remained strong when extended to 12 steps and is the best
  observed candidate so far.
- Compared with runner baseline raw HF val_loss `14.962966`, the 12-step
  candidate improved by `1.865873`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the 12-step
  candidate improved by `1.880624`.

## v0.3.3 - 2026-07-03

Added bilingual README support and extended the MindSpeed-LLM LR boundary
search.

### Added

- Added `README.zh-CN.md` and language links between English and Chinese
  README files.
- Added `framework_adapters/mindspeed_llm/candidates/lr_1p5em5_6step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_2em5_6step.env`.

### Results

Both candidates used the same 6-step runner loop, deepscaler-derived SFT data,
MCore-to-HF conversion, and fixed raw HF validation.

| Candidate | LR | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: |
| `lr_1p5em5_6step.env` | `1.5e-5` | `14.061219` | `0.329362` | `0.547393` |
| `lr_2em5_6step.env` | `2.0e-5` | `13.792945` | `0.310141` | `0.513032` |

### Notes

- `LR=2.0e-5` is the current best 6-step MindSpeed candidate.
- Compared with the runner baseline raw HF val_loss `14.962966`, the best
  candidate improved by `1.170021`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the best
  candidate improved by `1.184772`.
- Since the loss still improved at `2.0e-5`, the over-shoot boundary is still
  above this point for the current tiny 6-step budget.

## v0.3.2 - 2026-07-03

Extended the MindSpeed-LLM LR range search.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_7p5em6_6step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_1em5_6step.env`.

### Results

Both candidates used the same 6-step runner loop, deepscaler-derived SFT data,
MCore-to-HF conversion, and fixed raw HF validation.

| Candidate | LR | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: |
| `lr_7p5em6_6step.env` | `7.5e-6` | `14.622040` | `0.448289` | `0.693946` |
| `lr_1em5_6step.env` | `1.0e-5` | `14.503275` | `0.408763` | `0.643258` |

### Notes

- `LR=1.0e-5` is the current best 6-step MindSpeed candidate.
- Compared with the runner baseline raw HF val_loss `14.962966`, the best
  candidate improved by `0.459691`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the best
  candidate improved by `0.474442`.
- Since the loss continued improving at `1.0e-5`, the next search should probe
  the over-shoot boundary before increasing the training budget.

## v0.3.1 - 2026-07-03

First MindSpeed-LLM LR search pass.

### Added

- Added `framework_adapters/mindspeed_llm/candidates/lr_low_6step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_high_6step.env`.
- Added `framework_adapters/mindspeed_llm/candidates/lr_higher_6step.env`.

### Results

All candidates used the same 6-step runner loop, deepscaler-derived SFT data,
MCore-to-HF conversion, and fixed raw HF validation.

| Candidate | LR | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: | ---: |
| baseline | `1.25e-6` | `14.962966` | `0.612093` | `0.844955` |
| low | `6.25e-7` | `14.975369` | `0.615938` | `0.847379` |
| high | `2.5e-6` | `14.927566` | `0.590943` | `0.826132` |
| higher | `5.0e-6` | `14.819130` | `0.532507` | `0.772222` |

### Notes

- `LR=5.0e-6` is the best result in this first pass.
- Compared with the runner baseline, raw HF val_loss improved by `0.143836`.
- Compared with base Qwen3-0.6B raw HF val_loss `14.977717`, the best
  candidate improved by `0.158587`.

## v0.3.0 - 2026-07-03

MindSpeed-LLM autoresearch candidate runner.

### Added

- Added `framework_adapters/mindspeed_llm/run_autoresearch_candidate.sh`.
- Added `framework_adapters/mindspeed_llm/candidates/baseline_6step.env`.
- Documented the train -> convert -> fixed HF eval -> TSV record loop.

### Changed

- Made the MindSpeed training script honor the `LR` environment variable so
  learning-rate candidates are real experiments.

### Intent

- Move from one-off smoke commands to repeatable autoresearch candidates.
- Keep every candidate in its own run directory under
  `/workspace/runs/mindspeed_llm/`.
- Use `ascend_autoresearch/evaluate_hf.py` as the raw validation gate.

### Results

- Validated the baseline candidate end-to-end in `llin-autoresearch`.
- Runner output path:
  `/workspace/runs/mindspeed_llm/mindspeed_qwen3_0p6_baseline_6step/`.
- Converted checkpoint raw HF validation loss: `14.962966`.
- Base Qwen3-0.6B raw HF validation loss from the same run: `14.977717`.
- MindSpeed held-out validation loss: `0.612093`.
- Last MindSpeed train loss: `0.844955`.

## v0.2.2 - 2026-07-02

Same-surface HF validation for converted MindSpeed checkpoints.

### Added

- Added `ascend_autoresearch/evaluate_hf.py` for eval-only HF raw next-token
  validation using the fixed `prepare.py` data path.
- Added `framework_adapters/mindspeed_llm/convert_mcore_to_hf.sh` to convert
  MindSpeed/MCore checkpoints to Hugging Face format and copy required HF
  config/tokenizer metadata.

### Results

- Converted `/workspace/outputs/mindspeed_qwen3_0p6_deepscaler_eval_smoke_ckpt`
  to Hugging Face format.
- Base Qwen3-0.6B raw HF validation loss: `14.977717`.
- Converted MindSpeed 6-step checkpoint raw HF validation loss: `14.963873`.
- Both eval runs used `seq_len=512`, `batch_size=1`, and the same
  `prepare.py` validation tokens.

### Notes

- The improvement is intentionally tiny because the MindSpeed checkpoint only
  trained for 6 micro steps on 512 deepscaler-derived SFT records.
- This closes the first end-to-end comparison loop: MindSpeed train, convert to
  HF, evaluate with the existing raw HF validation surface.

## v0.2.1 - 2026-07-02

MindSpeed-LLM validation smoke setup.

### Changed

- Made `framework_adapters/mindspeed_llm/smoke_qwen3_0p6_full_sft.sh`
  configurable via `DATA_SPLIT`, `EVAL_INTERVAL`, `EVAL_ITERS`, and
  `SAVE_INTERVAL`.
- Updated framework evaluation notes with the first MindSpeed-LLM validation
  loss run.

### Results

- Ran Qwen3-0.6B full SFT with `DATA_SPLIT=90,10,0`, `TRAIN_ITERS=6`,
  `EVAL_INTERVAL=3`, and `EVAL_ITERS=2`.
- Validation loss at iteration 3: `1.254728`.
- Validation loss at iteration 6: `0.691403`.
- Final validation-set loss after checkpoint save: `0.611867`.
- Steady train step time after warmup was about `0.18-0.25s`.

### Notes

- This proves MindSpeed-LLM can produce a repeatable held-out validation signal.
- The number is not directly comparable to the HF thin-loop `val_loss` yet,
  because the MindSpeed run uses Alpaca-style SFT labels while the HF baseline
  uses raw next-token validation text.

## v0.2.0 - 2026-07-02

MindSpeed-LLM framework evaluation setup for Qwen3-0.6B on Ascend 910C.

### Added

- Added `framework_adapters/mindspeed_llm/smoke_qwen3_0p6_full_sft.sh`.
- Added `framework_adapters/mindspeed_llm/prepare_deepscaler_alpaca.sh`.
- Added MindSpeed-LLM adapter notes under `framework_adapters/mindspeed_llm/`.
- Added `docs/framework_evaluation.md` to track "can run / efficiency / effect".
- Added v0.2.0 update note.

### Findings

- HF + `torch_npu` thin loop is the current working baseline.
- MindSpeed-LLM is a better first acceleration-framework target than
  MindSpeed-MM for Qwen3-0.6B text-only SFT.
- The MindSpeed-LLM Python stack imports successfully in the isolated
  `llin-autoresearch` container when paired with the matching MindSpeed code.
- A 3-step MindSpeed-LLM full-SFT smoke completed on a 512-sample deepscaler
  conversion. Observed train loss moved from `1.0280` to `0.6329`, with steady
  step time around `0.24-0.25s` after warmup and about `10.3 GB` allocated HBM.

### Pending

- Add strict validation-loss evaluation so MindSpeed-LLM can be compared
  directly with the HF thin-loop `val_loss`.

## v0.1.0 - 2026-07-02

Initial public project setup for `autoresearch-for-ascend`.

### Added

- Added Ascend/Qwen3-0.6B autoresearch prototype under `ascend_autoresearch/`.
- Added fixed data and evaluation utilities in `prepare.py`.
- Added editable experiment surface in `train.py`.
- Added agent protocol in `program.md`.
- Added Ascend environment launcher in `run_train.sh`.
- Added project README and version update note.

### Results

- Baseline 5-minute validation loss: `6.411216`.
- Best first-pass result: `6.127654`.
- Best change so far: reduce `GRAD_ACCUM_STEPS` from `8` to `1`.

### Notes

- `reference/` is intentionally ignored because it contains external cloned repositories.
- `ssh_config/` is intentionally ignored because it can contain private login material.
- Runtime logs, caches, model files, and `results.tsv` are not committed.
