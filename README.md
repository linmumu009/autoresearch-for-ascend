# autoresearch-for-ascend

[English](README.md) | [中文](README.zh-CN.md)

Ascend/NPU adaptation of Andrej Karpathy's `autoresearch` idea.

This repository keeps the core autoresearch loop:

1. fix the data and evaluation code,
2. let an agent edit only the experiment file,
3. run each experiment under a fixed budget,
4. keep the commit when the metric improves,
5. discard it when it does not.

The current prototype targets a Huawei Ascend 910C server, runs inside an
isolated Docker container, and uses local Qwen3-0.6B weights.

## Where It Came From

This project started from studying:

- `karpathy/autoresearch`: the minimal autonomous research loop.
- `Ascend/MindSpeed-MM`: Ascend ecosystem reference code and environment.
- `Ascend/MindSpeed-LLM`: Ascend LLM training stack used for the current
  Qwen3-0.6B acceleration-framework evaluation.

The implementation here is not a direct fork of either repository. It is a
small Ascend-specific prototype created to test whether the autoresearch loop
can work on our available hardware and network constraints.

The external repositories are kept locally under `reference/` for study, but
they are intentionally not committed here.

## Current Shape

```text
ascend_autoresearch/
  README.md       local usage notes
  prepare.py      fixed data loading, tokenization, and validation
  train.py        editable experiment file
  program.md      agent experiment protocol
  run_train.sh    Ascend environment launcher
framework_adapters/
  mindspeed_llm/  MindSpeed-LLM smoke/evaluation scripts
    candidates/   versioned candidate env files
  ms_swift_dpo/   ms-swift DPO LoRA/FSDP2 LR search runner
docs/
  framework_evaluation.md
  updates/        version update notes
CHANGELOG.md      version history
```

## Runtime Assumptions

The tested remote runtime is:

- container name: `llin-autoresearch`
- container workspace: `/workspace/ascend_autoresearch`
- model path: `/models/Qwen3-0.6B`
- default device scope: one visible Ascend NPU

The container was created with a narrow mount/device surface: project workspace
is writable, model directories are read-only, and only one NPU device is exposed.

## Version History

| Version | Date | Summary |
| --- | --- | --- |
| v0.4.6 | 2026-07-04 | Compared saved `1.0e-4`, `1.5e-4`, and `2.0e-4` adapters on independent held-out pairs; `2.0e-4` maximizes margin, while `1.0e-4` is the more balanced LR. |
| v0.4.5 | 2026-07-04 | Added an independently authored 64-row held-out DPO set; `2.0e-4` adapter improved all margins and raised win_rate to 100%, mostly by lowering rejected logprob. |
| v0.4.4 | 2026-07-04 | Added offset-based DPO pair scoring and confirmed the `2.0e-4` adapter on rows 384-447: mean margin `0.697 -> 6.932`, win_rate `79.69% -> 100%`. |
| v0.4.3 | 2026-07-04 | Added a DPO chosen/rejected logprob scorer; the saved `2.0e-4` adapter improved 64/64 pair margins and raised win_rate from 81.25% to 100%. |
| v0.4.2 | 2026-07-04 | Saved a `2.0e-4` LoRA adapter checkpoint and ran base-vs-adapter generation checks; DPO metrics improved, but 16 short prompts produced identical text. |
| v0.4.1 | 2026-07-04 | Confirmed the `2.0e-4` ms-swift DPO LR with a 100-step holdout run: eval_loss `2e-8`, eval margin `19.84`. |
| v0.4.0 | 2026-07-04 | Added the ms-swift DPO LoRA/FSDP2 LR-search runner and found `2.0e-4` as the best 40-step holdout LR so far for Qwen3.6-27B. |
| v0.3.18 | 2026-07-03 | Probed a wider `2.0e-4` 6-step bracket; new best raw HF val_loss is `11.678953`. |
| v0.3.17 | 2026-07-03 | Probed `1.6e-4` on the 6-step boundary; new best raw HF val_loss is `11.906736`. |
| v0.3.16 | 2026-07-03 | Probed `1.5e-4` on the 6-step boundary; new best raw HF val_loss is `11.958964`. |
| v0.3.15 | 2026-07-03 | Probed `1.4e-4` on the 6-step boundary; new best raw HF val_loss is `12.018144`, with a much smaller gain than prior probes. |
| v0.3.14 | 2026-07-03 | Probed `1.3e-4` on the 6-step boundary; new best raw HF val_loss is `12.037202`. |
| v0.3.13 | 2026-07-03 | Probed `1.2e-4` on the 6-step boundary and tightened the runner to use a minimal verified Ascend runtime library path; new best raw HF val_loss is `12.152827`. |
| v0.3.12 | 2026-07-03 | Probed `1.1e-4` on the 6-step boundary and made the runner prefer the CANN aarch64 runtime path; new best raw HF val_loss is `12.271653`. |
| v0.3.11 | 2026-07-03 | Probed `1.0e-4` on the 6-step boundary; new best raw HF val_loss is `12.381389`. |
| v0.3.10 | 2026-07-03 | Probed `9.0e-5` on the 6-step boundary; new best raw HF val_loss is `12.420206`. |
| v0.3.9 | 2026-07-03 | Probed `8.0e-5` on the 6-step boundary; new best raw HF val_loss is `12.434883`. |
| v0.3.8 | 2026-07-03 | Refined the 12-step boundary at `4.5e-5` and probed `7.0e-5` on 6 steps; new best raw HF val_loss is `12.459524`. |
| v0.3.7 | 2026-07-03 | Verified `5.0e-5` on 12 steps and probed `6.0e-5` on 6 steps; new best raw HF val_loss is `12.540532`. |
| v0.3.6 | 2026-07-03 | Verified `4.0e-5` on 12 steps and probed `5.0e-5` on 6 steps; new best raw HF val_loss is `12.643408`. |
| v0.3.5 | 2026-07-03 | Compared `3.0e-5` on 12 steps and probed `4.0e-5` on 6 steps; new best raw HF val_loss is `12.725430`. |
| v0.3.4 | 2026-07-03 | Added `3.0e-5` 6-step LR probe and `2.0e-5` 12-step verification; best observed raw HF val_loss is `13.097093`. |
| v0.3.3 | 2026-07-03 | Added Chinese README and extended the MindSpeed-LLM LR boundary search to `2.0e-5`; new best raw HF val_loss is `13.792945`. |
| v0.3.2 | 2026-07-03 | Extended the MindSpeed-LLM LR search to `7.5e-6` and `1.0e-5`; new best raw HF val_loss is `14.503275`. |
| v0.3.1 | 2026-07-03 | Added first MindSpeed-LLM LR search candidates and recorded `5.0e-6` as best 6-step result. |
| v0.3.0 | 2026-07-03 | Added and validated MindSpeed-LLM autoresearch candidate runner with train-convert-evaluate-record loop. |
| v0.2.2 | 2026-07-02 | Added MindSpeed MCore-to-HF conversion and same-surface HF validation for converted checkpoints. |
| v0.2.1 | 2026-07-02 | Added configurable MindSpeed-LLM validation split/eval knobs and recorded first validation-loss smoke. |
| v0.2.0 | 2026-07-02 | Added MindSpeed-LLM adapter and framework evaluation notes for Qwen3-0.6B on Ascend 910C. |
| v0.1.0 | 2026-07-02 | Initial Ascend Qwen3-0.6B autoresearch prototype, baseline and first gradient-accumulation search. |

See [CHANGELOG.md](CHANGELOG.md) and [docs/updates](docs/updates) for details.

## Update Rule

For every meaningful code update:

1. update `CHANGELOG.md`,
2. add a note under `docs/updates/`,
3. keep this README's version table current,
4. do not commit secrets, SSH config, model files, cloned reference repos, logs, or runtime caches.

## Quick Start In The Container

```bash
cd /workspace/ascend_autoresearch
bash run_train.sh
```

For a smoke test:

```bash
bash run_train.sh --time-budget 60
```

## Current Best Result

The first search varied only `GRAD_ACCUM_STEPS`.

| Commit | val_loss | Memory | Status | Change |
| --- | ---: | ---: | --- | --- |
| `724174c` | 6.411216 | 4.6 GB | keep | baseline |
| `50ecb19` | 6.305202 | 4.6 GB | keep | grad accumulation 8 -> 4 |
| `b523bcf` | 6.278897 | 4.6 GB | keep | grad accumulation 4 -> 2 |
| `b49232a` | 6.127654 | 4.6 GB | keep | grad accumulation 2 -> 1 |

The current best is `b49232a`, a roughly 4.42% validation-loss improvement over
the baseline in the 5-minute exploration budget.

## Framework Evaluation Snapshot

| Framework | Can run? | Efficiency | Effect |
| --- | --- | --- | --- |
| HF + torch_npu thin loop | Yes | 5-minute budget completes; best smoke used about 4.6 GB HBM on one visible NPU. | Best observed val_loss: `6.127654`. |
| MindSpeed-LLM | Yes, autoresearch runner completed train -> convert -> HF eval -> TSV record. | Deepscaler smoke steady steps around 0.18-0.25 s after warmup; about 10.3 GB allocated HBM. | Best observed raw HF val_loss `11.678953` at `LR=2.0e-4`, 6 steps; base Qwen3 raw HF val_loss `14.977717`. |
| ms-swift DPO LoRA/FSDP2 | Yes, Qwen3.6-27B DPO + LoRA + FSDP2 runs. | 8 NPUs; 50-step adapter save runs take about 3m20s, with about 51.2 GiB HBM per NPU. | LR recommendation is now goal-dependent: `2.0e-4` maximizes DPO margin, while `1.0e-4` is the balanced held-out choice because it reaches 100% win_rate and slightly improves chosen logprob. |
| MindSpeed-MM | Not selected for the first Qwen3-0.6B text-only path. | Not measured. | Not measured. |

See [docs/framework_evaluation.md](docs/framework_evaluation.md) for the running
decision log.

## MindSpeed Autoresearch Loop

The first MindSpeed runner executes one complete candidate loop:

```bash
CANDIDATE_ENV=/workspace/framework_adapters/mindspeed_llm/candidates/baseline_6step.env \
  bash /workspace/framework_adapters/mindspeed_llm/run_autoresearch_candidate.sh
```

It trains a MindSpeed candidate, converts the checkpoint to Hugging Face format,
evaluates it with the fixed raw HF validation script, and appends metrics to
`/workspace/runs/mindspeed_llm/results.tsv`.

Current MindSpeed candidate results:

| Run | Raw HF Val Loss | MindSpeed Valid Loss | Last Train Loss |
| --- | ---: | ---: | ---: |
| `mindspeed_qwen3_0p6_baseline_6step` | 14.962966 | 0.612093 | 0.844955 |
| `mindspeed_qwen3_0p6_lr_low_6step` | 14.975369 | 0.615938 | 0.847379 |
| `mindspeed_qwen3_0p6_lr_high_6step` | 14.927566 | 0.590943 | 0.826132 |
| `mindspeed_qwen3_0p6_lr_higher_6step` | 14.819130 | 0.532507 | 0.772222 |
| `mindspeed_qwen3_0p6_lr_7p5em6_6step` | 14.622040 | 0.448289 | 0.693946 |
| `mindspeed_qwen3_0p6_lr_1em5_6step` | 14.503275 | 0.408763 | 0.643258 |
| `mindspeed_qwen3_0p6_lr_1p5em5_6step` | 14.061219 | 0.329362 | 0.547393 |
| `mindspeed_qwen3_0p6_lr_2em5_6step` | 13.792945 | 0.310141 | 0.513032 |
| `mindspeed_qwen3_0p6_lr_3em5_6step` | 13.120849 | 0.305617 | 0.491179 |
| `mindspeed_qwen3_0p6_lr_2em5_12step` | 13.097093 | 0.293768 | 0.477166 |
| `mindspeed_qwen3_0p6_lr_3em5_12step` | 12.725430 | 0.307299 | 0.481535 |
| `mindspeed_qwen3_0p6_lr_4em5_6step` | 12.786428 | 0.315171 | 0.509350 |
| `mindspeed_qwen3_0p6_lr_4em5_12step` | 12.668455 | 0.324619 | 0.492905 |
| `mindspeed_qwen3_0p6_lr_5em5_6step` | 12.643408 | 0.326887 | 0.528050 |
| `mindspeed_qwen3_0p6_lr_5em5_12step` | 12.686586 | 0.349943 | 0.516511 |
| `mindspeed_qwen3_0p6_lr_6em5_6step` | 12.540532 | 0.347458 | 0.555464 |
| `mindspeed_qwen3_0p6_lr_4p5em5_12step` | 12.678998 | 0.334108 | 0.503089 |
| `mindspeed_qwen3_0p6_lr_7em5_6step` | 12.459524 | 0.365993 | 0.577371 |
| `mindspeed_qwen3_0p6_lr_8em5_6step` | 12.434883 | 0.372704 | 0.590512 |
| `mindspeed_qwen3_0p6_lr_9em5_6step` | 12.420206 | 0.381243 | 0.601568 |
| `mindspeed_qwen3_0p6_lr_1em4_6step` | 12.381389 | 0.394805 | 0.612619 |
| `mindspeed_qwen3_0p6_lr_1p1em4_6step` | 12.271653 | 0.411056 | 0.626973 |
| `mindspeed_qwen3_0p6_lr_1p2em4_6step` | 12.152827 | 0.431454 | 0.644243 |
| `mindspeed_qwen3_0p6_lr_1p3em4_6step` | 12.037202 | 0.446334 | 0.656360 |
| `mindspeed_qwen3_0p6_lr_1p4em4_6step` | 12.018144 | 0.460213 | 0.667446 |
| `mindspeed_qwen3_0p6_lr_1p5em4_6step` | 11.958964 | 0.471812 | 0.688767 |
| `mindspeed_qwen3_0p6_lr_1p6em4_6step` | 11.906736 | 0.489890 | 0.703591 |
| `mindspeed_qwen3_0p6_lr_2em4_6step` | 11.678953 | 0.588330 | 0.789087 |

Current observed MindSpeed best: `lr_2em4_6step.env`. It improved the runner
baseline raw HF validation loss by `3.284013` and the base Qwen3-0.6B raw HF
validation loss by `3.298764` under the same fixed evaluation script.
