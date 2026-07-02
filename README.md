# autoresearch-for-ascend

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
| MindSpeed-LLM | Yes, full SFT smoke, validation smoke, HF conversion, and HF eval completed. | Deepscaler smoke steady steps around 0.18-0.25 s after warmup; about 10.3 GB allocated HBM. | Converted 6-step checkpoint raw HF val_loss `14.963873` vs base `14.977717`; local SFT validation loss `0.611867`. |
| MindSpeed-MM | Not selected for the first Qwen3-0.6B text-only path. | Not measured. | Not measured. |

See [docs/framework_evaluation.md](docs/framework_evaluation.md) for the running
decision log.
