# Ascend autoresearch

This is an experiment to let an AI agent do small Qwen3-0.6B training research
on one Ascend NPU.

## Scope

You may edit only `train.py`.

Do not edit:

- `prepare.py`
- raw data under `/workspace/inputs`
- model files under `/models`
- Docker, system packages, or other containers

The goal is to lower `val_loss` under the fixed validation function in
`prepare.py`. Lower is better.

## Setup

1. Work on a branch named `autoresearch/<tag>`.
2. Run the baseline first:

```bash
bash run_train.sh
```

3. Initialize `results.tsv`:

```text
commit	val_loss	memory_gb	status	description
```

## Experiment Loop

Repeat indefinitely until stopped:

1. Inspect current git state.
2. Change `train.py` with one clear idea.
3. Commit the change.
4. Run:

```bash
bash run_train.sh > run.log 2>&1
```

5. Extract metrics:

```bash
grep "^val_loss:\|^peak_npu_mem_mb:" run.log
```

6. Append one row to `results.tsv`.
7. If `val_loss` improved, keep the commit.
8. If it did not improve, reset back to the previous kept commit.

Crashes count as `crash` with `val_loss` set to `0.000000`.

## Things Worth Trying

- Learning rate and schedule.
- Optimizer type and weight decay.
- Batch size, sequence length, and gradient accumulation.
- Freezing or unfreezing parts of the model.
- Gradient checkpointing.
- Loss masking choices.

Prefer simple changes. A tiny improvement with a large fragile hack is usually
not worth keeping.
