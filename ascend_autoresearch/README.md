# Ascend Autoresearch Prototype

This is a small Ascend/NPU adaptation of Karpathy's `autoresearch` loop.

Goal: run fixed-budget Qwen3-0.6B experiments inside the isolated
`llin-autoresearch` Docker container. The human edits `program.md`; the agent
edits only `train.py`.

## Layout

- `prepare.py` fixes data parsing, tokenization, dataloading, and validation.
- `train.py` is the editable experiment file.
- `program.md` is the agent instruction sheet.
- `run_train.sh` sources Ascend environment variables and launches training.

## Default Server Paths

- Model: `/models/Qwen3-0.6B`
- Workspace: `/workspace/ascend_autoresearch`
- Raw data:
  - `/workspace/inputs/raw/deepscaler.json`
  - `/workspace/inputs/raw/ceval_val.jsonl`

## Quick Start In Container

```bash
cd /workspace/ascend_autoresearch
bash run_train.sh --time-budget 60
```

For real experiments, omit `--time-budget`; the default is 300 seconds.
