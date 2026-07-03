# ms-swift DPO LR Search

This adapter records the Qwen3.6-27B DPO + LoRA + FSDP2 learning-rate search
run in the existing `llin-rl-dpo` Ascend container.

The runner is intentionally small: it keeps the model, dataset, LoRA shape,
FSDP2 config, scheduler, and batch shape fixed, then exposes only the knobs
needed for short-budget LR search.

## Runner

```bash
RUN_ID=lr_2e-4_40step_eval10p \
LEARNING_RATE=2e-4 \
MAX_STEPS=40 \
SPLIT_DATASET_RATIO=0.1 \
EVAL_STRATEGY=steps \
EVAL_STEPS=20 \
MASTER_PORT=29901 \
bash scripts/run_ms_swift_lr_candidate.sh
```

Default fixed settings:

- `rlhf_type=dpo`
- `model=/models/Qwen3.6-27B`
- `dataset=/workspace/llin-rl-dpo/datasets/ops_dpo_512.jsonl`
- `tuner_type=lora`
- `lora_rank=8`
- `lora_alpha=32`
- `target_modules=all-linear`
- `fsdp=/workspace/llin-rl-dpo/configs/fsdp2_full_state.json`
- `lr_scheduler_type=cosine`
- `warmup_steps=0`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=1`

## Current Result

In the 40-step, 10% holdout search, `2.0e-4` is the current best LR:

- eval_loss: `1.2e-7`
- eval margin: `17.98`
- train speed: `0.236` steps/s
- HBM: about `51.85` GiB per NPU

This is a short-budget preference-training result. It should be confirmed with
a longer run and generation-side checks before being treated as final model
quality evidence.
