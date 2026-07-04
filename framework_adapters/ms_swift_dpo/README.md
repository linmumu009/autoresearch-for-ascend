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

To save a usable adapter checkpoint:

```bash
RUN_ID=lr_2e-4_50step_save_eval10p \
LEARNING_RATE=2e-4 \
MAX_STEPS=50 \
SPLIT_DATASET_RATIO=0.1 \
EVAL_STRATEGY=steps \
EVAL_STEPS=50 \
SAVE_STRATEGY=steps \
SAVE_STEPS=50 \
SAVE_TOTAL_LIMIT=1 \
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

In the 40-step, 10% holdout search, `2.0e-4` was the best LR. A 100-step
confirmation kept it stable:

- eval_loss: `2e-8`
- eval margin: `19.84`
- train speed: `0.281` steps/s
- HBM: about `51.19` GiB per NPU

A 50-step save run produced a usable LoRA checkpoint with eval_loss `5e-8` and
eval margin `18.38`. The checkpoint loads as `PeftModelForCausalLM`, but it did
not change short-form generation in the current smoke checks:

- 2 fixed prompts, greedy decoding: 0 changed outputs
- 16 ops prompts, greedy decoding: 0 changed outputs
- 16 ops prompts, `temperature=0.7`: 0 changed outputs

This is a short-budget preference-training result. The next check should use
preference/logprob scoring or a larger generation benchmark before treating it
as final model quality evidence.
