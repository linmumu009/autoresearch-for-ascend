# Changelog

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
