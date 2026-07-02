# Changelog

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
