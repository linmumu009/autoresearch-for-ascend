# Changelog

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
