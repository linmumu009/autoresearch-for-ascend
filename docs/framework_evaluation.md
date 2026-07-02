# Framework Evaluation

This document tracks framework choices for adapting the `autoresearch` idea to
our Ascend 910C environment. The scoring lens is intentionally practical:

1. can it run in our isolated environment,
2. is it efficient enough to justify using it,
3. does it improve or preserve the research signal.

## Current Task

- Model: Qwen3-0.6B
- Hardware: Ascend 910C server, one visible NPU for smoke tests
- Container: `llin-autoresearch`
- Network assumption: no non-China network dependency
- Training type: SFT / next-token supervised fine-tuning, not PPO or DPO

## Results So Far

| Framework | Status | Efficiency Signal | Effect Signal | Notes |
| --- | --- | --- | --- | --- |
| HF + `torch_npu` thin loop | Runs | 5-minute budget completes on one NPU; about 4.6 GB HBM in observed runs. | Best val_loss `6.127654`. | This is the working baseline and the simplest autoresearch loop. |
| MindSpeed-LLM | Runs | Deepscaler smoke steady steps around 0.24-0.25 s after warmup; about 10.3 GB allocated HBM. | Deepscaler smoke train loss `1.0280 -> 0.6329`; strict validation still pending. | Best fit for Qwen3-0.6B text SFT among the Ascend frameworks inspected so far. |
| MindSpeed-MM | Deferred | Not measured | Not measured | Useful Ascend reference, but less direct for pure text Qwen3-0.6B SFT. |

## MindSpeed-LLM Notes

The first MindSpeed-LLM check found a version mismatch: the local MindSpeed-LLM
code expected `mindspeed.core.optimizer.fix_duplicate_allgather`, which was
missing from one MindSpeed checkout. A paired MindSpeed checkout on the server
contained the module, so the adapter uses that matched MindSpeed code path for
the smoke run.

The container-only dependency additions are ordinary Python packages needed by
MindSpeed-LLM import paths. They were installed inside `llin-autoresearch`; the
physical machine and other containers were not changed.

The server had an existing preprocessed deepscaler dataset whose `labels`
records were shorter than `input_ids`, for example `input_ids` length 170 and
`labels` length 7. That format reaches the training forward pass but fails at
token-level cross entropy. The working path is to rebuild the data with
`AlpacaStyleInstructionHandler`, which produces equal-length `input_ids`,
`attention_mask`, and `labels`.

## MindSpeed-LLM Smoke Result

Run settings:

- checkpoint: `/models/Qwen3-0p6`
- tokenizer: `/models/Qwen3-0.6B`
- data: 512 deepscaler records converted to Alpaca-style SFT
- NPU scope: one visible NPU in `llin-autoresearch`
- steps: 3
- global batch size: 1
- sequence length argument: 2048, with no-pad instruction dataset mode

Observed output:

| Iteration | Step Time | Train Loss | Grad Norm |
| ---: | ---: | ---: | ---: |
| 1 | 1729.6 ms | 1.028000 | 27.809 |
| 2 | 249.9 ms | 0.977111 | 33.739 |
| 3 | 238.9 ms | 0.632855 | 17.502 |

Memory reported after the first iteration:

- allocated: about 10232.9 MB
- max allocated: about 10248.9 MB
- reserved: about 10308 MB

Interpretation: MindSpeed-LLM now clears the practical "can run" bar. The
efficiency signal is mixed but promising: it uses more memory than the thin HF
loop because it builds full Megatron optimizer/checkpoint machinery, but the
steady step time is low after startup. The effect signal is not yet comparable
to the HF validation loss because this run logs training loss only.

## 5-Minute Budget

Karpathy's original 5-minute loop is still valuable as the outer research
discipline: every attempt should have a tight time cap, a single measurable
result, and a keep/revert decision.

For Ascend/MindSpeed, the first run may need a separate smoke budget because
framework startup, extension loading, and dataset checks can dominate the first
few minutes. After a stable launch path exists, the comparable budget should be:

- 60 seconds for environment smoke,
- 300 seconds for autoresearch candidate evaluation,
- same fixed validation data and metric per framework when possible.
