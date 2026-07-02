#!/usr/bin/env bash
set -euo pipefail

MS_LLM_DIR="${MS_LLM_DIR:-/workspace/reference/MindSpeed-LLM}"
MINDSPEED_DIR="${MINDSPEED_DIR:-/workspace/reference/MindSpeed-rjx}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/models/Qwen3-0.6B}"

RAW_INPUT="${RAW_INPUT:-/workspace/inputs/raw/deepscaler.json}"
MAX_SAMPLES="${MAX_SAMPLES:-512}"
OUT_ROOT="${OUT_ROOT:-/workspace/datasets/deepscaler_alpaca}"
ALPACA_JSON="${ALPACA_JSON:-${OUT_ROOT}/raw/train.json}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-${OUT_ROOT}/processed/deepscaler}"

mkdir -p "$(dirname "${ALPACA_JSON}")" "$(dirname "${OUTPUT_PREFIX}")"

export RAW_INPUT MAX_SAMPLES ALPACA_JSON
python - <<'PY'
import json
import os

raw_input = os.environ["RAW_INPUT"]
max_samples = int(os.environ["MAX_SAMPLES"])
alpaca_json = os.environ["ALPACA_JSON"]

with open(raw_input, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for record in data[:max_samples]:
    solution = record.get("solution") or ""
    answer = record.get("answer") or ""
    output = solution.strip()
    if answer:
        output = (output + "\n\nAnswer:\n" + answer).strip()
    rows.append(
        {
            "instruction": record.get("problem", ""),
            "input": "",
            "output": output,
        }
    )

with open(alpaca_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)

print(f"wrote {len(rows)} records to {alpaca_json}")
PY

export PYTHONPATH="${MINDSPEED_DIR}:${MS_LLM_DIR}:${PYTHONPATH:-}"
export OUTPUT_PREFIX

cd "${MS_LLM_DIR}"

python preprocess_data.py \
  --input "${ALPACA_JSON}" \
  --tokenizer-name-or-path "${TOKENIZER_PATH}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --handler-name AlpacaStyleInstructionHandler \
  --tokenizer-type PretrainedFromHF \
  --workers 1 \
  --log-interval 100 \
  --enable-thinking true \
  --prompt-type qwen3

python - <<'PY'
import os
from megatron.core.datasets.indexed_dataset import IndexedDataset

prefix = os.environ["OUTPUT_PREFIX"]
lengths = {}
for field in ["input_ids", "attention_mask", "labels"]:
    dataset = IndexedDataset(f"{prefix}_packed_{field}_document", mmap=True)
    lengths[field] = len(dataset[0])
    print(field, "records", len(dataset), "first_len", lengths[field])

if len(set(lengths.values())) != 1:
    raise SystemExit(f"field length mismatch: {lengths}")
PY
