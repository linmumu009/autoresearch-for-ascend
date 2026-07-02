#!/usr/bin/env bash
set -euo pipefail

MS_LLM_DIR="${MS_LLM_DIR:-/workspace/reference/MindSpeed-LLM}"
MINDSPEED_DIR="${MINDSPEED_DIR:-/workspace/reference/MindSpeed-rjx}"

LOAD_DIR="${LOAD_DIR:-/workspace/outputs/mindspeed_qwen3_0p6_deepscaler_eval_smoke_ckpt}"
SAVE_DIR="${SAVE_DIR:-/workspace/outputs/mindspeed_qwen3_0p6_deepscaler_eval_smoke_hf}"
HF_CFG_DIR="${HF_CFG_DIR:-/models/Qwen3-0.6B}"
MODEL_TYPE_HF="${MODEL_TYPE_HF:-qwen3}"

export PYTHONPATH="${MINDSPEED_DIR}:${MS_LLM_DIR}:${PYTHONPATH:-}"

cd "${MS_LLM_DIR}"

python convert_ckpt_v2.py \
  --load-model-type mg \
  --save-model-type hf \
  --load-dir "${LOAD_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --hf-cfg-dir "${HF_CFG_DIR}" \
  --model-type-hf "${MODEL_TYPE_HF}"

for name in \
  config.json \
  configuration.json \
  generation_config.json \
  tokenizer.json \
  tokenizer_config.json \
  merges.txt \
  vocab.json \
  LICENSE \
  README.md; do
  if [[ -f "${HF_CFG_DIR}/${name}" && ! -f "${SAVE_DIR}/${name}" ]]; then
    cp "${HF_CFG_DIR}/${name}" "${SAVE_DIR}/${name}"
  fi
done

find "${SAVE_DIR}" -maxdepth 1 -type f | sort
