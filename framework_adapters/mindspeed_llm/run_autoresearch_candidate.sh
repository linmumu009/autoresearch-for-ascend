#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASCEND_RUNTIME_LD="${ASCEND_RUNTIME_LD:-/usr/local/Ascend/cann-9.0.0/aarch64-linux/lib64:/usr/local/Ascend/cann-9.0.0/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver}"
ASCEND_AARCH64_LIB="${ASCEND_RUNTIME_LD%%:*}"
if [[ -d "${ASCEND_AARCH64_LIB}" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_RUNTIME_LD}"
fi

if [[ -n "${CANDIDATE_ENV:-}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${CANDIDATE_ENV}"
  set +a
fi

RUN_NAME="${RUN_NAME:-mindspeed_qwen3_0p6_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/workspace/runs/mindspeed_llm}"
OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-${OUT_DIR}/mcore}"
HF_DIR="${HF_DIR:-${OUT_DIR}/hf}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
RESULTS_TSV="${RESULTS_TSV:-${RUN_ROOT}/results.tsv}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/models/Qwen3-0.6B}"
HF_EVAL_DIR="${HF_EVAL_DIR:-/workspace/ascend_autoresearch}"
EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_BASE="${EVAL_BASE:-0}"

mkdir -p "${CKPT_DIR}" "${HF_DIR}" "${LOG_DIR}" "$(dirname "${RESULTS_TSV}")"

TRAIN_LOG="${LOG_DIR}/train.log"
CONVERT_LOG="${LOG_DIR}/convert.log"
EVAL_LOG="${LOG_DIR}/eval_hf.log"
BASE_EVAL_LOG="${LOG_DIR}/eval_base_hf.log"
ENV_LOG="${LOG_DIR}/candidate.env"

{
  echo "RUN_NAME=${RUN_NAME}"
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "HF_DIR=${HF_DIR}"
  echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
  echo "DATA_PATH=${DATA_PATH:-/workspace/datasets/deepscaler_alpaca/processed/deepscaler}"
  echo "TRAIN_ITERS=${TRAIN_ITERS:-6}"
  echo "DATA_SPLIT=${DATA_SPLIT:-90,10,0}"
  echo "EVAL_INTERVAL=${EVAL_INTERVAL:-3}"
  echo "EVAL_ITERS=${EVAL_ITERS:-2}"
  echo "SEQ_LENGTH=${SEQ_LENGTH:-2048}"
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}"
  echo "MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}"
  echo "LR=${LR:-1.25e-6}"
  echo "EVAL_SEQ_LEN=${EVAL_SEQ_LEN}"
  echo "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}"
} > "${ENV_LOG}"

echo "[1/4] train MindSpeed candidate: ${RUN_NAME}"
CKPT_SAVE_DIR="${CKPT_DIR}" \
TRAIN_ITERS="${TRAIN_ITERS:-6}" \
DATA_SPLIT="${DATA_SPLIT:-90,10,0}" \
EVAL_INTERVAL="${EVAL_INTERVAL:-3}" \
EVAL_ITERS="${EVAL_ITERS:-2}" \
SEQ_LENGTH="${SEQ_LENGTH:-2048}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}" \
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
DATA_PATH="${DATA_PATH:-/workspace/datasets/deepscaler_alpaca/processed/deepscaler}" \
MASTER_PORT="${MASTER_PORT:-6041}" \
LR="${LR:-1.25e-6}" \
bash "${SCRIPT_DIR}/smoke_qwen3_0p6_full_sft.sh" 2>&1 | tee "${TRAIN_LOG}"

echo "[2/4] convert MindSpeed checkpoint to Hugging Face"
LOAD_DIR="${CKPT_DIR}" \
SAVE_DIR="${HF_DIR}" \
HF_CFG_DIR="${BASE_MODEL_PATH}" \
bash "${SCRIPT_DIR}/convert_mcore_to_hf.sh" 2>&1 | tee "${CONVERT_LOG}"

echo "[3/4] evaluate converted checkpoint on fixed HF validation"
cd "${HF_EVAL_DIR}"
python evaluate_hf.py \
  --model-path "${HF_DIR}" \
  --seq-len "${EVAL_SEQ_LEN}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  2>&1 | tee "${EVAL_LOG}"

if [[ "${EVAL_BASE}" == "1" ]]; then
  echo "[base] evaluate base checkpoint on fixed HF validation"
  python evaluate_hf.py \
    --model-path "${BASE_MODEL_PATH}" \
    --seq-len "${EVAL_SEQ_LEN}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    2>&1 | tee "${BASE_EVAL_LOG}"
fi

echo "[4/4] record metrics"
RAW_VAL_LOSS="$(awk '/^val_loss:/ {print $2}' "${EVAL_LOG}" | tail -1)"
RAW_VAL_PPL="$(awk '/^val_ppl:/ {print $2}' "${EVAL_LOG}" | tail -1)"
TOTAL_SECONDS="$(awk '/^total_seconds:/ {print $2}' "${EVAL_LOG}" | tail -1)"
PEAK_MEM_MB="$(awk '/^peak_npu_mem_mb:/ {print $2}' "${EVAL_LOG}" | tail -1)"
MS_VALID_LOSS="$(awk '/validation loss at iteration/ && /lm loss value:/ {for (i=1; i<=NF; i++) if ($i == "value:") print $(i+1)}' "${TRAIN_LOG}" | tail -1)"
LAST_TRAIN_LOSS="$(awk '/iteration/ && /lm loss:/ {for (i=1; i<=NF; i++) if ($i == "loss:") print $(i+1)}' "${TRAIN_LOG}" | tail -1)"

if [[ ! -f "${RESULTS_TSV}" ]]; then
  printf "run_name\traw_hf_val_loss\traw_hf_val_ppl\tmindspeed_valid_loss\tlast_train_loss\teval_seconds\tpeak_mem_mb\tckpt_dir\thf_dir\n" > "${RESULTS_TSV}"
fi

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${RUN_NAME}" \
  "${RAW_VAL_LOSS:-NA}" \
  "${RAW_VAL_PPL:-NA}" \
  "${MS_VALID_LOSS:-NA}" \
  "${LAST_TRAIN_LOSS:-NA}" \
  "${TOTAL_SECONDS:-NA}" \
  "${PEAK_MEM_MB:-NA}" \
  "${CKPT_DIR}" \
  "${HF_DIR}" >> "${RESULTS_TSV}"

echo "---"
echo "run_name:              ${RUN_NAME}"
echo "raw_hf_val_loss:       ${RAW_VAL_LOSS:-NA}"
echo "raw_hf_val_ppl:        ${RAW_VAL_PPL:-NA}"
echo "mindspeed_valid_loss:  ${MS_VALID_LOSS:-NA}"
echo "last_train_loss:       ${LAST_TRAIN_LOSS:-NA}"
echo "results_tsv:           ${RESULTS_TSV}"
