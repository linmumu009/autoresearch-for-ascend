#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

MS_LLM_DIR="${MS_LLM_DIR:-/workspace/reference/MindSpeed-LLM}"
MINDSPEED_DIR="${MINDSPEED_DIR:-/workspace/reference/MindSpeed-rjx}"

CKPT_LOAD_DIR="${CKPT_LOAD_DIR:-/models/Qwen3-0p6}"
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/workspace/outputs/mindspeed_qwen3_0p6_smoke_ckpt}"
DATA_PATH="${DATA_PATH:-/workspace/datasets/deepscaler_alpaca/processed/deepscaler}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/models/Qwen3-0.6B}"

NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6027}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

TP="${TP:-1}"
PP="${PP:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
TRAIN_ITERS="${TRAIN_ITERS:-3}"
DATA_SPLIT="${DATA_SPLIT:-100,0,0}"
EVAL_INTERVAL="${EVAL_INTERVAL:-${TRAIN_ITERS}}"
EVAL_ITERS="${EVAL_ITERS:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"

mkdir -p "${CKPT_SAVE_DIR}/logs"

export PYTHONPATH="${MINDSPEED_DIR}:${MS_LLM_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${DATA_PATH}_packed_input_ids_document.idx" ]]; then
  echo "Missing MindSpeed dataset at ${DATA_PATH}_packed_*."
  echo "Run prepare_deepscaler_alpaca.sh first, or set DATA_PATH to a prepared prefix."
  exit 2
fi

DISTRIBUTED_ARGS=(
  --nproc_per_node "${NPUS_PER_NODE}"
  --nnodes "${NNODES}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
)

GPT_ARGS=(
  --use-mcore-models
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --sequence-parallel
  --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
  --kv-channels 128
  --qk-layernorm
  --use-flash-attn
  --num-layers 28
  --hidden-size 1024
  --use-rotary-position-embeddings
  --num-attention-heads 16
  --ffn-hidden-size 3072
  --max-position-embeddings 32768
  --seq-length "${SEQ_LENGTH}"
  --train-iters "${TRAIN_ITERS}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --make-vocab-size-divisible-by 1
  --padded-vocab-size 151936
  --rotary-base 1000000
  --disable-bias-linear
  --swiglu
  --tokenizer-type PretrainedFromHF
  --tokenizer-name-or-path "${TOKENIZER_PATH}"
  --normalization RMSNorm
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --hidden-dropout 0
  --attention-dropout 0
  --no-gradient-accumulation-fusion
  --attention-softmax-in-fp32
  --exit-on-missing-checkpoint
  --no-masked-softmax-fusion
  --group-query-attention
  --num-query-groups 8
  --seed 42
  --bf16
  --min-lr 1.25e-7
  --weight-decay 1e-1
  --lr-warmup-fraction 0.01
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --no-load-optim
  --no-load-rng
  --lr 1.25e-6
)

DATA_ARGS=(
  --data-path "${DATA_PATH}"
  --split "${DATA_SPLIT}"
)

OUTPUT_ARGS=(
  --log-interval 1
  --save-interval "${SAVE_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --eval-iters "${EVAL_ITERS}"
  --log-throughput
)

TUNE_ARGS=(
  --finetune
  --stage sft
  --is-instruction-dataset
  --prompt-type qwen3
  --no-pad-to-seq-lengths
)

cd "${MS_LLM_DIR}"

torchrun "${DISTRIBUTED_ARGS[@]}" posttrain_gpt.py \
  "${GPT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "${TUNE_ARGS[@]}" \
  --distributed-backend nccl \
  --load "${CKPT_LOAD_DIR}" \
  --save "${CKPT_SAVE_DIR}" \
  2>&1 | tee "${CKPT_SAVE_DIR}/logs/smoke_qwen3_0p6_full_sft.log"
