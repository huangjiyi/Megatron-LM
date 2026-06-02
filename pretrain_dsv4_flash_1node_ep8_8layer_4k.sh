#!/usr/bin/env bash

set -euo pipefail

source /root/paddlejob/share-storage/gpfs/system-public/huangjiyi/dsv4-flash-workspace/Megatron-LM-CleanAlign/.venv/bin/activate
unset PYTHONPATH

MASTER_PORT="6091"
TRAIN_ITERS="${TRAIN_ITERS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

DATA_PATH="/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/data/pre-training/deepseek_v4_flash_train_text_document"
TOKENIZER_MODEL="/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/Models/DeepSeek-V4-Flash"
LOAD_CHECKPOINT="/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/dsv4-flash-workspace/alignment_data/dsv4_flash_hf_megatron_ckpt_qkln_pp1ep8_8layer_official_clean"

OUTPUT_DIR="outputs"
DATA_CACHE_PATH="${OUTPUT_DIR}/data-cache"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/tensorboard" "${DATA_CACHE_PATH}" "${LOG_DIR}"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVTE_CPU_OFFLOAD_V1=0
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#export DSV4_MEGATRON_FIXED_TOKENS="${SCRIPT_DIR}/alignment_data/real_tokens_seq4097.json"
export LOG_DATA_MD5="${LOG_DATA_MD5:-1}"
export LOG_LOSS_MD5="${LOG_LOSS_MD5:-1}"
export DSV4_DISABLE_MEGATRON_JIT_FUSER="${DSV4_DISABLE_MEGATRON_JIT_FUSER:-1}"
export FLAGS_use_accuracy_compatible_kernel="${FLAGS_use_accuracy_compatible_kernel:-1}"

mkdir -p "${OUTPUT_DIR}/tensorboard" "${DATA_CACHE_PATH}" "${LOG_DIR}"

# 单机 EP8 缩层拓扑：所有模块都在同一个 PP stage。
PARALLEL_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --pipeline-model-parallel-layout "Et*8mL"
)

if [[ "${USE_DISTRIBUTED_OPTIMIZER:-1}" == "1" ]]; then
    PARALLEL_ARGS+=(
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
    )
fi

# 与 DSv4-Flash 全量形状保持一致，只缩 decoder layer 数。
MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 8
    --hidden-size 4096
    --ffn-hidden-size 2048
    --num-attention-heads 64
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --disable-bias-linear
    --swiglu
    --activation-func-clamp-value 10.0
    --position-embedding-type rope
    --no-position-embedding
    --rotary-base 10000
    --rope-type yarn
    --rotary-scaling-factor 16
    --original-max-position-embeddings 65536
    --no-rope-fusion
    --seq-length 4096
    --max-position-embeddings 4096
    --untie-embeddings-and-output-weights
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend unfused
    --qk-layernorm
)

DSV4_ATTENTION_ARGS=(
    --multi-latent-attention
    --q-lora-rank 1024
    --v-head-dim 512
    --qk-pos-emb-head-dim 64
    --o-groups 8
    --o-lora-rank 1024
    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios "([0,0]+[4,128]*3+[0])"
    --csa-compress-rotary-base 160000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff "${DSA_INDEXER_LOSS_COEFF:-0.01}"
    --dsa-indexer-use-sparse-loss
)

MHC_MTP_ARGS=(
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --mtp-num-layers 1
    --mtp-loss-scaling-factor "${MTP_LOSS_SCALING_FACTOR:-0.1}"
)

MOE_ARGS=(
    --num-experts 256
    --moe-layer-freq 1
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-topk 6
    --moe-router-load-balancing-type none
    --moe-aux-loss-coeff 0.0
    --moe-router-dtype fp32
    --moe-router-score-function sqrtsoftplus
    --moe-router-topk-scaling-factor 1.5
    --moe-router-enable-expert-bias
    --moe-n-hash-layers 3
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
)

if [[ "${MOE_PERMUTE_FUSION:-1}" == "1" ]]; then
    MOE_ARGS+=(--moe-permute-fusion)
fi

DATA_ARGS=(
    --data-path "${DATA_PATH}"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${TOKENIZER_MODEL}"
    --trust-remote-code
    --make-vocab-size-divisible-by 128
    --data-cache-path "${DATA_CACHE_PATH}"
    --split 949,50,1
    --no-mmap-bin-files
    --no-create-attention-mask-in-dataloader
    --num-workers 0
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr-warmup-iters 0
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.999
    --weight-decay 0.1
    --clip-grad "${CLIP_GRAD:-1.0}"
    --seed 1234
    --bf16
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --empty-unused-memory-level 2
)

MEMORY_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act layernorm mla_up_proj mlp shared_experts
)

LOAD_ARGS=(
    --load "${LOAD_CHECKPOINT}"
    --ckpt-format torch_dist
    --finetune
    --no-load-optim
    --no-load-rng
)

LOGGING_ARGS=(
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --log-throughput
    --log-memory-to-tensorboard
    --log-timers-to-tensorboard
    --tensorboard-dir "${OUTPUT_DIR}/tensorboard"
)

CMD=(
    python -m torch.distributed.run
    --nnodes 1
    --nproc-per-node 8
    --node-rank 0
    --master-addr localhost
    --master-port "${MASTER_PORT}"
    pretrain_gpt.py
    "${PARALLEL_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${DSV4_ATTENTION_ARGS[@]}"
    "${MHC_MTP_ARGS[@]}"
    "${MOE_ARGS[@]}"
    "${DATA_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${MEMORY_ARGS[@]}"
    "${LOAD_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
)


TIME="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="${LOG_DIR}/${TIME}_pretraining_dsv4_flash_full_4nodes_4k.log"

{
    echo "DSv4-Flash 1-node 4K pretrain: TP=1 PP=1 EP=8 CP=1 layout=Et*8mL"
    echo "Train: TRAIN_ITERS=${TRAIN_ITERS} MBS=1 GBS=${GLOBAL_BATCH_SIZE} SEQ_LENGTH=4096"
    echo "Warm-start checkpoint: ${LOAD_CHECKPOINT}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Log file: ${LOG_FILE}"
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
} | tee -a "${LOG_FILE}"

"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
