#!/usr/bin/env bash

set -euo pipefail

source .venv/bin/activate
unset PYTHONPATH

NNODES="${LSHRUN_NNODES:-4}"
NODE_RANK="${LSHRUN_RANK:-0}"
MASTER_ADDR="${LSHRUN_MASTER:-localhost}"
MASTER_PORT="6080"

TRAIN_ITERS="${TRAIN_ITERS:-10}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"

DATA_PATH="./data/dsv4_flash_megatron_train_text_document"
TOKENIZER_MODEL="/root/paddlejob/share-storage/gpfs/system-public/huangjiyi/Models/DeepSeek-V4-Flash"
LOAD_CHECKPOINT="./ckpts/dsv4_flash_hf2megatron_pp4ep8_ckpt"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NVTE_CPU_OFFLOAD_V1=0
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"

OUTPUT_DIR="outputs"
DATA_CACHE_PATH="${OUTPUT_DIR}/data-cache"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/tensorboard" "${DATA_CACHE_PATH}" "${LOG_DIR}"

# Distributed topology and pipeline layout.
PARALLEL_ARGS=(
    --distributed-timeout-minutes 60
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    --pipeline-model-parallel-layout "Et*11|t*11|t*11|t*10mL"
)

# Dense model shape and transformer defaults.
MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --num-layers 43
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

# DSv4 hybrid MLA/CSA/DSA attention settings.
DSV4_ATTENTION_ARGS=(
    --multi-latent-attention
    --q-lora-rank 1024
    --v-head-dim 512
    --qk-pos-emb-head-dim 64
    --o-groups 8
    --o-lora-rank 1024
    --experimental-attention-variant dsv4_hybrid
    --csa-window-size 128
    --csa-compress-ratios "([0,0]+[4,128]*20+[4,0])"
    --csa-compress-rotary-base 160000
    --dsa-indexer-n-heads 64
    --dsa-indexer-head-dim 128
    --dsa-indexer-topk 512
    --dsa-indexer-loss-coeff 0.01
    --dsa-indexer-use-sparse-loss
)

# HyperConnections and one MTP layer.
MHC_MTP_ARGS=(
    --enable-hyper-connections
    --num-residual-streams 4
    --mhc-sinkhorn-iterations 20
    --mtp-num-layers 1
    --mtp-loss-scaling-factor 0.1
)

# DSv4 MoE routing and expert execution.
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
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --moe-grouped-gemm
    --moe-permute-fusion
)

# Real pretraining data and DeepSeek tokenizer.
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

# 4K short convergence run settings.
TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --train-iters "${TRAIN_ITERS}"
    --lr-decay-iters "${TRAIN_ITERS}"
    --lr 1.0e-4
    --min-lr 1.0e-5
    --lr-decay-style cosine
    --weight-decay 0.1
    --clip-grad 1.0
    --bf16
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --cross-entropy-loss-fusion
    --cross-entropy-fusion-impl te
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 10
    --empty-unused-memory-level 2
)

# Memory profile from the successful 10-step run: recompute + precision-aware optimizer, no offload.
MEMORY_ARGS=(
    --recompute-granularity selective
    --recompute-modules moe_act layernorm mla_up_proj mlp shared_experts
    --fp8-format e4m3
    --fp8-recipe mxfp8
    --use-precision-aware-optimizer
    --main-grads-dtype fp32
    --main-params-dtype fp32
    --exp-avg-dtype bf16
    --exp-avg-sq-dtype bf16
)

# Model-only warm start from the qk-layernorm HF->Megatron checkpoint.
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


TIME="$(date +%Y-%m-%d_%H%M)"
LOG_FILE="${LOG_DIR}/${TIME}_pretraining_dsv4_flash_full_4nodes_4k.log"

python -m torch.distributed.run \
    --nnodes "${NNODES}" \
    --nproc-per-node 8 \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    pretrain_gpt.py \
    "${PARALLEL_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${DSV4_ATTENTION_ARGS[@]}" \
    "${MHC_MTP_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MEMORY_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee -a "${LOG_FILE}"
