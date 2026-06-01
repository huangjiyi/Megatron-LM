export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

FLAGS_use_accuracy_compatible_kernel=1 \
MOE_PERMUTE_FUSION=0 \
RUN_ID=cleanalign_minpatch_ep8_8layer_fixed_loss_1step \
TRAIN_ITERS=1 \
GLOBAL_BATCH_SIZE=8 \
bash pretrain_dsv4_flash_1node_ep8_8layer_4k.sh
