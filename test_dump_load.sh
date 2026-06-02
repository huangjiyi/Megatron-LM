#!/usr/bin/env bash
set -euo pipefail

cd /root/paddlejob/share-storage/gpfs/system-public/xuxinyi/ds_new/Megatron-LM

DUMP_DIR="outputs/dump_data"
rm -rf "${DUMP_DIR}"

echo "========== Step 1: Run with DUMP_DATA_PATH to dump data =========="
DUMP_DATA_PATH="${DUMP_DIR}" bash align.sh > ml_dump.log 2>&1 || true
echo "Dump run finished. Log: ml_dump.log"

echo "========== Step 2: Run with LOAD_FIXED_DATA_PATH to load dumped data =========="
sleep 5
LOAD_FIXED_DATA_PATH="${DUMP_DIR}" bash align.sh > ml_load.log 2>&1 || true
echo "Load run finished. Log: ml_load.log"

echo "========== Step 3: Compare first loss =========="
LOSS_DUMP=$(grep -oP "lm loss: \K[0-9.eE+-]+" ml_dump.log | head -1)
LOSS_LOAD=$(grep -oP "lm loss: \K[0-9.eE+-]+" ml_load.log | head -1)

echo "First loss (dump run): ${LOSS_DUMP}"
echo "First loss (load run): ${LOSS_LOAD}"

if [[ "${LOSS_DUMP}" == "${LOSS_LOAD}" ]]; then
    echo "PASS: Loss values are identical."
else
    echo "DIFF: Loss values differ."
fi
