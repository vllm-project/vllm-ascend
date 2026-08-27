#!/bin/bash
set +e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ltc
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /home/l00832868/codexWork/vllm-ascend || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1
timeout 300 mssanitizer --tool=memcheck python /tmp/test_gmm_rank.py 256 512 2048 \
  > /tmp/moe_gmm_memcheck.log 2>&1
rc=$?
echo "rc=$rc"
grep -E 'ERROR:|WARNING:|Illegal|Misaligned|MoeGroupedMatmul|fixp_error|507015' \
  /tmp/moe_gmm_memcheck.log | tail -160
