source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data1/qvenv/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0
cd /data1
echo "BF16DRAFT_QUANT_START $(date)"
msmodelslim quant \
  --model_path /data1/DeepSeek-V4-Flash-DSpark \
  --save_path /data1/DeepSeek-V4-Flash-DSpark-bf16draft \
  --model_type DeepSeek-V4-Flash-DSpark \
  --config_path /data1/dspark_msslim/deepseek_v4_flash_dspark_bf16draft.yaml \
  --trust_remote_code True
echo "BF16DRAFT_QUANT_END rc=$? $(date)"
