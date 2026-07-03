source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data1/qvenv/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0
cd /data1
msmodelslim quant \
  --model_path /data1/DeepSeek-V4-Flash-DSpark \
  --save_path  /data1/DeepSeek-V4-Flash-DSpark-w8a8 \
  --model_type DeepSeek-V4-Flash-DSpark \
  --config_path /data1/msmodelslim/lab_practice/deepseek_v4/deepseek_v4_flash_dspark_w8a8.yaml \
  --trust_remote_code True
