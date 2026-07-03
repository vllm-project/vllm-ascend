#!/bin/bash
# Deploy the DeepSeek-V4-Flash-DSpark quant adapter into msModelSlim + run w8a8 quant.
# Run on m18 host. Prereqs: full ckpt downloaded (/data1/DeepSeek-V4-Flash-DSpark),
# 1 free NPU card, the 4 staged files in /data1/dspark_msslim/.
set -e
MS=/data1/msmodelslim
DV=$MS/msmodelslim/model/deepseek_v4
STAGE=/data1/dspark_msslim          # where the 4 component files were pushed

# 1) place the adapter files (filenames matter -> match the loader/adapter import paths)
cp $STAGE/msslim_dspark_model.py   $DV/dspark_model.py
cp $STAGE/msslim_dspark_adapter.py $DV/dspark_adapter.py
cp $STAGE/msslim_dspark_loader.py  $DV/dspark_loader.py
cp $STAGE/deepseek_v4_flash_dspark_w8a8.yaml $MS/lab_practice/deepseek_v4/

# 2) register a new "deepseek_v4_dspark" plugin group in config.ini (idempotent)
python - <<'PY'
ini="/data1/msmodelslim/config/config.ini"
s=open(ini).read()
edits=[
 ("deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro",
  "\ndeepseek_v4_dspark = DeepSeek-V4-Flash-DSpark"),
 ("deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader",
  "\ndeepseek_v4_dspark = msmodelslim.model.deepseek_v4.dspark_loader:DeepseekV4DSparkAdapterLoader"),
 ('deepseek_v4 = {"transformers": "==4.48.2"}',
  '\ndeepseek_v4_dspark = {"transformers": "==4.48.2"}'),
]
for anchor, add in edits:
    if add.strip() in s:
        continue
    assert anchor in s, f"anchor not found: {anchor}"
    s=s.replace(anchor, anchor+add, 1)
open(ini,"w").write(s)
print("config.ini patched")
PY

# 3) (re)install so entry points pick up the new model_type
cd $MS && bash install.sh
pip install transformers==4.48.2

# 4) run the quant (Flash = single card). Input fp4 native OR pre-dequant bf16.
export ASCEND_RT_VISIBLE_DEVICES=0
cd /data1
msmodelslim quant \
  --model_path /data1/DeepSeek-V4-Flash-DSpark \
  --save_path  /data1/DeepSeek-V4-Flash-DSpark-w8a8 \
  --model_type DeepSeek-V4-Flash-DSpark \
  --config_path lab_practice/deepseek_v4/deepseek_v4_flash_dspark_w8a8.yaml \
  --trust_remote_code True
