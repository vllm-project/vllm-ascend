set -e
exec > /data1/dspark_msslim/_setup.log 2>&1
echo "=== [1] venv (继承 system torch_npu) ==="
python -m venv --system-site-packages /data1/qvenv
source /data1/qvenv/bin/activate
echo "py: $(which python)"
echo "=== [2] transformers 4.48.2 (shadow system 5.5.4) ==="
pip install -q transformers==4.48.2
echo "=== [3] 放 DSpark adapter 文件 ==="
MS=/data1/msmodelslim; DV=$MS/msmodelslim/model/deepseek_v4
cp /data1/dspark_msslim/msslim_dspark_model.py   $DV/dspark_model.py
cp /data1/dspark_msslim/msslim_dspark_adapter.py $DV/dspark_adapter.py
cp /data1/dspark_msslim/msslim_dspark_loader.py  $DV/dspark_loader.py
cp /data1/dspark_msslim/deepseek_v4_flash_dspark_w8a8.yaml $MS/lab_practice/deepseek_v4/
echo "=== [4] patch config.ini (加 deepseek_v4_dspark group) ==="
python - <<PY
ini="/data1/msmodelslim/config/config.ini"
s=open(ini).read()
edits=[
 ("deepseek_v4 = DeepSeek-V4-Flash, DeepSeek-V4-Pro","\ndeepseek_v4_dspark = DeepSeek-V4-Flash-DSpark"),
 ("deepseek_v4 = msmodelslim.model.deepseek_v4.loader:DeepseekV4AdapterLoader","\ndeepseek_v4_dspark = msmodelslim.model.deepseek_v4.dspark_loader:DeepseekV4DSparkAdapterLoader"),
 ('deepseek_v4 = {"transformers": "==4.48.2"}','\ndeepseek_v4_dspark = {"transformers": "==4.48.2"}'),
]
for a,add in edits:
    if add.strip() in s: continue
    assert a in s, "anchor: "+a[:40]
    s=s.replace(a,a+add,1)
open(ini,"w").write(s); print("config.ini patched")
PY
echo "=== [5] 装 msModelSlim (注册 entry points 到 venv) ==="
cd $MS && bash install.sh
echo "=== DONE ==="
