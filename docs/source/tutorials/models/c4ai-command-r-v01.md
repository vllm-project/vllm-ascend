# CohereLabs/c4ai-command-r-v01
This tutorial shows how to run `CohereLabs/c4ai-command-r-v01` with vLLM-Ascend on Atlas A2.
## Prerequisites
- Ascend driver/toolkit is installed and healthy.
- vLLM-Ascend environment is ready.
- Model files are available (local path or Hugging Face access).
## Environment Setup
```bash
cd /data/vllm-ascend
PY=/usr/local/python3.11.13/bin/python
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnrt/set_env.sh 2>/dev/null || true
unset ASCEND_DEVICE_ID DEVICE_ID ASCEND_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export HCCL_INTRA_ROCE_ENABLE=1
```

## Launch Server (TP=4)
```bash
$PY -m vllm.entrypoints.openai.api_server \
  --model /data/models/c4ai-command-r-v01 \
  --served-model-name c4ai-command-r-v01 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 8192
```

## Quick Verification
Open another terminal and run:
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"c4ai-command-r-v01",
    "messages":[
      {"role":"system","content":"Please answer in simplified Chinese only."},
      {"role":"user","content":"你好，请回复：测试成功"}
    ],
    "temperature":0.2
  }'
```
Expected result:
- Response contains `choices`.
- `finish_reason` is `stop`.
## Troubleshooting
### HCCL init failure (EI0010 / error code 5)
Typical symptom:
- `hcclCommInitRootInfoConfig ... error code is 5`
- `P2P_Communication_Failed(EI0010)`
Checklist:
- Ensure `ASCEND_RT_VISIBLE_DEVICES` matches `--tensor-parallel-size`.
- Keep device mapping consistent across all worker processes.
- Set `HCCL_INTRA_ROCE_ENABLE=1`.
- If needed, try single-card first (`--tensor-parallel-size 1`) to verify baseline startup.
