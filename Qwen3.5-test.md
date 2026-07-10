这是我的部署脚本
#!/bin/sh

export VLLM_USE_MODELSCOPE=True

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1

vllm serve /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/Qwen3.5-27B-w8a8-mtp \
--host 0.0.0.0 \
--port 8000 \
--data-parallel-size 1 \
--tensor-parallel-size 2 \
--seed 1024 \
--quantization ascend \
--served-model-name qwen3.5 \
--max-num-seqs 32 \
--max-model-len 133000 \
--max-num-batched-tokens 8096 \
--trust-remote-code \
--gpu-memory-utilization 0.90 \
--no-enable-prefix-caching \
--speculative_config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3, "enforce_eager": true}' \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
--additional-config '{"enable_cpu_binding":true}' \
--async-scheduling
<img width="1363" height="516" alt="image" src="https://github.com/user-attachments/assets/32751cfa-f784-4504-996c-e036b21e56e4" />
<img width="1441" height="578" alt="image" src="https://github.com/user-attachments/assets/b540ef19-0513-4147-a0df-ef602343af92" />
