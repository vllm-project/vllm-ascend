## 启动脚本
```
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

vllm serve \
    --model /workspace/shared_assets/GeekCamp/Infer/Model/zouchangjiang/MiniMax-M2.5-w8a8-QuaRot \
    --served-model-name MiniMax-M2.5 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --quantization ascend \
    --enable-expert-parallel \
    --max-num-seqs 32 \
    --seed 1024 \
    --max-num-batched-tokens 32768 \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --gpu-memory-utilization 0.95 \
    --additional-config '{"enable_cpu_binding":true,
                          "enable_flashcomm1":true}' \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":16}'
```
## 启动成功截图
<img width="1533" height="1671" alt="745648f71876139eb8ae5cddba81895b" src="https://github.com/user-attachments/assets/90431c71-1002-42f0-995e-6663bcc257e3" />

## 推理测试脚本
```
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "stream": false,
    "temperature": 0.8,
    "max_tokens": 200
  }'
```
<img width="1549" height="639" alt="90a99753251a05bd473102225188f88c" src="https://github.com/user-attachments/assets/d3549718-0382-4541-ada7-7c8c5a4e59d3" />
