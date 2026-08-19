# 服务拉起脚本

## 环境变量设置

```shell
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=512
export NPU_MEMORY_FRACTION=0.95
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export OMP_PROC_BIND=false
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
export VLLM_ASCEND_ENABLE_NZ=2
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
```

## 服务拉起命令

- 图模式

```shell
# 图模式
vllm serve /home/twx1146604/Qwen3-30B-A3B                          --served-model-name Qwen3_30B_A3B                          --async-scheduling                          --tensor-parallel-size 4                          --port 8000                          --max-num-seqs 16                          --max-model-len 16384                          --max-num-batched-tokens 16384                          --gpu-memory-utilization 0.9                          --trust-remote-code                          --additional-config '{"enable_cpu_binding":true}'                          --compilation-config '{"cudagraph_mode": "FULL","cudagraph_capture_sizes":[1,2,4,8,16,32,48,64,96,128,160,192,256,512,1024,2048,4096,8192]}'
```

- eager 模式

```shell
# eager 模式
vllm serve /home/twx1146604/Qwen3-30B-A3B                          --served-model-name Qwen3_30B_A3B                          --async-scheduling                          --tensor-parallel-size 4                          --port 8000                          --max-num-seqs 16                          --max-model-len 16384                          --max-num-batched-tokens 16384                          --gpu-memory-utilization 0.9                          --trust-remote-code                          --additional-config '{"enable_cpu_binding":true}'                          --enforce-eager
```