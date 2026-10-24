---
license: mit
---
# Qwen3.5-397B-A17B-W8A8基于vLLMAscend框架部署指导 -- A2/A3
Qwen3.5-397B-A17作为原生视觉-语言模型，在推理、编程、智能体能力与多模态理解等全方位基准评估中表现优异，助力开发者与企业显著提升生产力。该模型采用创新的混合架构，将线性注意力（Gated Delta Networks）与稀疏混合专家（MoE）相结合，实现出色的推理效率：总参数量达 3970 亿，每次前向传播仅激活 170 亿参数，在保持能力的同时优化速度与成本。并且将语言与方言支持从 119 种扩展至 201 种，为用户提供更广泛的可用性与更完善的支持。

## 一、环境准备

###  1、下载镜像

通过[镜像链接](https://modelers.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp/tree/main/vllm-image)下载对应服务器的镜像版本，下载后通过docker load解压镜像包，如：

```sh
docker load -i Vllm-ascend-Qwen3_5-A3-Ubuntu-v0.tar   # A3
docker load -i Vllm-ascend-Qwen3_5-A2-Ubuntu-v0.tar   # A2
```

### 2、环境信息

```sh
CANN                              8.5.0
vllm                              0.16.0rc2.dev55+g65bb4942b.empty /vllm-workspace/vllm
vllm_ascend                       0.14.0rc2.dev119+g52aa9c006      /vllm-workspace/vllm-ascend
torch                             2.9.0+cpu
torch_npu                         2.9.0
torchvision                       0.24.0
```

### 3、创建容器

#### A3 单机部署

```shell
#!/bin/sh
NAME=qwen3-5
PORT=10005
DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
IMAGE="vllm-ascend:qwen3_5-v0-a3"    # 加载镜像
docker run -itd -u 0  --ipc=host  --privileged \
-e VLLM_USE_MODELSCOPE=True -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-e  ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
--name $NAME \
--net=host \
--shm-size=100g \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home/:/home/ \
-v /opt/data/:/opt/data/ \
-v /mnt/weight/:/mnt/weight/ \
-v /root/.cache:/root/.cache \
-p $PORT:10005 \
-it $IMAGE bash
```

#### A2 双机部署

```
#!/bin/sh
NAME=qwen3-5
PORT=10005
DEVICES="0,1,2,3,4,5,6,7"
IMAGE="vllm-ascend:qwen3_5-v0-a2"    # 加载镜像
docker run -itd -u 0  --ipc=host  --privileged \
-e VLLM_USE_MODELSCOPE=True -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
-e  ASCEND_RT_VISIBLE_DEVICES=$DEVICES \
--name $NAME \
--net=host \
--shm-size=100g \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /home/:/home/ \
-v /opt/data/:/opt/data/ \
-v /mnt/weight/:/mnt/weight/ \
-v /root/.cache:/root/.cache \
-p $PORT:10005 \
-it $IMAGE bash
```

### 4、下载权重

BF16权重链接：https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B

W8A8（无MTP的量化权重）：https://modelers.cn/models/Eco-Tech/Qwen3.5-397B-A17B-w8a8-mtp

- 可使用 [msmodelslim](https://modelers.cn/link?target=https%3A%2F%2Fgitcode.com%2FAscend%2Fmsmodelslim) 对模型进行基础量化。

## 二、拉起服务

### 1、创建脚本

#### A3 单机部署
tp=16时，验证纯语言测试性能更好

```
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_IF_IP="x.x.x.x"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1

vllm serve /mnt/weight/Qwen3.5-397B-A17B-w8a8 \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 1 \
    --tensor-parallel-size 16 \
    --max-model-len 30000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.9 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,80,96,128]}' \
    --trust-remote-code \
    --async-scheduling \
    --enable_expert_parallel \
    --allowed-local-media-path / \
    --quantization ascend \
    --mm_processor_cache_type="shm" \
    --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}'
```

#### A2 双机部署

主节点node0：

```
export HCCL_IF_IP=x.x.x.x
export GLOO_SOCKET_IFNAME="xxxx"
export TP_SOCKET_IFNAME="xxxx"
export HCCL_SOCKET_IFNAME="xxxx"
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

vllm serve /opt/data/verification/models/Qwen3.5-397B-A17B-w8a8 \
    --served-model-name "qwen35" \
    --host x.x.x.x \
    --port 10010 \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address {主节点ip} \
    --data-parallel-rpc-port 2347 \
    --max-num-seqs 64 \
    --max-model-len 10000 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-expert-parallel \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,80,96,128]}' \
    --mm_processor_cache_type="shm" \
    --quantization ascend \
    --allowed-local-media-path / \
    --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert": true}'
```

从节点node1：

```
export HCCL_IF_IP=x.x.x.x
export GLOO_SOCKET_IFNAME="xxxx"
export TP_SOCKET_IFNAME="xxxx"
export HCCL_SOCKET_IFNAME="xxxx"
export HCCL_BUFFSIZE=1024
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100
export VLLM_USE_V1=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0

export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

export TASK_QUEUE_ENABLE=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

# profiling
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_DIR="/home/d00656702/Qwen35/profiling/"

vllm serve /opt/data/verification/models/Qwen3.5-397B-A17B-w8a8 \
    --served-model-name "qwen35" \
    --host 0.0.0.0 \
    --port 10010 \
    --headless \
    --tensor-parallel-size 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address {主节点ip} \
    --data-parallel-rpc-port 2347 \
    --max-num-seqs 64 \
    --max-model-len 10000 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.92 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-expert-parallel \
    --trust-remote-code \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32,64,80,96,128]}' \
    --mm_processor_cache_type="shm" \
    --quantization ascend \
    --allowed-local-media-path / \
    --additional-config '{"enable_cpu_binding":true,"multistream_overlap_shared_expert": true}'
```

### 2、curl测试

服务拉起后，用curl命令测试服务可用性

纯语言推理测试：

```
curl http://{主节点ip}:10010/v1/completions \
 -H "Content-Type: application/json" \
 -d '{
 "model": "qwen35",
 "prompt": "介绍一下你自己，用中文回答",
 "max_tokens": 200,
 "temperature": 0
 }'
```

图片推理测试：

```
curl http://{主节点ip}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.5",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}},
                {"type": "text", "text": "What is the text in the illustrate?"}
            ]}
        ]
    }'
```

## 三、精度测试

详细步骤请参阅 [使用 AISBench 进行精度评估](https://modelers.cn/link?target=https%3A%2F%2Fdocs.vllm.ai%2Fprojects%2Fascend%2Fen%2Flatest%2Fdeveloper_guide%2Fevaluation%2Fusing_ais_bench.html)。执行后即可获得评估结果。
```shell
ais_bench --models vllm_api_general_chat --datasets ceval_gen_0_shot_cot_chat_prompt --mode all --max-num-workers=52
```

本次在A3上使用ceval数据集进行精度测试：

|  数据集 (Dataset)  |  得分 (Score)  |
|-------------------|---------------|
| ceval             | 92.52         |

## 四、性能测试

### 1、480p单图测试
输入480p* 1图+30 token，输出1024 token，并发64
```shell
# 使用vllm bench工具进行测试
bs=64
in=30
out=1024
vllm bench serve \
--host {主节点ip} \
--port 8000 \
--backend openai-chat \
--endpoint /v1/chat/completions \
--model qwen3.5 \
--tokenizer /mnt/weight/Qwen3.5-397B-A17B-w8a8 \
--dataset-name random-mm \
--num-prompts $((bs*4)) \
--max-concurrency $bs \
--random-input-len $in \
--random-output-len $out \
--random-mm-bucket-config "{(480,640,1):1}"
```
#### A3
- Mean TTFT：`4.85 s`
- Mean TPOT：`47.6 ms`
- Output Token Throughput：`1221.38 tok/s`
- Total Token Throughput：`1257.16 tok/s`

#### A2
- Mean TTFT：`11.71 s`
- Mean TPOT：`52.53 ms`
- Output Token Throughput：`1000.74 tok/s`
- Total Token Throughput：`1030.05 tok/s`

### 2、纯语言测试
输入8192 token，输出1024 token，并发7

```shell
# 使用aisbench工具进行测试
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --debug --summarizer default_perf --mode perf --num-prompts 28
```

#### A3

- Mean TTFT：`2.73 s`
- Mean TPOT：`39.0 ms`
- Output Token Throughput：`168.19 tok/s`
- Total Token Throughput：`1515.69 tok/s`

#### A2

- Mean TTFT：`5.88 s`
- Mean TPOT：`40.8 ms`
- Output Token Throughput：`150.31 tok/s`
- Total Token Throughput：`1354.57 tok/s`
