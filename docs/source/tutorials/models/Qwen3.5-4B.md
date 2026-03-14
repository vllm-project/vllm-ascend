---
license: apache-2.0
---
# Qwen3.5-4B
## 简介
Qwen3.5 是 Qwen 系列最新的旗舰多模态模型，采用 MoE (Mixture of Experts)架构，在保持极强模型能力的同时显著降低推理成本。核心架构特点包括：
- 原生多模态能力（Vision Encoder + 图文融合）
- 混合注意力机制（Full Attention 与 Linear-Attention 交替）
- MTP 多 Token预测分支
- 高性能 MoE 专家路由与共享专家机制。
本文档将展示该模型的主要验证步骤，包括支持特性、特性配置、环境准备、单节点与多节点部署、精度评估及性能评估。

## 支持特性

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|
| Qwen3.5-4B | ✅ | ✖️ | ✅ | A3 | ✖️ | ✅ | ✖️ | - | ✖️ | ✅ | ✅ | - | ✅ | ✅ | ✖️ | - | ✅ | 256K | -

请参阅 [特性指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html) 获取特性配置说明。

## 环境准备
### 模型权重
- Qwen3.5-4B（BF16 版本）：[下载模型权重][https://modelers.cn/models/Qwen-AI/Qwen3.5-4B]
注： 建议将模型权重下载至多节点共享目录，例如 `/root/.cache/`。

### 安装
#### 1) 官方Docker镜像
您可以通过[镜像链接](https://)下载镜像压缩包来进行部署，具体流程如下：
```{code-block} bash
# 使用docker加载下载的镜像压缩包
# 根据您的环境更新要加载的vllm-ascend镜像压缩包名称,以下以A3 arm为例：
docker load -i Vllm-ascend-Qwen3_5-A3-Ubuntu-v0.tar 
# 根据您的设备更新 --device（Atlas A3：/dev/davinci[0-15]）。

# 注意：您需要提前将权重下载至 /root/.cache。
# 更新 vllm-ascend 镜像，并配置对应的Image名
export IMAGE=vllm-ascend:qwen3_5-v0-a3 
export NAME=vllm-ascend

# 使用定义的变量运行容器
# 注意：若使用 Docker 桥接网络，请提前开放可供多节点通信的端口
docker run --rm \
--name $NAME \
--net=host \
--shm-size=100g \
--device /dev/davinci0 \
--device /dev/davinci1 \
--device /dev/davinci2 \
--device /dev/davinci3 \
--device /dev/davinci4 \
--device /dev/davinci5 \
--device /dev/davinci6 \
--device /dev/davinci7 \
--device /dev/davinci_manager \
--device /dev/devmm_svm \
--device /dev/hisi_hdc \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /root/.cache:/root/.cache \
-it $IMAGE bash
```

#### 2）源码构建

如果您不希望使用上述 Docker 镜像，也可通过源码完整构建：
- 保证你的环境成功安装了CANN 8.5.0

- 从源码安装 `vllm-ascend`，请参考 [安装指南](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)。

从源码安装 `vllm-ascend`后，您需要将 vllm、vllm-ascend、transformers 升级至主分支：
```shell
# 升级 vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a75a5b54c7f76bc2e15d3025d6
git fetch origin pull/34521/head:pr-34521
git merge pr-34521
VLLM_TARGET_DEVICE=empty pip install -v .

# 升级 vllm-ascend
pip uninstall vllm-ascend -y
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout c63b7a11888e9e1caeeff8
git fetch origin pull/6742/head:pr-6742
git merge pr-6742
pip install -v .

# 重新安装 transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git reset --hard fc9137225880a9d03f130634c20f9dbe36a7b8bf
pip install .
``` 
如需部署多节点环境，您需要在每个节点上分别完成环境配置。

### 部署

#### 单节点部署

##### A2 系列

尚未测试。

##### A3 系列

执行以下脚本进行在线推理。

```shell
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
export TASK_QUEUE_ENABLE=1

vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/Qwen3.5-4B/ \
    --served-model-name "qwen3.5" \
    --host 0.0.0.0 \
    --port 8010 \
    --data-parallel-size 1 \
    --tensor-parallel-size 4 \
    --max-model-len 5000 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.8 \
    --skip-mm-profiling \
    --trust-remote-code \
    --async-scheduling \
    --allowed-local-media-path / \
    --mm-processor-cache-gb 0 \
	  --enforce-eager \
    --additional-config '{"enable_cpu_binding":true, "multistream_overlap_shared_expert": true}' 
```

执行以下脚本向模型发送一条请求：
```shell
curl http://localhost:8010/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
        "prompt": "The future of AI is",
        "path": "/path/to/model/Qwen3.5-27B/",
        "max_tokens": 100,
        "temperature": 0
        }'
```

执行结束后，您可以看到模型回答如下：

```shell
Prompt: 'The future of AI is', Generated text: ' not just about building smarter machines, but about creating systems that can collaborate with humans in meaningful, ethical, and sustainable ways. As AI continues to evolve, it will increasingly shape how we live, work, and interact — and the decisions we make today will determine whether this future is one of shared prosperity or deepening inequality.\n\nThe rise of generative AI, for example, has already begun to transform creative industries, education, and scientific research. Tools like ChatGPT, Midjourney, and'

```
也可执行以下脚本向模型发送一条多模态请求：
```shell
curl http://localhost:8010/v1/completions \
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
执行结束后，您可以看到模型回答如下：
```shell
{"id":"chatcmpl-9dab99d55addd8c0","object":"chat.completion","created":1771060145,"model":"qwen3.5","choices":[{"index":0,"message":{"role":"assistant","content":"TONGYI Qwen","refusal":null,"annotations":null,"audio":null,"function_call":null,"tool_calls":[],"reasoning":null},"logprobs":null,"finish_reason":"stop","stop_reason":null,"token_ids":null}],"service_tier":null,"system_fingerprint":null,"usage":{"prompt_tokens":112,"total_tokens":119,"completion_tokens":7,"prompt_tokens_details":null},"prompt_logprobs":null,"prompt_token_ids":null,"kv_transfer_params":null}
```



#### 多节点部署

##### A2 系列

尚未测试。

##### A3 系列

尚未测试。

#### PD分离
