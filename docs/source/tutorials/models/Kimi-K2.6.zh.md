# Kimi-K2.6

<p align="center">
  <a href="Kimi-K2.6.md"><b>English</b></a> | <a href="Kimi-K2.6.zh.md"><b>中文</b></a>
</p>

## 1 简介

Kimi K2.6 是一个开源的原生多模态 agentic 模型，在 Kimi-K2-Base 的基础上通过约 15 万亿混合视觉和文本 token 的持续预训练构建。它无缝融合了视觉与语言理解能力，具备高级 agentic 能力、instant 与 thinking 模式，以及对话与 agentic 两种范式。

本文档将展示模型的主要验证步骤，包括支持的功能、功能配置、环境准备、单节点和多节点部署、精度和性能评估。

本文档基于 **vLLM-Ascend v0.20.0rc1** 版本进行验证和编写。当前模型（Kimi-K2.6）在该版本中首次支持。

## 2 支持的特性

请参考[支持的功能列表](../../user_guide/support_matrix/supported_models.md)，获取模型支持的功能矩阵。

请参考[特性指南](../../user_guide/feature_guide/index.md)获取功能配置信息。

## 3 前置准备

### 3.1 模型权重

- `Kimi-K2.6-w4a8`（w4a8 量化版）：需要 1 台 Atlas 800 A3（64G × 16）节点或 2 台 Atlas 800 A2（64G × 8）节点。[下载模型权重](https://modelscope.cn/models/Eco-Tech/Kimi-K2.6-W4A8)。
- `kimi-k2.6-eagle3`（用于加速 Kimi-K2.6 推理的 Eagle3 MTP 草稿模型）：[下载模型权重](https://huggingface.co/lightseekorg/kimi-k2.6-eagle3)
- `Kimi-K2.6-DFlash`（一种推测解码框架，利用轻量级块扩散模型进行并行草稿生成）：[下载模型权重](https://huggingface.co/z-lab/Kimi-K2.6-DFlash)

建议将模型权重下载至多节点共享目录，如 `/root/.cache/`。

### 3.2 验证多节点通信（可选）

若需部署多节点环境，请依据[验证多节点通信环境](../../installation.md#verify-multi-node-communication)指南进行通信验证。

## 4 安装

### 4.1 Docker 镜像安装

根据机器类型选择镜像并在节点上启动 docker 镜像，请参考[使用 docker](../../installation.md#set-up-using-docker)。

**A3 系列**

在每个节点上启动 docker 镜像。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
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

**A2 系列**

在每个节点上启动 docker 镜像。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
    --name vllm-ascend \
    --shm-size=1g \
    --net=host \
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

docker 成功运行后，可通过执行 `docker ps` 命令验证容器服务是否正在运行。

### 4.2 源码安装

如果不希望使用上述 docker 镜像，也可以从源码构建：

- 从源码安装 `vllm-ascend`，请参考[安装指南](../../installation.md)。

如需部署多节点环境，需要在每个节点上进行环境配置。

如需使用 tools_call 功能，请确保 transformers 版本为 4.57.6 或更低。

## 5 在线服务化部署

### 5.1 单机在线部署

单机部署将 Prefill 与 Decode 在同一节点内完成。量化模型 `Kimi-K2.6-w4a8` 可部署在 1 台 Atlas 800 A3（64G × 16）上。

单机部署支持所有输入/输出场景，但要获得最佳性能，建议使用多节点部署。

启动命令：

```bash
#!/bin/sh
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_MLAPO=1

# [可选] jemalloc
# jemalloc 可提升性能，如果机器上已安装 libjemalloc.so，可以启用。
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export HCCL_BUFFSIZE=800
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_BALANCE_SCHEDULING=1
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

vllm serve Eco-Tech/Kimi-K2.6-W4A8 \
    --quantization ascend \
    --served-model-name kimi_k26 \
    --allowed-local-media-path / \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --data-parallel-size 4 \
    --no-enable-prefix-caching \
    --enable-expert-parallel \
    --port 8088 \
    --max-num-seqs 4 \
    --max-model-len 32768 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.9 \
    --seed 42 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --mm-processor-cache-gb 0 \
    --mm-encoder-tp-mode data \
    --speculative-config '{"method": "dflash","model": "z-lab/Kimi-K2.6-DFlash", "num_speculative_tokens": 15}'
```

关键参数说明：

- 设置环境变量 `VLLM_ASCEND_BALANCE_SCHEDULING=1` 可启用均衡调度。这有助于在 v1 调度器中提高输出吞吐量并降低 TPOT，但在某些场景下可能会使 TTFT 变差。此外，在 PD 分离场景下不建议启用此功能。
- `--max-model-len` 指定最大上下文长度，即单个请求的输入和输出 token 总和。对于输入长度为 3.5K、输出长度为 1.5K 的性能测试，设置为 `16384` 即可满足需求；但对于精度测试，请至少设置为 `35000`。
- `--no-enable-prefix-caching` 表示禁用前缀缓存。如需启用，请移除此选项。
- `--mm-encoder-tp-mode` 指示如何使用张量并行（TP）优化多模态编码器推理。如需测试多模态输入，推荐使用 `data`。
- 如果使用 w4a8 权重，将为 kvcache 分配更多内存，可尝试提高系统吞吐量以获得更大的吞吐量。
- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`：启用大规模融合算子以替换原有的细粒度小算子，可显著降低 kernel 启动开销，提升整体执行性能。

常见问题提示：如遇问题，请参考[公共 FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) 进行检查。

服务验证：

```shell
curl http://<node0_ip>:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k26",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

预期结果：

服务返回 HTTP 200 OK，JSON 响应中包含 `choices` 字段。示例输出（内容有删减）：

```json
{
    "id": "chatcmpl-9df13fd5e539af93",
    "object": "chat.completion",
    "created": 1780971952,
    "model": "kimi_k26",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The future of AI is not a destination we are passively approaching, but a design problem we are actively solving right now...",
                "reasoning": "The user is asking for my thoughts on \"The future of AI is\"...",
                "refusal": null,
                "annotations": null,
                "audio": null,
                "function_call": null
            },
            "logprobs": null,
            "finish_reason": "length",
            "stop_reason": null,
            "token_ids": null
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 1037,
        "completion_tokens": 1024,
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": null,
            "accepted_prediction_tokens": null,
            "rejected_prediction_tokens": null
        }
    }
}
```

### 5.2 多机 PD 分离部署

推荐使用 Mooncake 进行部署：[Mooncake](../features/pd_disaggregation_mooncake_multi_node.md)。

在标准的单节点部署模式下，Prefill（提示词处理）和 Decode（token 生成）任务在同一组 NPU 上运行。这可能导致两个问题：

1. **Prefill 抢占中断 Decode**：Prefill 是计算密集型任务，一次性处理整个输入上下文；而 Decode 是逐个生成 token。当新用户请求到达时，其 Prefill 阶段会抢占并中断正在进行的 Decode 任务，导致抖动和更高的每输出 token 时间（TPOT）延迟。
2. **资源分配不灵活**：Prefill 和 Decode 具有根本不同的计算特性——Prefill 受计算和内存带宽限制，而 Decode 仅受内存带宽限制。在同一硬件上运行两者会迫使做出妥协，无法最优地满足各自需求。

PD（Prefill-Decode）分离通过将 Prefill 和 Decode 运行在专用的节点组上来解决这些问题，每组可独立配置：

- **Prefill 节点**专注于高吞吐的提示词处理，针对计算和通信进行优化（例如，启用 FlashComm 加速 Allreduce）。
- **Decode 节点**专注于低延迟的 token 生成，针对内存带宽进行优化（例如，启用 MLAPO 融合算子）。

该架构推荐用于具有并发多用户工作负载的生产部署，同时对稳定延迟和高吞吐量都有要求。

以 Atlas 800 A3（64G × 16）为例，我们建议部署 2P1D（4 节点）而非 1P1D（2 节点），因为 1P1D 场景下 NPU 内存不足以支持高并发。

- `Kimi-K2.6-w4a8 2P1D`：需要 4 台 Atlas 800 A3（64G × 16）节点。

要运行 vllm-ascend 的 `Prefill-Decode 分离` 服务，需要在每个节点上部署 `launch_online_dp.py` 脚本和 `run_dp_template.sh` 脚本，并在 prefill 主节点上部署 `proxy.sh` 脚本用于转发请求。

1. `launch_online_dp.py` 用于启动外部 DP vllm 服务器。
    [launch_online_dp.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/external_online_dp/launch_online_dp.py)

    参数说明：

    |参数|类型|是否必填|默认值|描述|
    |----|----|--------|------|------|
    |`--dp-size`|int|是|-|数据并行规模（所有节点上的 DP rank 总数）。|
    |`--tp-size`|int|否|1|每个 DP rank 的张量并行规模。|
    |`--dp-size-local`|int|否|（与 `--dp-size` 相同）|当前节点上的 DP rank 数量。如未设置，默认与 `--dp-size` 一致。|
    |`--dp-rank-start`|int|否|0|当前节点上数据并行 rank 的起始编号偏移。|
    |`--dp-address`|str|是|-|数据并行主节点（节点 0）的 IP 地址。|
    |`--dp-rpc-port`|str|否|12345|数据并行主节点通信的 RPC 端口。|
    |`--vllm-start-port`|int|否|9000|当前节点每个 vLLM 引擎实例的起始端口。每个 DP rank 的引擎端口 = `vllm_start_port` + 本地 rank 索引。|

2. Prefill 节点 0 `run_dp_template.sh` 脚本

    ```shell
    # 通过 ifconfig 获取
    # nic_name 是当前节点 local_ip 对应的网卡名称
    nic_name="xxx"
    local_ip="141.xx.xx.1"

    # node0_ip 的值必须与 node0（主节点）中设置的 local_ip 值一致
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [可选] jemalloc
    # jemalloc 可提升性能，如果机器上已安装 libjemalloc.so，可以启用。
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=800
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve Eco-Tech/Kimi-K2.6-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k26 \
      --trust-remote-code \
      --max-num-seqs 4 \
      --max-model-len 32768 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.95 \
      --enforce-eager \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.6-eagle3", "num_speculative_tokens": 3}' \
      --additional-config '{"recompute_scheduler_enable":true}' \
      --mm-encoder-tp-mode data \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30000",
      "engine_id": "0",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
          }
      }'
    ```

3. Prefill 节点 1 `run_dp_template.sh` 脚本

    ```shell
    # 通过 ifconfig 获取
    # nic_name 是当前节点 local_ip 对应的网卡名称
    nic_name="xxx"
    local_ip="141.xx.xx.2"

    # node0_ip 的值必须与 node0（主节点）中设置的 local_ip 值一致
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [可选] jemalloc
    # jemalloc 可提升性能，如果机器上已安装 libjemalloc.so，可以启用。
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=800
    export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve Eco-Tech/Kimi-K2.6-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k26 \
      --trust-remote-code \
      --max-num-seqs 4 \
      --max-model-len 32768 \
      --max-num-batched-tokens 16384 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.95 \
      --enforce-eager \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.6-eagle3", "num_speculative_tokens": 3}' \
      --additional-config '{"recompute_scheduler_enable":true}' \
      --mm-encoder-tp-mode data \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_producer",
      "kv_port": "30100",
      "engine_id": "1",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
          }
      }'
    ```

4. Decode 节点 0 `run_dp_template.sh` 脚本

    ```shell
    # 通过 ifconfig 获取
    # nic_name 是当前节点 local_ip 对应的网卡名称
    nic_name="xxx"
    local_ip="141.xx.xx.3"

    # node0_ip 的值必须与 node0（主节点）中设置的 local_ip 值一致
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [可选] jemalloc
    # jemalloc 可提升性能，如果机器上已安装 libjemalloc.so，可以启用。
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=800
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve Eco-Tech/Kimi-K2.6-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k26 \
      --trust-remote-code \
      --max-num-seqs 8 \
      --max-model-len 32768 \
      --max-num-batched-tokens 32 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.91 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}' \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.6-eagle3", "num_speculative_tokens": 3}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "2",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
          }
      }'
    ```

5. Decode 节点 1 `run_dp_template.sh` 脚本

    ```shell
    # 通过 ifconfig 获取
    # nic_name 是当前节点 local_ip 对应的网卡名称
    nic_name="xxx"
    local_ip="141.xx.xx.4"

    # node0_ip 的值必须与 node0（主节点）中设置的 local_ip 值一致
    node0_ip="xxxx"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name

    # [可选] jemalloc
    # jemalloc 可提升性能，如果机器上已安装 libjemalloc.so，可以启用。
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sysctl -w vm.swappiness=0
    sysctl -w kernel.numa_balancing=0
    sysctl kernel.sched_migration_cost_ns=50000
    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000

    export HCCL_OP_EXPANSION_MODE="AIV"
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    export ASCEND_BUFFER_POOL=4:8
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH

    export HCCL_BUFFSIZE=1100
    export VLLM_ASCEND_ENABLE_MLAPO=1
    export ASCEND_RT_VISIBLE_DEVICES=$1
    export VLLM_ASCEND_ENABLE_FUSED_MC2=1

    vllm serve Eco-Tech/Kimi-K2.6-W4A8 \
      --host 0.0.0.0 \
      --port $2 \
      --data-parallel-size $3 \
      --data-parallel-rank $4 \
      --data-parallel-address $5 \
      --data-parallel-rpc-port $6 \
      --tensor-parallel-size $7 \
      --enable-expert-parallel \
      --seed 1024 \
      --quantization ascend \
      --served-model-name kimi_k26 \
      --trust-remote-code \
      --max-num-seqs 8 \
      --max-model-len 32768 \
      --max-num-batched-tokens 4 \
      --no-enable-prefix-caching \
      --gpu-memory-utilization 0.91 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --additional-config '{"recompute_scheduler_enable":true,"multistream_overlap_shared_expert": false}' \
      --speculative-config '{"method": "eagle3", "model":"lightseekorg/kimi-k2.6-eagle3", "num_speculative_tokens": 3}' \
      --kv-transfer-config \
      '{"kv_connector": "MooncakeConnectorV1",
      "kv_role": "kv_consumer",
      "kv_port": "30200",
      "engine_id": "2",
      "kv_connector_extra_config": {
                "prefill": {
                        "dp_size": 4,
                        "tp_size": 4
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 4
                }
          }
      }'
    ```

关键参数说明：

- `VLLM_ASCEND_ENABLE_FUSED_MC2=1`：启用大规模融合算子以替换原有的细粒度小算子，可显著降低 kernel 启动开销，提升整体执行性能。
- `VLLM_ASCEND_ENABLE_FLASHCOMM1=1`：在 prefill 节点上启用通信优化功能。
- `VLLM_ASCEND_ENABLE_MLAPO=1`：启用融合算子，可显著提升性能，但会消耗更多 NPU 内存。在 Prefill-Decode（PD）分离场景下，请仅在 Decode 节点上启用 MLAPO。
- `recompute_scheduler_enable: true`：启用重计算调度器。当 Decode 节点的 Key-Value Cache（KV Cache）不足时，请求将发送至 Prefill 节点重新计算 KV Cache。在 PD 分离场景下，建议在 Prefill 和 Decode 节点上同时启用此配置。
- `multistream_overlap_shared_expert: true`：当张量并行（TP）规模为 1 或 `enable_shared_expert_dp: true` 时，启用额外的流来重叠共享专家的计算过程，以提高效率。

6. 各节点启动服务：

    ```shell
    # p0
    python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address 141.xx.xx.1 --dp-rpc-port 12321 --vllm-start-port 7100
    # p1
    python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address 141.xx.xx.2 --dp-rpc-port 12321 --vllm-start-port 7100
    # d0
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 8 --dp-rank-start 0 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    # d1
    python launch_online_dp.py --dp-size 8 --tp-size 4 --dp-size-local 8 --dp-rank-start 8 --dp-address 141.xx.xx.3 --dp-rpc-port 12321 --vllm-start-port 7100
    ```

7. 在 prefill 主节点上运行 `proxy.sh` 脚本

    在与 prefiller 服务实例相同的节点上运行代理服务器。可以从仓库的示例中获取代理程序：[load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py)

    ```shell
    python load_balance_proxy_server_example.py \
      --port 1999 \
      --host 141.xx.xx.1 \
      --prefiller-hosts \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.1 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
        141.xx.xx.2 \
      --prefiller-ports \
        7100 7101 7102 7103 7100 7101 7102 7103 \
      --decoder-hosts \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.3 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
        141.xx.xx.4 \
      --decoder-ports \
        7100 7101 7102 7103 \
        7100 7101 7102 7103 \
    ```

    ```shell
    cd vllm-ascend/examples/disaggregated_prefill_v1/
    bash proxy.sh
    ```

部署验证：

PD 分离服务完全启动后，通过 prefill 主节点上的代理端口发送请求，验证 Prefill 和 Decode 节点是否能够正常协同工作：

```shell
curl http://141.xx.xx.1:1999/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k26",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

预期结果：

代理返回 HTTP 200 OK。JSON 响应中包含 `choices` 字段及生成的文本，确认 Prefill 节点已成功处理提示词、Decode 节点已生成响应：

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "kimi_k26",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The future of AI is not a destination we are passively approaching...",
                "finish_reason": "length"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 1037,
        "completion_tokens": 1024
    }
}
```

常见问题提示：如遇 PD 分离部署问题，请参考[公共 FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) 进行检查。

## 6 功能验证

服务启动后，即可通过发送提示词来调用模型：

```shell
curl http://<node0_ip>:<port>/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "kimi_k26",
        "messages": [{
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "The future of AI is"
            }]
        }],
        "max_tokens": 1024,
        "temperature": 1.0,
        "top_p": 0.95
    }'
```

预期结果：

服务返回 HTTP 200 OK。JSON 响应中包含 `choices` 字段及生成的文本，以及用量统计信息：

```json
{
    "id": "chatcmpl-9df13fd5e539af93",
    "object": "chat.completion",
    "created": 1780971952,
    "model": "kimi_k26",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The future of AI is not a destination we are passively approaching, but a design problem we are actively solving right now...",
                "reasoning": "The user is asking for my thoughts on...",
                "finish_reason": "length"
            }
        }
    ],
    "usage": {
        "prompt_tokens": 13,
        "total_tokens": 1037,
        "completion_tokens": 1024
    }
}
```

## 7 精度评估

以下提供两种精度评估方法。

### AISBench 的使用

1. 详情请参考 [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md)。

2. 执行后可获得评估结果。以下为 `Kimi-K2.6-w4a8` 在 `vllm-ascend:v0.20.0rc1` 中的测试结果，仅供参考。

| dataset | version | metric | mode | vllm-api-general-chat | note |
| ----- | ----- | ----- | ----- | ----- | ----- |
| AIME2026 | - | accuracy | gen | 90.00 | 1 Atlas 800 A3 (64G × 16) |
| GPQA | - | accuracy | gen | 89.90 | 1 Atlas 800 A3 (64G × 16) |
| MMMU | - | accuracy | gen | 82.67 | 1 Atlas 800 A3 (64G × 16) |

## 8 性能

### AISBench 的使用

详情请参考 [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation)。

### vLLM Benchmark 的使用

以 `Kimi-K2.6-w4a8` 为例，运行性能评估。

更多详情请参考 [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/)。

`vllm bench` 有三个子命令：

- `latency`：测试单批请求的延迟。
- `serve`：测试在线服务吞吐量。
- `throughput`：测试离线推理吞吐量。

以 `serve` 为例，运行以下命令。

```shell
export VLLM_USE_MODELSCOPE=True
vllm bench serve --model Eco-Tech/Kimi-K2.6-w4a8 --dataset-name random --random-input 1024 --num-prompts 200 --request-rate 1 --save-result --result-dir ./
```

大约几分钟后即可获得性能评估结果。

## 9 性能调优

### 9.1 推荐配置

> **说明**：以下配置基于特定测试环境验证，仅作参考。实际最优配置取决于最大输入/输出长度、前缀缓存命中率、精度要求、部署机器配比等因素，建议根据实际情况参考 9.2 节进行调优。

**最佳实践建议（来自源文档）：**

- **单节点混合部署**
    - 16K 长上下文场景：设置为 dp2 tp8，以平衡内存容量和计算效率。
    - 128K 长上下文（无前缀缓存）：设置为 dp1 tp16，最大化张量并行以在单个节点内支持极长上下文。
    - 128K 长上下文（有前缀缓存）：设置为 dp2 tp8，优化内存带宽并提高前缀缓存利用率。
    - 1080P 多模态场景：设置为 dp1 tp16，以应对高分辨率视觉输入的高计算和内存需求。

- **双节点 1P1D 场景（1 个 Prefill 节点，1 个 Decode 节点）**
    - 通用 P/D 节点配置：Prefill 和 Decode 节点均设置为 dp2 tp8，确保延迟和吞吐量之间的平衡。
    - 1080P 多模态场景：根据具体的内存约束和并发需求，配置为 dp2 tp8 或 dp16 tp1。

- **四节点 2P2D 场景（2 个 Prefill 节点，2 个 Decode 节点）**
    - 通用 P/D 节点配置：将配置从 dp4 tp4 扩展到 dp8 tp4，有效利用增加的分布式计算资源。
    - 1080P 多模态场景：配置为更高的数据并行度，从 dp8 tp2 到 dp32 tp1，以最大化吞吐量并处理多个 Decode 节点上的重多模态工作负载。

#### 表 1：场景概览

> `*总卡数` 表示所有节点使用的 NPU 总数。1 节点 = 1 台 Atlas 800 A3 服务器（64G × 16 NPU）。

|场景|部署形态|*总卡数|权重版本|场景要点|
|----|--------|------|--------|--------|
|高吞吐 / 低时延<br>（16K 上下文）|单节点混合|16（A3）|kimi-k2.6-w4a8|使用 dp2 tp8 平衡内存容量和计算效率|
|高吞吐 / 低时延<br>（16K 上下文）|1P1D 部署|32（A3）|kimi-k2.6-w4a8|P 和 D 节点均为 dp2 tp8；延迟和吞吐量均衡|
|高吞吐 / 低时延<br>（16K 上下文）|2P2D 部署|64（A3）|kimi-k2.6-w4a8|各节点从 dp4 tp4 扩展到 dp8 tp4|
|长上下文<br>（128K，无前缀缓存）|单节点混合|16（A3）|kimi-k2.6-w4a8|dp1 tp16，最大化 TP，支持极长上下文|
|长上下文<br>（128K，有前缀缓存）|单节点混合|16（A3）|kimi-k2.6-w4a8|dp2 tp8，优化内存带宽，提高缓存利用率|
|多模态<br>（1080P）|单节点混合|16（A3）|kimi-k2.6-w4a8|dp1 tp16，用于高分辨率视觉输入|
|多模态<br>（1080P）|1P1D 部署|32（A3）|kimi-k2.6-w4a8|dp2 tp8 或 dp16 tp1，取决于内存和并发需求|
|多模态<br>（1080P）|2P2D 部署|64（A3）|kimi-k2.6-w4a8|dp8 tp2 到 dp32 tp1，最大化吞吐量以处理重多模态工作负载|

#### 表 2：节点详细配置

|场景|配置|卡数|TP|DP|最大上下文长度|MTP 投机数|FUSED_MC2|
|----|------|----|--|--|--------------|---------|---------|
|高吞吐 / 低时延（16K）|服务端 / 单机|16|8|2|~16K|15|开|
|高吞吐 / 低时延（16K）|服务端-P 节点|16|8|2|~16K|3|开|
|高吞吐 / 低时延（16K）|服务端-D 节点|16|8|2|~16K|3|开|
|长上下文（128K，无缓存）|服务端 / 单机|16|16|1|128K|15|开|
|长上下文（128K，有缓存）|服务端 / 单机|16|8|2|128K|15|开|
|多模态（1080P）|服务端 / 单机|16|16|1|~16K|15|开|
|多模态（1080P）|服务端-P 节点|16|8|2|~16K|3|开|
|多模态（1080P）|服务端-D 节点|16|1|16|~16K|3|开|

> 完整启动命令及参数含义请参考[第 5 章](#5-在线服务化部署)部署示例。

**注意：**
`max-model-len` 和 `max-num-seqs` 需根据实际使用场景进行设置。其他设置请参考 **[部署](#5-在线服务化部署)** 章节。

## 10 FAQ

常见环境、安装、通用参数问题请参考[公共 FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)；本章仅收录本模型特有疑难问题。

- **问：启动时出现 HCCL 端口冲突（地址已被占用），如何处理？**

  答：清理旧进程后重启：`pkill -f vLLM*`。

- **问：遇到 OOM 或启动不稳定如何处理？**

  答：首先降低 `--max-num-seqs` 和 `--max-model-len`。如有需要，减少并发数和压力测试压力（如 `max-concurrency` / `num-prompts`）。

- **问：tools_call 功能需要什么 transformers 版本？**

  答：如需使用 tools_call 功能，请确保 transformers 版本为 4.57.6 或更低。
