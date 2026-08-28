# GLM-5.3-Flash

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash`(BF16 version): requires Atlas 800 A5 (96GB × 8) node or 2 Atlas 800 A3 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/ZhipuAI/GLM-5.2).
- `GLM-5.3-Flash-w8a8`: requires 1 Atlas 800 A3 (64GB × 8) node or Atlas 800 A2 (64GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w8a8).
- `GLM-5.3-Flash-MXFP8`: requires 1 Atlas A5 DT Server (96GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.2-w4a8c8).
- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

### 3.2 Verify Multi-node Communication (Optional)

If you want to deploy multi-node environment, you need to verify multi-node communication according to [verify multi-node communication environment](../../installation.md#verify-multi-node-communication).

## 4 Installation

### 4.1 Docker Image Installation

- You can use our official docker image to run GLM-5.2 directly.
















=== "A5 series"

    Start the docker image on each node.

    ```shell

    # Reduce memory fragmentation and avoid out-of-memory errors.
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # export HCCL_OP_EXPANSION_MODE="AIV"
    export HCCL_BUFFSIZE=1024
    export OMP_NUM_THREADS=1
    export TASK_QUEUE_ENABLE=1
    # export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
    #echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    #sysctl -w vm.swappiness=0
    #sysctl -w kernel.numa_balancing=0
    #sysctl kernel.sched_migration_cost_ns=50000
    #export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

    #vllm serve /mnt/share/wyzhou/GLM-5-Next-0808/hf/ \
    #vllm serve /mnt/share/w00936111/GLM-5-Next-W8A8-0814-A5/GLM-5-Next-W8A8-0814-A5/ \
    vllm serve /mnt/share/w00936111/weights/GLM-5.3-Flash-0day-A5-convert \
    --host 0.0.0.0 \
    --port 8000 \
    --data-parallel-size 8 \
    --data-parallel-size-local 8 \
    --enable-expert-parallel \
    --seed 1024 \
    --dtype bfloat16 \
    --served-model-name glm \
    --max-num-seqs 128 \
    --max-model-len 168000 \
    --async-scheduling \
    --enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'
    ```



=== "A3 series"

    Start the docker image on each node.

    ```shell

    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
    export NAME=vllm-ascend

    # Run the container using the defined variables
    # Note: If you are running bridge network with docker, please expose available ports for multiple nodes communication in advance
    docker run --rm \
    --name $NAME \
    --net=host \
    --shm-size=1g \
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


### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../installation.md).

## 5 Deployment

The deployment scenarios validated for this release are organized by context window size (below 1M / 1M), hardware (Atlas 800 A3 / A2), and deployment mode (single-node, multi-node co-located, Prefill-Decode disaggregation). All startup scripts below are the verified reference commands; key parameters are explained after each scenario.

### 5.1 Context Below 1
#### 
.1.1Ascend0950DT3


####25.1.1 Atlas 800 A3

#####25.1.1.1 Single-Node Deployment

- Quantized model `GLM-5.2-w4a8c8` can be deployed on 1 Atlas 800 A3 (64GB × 16) .

Run the following script to execute online inference.

```shell
export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_TRANSFER_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=3600
export HCCL_CONNECT_TIMEOUT=3600
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=200
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
vllm serve /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.2-w4a8c8 \
--host 0.0.0.0 \
--port 8077 \
--safetensors-load-strategy prefetch \
--api-server-count 1 \
--data-parallel-size 2 \
--enable-expert-parallel \
--tensor-parallel-size 8 \
--seed 1024 \
--served-model-name glm-5 \
--tool-call-parser glm47 \
--reasoning-parser glm45 \
--enable-auto-tool-choice \
--max-num-seqs 12 \
--max-model-len 135000 \
--max-num-batched-tokens 8192 \
--trust-remote-code \
--gpu-memory-utilization 0.92 \
--quantization ascend \
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
--additional-config '{"enable_dsa_cp": true,"enable_sparse_li_c8": true,"enable_balance_scheduling": true,"multistream_overlap_shared_expert":true,"enable_fused_mc2":0}' \
--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp","enforce_eager":true}'

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.2.
- `--quantization ascend`: Enables Ascend quantization for the w4a8c8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w4a8c8 weights. For low-latency scenarios, use `dp1tp16` instead and turn off expert parallel, at the cost of lower throughput.
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable function calling for GLM-5.2.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.2. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.2 does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.
- `additional_config.enable_fused_mc2=0`: Disables the fused `dispatch_ffn_combine`/`mega_moe` operators in this scenario because they conflict with `multistream_overlap_shared_expert`; turn them on in multi-node scenarios where they are beneficial.

**`--additional-config` fields (Ascend-specific optimizations):**

- `"enable_dsa_cp": true`: Enables DSA context parallelism to accelerate long-context prefill. With DSA-CP enabled, `layer_sharding` cannot include `o_proj`.
- `"enable_sparse_li_c8": true`: Sparse attention optimizations of the C8 quantized model. `enable_sparse_li_c8` accelerates the layer-index (LI) sparse attention and is recommended to keep `true`; If the GPU memory is insufficient due to a long sequence length, you are advised to enable `enable_sparse_sfa_c8`.
- `"enable_balance_scheduling": true`: Balance scheduling improves output throughput and reduces TPOT in the v1 scheduler. This config field replaces the removed environment setting; TTFT may degrade in some scenarios, and it is not recommended when Prefill-Decode is separated.
- `"multistream_overlap_shared_expert": true`: Overlaps shared-expert computation on an additional stream, improving decode efficiency.



## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-52",
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
        "temperature": 0
    }'
```

## 7 Accuracy Evaluation

Here are two accuracy evaluation methods.

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

2. After execution, you can get the result.

### 7.2 Using Language Model Evaluation Harness

Not tested yet.

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to [Tuning Guidelines](#92-tuning-guidelines) for tuning based on actual conditions.

The tables below provide recommended parameter configurations for different deployment scenarios. All scenarios are categorized by use case (Low Latency, High Throughput, Long Context) and correspond to the deployment modes documented in [Deployment](#5-deployment).

#### 9.1.1 Table 1: Scenario Overview

> `*Total NPUs` indicates the total number of NPUs used across all nodes. 1 node = 1 Atlas 800 A3 server (64GB × 16 NPUs) or 1 Atlas 800 A2 server (64GB × 8 NPUs).

|Scenario|Deployment Mode|*Total NPUs|Weight Version|Key Considerations|
|--------|---------------|-----------|--------------|------------------|
|Low Latency<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, DSA CP|
|Low Latency<br>(128K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp2 tp16, MTP3, max-num-seqs 8, max-model-len 135000, DSA CP|
|High Throughput<br>(64K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp4 tp8, fused MC2, MTP3, max-num-seqs 16, max-model-len 66000, DSA CP|
|High Throughput|PD Disaggregation (A3), [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|4 nodes (A3)|w4a8c8|P: dp4 tp8 (max-num-seqs 64, max-num-batched-tokens 8192, MTP1); D: dp32 tp1 (max-num-seqs 32, max-num-batched-tokens 164, MTP5), max-model-len 133120, dedicated P/D nodes, Mooncake KV transfer|
|Long Context<br>(1M)|PD Disaggregation (A3), [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|4 nodes (A3)|w4a8c8|P/D: dp4 tp8, DCP8, max-model-len 1040000, Mooncake KV transfer|

#### 9.1.2 Table 2: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a node hosting 2 DP ranks of TP8, or 1 DP rank of TP16, uses 16 NPUs).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-deployment)** chapter.

|Scenario|Configuration|NPUs (per node)|TP|DP (per node)|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|Low Latency 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|Low Latency 128K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|16|1|8|8192|135000|3|
|High Throughput 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|16|8|2|16|8192|66000|3|
|High Throughput (A3)|PD — Server-P Node, [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|16|8|2|64|8192|133120|1|
|High Throughput (A3)|PD — Server-D Node, [Prefill-Decode Disaggregation](#5113-prefill-decode-disaggregation)|16|1|16|32|164|133120|5|
|Long Context 1M (A3)|PD — Server-P Node, [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|16|8|2|8|16384|1040000|1|
|Long Context 1M (A3)|PD — Server-D Node, [PD Disaggregation 1M Deployment](#523-pd-disaggregation-1m-deployment)|16|8|2|32|128|1040000|3|

> On PD decode nodes, `--max-num-batched-tokens` is configured as `(MTP Spec Num + 1) × Max Num Seqs` — each sequence generates one target token plus the speculated MTP tokens per decode step.
>
> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#5-deployment).

#### 9.1.3 Table 3: Performance-Related Parameter Tuning Guide

|Parameter|Low Latency|High Throughput|Long Context|Description|
|---------|-----------|---------------|-------------|-----------|
|`--max-num-batched-tokens`|Lower (4096–8192)|Higher for prefill (8192)|Higher for prefill (16384)|Controls batch size per step. Lower values reduce per-step latency; higher values improve prefill throughput.|
|`--gpu-memory-utilization`|0.92–0.95|0.90–0.95|0.75–0.93|NPU memory fraction. 1M scenarios must reserve memory for the huge KV cache, so use lower values (0.75–0.80 on prefill/co-located).|
|`num_speculative_tokens` (MTP)|3–5|3–5|1 (prefill) / 3 (decode)|MTP speculation count. Higher values improve decode throughput at the cost of memory for the draft model KV cache. Use `1` on prefill nodes in PD mode.|
|`enable_dsa_cp`|Optional|Enable (prefill nodes)|Enable (prefill nodes)|DSA context parallelism accelerates long-context prefill.|
|`enable_sparse_sfa_c8` / `enable_sparse_li_c8`|`false` / `true`|`true` / `true` (PD)|`false` / `true`|`enable_sparse_sfa_c8` is an experimental SFA optimization for the C8 quantized model; do not combine it with DCP in v0.23.0 because of known issues. `enable_sparse_li_c8` is independent and remains recommended.|
|`enable_balance_scheduling`|Enable (single-node)|Enable|Disable in PD mode|Improves output throughput and reduces TPOT in the v1 scheduler. TTFT may degrade in some scenarios; not recommended when Prefill-Decode is separated.|
|`VLLM_ASCEND_ENABLE_MLAPO`|—|1 (A2 P/D nodes)|—|Fusion operator that significantly improves performance but consumes more NPU memory. On A3 used on decode nodes only.|
|`cudagraph_mode`|FULL_DECODE_ONLY|FULL_DECODE_ONLY (decode)|FULL_DECODE_ONLY (decode)|Graph capture for the decode phase only. Prefill nodes in PD mode use `--enforce-eager` instead.|

### 9.2 Tuning Guidelines

For general performance tuning methods, refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md).

For detailed feature descriptions and configuration options, refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md).

For environment variable descriptions and constraints, refer to [envs.py](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/envs.py).

## 10 FAQ

- **Q: How to enable function calling for GLM-5.2?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
