# GLM-5.3-Flash

## 1 Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the first natively multimodal model in the GLM-5 series. Built on a hybrid architecture that combines sparse and linear attention for the first time in the GLM series, it adopts Manifold-Constrained Hyper-Connections (mHC) and is trained on a 30T-token multimodal pre-training corpus. With 320B total parameters and only 18B active parameters, it outperforms GLM-5.2 across benchmarks and real-world workloads at one-tenth the price, while approaching Claude Opus 4.8 on coding and agentic benchmarks. GLM-5.3-Flash also supports controlling the thinking budget through the `reasoning_effort` parameter (`low`, `high`, `max`).

This document will show the main verification steps of the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, accuracy and performance evaluation.

## 2 Supported Features

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get the feature's configuration.

## 3 Prerequisites

### 3.1 Model Weight

- `GLM-5.3-Flash-w8a8 (Ascend950DT mxfp8 Quantized)`: requires 1 Ascend950DT (96GB × 8) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).
- `GLM-5.3-Flash-w8a8`: requires 1 Atlas 800 A3 (64GB × 16) node.[Download model weight](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-Flash-w8a8).

- You can use [msmodelslim](https://gitcode.com/Ascend/msmodelslim) to quantize the model directly.

It is recommended to download the model weight to the shared directory of multiple nodes, such as `/root/.cache/`

## 4 Installation

### 4.1 Docker Image Installation

=== "Ascend950DT series"

    Start the docker image on each node.

    ```shell

    ```

=== "A3 series"

    Start the docker image on each of your nodes.

    ```shell

    ```


## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

#### 5.1.1 Ascend950DT

- Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 Ascend950DT (96GB × 8) .

Run the following script to execute online inference.

```shell

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.3-Flash.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w8a8 weights. 
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable function calling for GLM-5.3-Flash.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.3-Flash does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.

#### 5.1.2 Atlas 800 A3

- Quantized model `GLM-5.3-Flash-w8a8` can be deployed on 1 A3 (64GB × 16) .

Run the following script to execute online inference.

```shell

```

Key Parameter Descriptions:

Only the key parameters specific to this model/scenario are described below. `max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario.

**Model-specific parameters:**

- `--enable-expert-parallel`: Must be enabled for the MoE architecture of GLM-5.3-Flash.
- `--quantization ascend`: Enables Ascend quantization for the w8a8 quantized weights.
- `--data-parallel-size 2` / `--tensor-parallel-size 8`: DP2 TP8 parallelism layout, recommended to balance memory capacity and compute efficiency for the w8a8 weights. 
- `--tool-call-parser glm47` / `--reasoning-parser glm45` / `--enable-auto-tool-choice`: Enable function calling for GLM-5.3-Flash.
- `--speculative-config '{"num_speculative_tokens": 3, "method": "deepseek_mtp", "enforce_eager": true}'`: Enables Multi-Token Prediction (MTP) speculative decoding with the DeepSeek-style MTP draft head of GLM-5.3-Flash. `num_speculative_tokens` (3-5) controls how many tokens are speculated per step; `enforce_eager: true` is required because GLM-5.3-Flash does not support graph-mode speculative decoding.
- `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`: Enables graph capture for the decode phase only, improving decode performance by reducing kernel launch overhead.

## 6 Functional Verification

Once your server is started, you can query the model with input prompts:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "glm-52",#todo
        "prompt": "The future of AI is",
        "max_completion_tokens": 50,
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
|High Throughput<br>(128K input)|Dual-Node Co-Located (A3), [Multi-Node Co-Located Deployment](#5112-multi-node-co-located-deployment)|32 (A3)|w4a8c8|dp4 tp8, fused MC2, MTP3, max-num-seqs 16, max-model-len 66000, DSA CP|

#### 9.1.2 Table 2: Detailed Node Configuration

> The TP/DP columns show the values **per node** as configured in the Deployment scripts (a node hosting 2 DP ranks of TP8, or 1 DP rank of TP16, uses 16 NPUs).

**Notice:**
`max-model-len` and `max-num-seqs` need to be set according to the actual usage scenario. For other settings, please refer to the **[Deployment](#5-deployment)** chapter.

|Scenario|Configuration|NPUs (per node)|TP|DP (per node)|Max Num Seqs|Max Num Batched Tokens|Max Model Len|MTP Spec Num|
|--------|-------------|-----|--|--|------------|----------------------|--------------|-------------|
|High Throughput 64K (A3)|Dual-Node (per node), [Multi-Node Co-Located Deployment]

> On PD decode nodes, `--max-num-batched-tokens` is configured as `(MTP Spec Num + 1) × Max Num Seqs` — each sequence generates one target token plus the speculated MTP tokens per decode step.
>
> For complete startup commands and detailed parameter descriptions, please refer to the deployment examples and Key Parameter Descriptions in [Deployment](#5-deployment).

## 10 FAQ

- **Q: How to enable function calling for GLM-5.3-Flash?**

  A: Please add following configurations in vLLM startup command

  ```shell
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  ```
