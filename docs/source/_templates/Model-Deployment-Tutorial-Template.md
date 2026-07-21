# Technical Documentation Template for Deployment Tutorials Based on the XXX Model

<p align="center">
  <a href="Model-Deployment-Tutorial-Template.md"><b>English</b></a> | <a href="Model-Deployment-Tutorial-Template.zh.md"><b>中文</b></a>
</p>

This template is based on deployment tutorials for models such as DeepSeek-V3.2 and Qwen-VL-Dense, and is intended to serve as a reference for technical documentation writing. Users can systematically construct relevant technical documentation by following the guidelines provided in this template.

## 1 Introduction

**Content Writing Requirements:**

- Provide a one-sentence description of the model's basic architecture, core features, and primary application scenarios.
- Provide a one-sentence description of the document's purpose and the objectives to be achieved.
- Specify the version of vLLM-Ascend used in the document and the version support status of the model.

**Example 1: Model Introduction**

DeepSeek-V3.2 is a sparse attention model. Its core architecture is similar to that of DeepSeek-V3.1, but it employs a sparse attention mechanism, aiming to explore and validate optimization solutions for training and inference efficiency in long-context scenarios.

**Example 2: Document Purpose**

This document will demonstrate the primary validation steps for the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, as well as accuracy and performance evaluation.

**Example 3: Version Information**

This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (XXX) is fully supported in this version, and all **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use the latest release candidate or official version.

## 2 Supported Features

This section introduces the features supported by the model, including supported hardware, quantization methods, data parallelism, long-sequence features, etc.

**Content Writing Requirements:**

- Present the support status of models and features in a table format.
- Or provide cross-references with jump links (recommended).

**Example 1: Feature Support List**

| Model Name | Support Status | Remarks | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Caching | LoRA | Speculative Decoding | Asynchronous Scheduling | Tensor Parallelism | Pipeline Parallelism | Expert Parallelism | Data Parallelism | Prefill-Decode Separation | Segmented ACL Graph Execution | Full ACL Graph Execution | Max Model Length | Documentation |
| ------ | ---------- | ------ | ------ | ---------- | ------ | ------------ | -------------- | ------ | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ------------------- | ----------- | ----------- | ---------- | ---- |
| DeepSeek V3/3.1 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
| DeepSeek V3.2 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md) |
| Qwen3 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |

>**Note**: This is a simplified example. Please refer to the complete feature matrix for the full table.

**Example 2: Reference Citation**

Please refer to the [Supported Features List](../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

**Content Writing Requirements:** Describe the hardware resources, software environment, and model files required for deployment.

**Example:**

- `DeepSeek-V3.2-Exp-W8A8` (Quantized version): requires 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Model Weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8)
- `DeepSeek-V3.2-w8a8` (Quantized version): requires 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Model Weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/)

It is recommended to download the model weight to a shared directory across multiple nodes.

### 3.2 Verify Multi-node Communication (Optional)

**Example:**

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

**Content Writing Requirements:**

- Provide specific installation steps and commands (parameters should be explained with meaning, value range, units, etc.).
- **Version Number Writing Specification:** Prefer using placeholders (values are centrally configured). If a fixed value is used and it differs from the documented validation version, a comment MUST be added stating: "Please replace with your actual version."
- Provide verification commands and expected status: guide users to check the installation result by executing commands (e.g., docker ps), specifying success criteria such as status codes or output characteristics.
- When multiple hardware series (e.g., A3/A2 series) are involved, the tab markup syntax MUST be used to present them in separate tabs, with tabs ordered by newer hardware series first.

### 4.1 Docker Image Installation

**Example: (Applicable to v0.24.0rc and later versions (MkDocs syntax). For v0.23.0 and earlier versions (Sphinx framework), please refer to the Sphinx official documentation for more syntax details.)**

=== "A3 series"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run ...
    ```

=== "A2 series"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker run ...
    ```

### 4.2 Source Code Installation

**Example:** Omitted

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

**Content Writing Requirements:**

- Describe the architectural characteristics and applicable scenarios of single-node deployment.
- Provide startup command templates and key parameter descriptions.
- Provide service verification methods (e.g., curl commands) and expected results, specifying success indicators (e.g., 200 OK).
- Below the startup command, provide guidance on common issues; if already described in the public FAQ, a direct link may be provided.
- When multiple hardware series (e.g., A3/A2 series) are involved, the tab markup syntax MUST be used to present them in separate tabs, with tabs ordered by newer hardware series first.

**Example:**

Single-node deployment completes both Prefill and Decode within the same node, suitable for XXX scenarios.

Startup Command:

```bash
# Omitted
```

Common Issues Tip: If you encounter XXX issues, please refer to the [Public FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) for troubleshooting.

Service Verification:

```bash
# Omitted
```

Expected Result: Omitted (fill in according to actual output).

### 5.2 Multi-Node PD Separation Deployment

**Content Writing Requirements:**

- Describe the principles of PD separation architecture and applicable scenarios.
- Provide startup procedures, key configurations, and **deployment verification instructions**, and indicate performance metrics.
- Below the startup command, provide guidance on common issues; if already described in the public FAQ, a direct link may be provided.
- When multiple hardware series (e.g., A3/A2 series) are involved, the tab markup syntax MUST be used to present them in separate tabs, with tabs ordered by newer hardware series first.

**Example:** Omitted

### 5.3 Special Deployment Modes (Optional)

**Content Writing Requirements:**

- If the model features non‑standard deployment modes (e.g., offline batch processing for embedding models, low‑latency online serving for reranker models), the corresponding deployment solutions must be explicitly documented.
- Section 5.1 and 5.2 above can be referenced for extension.

## 6 Functional Verification

**Content Writing Requirements:**

- Guide users on how to test the basic functionality of the model through simple interface calls after the service is started.
- Provide expected results, specifying success indicators (e.g., HTTP 200, JSON response containing a choices field).

**Example:**

After the service is started, the model can be invoked by sending a prompt:

```shell
curl http://<node0_ip>:<port>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek_v3.2",
        "prompt": "The future of AI is",
        "max_tokens": 50,
        "temperature": 0
    }'
```

Expected Result: Omitted (fill in according to actual output).

## 7 Accuracy Evaluation

**Content Writing Requirements:** Introduce standardized methods and tools for evaluating model output quality (accuracy). Two accuracy evaluation methods are provided below as examples; alternatively, provide direct links to existing documentation.

### Using AISBench

For details, please refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md).

### Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example test dataset, run the accuracy evaluation for `DeepSeek-V3.2-W8A8` in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation.

```shell
lm_eval \
  --model local-completions \
  --model_args model=/root/.cache/Eco-Tech/DeepSeek-V3.2-w8a8-mtp-QuaRot,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gsm8k \
  --output_path ./
```

## 8 Performance Evaluation

Omitted. Requirements are the same as for Accuracy Evaluation.

## 9 Performance Tuning

### 9.1 Recommended Configurations

**Content Writing Requirements:**

Provide recommended configurations for three typical scenarios (long context, low latency, high throughput). Clearly state that the configurations are not globally optimal and guide users to perform tuning based on their actual circumstances.

**Example:**

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

#### Table 1: Scenario Overview

| Scenario | Deployment Mode | *Total NPUs | Weight Version | Key Considerations |
| ---------- | ---------------- | ------------- | ---------------- | ------------------------ |
| High Throughput<br>(32K context → 1K output) | 1P1D deployment | 16 (A3) | glm5.1w4a8 | For short-sequence high throughput, try adjusting xxx parameters |
| Long Context | | | | |
| Low Latency | | | | |

> `*Total NPUs` indicates the total number of NPUs used across all nodes.

#### Table 2: Detailed Node Configuration

| Scenario | Configuration | NPUs | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Speculation Num | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
| ---------- | --------------- | ------- | ---- | ---- | ---- | ------------- | -------------------- | --------------------- | ----------- | ----------- | -------------- | ------------------ |
| High Throughput (32K→1K) | Server-P Node / Single Machine | 8 | 8 | 2 | 32 | 4096 | 30k | 3 | Off | On | On | On |
| High Throughput (32K→1K) | Server-D Node | 8 | 2 | 8 | 8 | 4096 | 30k | 12 | Off | On | Off | On |
| Long Context | Server-P Node / Single Machine | | | | | | | | | | | |
| Long Context | Server-D Node | | | | | | | | | | | |
| Low Latency | Server-P Node / Single Machine | | | | | | | | | | | |
| Low Latency | Server-D Node | | | | | | | | | | | |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in Chapter 5.

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

**Content Writing Requirements:**

If no special tuning is involved, directly provide a feature combination table and a link to the public performance tuning documentation.

**Example:**

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Guide](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

#### 9.2.2 Model-Specific Optimizations (Optional)

**Documentation Requirements:**

If the model has specific optimizations, summarize the key optimization techniques and tuning experience for this model.

**Example:**

#### Optimizations Enabled by Default

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| --------- | --------- | --------- |
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero-tensor operations in Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### Optimizations That Require Explicit Enabling

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
| --------------------- | -------------------- | ----------------- | ------------------- | ----------- |
| FlashComm_v1 | High-concurrency, Tensor Parallelism (TP) scenarios | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | Threshold protection: Only takes effect when the actual number of tokens exceeds the threshold to avoid performance degradation in low-concurrency scenarios |
| Matmul-ReduceScatter Fusion | Large-scale distributed environments | Automatically enabled after enabling FlashComm_v1 | Fuses matrix multiplication and Reduce-Scatter operations to achieve pipelined parallel processing | Same as FlashComm_v1, has threshold protection |

## 10 FAQ

**Content Writing Requirements:**

- Add a note at the beginning of the section: For common environment, installation, and general parameter issues, please refer to the [Public FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html); this chapter only covers model-specific issues.
- For **model-specific issues**, provide the following elements: problem phenomenon description, cause analysis, and solution measures.
