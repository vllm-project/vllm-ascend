# Deployment Tutorial Template Based on the XXX Model

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

This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (XXX) is fully supported in this version, and all **v0.13.0 and later versions** can run stably. To use the latest features (e.g., PD separation, MTP), it is recommended to use v0.13.0 or a later version.

## 2 Feature Matrix

This section introduces the features supported by the model, including supported hardware, quantization methods, data parallelism, long-sequence features, etc.

**Content Writing Requirements:**

- Present the support status of models and features in a table format.
- Alternatively, provide references with hyperlinks.

**Example 1: Feature Support List**

| Model Name | Support Status | Remarks | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Caching | LoRA | Speculative Decoding | Asynchronous Scheduling | Tensor Parallelism | Pipeline Parallelism | Expert Parallelism | Data Parallelism | Prefill-Decode Separation | Segmented ACL Graph Execution | Full ACL Graph Execution | Max Model Length | MLP Weight Prefetch | Documentation |
| ------ | ---------- | ------ | ------ | ---------- | ------ | ------------ | -------------- | ------ | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ------------------- | ----------- | ----------- | ------------- | ------------- | ---------- |
| DeepSeek V3/3.1 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k | | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
| DeepSeek V3.2 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md) |
| DeepSeek R1 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k | | [DeepSeek R1](../../tutorials/models/DeepSeek-R1.md) |
| Qwen3 | ✅ | | ✅ | Atlas 800I A2:<br>Minimum card requirement: xx | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k | ✅ | [Qwen3](../../tutorials/models/Qwen3-Dense.md) |

**Note**: This is a simplified example. Please refer to the complete feature matrix for the full table.

**Example 2: Reference Citation**

Please refer to the [Supported Features List](../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Environment Preparation

### 3.1 Model Weight

**Content Writing Requirements:** Describe the hardware resources, software environment, and model files required for deployment.

**Example:**

| Model Version | Hardware Requirements | Download Link |
| ---------- | ---------- | ---------- |
| DeepSeek-V3.2-Exp (BF16) | 2×Atlas 800 A3 (64G×16)<br>4×Atlas 800 A2 (64G×8) | [Model Weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-BF16) |
| DeepSeek-V3.2-Exp-w8a8 (Quantized) | 1×Atlas 800 A3 (64G×16)<br>2×Atlas 800 A2 (64G×8) | [Model Weight](https://modelers.cn/models/Modelers_Park/DeepSeek-V3.2-Exp-w8a8) |
| DeepSeek-V3.2-w8a8 (Quantized) | 1×Atlas 800 A3 (64G×16)<br>2×Atlas 800 A2 (64G×8) | [Model Weight](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/) |

### 3.2 Verify Multi-node Communication (Optional)

**Example:**

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../installation.md#verify-multi-node-communication) guide for communication verification.

## 4 Installation

**Content Writing Requirements:**

- Provide specific steps and startup commands, covering both single-node and multi-node configurations.
- Provide explanations for parameters, including meaning, value range, and units.
- Specify the basic environment variables and communication environment variables that need to be enabled, with explanations including meaning, value range, and units.

### 4.1 Docker Image Installation

**Example:** Omitted

### 4.2 Source Code Installation

**Example:** Omitted

## 5 Online Service Deployment

### 5.1 Single-Node Online Deployment

**Content Writing Requirements:**

- Describe the architectural characteristics and applicable scenarios of single-node deployment.
- Provide startup command templates and key parameter descriptions.
- Provide service verification methods.

**Example:**

Single-node deployment completes both Prefill and Decode within the same node, suitable for XXX scenarios.

Startup Command:

```bash
# Omitted
```

Service Verification:

```bash
# Omitted
```

### 5.2 Multi-Node PD Separation Deployment

**Content Writing Requirements:**

- Describe the principles of PD separation architecture and applicable scenarios.
- List prerequisites (network, storage, permissions).
- Provide script frameworks and key configuration item descriptions.
- Specify node role division and startup procedures.
- Indicate performance metrics.

**Example:** Omitted

### 5.3 Special Deployment Modes (Optional)

**Content Writing Requirements:**

- If the model features non‑standard deployment modes (e.g., offline batch processing for embedding models, low‑latency online serving for reranker models), the corresponding deployment solutions must be explicitly documented.
- Section 5 "Online Service Deployment" provides examples for single‑node online service deployment and multi‑node PD‑separated deployment, which can be referenced and extended.

## 6 Functional Verification

**Content Writing Requirements:** Guide users on how to test the basic functionality of the model through simple interface calls after the service is started.

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

## 8 Performance

Omitted. Requirements are the same as for Accuracy Evaluation.

## 9 Best Practices

**Content Writing Requirements:**

Provide recommended configurations for three scenarios (long sequence, low latency, high throughput) for each model that can achieve optimal performance, but do not provide specific performance data.

## 10 Performance Tuning (Optional)

**Content Writing Requirements:**

- Summarize key optimization techniques and parameter tuning experiences for the model to help users achieve optimal performance in specific scenarios. Include optimization technique descriptions, enablement methods, parameter tuning recommendations, and typical configuration examples.
- Hyperlinks to the features guide may be used to allow users to view detailed descriptions of specific features.

### 10.1 Key Optimization Points

In this section, we will introduce the key optimization points that can significantly improve the performance of the XX model. These techniques aim to improve throughput and efficiency in various scenarios.

#### 10.1.1 Basic Optimizations

**Example:**

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| --------- | --------- | --------- |
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero-tensor operations in Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

#### 10.1.2 Advanced Optimizations (Require Explicit Enablement)

**Example:**

| Optimization Technique | Technical Principle | Enablement Method | Applicable Scenarios | Precautions |
| --------- | --------- | --------- | --------- | --------- |
| FlashComm_v1 | Decomposes traditional Allreduce into Reduce-Scatter and All-Gather, reducing RMSNorm computation dimensions | `export VLLM_ASCEND_ENABLE_FLASHCOMM1=1` | High-concurrency, Tensor Parallelism (TP) scenarios | Threshold protection: Only takes effect when the actual number of tokens exceeds the threshold to avoid performance degradation in low-concurrency scenarios |
| Matmul-ReduceScatter Fusion | Fuses matrix multiplication and Reduce-Scatter operations to achieve pipelined parallel processing | Automatically enabled after enabling FlashComm_v1 | Large-scale distributed environments | Same as FlashComm_v1, has threshold protection |
| Weight Prefetch | Utilizes vector computation time to prefetch MLP weights into L2 cache in advance | `export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1` | MLP-intensive scenarios (Dense models) | Requires coordination with prefetch buffer size adjustment |
| Asynchronous Scheduling | Non-blocking task scheduling to improve concurrent processing capability | `--async-scheduling` | Large-scale models, high-concurrency scenarios | Should be used in coordination with FullGraph optimization |

### 10.2 Optimization Highlights

**Content Writing Requirements:**

Summarize the most noteworthy optimization points during the actual tuning process, distill core experiences, and provide readers with tuning ideas for getting started quickly.

**Example:**

During the actual tuning process, the following points are most critical for performance improvement: The prefetch buffer size needs to be determined through empirical measurement to find the optimal overlap between computation and prefetching; the setting of `max-num-batched-tokens` needs to balance throughput and video memory to avoid excessive chunking or OOM risk; `cudagraph_capture_sizes` must be manually specified and cover the target concurrency; when FlashComm_v1 is enabled, it is also necessary to ensure that the values are multiples of TP; `pa_shape_list` is a temporary tuning parameter that only takes effect for specific batch sizes, requiring attention to version evolution for timely adjustments. The coordinated configuration of the above parameters and environment variables is key to achieving extreme performance.

## 11 FAQ

**Content Writing Requirements:**

 Provide solutions to common problems, including but not limited to problem phenomenon description, cause analysis, and solution measures.
