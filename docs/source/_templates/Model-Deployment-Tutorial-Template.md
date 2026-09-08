# Model Deployment Tutorial Template

<p align="center">
  <a href="Model-Deployment-Tutorial-Template.md"><b>English</b></a> | <a href="Model-Deployment-Tutorial-Template.zh.md"><b>中文</b></a>
</p>

**Document Version**: v2.0<br>
**Updated**: 2026-08-27

|   Version  |     Date      | Change Description |
|------------|---------------|--------------------|
|    v2.0    |    2026-08    | Added 0day/POC and weight path constraint specifications; restructured Chapter 9; <br>supplemented echo output examples and hardware support descriptions; fixed documentation issues. |
|    v1.0    |    2026-03    | Initial version creation |

This template aims to serve as a reference for writing model deployment documentation. Users can follow the guidelines provided in the template to systematically complete the construction of relevant technical documentation.

**Title conventions:**

- **Official releases**: Named after the model, e.g., `DeepSeek-V3.2`
- **Experimental releases (0day/POC)**: Named after the model with the `(Experimental)` suffix, e.g., `Kimi-K3 (Experimental)`

## 1 Introduction

**Content Writing Requirements:**

- Provide a single paragraph that describes the model's basic architecture, core features, and primary application scenarios, along with the purpose and intended outcomes of this document.
- Specify the version of vLLM-Ascend used in the document and the version support status of the model,**If 0day or POC models are involved, a relevant description must be provided to explain the model's adaptation status.**

**Example 1: Model Introduction**

DeepSeek-V3.2 is a sparse attention model. Its core architecture is similar to that of DeepSeek-V3.1, but it employs a sparse attention mechanism, aiming to explore and validate optimization solutions for training and inference efficiency in long-context scenarios.This document will demonstrate the primary validation steps for the model, including supported features, feature configuration, environment preparation, single-node and multi-node deployment, as well as accuracy and performance evaluation.

**Example 2: Version Information and Adaptation Status Description**

- **Official releases**: This document is validated and written based on **vLLM-Ascend v0.13.0**. The current model (XXX) is fully supported in this version and can run stably on **v0.13.0 and later versions**. For access to the latest features (e.g., PD separation, MTP, etc.), it is recommended to use the latest release candidate or official release.

- **Experimental releases (0day/POC)**: The Kimi K3 support in vLLM-Ascend 0.26.0rc is an initial experimental release. It is intended solely for evaluation and validation following the fixed deployment configuration outlined in this guide. Supported scenarios, performance, and configuration interfaces are subject to change in subsequent releases; please do not interpret this guide as a production support commitment.

## 2 Supported Features

This section introduces the features supported by the model, including supported hardware, quantization methods, data parallelism, long-sequence features, etc.

**Content Writing Requirements:**

- Present the support status of models and features in a table format.
- Or provide cross-references with jump links (recommended).

**Example 1: Feature Support List**

| Model Name | Support Status | Remarks | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Caching | LoRA | Speculative Decoding | Asynchronous Scheduling | Tensor Parallelism | Pipeline Parallelism | Expert Parallelism | Data Parallelism | Prefill-Decode Separation | Segmented ACL Graph Execution | Full ACL Graph Execution | Max Model Length | Documentation |
| ------ | ---------- | ------ | ------ | ---------- | ------ | ------------ | -------------- | ------ | ---------- | ---------- | ---------- | ------------ | ---------- | ---------- | ------------------- | ----------- | ----------- | ---------- | ---- |
| DeepSeek-V3.2 | ✅ | | ✅ | Atlas 800I A2 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | [DeepSeek-V3.2](../tutorials/models/DeepSeek-V3.2.md) |
| Qwen3-Dense | ✅ | | ✅ | Atlas 800I A2 | ✅ | ✅ | ✅ | | | ✅ | ✅ | | | ✅ | | ✅ | ✅ | 128k | [Qwen3-Dense](../tutorials/models/Qwen3-Dense.md) |

>**Note**: This is a simplified example. Please refer to the complete feature matrix for the full table.

**Example 2: Reference Citation**

Please refer to the [Supported Models](../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../user_guide/feature_guide/index.md) for feature configuration information.

## 3 Prerequisites

### 3.1 Model Weight

**Content Writing Requirements:**

- Describe the hardware resources, software environment, and model files required for deployment.
- Weight download links from both `HuggingFace` and `ModelScope` must be provided.
- Path description: After providing the download link, a path description example must be included, clearly reminding users to note the actual storage path and explaining that this path will be used in subsequent deployment commands.

**Example:**

|  Weight Version | Hardware Requirements | Download Links |
|-----------------|-----------------------|----------------|
| `DeepSeek-V3.2-Exp-W8A8` |  1 Atlas 800 A3 (64GB × 16) node or 2 Atlas 800 A2 (64GB × 8) nodes | [Modelscope](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-Exp-W8A8) \| [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |
| `DeepSeek-V3.2-W8A8` | 1 Atlas 800 A3 (64GB × 16) node or 2 Atlas 800 A2 (64GB × 8) nodes | [Modelscope](https://www.modelscope.cn/models/vllm-ascend/DeepSeek-V3.2-W8A8/) \| [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) |

>**Path description:** Please download the model weights to a directory of your choice and record this path. For example: `/root/.cache/modelscope/hub/models/vllm-ascend/DeepSeek-V3.2-W8A8`. In subsequent deployment commands, the placeholder `<YOUR_MODEL_PATH>` will be used; please replace it with the path you have recorded here.

### 3.2 Verify Multi-node Communication (Optional)

**Example:**

If multi-node deployment is required, please follow the [Verify Multi-node Communication Environment](../getting_started/installation.md#installation-multi-node-interconnect) guide for communication verification.

## 4 Installation

**Content Writing Requirements:**

- Provide specific installation steps and commands (parameters should be explained with meaning, value range, units, etc.).
- **Version Number Writing Specification:** Prefer using placeholders (values are centrally configured). If a fixed value is used and it differs from the documented validation version, a comment MUST be added stating: "Please replace with your actual version."
- Provide verification commands and expected status: guide users to check the installation result by executing commands (e.g., docker ps), specifying success criteria such as status codes or output characteristics,**and provide a complete example of the echo output**.
- If the model supports only a single hardware series (e.g., Atlas 300I DUO only), explicitly state this at the beginning of the installation section. If multiple hardware series are supported(e.g., A3/A2 series), use tabbed syntax to present them separately, with newer models listed first. For syntax differences between MkDocs and Sphinx frameworks, refer to [Syntax Supplement](template-supplement.md#3-tabs).

### 4.1 Docker Image Installation

**Example: (For MkDocs + Material tab syntax (`=== "label"`) and Sphinx + MyST-Parser tab syntax (`::::{tab-item}`), please refer to the [Syntax Supplement](template-supplement.md#3-tabs).)**

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

## 5 Online Service Deployment {: #5-online-service-deployment }

**Content Writing Requirements (Applicable to Subsections 5.1, 5.2, and 5.3):**

- Provide troubleshooting guidance below the startup commands. If the issue is already covered in the public FAQ, a direct link to it may be used.
- **Model path specification**: In deployment commands, the model path must use the variable placeholder `<YOUR_MODEL_PATH>`, with a comment reminding users to replace this placeholder with the path recorded in Section 3.1.
- If the model supports only a single hardware series (e.g., Atlas 300I DUO only), explicitly state this at the beginning of the installation section. If multiple hardware series are supported(e.g., A3/A2 series), use tabbed syntax to present them separately, with newer models listed first. For syntax differences between MkDocs and Sphinx frameworks, refer to [Syntax Supplement](template-supplement.md#3-tabs).

### 5.1 Single-Node Online Deployment

**Content Writing Requirements:**

- Describe the architectural characteristics and applicable scenarios of single-node deployment.
- Provide startup command templates and key parameter descriptions.
- Provide service verification methods (e.g., curl commands) and expected results, specifying success indicators (e.g., 200 OK),**and provide a complete example of the echo output**.

**Example:**

Single-node deployment completes both Prefill and Decode within the same node, suitable for XXX scenarios.

Startup Command:

```bash
# Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1
vllm serve <YOUR_MODEL_PATH> \
  --port 8000 \
  --served-model-name DeepSeek-V3.2-W8A8 \
```

Common Issues Tip: If you encounter XXX issues, please refer to the [Public FAQs](../faqs.md) for troubleshooting.

Service Verification:

```bash
# Omitted
```

Expected Result: Omitted (fill in according to actual output).

### 5.2 Multi-Node PD Separation Deployment

**Content Writing Requirements:**

- Describe the principles of PD separation architecture and applicable scenarios.
- Provide startup procedures, key configurations, and **deployment verification instructions**, and indicate performance metrics.

**Example:** Omitted

### 5.3 Special Deployment Modes (Optional)

**Content Writing Requirements:**

- If the model features non‑standard deployment modes (e.g., offline batch processing for embedding models, low‑latency online serving for reranker models), the corresponding deployment solutions must be explicitly documented.
- Section 5.1 and 5.2 above can be referenced for extension.

## 6 Functional Verification

**Content Writing Requirements:**

- Guide users on how to test the basic functionality of the model through simple interface calls after the service is started.
- Provide expected results, specifying success indicators (e.g., HTTP 200, JSON response containing a choices field),**and provide a complete example of the echo output**.

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

### 7.1 Using AISBench

For details, please refer to [Using AISBench](../developer_guide/evaluation/using_ais_bench.md).

### 7.2 Using Language Model Evaluation Harness

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

**Content Writing Requirements:** The requirements are the same as the accuracy evaluation in Chapter 7. Basic command examples and complete echo output must be provided.
**Example:** Omitted.

## 9 Performance Tuning

### 9.1 Recommended Configurations

**Content Writing Requirements:**

Provide recommended configurations for three typical scenarios (long context, low latency, high throughput). Clearly state that the configurations are not globally optimal and guide users to perform tuning based on their actual circumstances.**For the detailed configuration tables in the examples below, present only the columns applicable to your actual use case.**

**Example:**

> **Note**: The following configurations are validated in specific test environments and are for reference only. The optimal configuration depends on factors such as maximum input/output length, prefix cache hit rate, precision requirements, and deployment machine ratios. It is recommended to refer to Section 9.2 for tuning based on actual conditions.

**Detailed Configuration**

| Scenario                 | Configuration                  | NPUs | Weight Version | TP | DP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | MTP Speculation Num | FUSED_MC2 | EP Switch | FC+CP Switch | Async Scheduling |
|--------------------------|--------------------------------|------|----------------|----|----|--------------|------------------------|---------------|---------------------|-----------|-----------|--------------|------------------|
| High Throughput (32k→1k) | Server-P Node / Single Machine |   8  |  GLM5.1 W4A8   | 8  | 2  |      32      |          4096          |       30k     |           3         |     Off   |     On    |      On      |  On  |
| High Throughput (32k→1k) | Server-D Node                  |   8  |  GLM5.1 W4A8   | 2  | 8  |      8       |          4096          |       30k     |           12        |     Off   |     On    |      Off     |  On  |
| Long Context             | Server-P Node / Single Machine |      |                |    |    |              |                        |               |                     |           |           |              |      |
| Long Context             | Server-D Node                  |      |                |    |    |              |                        |               |                     |           |           |              |      |
| Low Latency              | Server-P Node / Single Machine |      |                |    |    |              |                        |               |                     |           |           |              |      |
| Low Latency              | Server-D Node                  |      |                |    |    |              |                        |               |                     |           |           |              |      |

> For complete startup commands and parameter descriptions, please refer to the deployment examples in Chapter 5.

### 9.2 Tuning Guidelines

#### 9.2.1 Model-Specific Optimizations

**Documentation Requirements:**

If the model has specific optimizations, summarize the key optimization techniques and tuning experience for this model.

**Example:**

**Optimizations Enabled by Default**

The following optimizations are enabled by default and require no additional configuration:

| Optimization Technique | Technical Principle | Performance Benefit |
| ---------------------- | ------------------- | ------------------- |
| Rope Optimization | The cos_sin_cache and indexing operations of positional encoding are executed only in the first layer, and subsequent layers reuse them directly | Reduces redundant computation during the decoding phase, accelerating inference |
| AddRMSNormQuant Fusion | Merges address-wise multi-scale normalization and quantization operations into a single operator | Optimizes memory access patterns, improving computational efficiency |
| Zero-like Elimination | Removes unnecessary zero-tensor operations in Attention forward pass | Reduces memory footprint, improves matrix operation efficiency |
| FullGraph Optimization | Captures and replays the entire decoding graph at once using `compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"}` | Significantly reduces scheduling latency, stabilizes multi-device performance |

**Optimizations That Require Explicit Enabling**

| Optimization Technique | Applicable Scenarios | Enablement Method | Technical Principle | Precautions |
| ---------------------- | -------------------- | ----------------- | ------------------- | ----------- |
| Matmul-ReduceScatter Fusion | Large-scale distributed environments | Automatically enabled after enabling sequence parallelism | Fuses matrix multiplication and Reduce-Scatter operations to achieve pipelined parallel processing | Same as sequence parallelism, has threshold protection |

#### 9.2.2 General Tuning Reference

**Content Writing Requirements:**

If no special tuning is involved, directly provide a feature combination table and a link to the public performance tuning documentation.

**Example:**

Please refer to the [Public Performance Tuning Documentation](../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.
Please refer to the [Feature Matrix](../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

**Content Writing Requirements:**

- Add a note at the beginning of the section: For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../faqs.md); this chapter only covers model-specific issues.
- For **model-specific issues**, provide the following elements: problem phenomenon description, cause analysis, and solution measures.
