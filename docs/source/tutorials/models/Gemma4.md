# Gemma4

## 1 Introduction

Gemma4 is a Gemma-family multimodal model that includes dense and Mixture-of-Experts (MoE) variants, and is suitable for text generation, image understanding, reasoning, and instruction-following scenarios.

This document describes the main validation steps for Gemma4 on Atlas A2, Atlas A3, and 950PR&950DT Products, including supported features, prerequisites, installation, single-node online deployment, functional verification, offline inference, accuracy and performance evaluation, performance tuning, and FAQs.

This document is written based on the latest vLLM Ascend main branch. Gemma4 is supported on Atlas A2, Atlas A3, and 950PR&950DT Products in this version.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model support matrix, including BF16, chunked prefill, automatic prefix caching, tensor parallelism, expert parallelism, and ACLGraph support.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

Gemma4 supports both eager execution and ACLGraph execution on Atlas A2, Atlas A3, and 950PR&950DT Products. For graph execution, `FULL_DECODE_ONLY` can reduce decode-phase dispatch overhead, while `PIECEWISE` is also supported. MXFP4 quantization is supported on 950PR&950DT Products.

## 3 Prerequisites

### 3.1 Model Weight

Download the Gemma4 model weight to a local or shared directory, such as `/root/.cache/`. In the following examples, replace `/root/.cache/path/to/gemma4` with your actual model path.

| Weight version | Description | Recommended hardware | Download |
| -------------- | ----------- | -------------------- | -------- |
| `google/gemma-4-31B-it` | BF16 dense model. | Atlas A2, Atlas A3, or 950PR&950DT Products. Adjust the number of visible NPUs according to the available memory. | [ModelScope](https://www.modelscope.cn/models/google/gemma-4-31B-it) |
| `Eco-Tech/gemma-4-31B-it-w4a4-mxfp4` | MXFP4 quantized dense model. | 1 950DT Products (96GB × 8) node. | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/gemma-4-31B-it-w4a4-mxfp4) |
| `google/gemma-4-26B-A4B-it` | BF16 MoE model with approximately 26 billion total and 4 billion active parameters. | Atlas A2, Atlas A3, or 950PR&950DT Products. Use tensor parallelism or expert parallelism according to the deployment plan. | [ModelScope](https://www.modelscope.cn/models/google/gemma-4-26B-A4B-it) |
| `Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4` | MXFP4 quantized MoE model. | 1 950DT Products (96GB × 8) node. | [ModelScope](https://www.modelscope.cn/models/Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4) |

Both MXFP4 checkpoints contain a `quant_model_description.json` file and require `--quantization ascend` when served. For information about preparing quantized weights, refer to the [Quantization Guide](../../user_guide/feature_guide/quantization.md).

## 4 Installation

### 4.1 Docker Image Installation

You can use the official all-in-one vLLM Ascend Docker image. For 950DT Products, pull the A5 image that matches the current vLLM Ascend version:

```shell
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a5
docker pull ${IMAGE}
```

For other supported hardware and operating systems, select the corresponding image by following [Using a prebuilt image](../../getting_started/installation.md#installation-prebuilt-image). Follow the hardware-specific `docker run` command in the installation guide and mount the directory containing the Gemma4 weights into the container. The prebuilt image includes the matching CANN user-space environment, PyTorch, TorchNPU, vLLM, and vLLM Ascend versions.

After entering the container, verify the environment:

```shell
npu-smi info
python -m pip show vllm vllm-ascend
```

Expected result: `npu-smi info` lists the expected Ascend devices, and both Python packages are installed.

### 4.2 Source Code Installation

Install vLLM Ascend from the main branch:

```shell
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

To verify the source installation, run the following command:

```shell
python -c "import vllm_ascend; print(vllm_ascend.__version__)"
```

Expected result: the command prints the installed vLLM Ascend version without errors.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment runs both Prefill and Decode on the same node, and is suitable for functional verification and single-node serving. The following examples show eager and ACLGraph startup commands. The commands use 4 visible NPUs as an example. Adjust `ASCEND_RT_VISIBLE_DEVICES` and `--tensor-parallel-size` when using a different device count.

> **Note**: In this tutorial, `/root/.cache/path/to/gemma4` is a placeholder. Replace it with your actual Gemma4 model path.

#### Eager Mode

Use eager mode as the baseline configuration for functional verification.

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/root/.cache/path/to/gemma4

vllm serve ${MODEL_PATH} \
  --served-model-name gemma4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --enforce-eager
```

#### ACLGraph Mode

Use ACLGraph mode when graph execution is required. `FULL_DECODE_ONLY` can reduce decode-phase dispatch overhead.

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export MODEL_PATH=/root/.cache/path/to/gemma4

vllm serve ${MODEL_PATH} \
  --served-model-name gemma4 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

#### 950PR&950DT Products MXFP4 Quantized Models

The following commands use TP4DP2 across all 8 NPUs on one 950DT Products node. The quantized checkpoints require `--quantization ascend`.

=== "31B Dense"

    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_MODELSCOPE=True
    export MODEL_PATH=Eco-Tech/gemma-4-31B-it-w4a4-mxfp4

    vllm serve ${MODEL_PATH} \
      --served-model-name gemma4 \
      --trust-remote-code \
      --tensor-parallel-size 4 \
      --data-parallel-size 2 \
      --quantization ascend \
      --max-model-len 32768 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```

=== "26B A4B MoE"

    ```shell
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export VLLM_USE_MODELSCOPE=True
    export MODEL_PATH=Eco-Tech/gemma-4-26B-A4B-it-w4a4-mxfp4

    vllm serve ${MODEL_PATH} \
      --served-model-name gemma4 \
      --trust-remote-code \
      --tensor-parallel-size 4 \
      --data-parallel-size 2 \
      --enable-expert-parallel \
      --quantization ascend \
      --max-model-len 32768 \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
    ```

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [Public FAQs](../../faqs.md) first, and then check the model-specific FAQ in Section 10.

#### Key Parameters

- `--tensor-parallel-size`: sets the tensor parallel size. Adjust it according to model size and available NPU devices.
- `--data-parallel-size`: sets the number of data parallel ranks. The MXFP4 examples use DP2 together with TP4 to use all 8 NPUs.
- `--enable-expert-parallel`: enables expert parallelism for MoE variants when the deployment plan uses EP.
- `--quantization ascend`: enables Ascend quantization. It is required for the two MXFP4 checkpoints listed in Section 3.1 and must be omitted for BF16 weights.
- `--max-model-len`: sets the maximum input plus output length for a single request. Increase it only when enough KV cache is available.
- `--enforce-eager`: enables eager execution for baseline verification.
- `--compilation-config`: configures graph execution. `FULL_DECODE_ONLY` can reduce decode-phase dispatch overhead, and `PIECEWISE` is also supported.
- `--trust-remote-code`: allows vLLM to load model-specific code when required by the model repository.

#### Service Verification

After the service is started, send a request to verify basic model functionality.

```shell
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "prompt": "Explain why graph execution improves decode performance.",
    "max_tokens": 128,
    "temperature": 0
  }'
```

Expected result: the HTTP status is 200 and the JSON response contains a `choices` field with generated text.

### 5.2 Multi-Card Deployment

This tutorial provides single-node multi-card deployment examples for Gemma4 on Atlas A2, Atlas A3, and 950PR&950DT Products. For different model sizes or device counts, adjust `ASCEND_RT_VISIBLE_DEVICES` and `--tensor-parallel-size` according to the available NPUs.

## 6 Functional Verification

After the service is started, Gemma4 can be invoked through OpenAI-compatible APIs.

### 6.1 Completions API

```shell
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "prompt": "Explain why graph execution improves decode performance.",
    "max_tokens": 128,
    "temperature": 0
  }'
```

### 6.2 Chat Completions API

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [
      {"role": "user", "content": "Explain why graph execution improves decode performance."}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Expected result: the request returns HTTP 200, and the JSON response contains a `choices` field with generated text from the model.

### 6.3 Multimodal Chat Completions API

Send an image request to verify Gemma4 multimodal inference:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/tiger.jpeg"
            }
          },
          {
            "type": "text",
            "text": "Describe this image."
          }
        ]
      }
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

Expected result: the request returns HTTP 200, and the response describes the image in the `choices` field.

### 6.4 Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/root/.cache/path/to/gemma4",
    trust_remote_code=True,
    tensor_parallel_size=4,
    compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
)

sampling_params = SamplingParams(temperature=0, max_tokens=128)
outputs = llm.generate(
    ["Explain why graph execution improves decode performance."],
    sampling_params,
)

for output in outputs:
    print(output.outputs[0].text)
```

## 7 Accuracy Evaluation

> **Note**: Post-processing parameters, such as `max_tokens`, `temperature`, and stop tokens, should match those defined in the model weight's `generation_config.json`.

### 7.1 Using AISBench

Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for details.

### 7.2 Using Language Model Evaluation Harness

Using the GPQA-Diamond dataset as an example, run the accuracy evaluation for the selected Gemma4 dense or MoE model in online mode. Refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md) for installation and usage details.

```shell
lm_eval \
  --model local-completions \
  --model_args model=gemma4,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False,trust_remote_code=True \
  --tasks gpqa_diamond \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vLLM benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

The following configurations are for reference only. The optimal configuration depends on hardware resources, model size, maximum input/output length, request concurrency, prefix cache hit rate, and whether the selected Gemma4 weight is dense or MoE. Tune the parameters in Section 9.2 based on your actual workload.

| Scenario | Deployment Mode | Total NPUs | Model Type | Key Considerations |
| -------- | --------------- | ---------- | ---------- | ------------------ |
| Functional verification | Single-node online serving | 4 | Dense or MoE | Use eager mode first to verify model loading and basic generation. |
| Generation serving | Single-node online serving | 4 | Dense or MoE | Use ACLGraph with `FULL_DECODE_ONLY` to reduce decode dispatch overhead. |
| Long context | Single-node online serving | 4 | Dense or MoE | Increase `--max-model-len` only when enough KV cache is available; lower concurrency if OOM occurs. |
| MoE serving | Single-node online serving | 4 | MoE | Use tensor parallelism or expert parallelism according to model size and deployment plan. |
| Quantized serving on 950PR&950DT Products | Single-node online serving | 8 | Dense or MoE | Use TP4DP2 with the MXFP4 checkpoint and add `--quantization ascend`. |

### 9.2 Tuning Guidelines

Please refer to [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for general tuning methods.

Please refer to [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

Recommended tuning order:

1. Use eager mode to verify model loading and generation correctness.
2. Enable ACLGraph with `FULL_DECODE_ONLY` for generation serving and compare it with the eager baseline.
3. Choose `--max-model-len` according to the required context length. Long context increases KV cache usage, so reduce `--max-num-seqs` if OOM occurs.
4. Tune `--max-num-batched-tokens`. Larger values can improve prefill efficiency but consume more activation memory.
5. Tune `--max-num-seqs` according to service concurrency. Requests above this value wait in the queue.
6. Tune `--gpu-memory-utilization` when more KV cache is needed, while leaving enough HBM headroom for runtime memory fluctuation.
7. For MoE variants, choose tensor parallelism or expert parallelism according to the model size and deployment plan, and compare throughput and latency under the same request workload.

### 9.3 Model-Specific Optimizations

| Optimization | Enablement | Benefit | Notes |
| ------------ | ---------- | ------- | ----- |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` | Reduces decode dispatch overhead. | Recommended when decode tokens are generated. |
| Piecewise ACLGraph | `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'` | Enables segmented graph execution. | Supported when piecewise graph execution is required. |
| Tensor parallelism | `--tensor-parallel-size` | Splits model computation across multiple NPUs. | Adjust according to model size and available devices. |
| Expert parallelism | `--enable-expert-parallel` | Distributes experts for MoE variants. | Enable only for MoE deployment plans that use EP. |
| MXFP4 quantization | `--quantization ascend` | Reduces weight and activation memory and uses Ascend MXFP4 operators. | Required for the MXFP4 checkpoints; omit it for BF16 weights. |

## 10 FAQ

For common environment, installation, and general parameter issues, refer to [Public FAQs](../../faqs.md). This section only covers model-specific issues for Gemma4.

### Q1: Which execution mode should be used first?

**Phenomenon:** Users need to choose between eager mode and ACLGraph mode during initial deployment.

**Cause:** Eager mode is easier to use as a baseline, while ACLGraph mode is designed to reduce graph dispatch overhead during serving.

**Solution:** Use eager mode first to verify model loading and basic generation. After the baseline is confirmed, use ACLGraph mode with `FULL_DECODE_ONLY` for generation serving.

### Q2: How should the model path be configured?

**Phenomenon:** The service fails to start because the model path cannot be found or the wrong model is loaded.

**Cause:** The examples use `/root/.cache/path/to/gemma4` as a placeholder.

**Solution:** Replace `MODEL_PATH` and all benchmark or offline inference model paths with the actual local Gemma4 model directory.

### Q3: How should Gemma4 MoE variants be deployed?

**Phenomenon:** Users need to decide whether to use tensor parallelism or expert parallelism for MoE variants.

**Cause:** MoE models can use different parallel strategies depending on model size, number of devices, and serving workload.

**Solution:** Start with the same tensor parallel configuration used for the baseline. If the deployment plan uses expert parallelism, add `--enable-expert-parallel` and compare throughput, latency, and memory usage under the same workload.
