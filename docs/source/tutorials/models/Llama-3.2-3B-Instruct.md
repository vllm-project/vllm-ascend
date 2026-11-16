# Llama-3.2-3B-Instruct

## 1 Introduction

Llama 3.2 is Meta's lightweight instruction-tuned model family, and `Llama-3.2-3B-Instruct` is its 3.2B-parameter dense variant. It keeps the standard Llama decoder-only architecture (28 layers, grouped-query attention with 24 query heads and 8 KV heads, RoPE, SwiGLU, RMSNorm), supports a 128K context window, and is instruction-tuned for multilingual dialogue, summarization, and tool calling. Because of its small footprint it fits on a single NPU, which makes it a practical choice for edge-side assistants, data filtering, and evaluation baselines. This document describes the main validation steps for the model on vLLM-Ascend, including supported features, environment preparation, single-node deployment, and accuracy and performance evaluation.

`Llama-3.2-3B-Instruct` uses the upstream `LlamaForCausalLM` architecture, which vLLM already implements natively, so no model-level code change is required in vLLM-Ascend; the work is to verify that the model loads real weights and serves correctly on Ascend hardware.

This document is validated and written based on **vLLM-Ascend v0.30.0**, which aligns with upstream vLLM v0.30.0 and bundles CANN 9.2.1, PyTorch 2.10.0 and torch_npu 2.10.0.post7. Validation covered single-card deployments on Atlas A2 (910B4, 32GB) and Atlas A3 (910_9382, 64GB), and both reproduce the accuracy baseline recorded in CI. The model is expected to run stably on v0.30.0 and later versions. To use the latest features, it is recommended to use the latest release candidate or official release.

## 2 Supported Features

Please refer to the [Supported Models](../../user_guide/support_matrix/supported_models.md) for the model support matrix.

Please refer to the [Feature Guide](../../user_guide/feature_guide/index.md) for feature configuration information.

Key points for this model:

| Item | Status | Remarks |
| ---- | ------ | ------- |
| Architecture | ✅ | `LlamaForCausalLM`, natively supported by upstream vLLM |
| BF16 | ✅ | Default precision on Atlas A2/A3 inference products |
| FP16 | ✅ | Required on Atlas 300I DUO |
| Chunked Prefill | ✅ | Enabled by default |
| Automatic Prefix Caching | ✅ | `--enable-prefix-caching` |
| Tensor Parallelism | ✅ | TP=1 is sufficient for a 3B model |
| Data Parallelism | ✅ | `--data-parallel-size` |
| Speculative Decoding | 🔵 | Only with an external draft model; no MTP layer in the checkpoint |
| Quantization (W8A8) | 🔵 | Requires a separately produced quantized checkpoint |
| ACLGraph (piecewise + full decode) | ✅ | `cudagraph_mode=FULL_AND_PIECEWISE`, the default on this version |

!!! note

    The model ships no MTP (multi-token prediction) head, so `--speculative-config` with `method: mtp` is not applicable. Use a draft-model based speculative method if you need it.

## 3 Prerequisites

### 3.1 Model Weight

| Weight Version | Hardware Requirements | Download Links |
| -------------- | --------------------- | -------------- |
| `Llama-3.2-3B-Instruct` | 1 Atlas A3 inference product (64GB × 1) or 1 Atlas A2 inference product (64GB × 1) | [ModelScope](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct) \| [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `Llama-3.2-3B-Instruct` | 1 Atlas 300I DUO device | [ModelScope](https://www.modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct) \| [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |

The Hugging Face repository is gated and requires accepting Meta's license first; set `export HF_TOKEN=<YOUR_TOKEN>` before downloading. The ModelScope copy (`LLM-Research/Llama-3.2-3B-Instruct`) is the one used by the end-to-end test configuration in `tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml`.

>**Path description:** Please download the model weights to a directory of your choice and record this path. For example: `/mnt/weight/Llama-3.2-3B-Instruct`. In subsequent deployment commands, the placeholder `<YOUR_MODEL_PATH>` will be used; please replace it with the path you have recorded here.

### 3.2 Verify Multi-node Communication (Optional)

A 3B model fits on one device, so multi-node deployment is not required. If you still want to deploy it across nodes, please follow the [Verify Multi-node Communication Environment](../../getting_started/installation.md#installation-multi-node-interconnect) guide first.

## 4 Installation

### 4.1 Docker Image Installation

Select an image based on your machine type and start the docker image on your node. Refer to [using docker](../../getting_started/installation.md#installation-prebuilt-image) for details.

It is recommended to use the latest release candidate (rc) version or the latest official version of the `vllm-ascend` image.

=== "Atlas A3 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3

    docker run --rm \
        --name vllm-ascend-env \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
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

=== "Atlas A2 inference products"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}

    docker run --rm \
        --name vllm-ascend-env \
        --shm-size=1g \
        --net=host \
        --device /dev/davinci0 \
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

=== "Atlas 300I DUO"

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-310p

    docker run --rm \
        --name vllm-ascend \
        --shm-size=10g \
        --device /dev/davinci0 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /root/.cache:/root/.cache \
        -it $IMAGE bash
    ```

Verify the container is running:

```bash
docker ps | grep vllm-ascend
```

Expected result: one container in `Up` status whose name is `vllm-ascend-env` (or `vllm-ascend` on Atlas 300I DUO).

### 4.2 Source Code Installation

If you prefer to build from source instead of using the Docker image, install vLLM-Ascend by following the [Installation Guide](../../getting_started/installation.md).

!!! note

    Triton Ascend is not supported on Atlas 300I DUO. If you install from source there, uninstall both packages before running the service:

    ```bash
    pip uninstall -y triton-ascend triton
    ```

To verify the source installation:

```bash
pip show vllm-ascend
```

Expected result: the version information is displayed, confirming a successful installation.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

Single-node deployment completes both Prefill and Decode within the same node. A 3B dense model fits on a single device, so `TP=1` is enough and no cross-node communication is involved; this is the recommended mode for development, testing, and small-scale inference.

=== "Atlas A3 inference products / Atlas A2 inference products"

    Startup Command:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

    # Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1
    vllm serve <YOUR_MODEL_PATH> \
        --served-model-name llama3.2 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.8 \
        --port 8000
    ```

    Key parameters:

    - `--tensor-parallel-size 1`: one device is sufficient for a 3.2B BF16 checkpoint (about 6.4GB of weights).
    - `--max-model-len`: the checkpoint supports 128K, but raise it in steps and watch NPU memory; 4096 is enough for the functional check below.
    - `--gpu-memory-utilization 0.8`: 0.9 works on 64GB cards but leaves too little headroom on a 32GB card (e.g. 910B4); on 32GB the engine fails during graph capture with `Engine core initialization failed`. Lower it to 0.8, or keep 0.9 only on 64GB.
    - `--served-model-name`: the model name used in the request body of Section 6.
    - `--trust-remote-code` is **not** needed: the architecture is upstream native.

=== "Atlas 300I DUO"

    Startup Command:

    ```bash
    export ASCEND_RT_VISIBLE_DEVICES=0

    # Replace <YOUR_MODEL_PATH> with the actual path recorded in Section 3.1
    vllm serve <YOUR_MODEL_PATH> \
        --served-model-name llama3.2 \
        --dtype float16 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --port 8080 \
        --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false}}'
    ```

    Key parameters:

    - `--dtype float16`: Atlas 300I DUO does not support BF16, so the checkpoint must be served in FP16.
    - `--max-model-len`: keep it modest (4096 in this example) because the device memory is shared by weights and KV cache.
    - `--additional-config`: keeps `npugraph_ex` explicitly off. It is already the default on Atlas 300I DUO, so this line is only for clarity and can be omitted.

The output upon successful startup is:

```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

Common issues: if startup fails with an unresolved architecture error or missing weight keys, confirm that the checkpoint is complete (`config.json` plus all `*.safetensors` shards) and that `--dtype` matches the platform. For other environment problems, refer to the [Public FAQs](../../faqs.md).

### 5.2 Multi-Node Online Deployment

Not applicable: a 3B dense model runs on a single device and does not benefit from pipeline or expert parallelism. Use Section 5.1.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a chat request:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a short introduction to large language models."}
        ],
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 200
    }'
```

Expected result: HTTP 200 with a JSON body containing a non-empty `choices[0].message.content`, for example:

```json
{
  "id": "chatcmpl-8f0f0f0f0f0f0f0f",
  "object": "chat.completion",
  "model": "llama3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Large language models are neural networks trained on massive text corpora..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

The response content is generated, so the exact text differs between runs; only the structure and a non-empty `content` are guaranteed.

## 7 Accuracy Evaluation

### 7.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md).

The following is an example configuration for the accuracy evaluation config file:

```bash
# Example configuration: benchmarks/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="<YOUR_MODEL_PATH>",
        model="llama3.2",
        request_rate=0,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8000,
        max_out_len=2048,
        batch_size=32,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.0,
        ),
    )
]
```

Run the accuracy evaluation with the `gsm8k` dataset:

```bash
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_5_shot_chat_prompt --debug
```

Results measured with `num_fewshot: 5`, chat template applied, greedy decoding, on the full 1319-question `test` split, on a single NPU (TP=1, BF16, `--max-model-len 4096`):

| Hardware | `exact_match,strict-match` | `exact_match,flexible-extract` |
| -------- | -------------------------- | ------------------------------ |
| Atlas A2 (910B4, 32GB) | 0.6990 (922/1319) | 0.7635 (1007/1319) |
| Atlas A3 (910_9382, 64GB) | 0.6990 (922/1319) | 0.7574 (999/1319) |
| Reference in CI config | 0.71 | 0.76 |

Both platforms reproduce the baseline recorded in `tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml` within run-to-run variance, and agree with each other on the strict metric, so the checkpoint runs at its expected accuracy on Ascend.

The end-to-end case in `tests/e2e/models` also passes. Running

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml
```

reports `exact_match,strict-match` 0.718 and `exact_match,flexible-extract` 0.7672 against the baseline of 0.71 / 0.76 recorded in the config, i.e. within the harness tolerance `RTOL=0.05`.

!!! note

    The harness runs vLLM in-process. Without `VLLM_WORKER_MULTIPROC_METHOD=spawn`, the engine core subprocess fails with `RuntimeError: Cannot re-initialize NPU in forked subprocess`.

!!! note

    Accuracy depends on the decoding parameters and the tokenizer configuration. Use the same `num_fewshot`, chat-template setting, and temperature as the reference run before comparing numbers.

### 7.2 Using Language Model Evaluation Harness

Using the `gsm8k` dataset as an example, run the accuracy evaluation in online mode.

1. For `lm_eval` installation, please refer to [Using lm_eval](../../developer_guide/evaluation/using_lm_eval.md).
2. Run `lm_eval` to execute the accuracy evaluation.

```bash
lm_eval \
  --model local-completions \
  --model_args model=<YOUR_MODEL_PATH>,base_url=http://127.0.0.1:8000/v1/completions,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5 \
  --output_path ./
```

## 8 Performance Evaluation

### 8.1 Using AISBench

For setup details, including installation, dataset download, and configuration, please refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation).

Run the performance evaluation with a shared-prefix prompt set:

```bash
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_str_perf --mode perf --debug
```

!!! note

    AISBench was not used for the runs recorded in this document. The measured throughput, TTFT and TPOT on Atlas A2 and Atlas A3 are listed in Section 8.2, together with the exact `vllm bench serve` command that produced them.

### 8.2 Using vLLM Benchmark

```bash
vllm bench serve \
    --model <YOUR_MODEL_PATH> \
    --served-model-name llama3.2 \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 200 \
    --max-concurrency 32
```

Measured with 200 prompts of 1024 input / 256 output tokens, TP=1, BF16, `--max-model-len 4096`:

| Hardware | Concurrency | Request Throughput (req/s) | Output Token Throughput (tok/s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) |
| -------- | ----------- | -------------------------- | ------------------------------- | -------------- | ------------- | -------------- |
| Atlas A2 (910B4, 32GB) | 32 | 3.84 | 982.49 | 427.95 | 1663.84 | 28.86 |
| Atlas A2 (910B4, 32GB) | 64 | 5.07 | 1299.02 | 808.77 | 3217.72 | 41.32 |
| Atlas A3 (910_9382, 64GB) | 32 | 5.38 | 1378.25 | 289.81 | 1082.26 | 20.63 |
| Atlas A3 (910_9382, 64GB) | 64 | 8.56 | 2191.45 | 212.29 | 448.15 | 24.87 |

Throughput scales with concurrency while the KV cache holds. The 32GB device reaches a lower plateau and degrades faster (TPOT 28.86 ms to 41.32 ms), because the smaller card limits how much KV cache can back concurrent sequences; choose `--max-num-seqs` accordingly on such devices. Report `Request throughput`, `Output token throughput`, and `Mean TTFT` from your own summary table when comparing runs.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> **Note**: The configurations below are for reference only and were not tuned as a global optimum. The best setting depends on the input/output length ratio, prefix cache hit rate, and device memory.

| Scenario | NPUs | Weight Version | TP | Max Num Seqs | Max Num Batched Tokens | Max Model Len | Async Scheduling |
| -------- | ---- | -------------- | -- | ------------ | ---------------------- | ------------- | ---------------- |
| Low latency | 1 | BF16 | 1 | 8 | 2048 | 4096 | Off |
| High throughput | 1 | BF16 | 1 | 64 | 8192 | 8192 | On |
| Long context | 1 | BF16 | 1 | 8 | 8192 | 32768 | On |

### 9.2 Tuning Guidelines

#### 9.2.1 Model-Specific Optimizations

| Optimization Technique | Applicable Scenarios | Enablement Method | Precautions |
| ---------------------- | -------------------- | ----------------- | ----------- |
| Automatic prefix caching | Repeated system prompts / shared prefixes | `--enable-prefix-caching` | Costs extra NPU memory for the cached blocks |
| Chunked prefill | Long inputs mixed with decode traffic | enabled by default | Larger `--max-num-batched-tokens` improves prefill throughput but increases TTFT jitter |
| Async scheduling | High-concurrency serving | `--async-scheduling` | Small gains for a 3B model; verify accuracy after enabling |

#### 9.2.2 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and general parameter issues, please refer to the [Public FAQs](../../faqs.md); this chapter only covers model-specific issues.

**Q1: Startup fails with `ValueError: The model's max seq len ... is larger than the maximum number of tokens that can be stored in KV cache`.**

Cause: `--max-model-len` is larger than what the remaining NPU memory can hold after loading the weights.

Solution: lower `--max-model-len` or raise `--gpu-memory-utilization`, and make sure no other process is occupying the device.

**Q2: Output contains raw special tokens such as `<|eot_id|>` or the turns are not separated.**

Cause: the service was started with `--chat-template` pointing at a wrong template, or the client called `/v1/completions` with a manually concatenated prompt while the checkpoint expects the Llama 3 chat format.

Solution: drop the custom `--chat-template` so the template shipped in the checkpoint is used, and call `/v1/chat/completions` instead of `/v1/completions`.

**Q3: On Atlas 300I DUO the service fails with an unsupported dtype or an operator error in BF16.**

Cause: Atlas 300I DUO does not support BF16.

Solution: start with `--dtype float16` as shown in Section 5.1.

**Q4: Startup fails with `RuntimeError: Engine core initialization failed` and no other detail, on a 32GB card.**

Cause: `--gpu-memory-utilization 0.9` on a 32GB device (e.g. 910B4) leaves too little headroom for the KV cache and the ACLGraph buffers, so the engine core dies during graph capture while the API server is still starting.

Solution: lower it to `0.8` and retry (this is what Section 5.1 uses). If it still fails, add `--enforce-eager` to confirm whether the failure is in graph capture, and lower `--max-model-len` as well.

**Q5: `import torch_npu` fails with `libhccl.so: cannot open shared object file`, or the engine core dies with `cannot open shared object file: libatb.so` / `undefined symbol ... Mki`.**

Cause: the CANN and ATB runtime is installed in the container but the environment scripts were not sourced, so the dynamic loader cannot find `libhccl.so`, `libatb.so` and their dependencies. The `undefined symbol` form is a downstream symptom: the library that would provide the symbol failed to load first.

Solution: source both environment scripts before starting the service, and make sure the architecture-specific library directory is on the loader path:

```bash
export PATH=/usr/local/python3.12.16/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.2.1/aarch64-linux/lib64:$LD_LIBRARY_PATH
```

Verify with `python3 -c "import torch, torch_npu, torch_npu.op_plugin.atb._atb_ops; print('ok')"`. If that prints `ok`, ATB is resolved and the engine will initialize.
