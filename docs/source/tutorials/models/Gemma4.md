# Gemma4

## 1 Introduction

Gemma4 is a Gemma-family language model that includes dense and Mixture-of-Experts (MoE) variants, and is suitable for general text generation, reasoning, and instruction-following scenarios.

This document describes the main validation steps for Gemma4 on Atlas A2, Atlas A3, and 950PR&950DT Products, including supported features, prerequisites, installation, single-node online deployment, functional verification, offline inference, accuracy and performance evaluation, performance tuning, and FAQs.

This document is written based on the latest vLLM Ascend main branch. Gemma4 is supported on Atlas A2, Atlas A3, and 950PR&950DT Products in this version.

## 2 Supported Features

Refer to [Supported Features List](../../user_guide/support_matrix/supported_models.md) to get the model support matrix, including BF16, chunked prefill, automatic prefix caching, tensor parallelism, expert parallelism, and ACLGraph support.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

Gemma4 supports both eager execution and ACLGraph execution on Atlas A2, Atlas A3, and 950PR&950DT Products. For graph execution, `FULL_DECODE_ONLY` can reduce decode-phase dispatch overhead, while `PIECEWISE` is also supported.

Gemma4 also supports MTP (Multi-Token Prediction) speculative decoding on Ascend 950 (A5). It drafts multiple tokens per decode step with a Gemma4 assistant checkpoint and verifies them against the target model in a single forward pass. See Section 5.3 for the serving configuration and Section 9.3 for tuning.

## 3 Prerequisites

### 3.1 Model Weight

Download the Gemma4 model weight to a local or shared directory, such as `/root/.cache/`. In the following examples, replace `/root/.cache/path/to/gemma4` with your actual model path.

| Model type | Description | Recommended hardware |
| ---------- | ----------- | -------------------- |
| Gemma4 dense model | Dense Gemma4 weight. | A single Atlas A2, Atlas A3, or 950PR&950DT node. Adjust the number of visible NPUs according to model size. |
| Gemma4 MoE model | Mixture-of-Experts Gemma4 weight. | A single Atlas A2, Atlas A3, or 950PR&950DT node. Use tensor parallelism or expert parallelism according to the model size and deployment plan. |

The examples below assume a single node with visible NPUs. The commands use 4 visible NPUs as an example. Adjust `ASCEND_RT_VISIBLE_DEVICES` and `--tensor-parallel-size` according to the model size and available devices.

## 4 Installation

### 4.1 Image Availability

If you need a prebuilt image, contact Huawei engineers to obtain the appropriate image and software stack for your target hardware.

After the environment is prepared, verify that the NPU devices are visible:

```shell
npu-smi info
```

Expected result: `npu-smi info` lists the expected Ascend devices.

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

Common Issues Tip: If the service fails to start, HBM is insufficient, or requests are not scheduled as expected, refer to [Public FAQs](../../faqs.md) first, and then check the model-specific FAQ in Section 10.

#### Key Parameters

- `--tensor-parallel-size`: sets the tensor parallel size. Adjust it according to model size and available NPU devices.
- `--enable-expert-parallel`: enables expert parallelism for MoE variants when the deployment plan uses EP.
- `--max-model-len`: sets the maximum input plus output length for a single request. Increase it only when enough KV cache is available.
- `--enforce-eager`: enables eager execution for baseline verification.
- `--compilation-config`: configures graph execution. `FULL_DECODE_ONLY` can reduce decode-phase dispatch overhead, and `PIECEWISE` is also supported.
- `--speculative-config`: enables MTP speculative decoding. Supported on Ascend 950 (A5) only. Unlike the MTP implementations of other models, it must name the Gemma4 assistant checkpoint explicitly in `model`. See Section 5.3.
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

### 5.3 MTP Speculative Decoding

MTP (Multi-Token Prediction) speculative decoding is supported for Gemma4 on **Ascend 950 (A5) only**. An assistant checkpoint drafts several tokens per decode step, and the target model verifies them in a single forward pass.

Two properties of the Gemma4 drafter affect how it is deployed:

- **It is a separate assistant checkpoint**, not a draft head embedded in the target weight. Unlike the MTP configuration of other models, `--speculative-config` must therefore set `model` to the assistant checkpoint.
- **Its layers do not share a uniform attention shape.** Sliding-window layers use `head_dim=256` and full-attention layers use `head_dim=512` (`global_head_dim` in the checkpoint configuration), so each KV cache group gets its own attention metadata. The drafter also reads K/V from the target model's cache (cross-model KV sharing) instead of maintaining its own.

Both the target checkpoint and the assistant checkpoint must be available locally; in the example below they are `/root/.cache/path/to/gemma4` and `/root/.cache/path/to/gemma4-assistant`. The assistant is resolved through the same loader as the target model.

The following example enables MTP together with `FULL_DECODE_ONLY` graph mode, serving three speculative tokens per step on a single Ascend 950 card (TP=1):

```shell
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL_PATH=/root/.cache/path/to/gemma4

vllm serve ${MODEL_PATH} \
  --served-model-name gemma4-mtp \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --speculative-config '{"method": "mtp", "model": "/root/.cache/path/to/gemma4-assistant", "num_speculative_tokens": 3}' \
  --max-model-len 5500 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

The `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, and `--gpu-memory-utilization` values above are examples. Tune them for your workload and available NPU memory as described in Section 9. Data-parallel serving (`--data-parallel-size N`, one TP=1 replica per card) uses the same speculative configuration.

**Readiness check.** When the service starts with MTP enabled, the log reports how many draft layers were wired to the target model's cache:

```text
Gemma4 MTP: propagated KV-sharing target to 4/4 draft layers.
```

The Gemma4 assistant checkpoint has four attention layers, so a correctly wired service reports `4/4`. A first number lower than the second means some draft layers would read their own (empty) cache instead of the target's, which collapses the acceptance rate.

**Verify the acceptance rate.** The `/metrics` endpoint exposes the number of accepted draft tokens per position as `vllm:spec_decode_num_accepted_tokens_per_pos`, and the number of draft rounds as `vllm:spec_decode_num_drafts`. Scrape both before and after a batch of requests, then divide the deltas:

```text
acceptance[pos] = accepted_tokens_per_pos_delta[pos] / drafts_delta
```

Acceptance normally decreases with the draft position (`pos0` > `pos1` > `pos2`) and depends on the workload and prompting style, so measure it on your own traffic rather than assuming a target value. A collapse at `pos1` and `pos2` while `pos0` stays high is not a workload effect — it usually indicates that the draft attention metadata is misconfigured.

**Limitations:**

- MTP is supported on Ascend 950 (A5) only. It is not supported on Atlas A2 or Atlas A3.
- The assistant checkpoint must be available in addition to the target checkpoint.
- Centroid-based draft sampling runs eagerly and is not captured in an ACL graph. Decode graphs are still captured as usual with `FULL_DECODE_ONLY`.
- The throughput benefit depends on the acceptance rate, which is workload-dependent. Compare against the configuration without MTP on your own traffic before enabling it in production.

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

### 6.3 Offline Inference

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
| MTP speculative decoding | `--speculative-config '{"method": "mtp", "model": "<ASSISTANT_PATH>", "num_speculative_tokens": 3}'` | Improves decode throughput when the acceptance rate is good. | Ascend 950 (A5) only; requires the assistant checkpoint. Measure the acceptance rate before tuning `num_speculative_tokens`. See Section 5.3. |
| Full decode ACLGraph | `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` | Reduces decode dispatch overhead. | Recommended when decode tokens are generated. |
| Piecewise ACLGraph | `--compilation-config '{"cudagraph_mode": "PIECEWISE"}'` | Enables segmented graph execution. | Supported when piecewise graph execution is required. |
| Tensor parallelism | `--tensor-parallel-size` | Splits model computation across multiple NPUs. | Adjust according to model size and available devices. |
| Expert parallelism | `--enable-expert-parallel` | Distributes experts for MoE variants. | Enable only for MoE deployment plans that use EP. |

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
