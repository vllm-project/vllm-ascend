# NVIDIA-Nemotron-3-Super-120B-A12B-BF16 Deployment Tutorial

## 1 Introduction

`NVIDIA-Nemotron-3-Super-120B-A12B-BF16` is a 120B-parameter,
12B-active text-generation model based on the Nemotron-H architecture. It
combines Mamba2, attention, and latent Mixture-of-Experts (MoE) layers. This
tutorial describes the verified BF16 deployment path on Atlas A2, including
chunked prefill, eager execution, `FULL_DECODE_ONLY` ACLGraph execution, and
contexts up to 1M tokens.

Support is experimental. The configuration in this tutorial was validated on
one server with eight Ascend 910B3 NPUs and tensor parallel size 8.

## 2 Supported Features

Refer to [Supported Models](../../user_guide/support_matrix/supported_models.md)
for the project-wide feature matrix.

The following configuration has been validated:

| Feature | Status | Notes |
| ------- | ------ | ----- |
| BF16 | Supported | BF16 model weights with FP32 Mamba state cache. |
| Tensor parallelism | Supported | `--tensor-parallel-size 8`. |
| Chunked prefill | Supported | Validated with `--max-num-batched-tokens 1024`. |
| Eager execution | Supported | Use `--enforce-eager`. |
| ACLGraph | Supported | `FULL_DECODE_ONLY`, capture sizes 1, 2, and 4. |
| OpenAI-compatible API | Supported | Chat completions and completions endpoints. |
| Maximum context | Supported | 1,048,576 configured; 1,048,448 input tokens validated. |

Features not marked as supported for this model in the support matrix have not
been validated by this integration.

## 3 Prerequisites

### 3.1 Hardware

The verified BF16 configuration uses eight Ascend 910B3 NPUs with 64 GiB HBM
per NPU. Other device counts and HBM capacities may require different parallel
and cache settings.

### 3.2 Model Weight

Download the model from
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
or prepare an equivalent local directory. The directory must include the
model-provided `chat_template.jinja` and `super_v3_reasoning_parser.py` files.

The model configuration defaults to a 256K context. NVIDIA documents up to a
1M context when `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` and
`--max-model-len 1048576` are set explicitly.

### 3.3 Runtime Environment

Use a vLLM Ascend image or source installation compatible with the current
main branch. Follow the [Installation Guide](../../installation.md) to prepare
the driver, CANN toolkit, PyTorch NPU, vLLM, and vLLM Ascend versions.

For this model, disable vLLM batch-invariant mode before starting the server:

```shell
export VLLM_BATCH_INVARIANT=0
```

The model uses Ascend custom operators that are disabled when batch-invariant
mode is enabled. Set this variable in every server process, including spawned
workers.

## 4 Installation

Install vLLM Ascend from the main branch when validating a source change:

```shell
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
pip install -e .
```

Verify that all eight devices are visible before deployment:

```shell
npu-smi info
```

## 5 Online Service Deployment

Set the model path once for the commands below:

```shell
export MODEL=/data/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
export VLLM_BATCH_INVARIANT=0
```

### 5.1 Eager Mode

Use eager mode as the functional baseline:

```shell
vllm serve "${MODEL}" \
  --served-model-name nemotron-super \
  --trust-remote-code \
  --chat-template "${MODEL}/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser super_v3 \
  --reasoning-parser-plugin "${MODEL}/super_v3_reasoning_parser.py" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --enforce-eager \
  --host 0.0.0.0 \
  --port 8000
```

### 5.2 `FULL_DECODE_ONLY` ACLGraph Mode

Use `FULL_DECODE_ONLY` to capture decode at the configured batch sizes while
keeping prefill outside the graph:

```shell
vllm serve "${MODEL}" \
  --served-model-name nemotron-super \
  --trust-remote-code \
  --chat-template "${MODEL}/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser super_v3 \
  --reasoning-parser-plugin "${MODEL}/super_v3_reasoning_parser.py" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --max-model-len 262144 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --compilation-config \
    '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --host 0.0.0.0 \
  --port 8000
```

The first request at a newly encountered capture size can include graph
compilation overhead. Warm up capture sizes 1, 2, and 4 before measuring steady
state latency.

### 5.3 1M Context Deployment

For 1M context, set the model-length override and reserve enough HBM for model
execution. The following cache settings are the validated 8x64 GiB profile:

```shell
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

vllm serve "${MODEL}" \
  --served-model-name nemotron-super \
  --trust-remote-code \
  --chat-template "${MODEL}/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser super_v3 \
  --reasoning-parser-plugin "${MODEL}/super_v3_reasoning_parser.py" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory 8589934592 \
  --enable-chunked-prefill \
  --compilation-config \
    '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
  --host 0.0.0.0 \
  --port 8000
```

For eager execution at 1M context, replace `--compilation-config ...` with
`--enforce-eager`. The 8 GiB KV-cache value is in bytes and is a reference for
the validated hardware, not a universal recommendation. Reduce
`--max-num-seqs` or adjust cache allocation if the target system has less HBM.

## 6 Functional Verification

### 6.1 Basic Chat Completion

Send a deterministic request with thinking disabled:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-super",
    "messages": [
      {
        "role": "user",
        "content": "Compute 23 * 19. Reply with only the integer."
      }
    ],
    "max_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The expected `message.content` is `437`.

The integration validation also checks exact answers for an English factual
prompt and a Chinese instruction. These checks are run with concurrency 1, 2,
and 4 in both eager and ACLGraph modes.

### 6.2 Long Context

Long-context validation uses an exact needle-retrieval prompt. A unique
passphrase is inserted at 25% of the prompt and the model must return only that
passphrase. Token counts are measured after applying the model chat template.

The following cases are part of the integration validation:

| Prompt tokens | Concurrency | Purpose |
| ------------- | ----------- | ------- |
| 32,768 | 1, 2, and 4 | Chunked-prefill and concurrent retrieval. |
| 1,048,448 | 1 | Near-maximum 1M-context retrieval. |

A successful HTTP response alone is insufficient. Verify the exact passphrase,
`usage.prompt_tokens`, and `finish_reason` for every request.

The validated TP8 BF16 configuration produced the following results on vLLM
Ascend base `92ff388f`. Short checks contain two sequential waves at each
concurrency to exercise cache reuse. The 32K row contains one wave at each
concurrency.

| Mode | Short exact match | 32K exact retrieval | 1,048,448-token retrieval |
| ---- | ----------------- | ------------------- | ------------------------- |
| Eager | 14 / 14 | 7 / 7 | 1 / 1 in 405.02 seconds |
| `FULL_DECODE_ONLY` | 14 / 14 | 7 / 7 | 1 / 1 in 398.18 seconds |

Every long-context response reported the requested prompt-token count and
`finish_reason=stop`. The near-maximum request returned the exact passphrase in
both modes.

## 7 Accuracy Evaluation

NVIDIA's published evaluation uses MMLU-Pro with the
`eval/aai/mcq-10choices-boxed` prompt, one sample, thinking enabled,
`temperature=1.0`, `top_p=0.95`, and special-token skipping disabled. The
[NVIDIA reproducibility guide](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator-launcher/examples/nemotron/nemotron-3-super/reproducibility.md)
contains the complete evaluator configuration.

To run the complete official task with NeMo Evaluator Launcher:

```shell
nemo-evaluator-launcher run \
  --config local_nemotron-3-super-120b-a12b.yaml \
  -t nemo_skills.ns_mmlu_pro \
  -o target.api_endpoint.url=http://127.0.0.1:8000/v1/chat/completions
```

For a shorter integration check, use a fixed, stratified subset of the official
MMLU-Pro test split and preserve all of the following in the report:

- Dataset revision and SHA-256.
- Selection seed and complete question-ID list.
- Prompt template and generation parameters.
- Per-category results and a 95% confidence interval.
- Request errors, unparsed answers, and finish-reason counts.

Do not present a subset result as the full official score. Unparsed or truncated
responses must count as incorrect. NVIDIA reports 83.73 on the complete
MMLU-Pro evaluation for this checkpoint; a small subset should be interpreted
with its confidence interval.

The following result is a reproducible integration check collected with
`FULL_DECODE_ONLY` at concurrency 4, not a replacement for NVIDIA's complete
evaluation. Four responses reached the initial 32,768-token output limit. A
diagnostic retry raised the limit only for those four samples; the latest
result for every question is shown in the second row.

The subset was collected on vLLM Ascend `731436a1` with vLLM `e5588e49`.
The 18-case performance matrix in Section 8 was collected after the
source-changing rebase to vLLM Ascend `b266da5d`, using vLLM `ee0da84a`
(`v0.24.0`). The deterministic service matrix in Section 6 was rerun on vLLM
Ascend `92ff388f`; eager and `FULL_DECODE_ONLY` each passed 22/22 exact-content
requests. After subsequent latest-main rebases, the targeted unit suite was
rerun and overlapping upstream changes were audited before each branch update.
This provenance is explicit because the accuracy subset was not rerun after
the source-changing runtime rebase and the performance matrix was not rerun
after the final-base rebase.

| Output limit | Samples | Correct | Accuracy | Wilson 95% CI | Errors | Unparsed |
| ------------ | ------- | ------- | -------- | ------------- | ------ | -------- |
| 32,768 tokens | 70 | 63 | 90.00% | 80.77%-95.07% | 0 | 4 |
| Truncated-only retry at 65,536 | 70 | 64 | 91.43% | 82.53%-96.01% | 0 | 2 |

The subset contains five deterministic samples from each of the 14 categories,
selected with seed `20260710`. It uses the MMLU-Pro test parquet at revision
`b189ec765aa7ed75c8acfea42df31fdae71f97be`, whose SHA-256 is
`0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8`.
Generation uses a fixed per-question seed and an initial 32,768-token limit.
All 70 selected samples are included in the denominator. The initial run had
66 parsed answers with `finish_reason=stop`; four responses reached the token
limit and were counted as incorrect. Retrying only those four at 65,536 tokens
produced two additional parsed answers, one correct and one incorrect. The
other two responses again reached the output limit. Sampling at
`temperature=1.0` is sensitive to runtime scheduling, so the retry is reported
as a diagnostic rather than merged into an official benchmark claim.

The per-category scores below use the latest result after the diagnostic
retry.

| Category | Question IDs | Correct |
| -------- | ------------ | ------- |
| Biology | 2973, 2995, 3161, 3169, 3325 | 5 / 5 |
| Business | 169, 184, 408, 488, 561 | 4 / 5 |
| Chemistry | 3527, 3559, 3825, 3915, 4065 | 5 / 5 |
| Computer science | 10424, 10452, 10573, 10648, 10748 | 5 / 5 |
| Economics | 6829, 6957, 6980, 7209, 7645 | 4 / 5 |
| Engineering | 11446, 11652, 11937, 12162, 12234 | 4 / 5 |
| Health | 6043, 6122, 6155, 6331, 6533 | 5 / 5 |
| History | 4698, 4800, 4894, 4957, 5023 | 4 / 5 |
| Law | 945, 1056, 1140, 1149, 1566 | 4 / 5 |
| Math | 8388, 8568, 8666, 8706, 8975 | 5 / 5 |
| Other | 5123, 5278, 5788, 5960, 5974 | 4 / 5 |
| Philosophy | 10975, 11140, 11168, 11177, 11228 | 5 / 5 |
| Physics | 9057, 9163, 9569, 9778, 10284 | 5 / 5 |
| Psychology | 2036, 2340, 2375, 2543, 2800 | 5 / 5 |

## 8 Performance Evaluation

Use `vllm bench serve` against a warmed server. Keep server flags, prompt
distribution, output length, and request count identical when comparing eager
and ACLGraph modes.

The integration performance matrix uses three workloads:

| Workload | Input / output tokens | Requests | Reported metrics |
| -------- | --------------------- | -------- | ---------------- |
| Prefill | 32,768 / 1 | One wave at concurrency 1, 2, and 4 | Input throughput and TTFT. |
| Decode | 256 / 512 | Two waves at concurrency 1, 2, and 4 | Output throughput and TPOT. |
| Mixed | 8,192 / 512 | Two waves at concurrency 1, 2, and 4 | TTFT, TPOT, and total throughput. |

Use `--ignore-eos` so every performance request produces the requested output
length. Run deterministic functional requests separately, because benchmark
throughput does not prove output correctness.

Performance depends on the device SKU, software stack, graph warmup, model
storage, scheduler configuration, and request distribution. Treat measured
values as a reference for a specific environment, not a guaranteed baseline.

The following warmed results were collected on vLLM Ascend base `b266da5d`
with vLLM `ee0da84a` (`v0.24.0`), on one server with eight Ascend 910B3 64 GiB
NPUs, TP8, BF16 weights, FP32 Mamba state cache,
`--max-num-batched-tokens 1024`, and `--max-num-seqs 4`. Each cell reports the
mean for the corresponding concurrency.

| Workload and metric | Mode | C1 | C2 | C4 |
| ------------------- | ---- | -- | -- | -- |
| Prefill total tok/s | Eager | 2,969.66 | 2,957.19 | 2,980.94 |
| Prefill TTFT (ms) | Eager | 11,034.29 | 16,657.73 | 27,525.14 |
| Prefill total tok/s | `FULL_DECODE_ONLY` | 2,937.30 | 2,964.44 | 3,022.39 |
| Prefill TTFT (ms) | `FULL_DECODE_ONLY` | 11,155.79 | 16,662.76 | 27,119.68 |
| Decode output tok/s | Eager | 4.47 | 8.91 | 17.29 |
| Decode TPOT (ms) | Eager | 223.32 | 223.39 | 227.32 |
| Decode output tok/s | `FULL_DECODE_ONLY` | 32.94 | 68.82 | 88.47 |
| Decode TPOT (ms) | `FULL_DECODE_ONLY` | 24.40 | 28.06 | 40.00 |
| Mixed output tok/s | Eager | 4.38 | 8.34 | 15.61 |
| Mixed total tok/s | Eager | 74.49 | 141.75 | 265.35 |
| Mixed TTFT / TPOT (ms) | Eager | 2,810.81 / 223.16 | 7,198.60 / 225.27 | 8,238.03 / 237.95 |
| Mixed output tok/s | `FULL_DECODE_ONLY` | 33.33 | 38.76 | 44.33 |
| Mixed total tok/s | `FULL_DECODE_ONLY` | 566.65 | 658.88 | 753.59 |
| Mixed TTFT / TPOT (ms) | `FULL_DECODE_ONLY` | 2,790.39 / 24.60 | 7,176.87 / 37.57 | 10,209.55 / 70.11 |

All 18 performance cases completed without request errors. Every decode and
mixed request produced exactly 512 output tokens. Relative to eager mode,
`FULL_DECODE_ONLY` improved decode output throughput by 7.36x at C1, 7.72x at
C2, and 5.12x at C4 while leaving prefill throughput approximately unchanged,
as expected for decode-only graph capture.

## 9 Performance Tuning

The Mamba path uses Triton kernels adapted for Ascend according to the
[Triton-Ascend migration guide](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide).

### 9.1 SSD Chunk Scan

The upstream GPU kernel launches a logical program for each chunk and head. On
Ascend, the physical `coreDim` is limited, so a 1M request can exceed a single
launch. The Ascend implementation:

- Uses an NPU-specific `M=128`, `N=64`, `K=64`, `num_stages=1` tile.
- Preserves tail masks and FP32 accumulation.
- Splits the chunk axis into launches capped at 32,768 programs.
- Passes valid non-null pointers for compile-time-disabled optional branches.
- Supports both zero and non-zero initial Mamba states.

For the model's TP-local shape, a 1M request has 8,192 chunks and 131,072
logical programs. It is executed as four launches without a host-side device
synchronization.

### 9.2 Selective State Update

The upstream SSU heuristic is retained as the fallback for unrecognized shapes.
For the model's Ascend 910B3 shape (`head_dim=64`, `dstate=128`, FP32 state
cache), the following TP-local effective-batch configurations were tuned and
validated against the upstream reference:

| Effective batch | Upstream heuristic | Tuned config | Kernel speedup | Validation |
| --------------- | ------------------ | ------------ | -------------- | ---------- |
| 16 | `M=4, warps=4` | `M=64, warps=1` | 5.52x | Pass |
| 32 | `M=4, warps=4` | `M=64, warps=4` | 7.24x | Pass |
| 64 | `M=4, warps=4` | `M=64, warps=1` | 8.89x | Pass |

The dispatcher applies these settings only to the exact validated device,
shape, and cache dtype. All production configs passed the upstream reference
comparison with `atol=1e-2` and `rtol=1e-2` across three fixed random seeds.

## 10 FAQs

### 10.1 Why are custom operators disabled at startup?

Confirm that `VLLM_BATCH_INVARIANT=0` is exported in the API server and worker
environment. Batch-invariant mode disables custom operators that this model
uses on Ascend.

### 10.2 Why does vLLM warn about a model length greater than 256K?

The checkpoint configuration defaults to 256K while the model card documents
an explicit 1M override. Set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` only when 1M
serving is required and validate the target memory configuration.

### 10.3 Why does a 1M request run out of memory?

Model weights, attention KV cache, Mamba state cache, graph captures, and
prefill workspace all consume HBM. Start from the validated TP8 profile, lower
`--max-num-seqs`, and tune `--kv-cache-memory` while retaining HBM headroom.

### 10.4 Why is the first graph request slower?

ACLGraph capture and Triton compilation occur on first use of a new shape or
capture size. Warm up concurrency 1, 2, and 4 before collecting performance
results.

### 10.5 Why is `message.content` empty when thinking is enabled?

Use the model-provided reasoning parser and plugin. Thinking tokens may be
returned in `reasoning_content`, while the parsed final answer is returned in
`content`. For deterministic functional checks, set
`chat_template_kwargs.enable_thinking` to `false`.
