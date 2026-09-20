# Qwen3.8-Flash-Next

## 1 Introduction

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts model whose
`Qwen4ExpForConditionalGeneration` architecture combines Gated DeltaNet,
Qwen Sparse Attention (QSA), HyperConnection, Position Learning Enhancement
(PLE), vision input, and an MTP draft model.

The portable Qwen4Exp model implementation is owned by vLLM. vLLM Ascend adds
the Ascend-specific kernels, cache integration, quantization support, and Model
Runner V1 runtime wiring described on this page.

## 2 Supported Configuration

The target validation configuration for a single Atlas 800I A3 node is:

| Item | Configuration |
|---|---|
| Weights | [Qwen3.8-Flash-Next-w8a8-mtp](https://modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp) |
| Quantization | Ascend W8A8 |
| Parallelism | DP1 x TP8 with expert parallelism |
| Runner | Model Runner V1 |
| Prefix caching | `align` Mamba cache mode |
| Speculative decoding | Qwen4Exp MTP, 3 speculative tokens |
| Input | Text and multimodal |

## 3 Online Serving

Install compatible vLLM and vLLM Ascend versions by following the
[installation guide](../../getting_started/installation.md). Do not mix the
Python package from one source checkout with a native extension built from
another checkout.

```bash
export MODEL_PATH=Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp
export VLLM_USE_MODELSCOPE=True
export VLLM_ASCEND_ENABLE_QSA_LIGHTNING_INDEXER=1
export VLLM_ASCEND_ENABLE_QSA_E3V=1
unset VLLM_ASCEND_FORCE_QSA_REFERENCE

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --quantization ascend \
  --max-model-len 135168 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.93 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[4,8,12,16,20,24,28,32]}' \
  --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
```

The command intentionally does not specify `--language-model-only`, so the
vision-language path remains available. Adjust the function-calling and
reasoning parsers separately when they are required by your API workload.

Prefix caching stores additional state on the NPU. Its memory cost depends on
the maximum context length, concurrency, and cache layout. In the A3 TP8
correctness validation above, `--gpu-memory-utilization 0.95` left insufficient
workspace for one QSA prefill shape, while `0.93` passed. This is a validation
observation, not a universal fixed value. If memory is tight, reduce
`--gpu-memory-utilization`, `--max-model-len`, or `--max-num-seqs` according to
the workload.

## 4 Verification

Wait for the service to become ready, then require HTTP 200 and a non-empty
response:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Say hi."}],
    "temperature": 0,
    "max_tokens": 16
  }'
```

For prefix-cache validation, send the same long prefix twice. The second
request must report cached prompt tokens and produce the same greedy token
sequence as the first request. Also inspect the MTP proposed/accepted-token
metrics and service logs for OOM, HCCL, illegal-memory-access, NaN, and
preemption errors.

For multimodal validation, send one image request through the same
OpenAI-compatible chat endpoint and require HTTP 200 with non-empty `choices`.
A successful startup without successful text and image requests is not a
functional pass.

## 5 Limitations

- This page documents the A3 TP8 validation. It does not claim validation on
  other Ascend products.
- PLE CPU offload is not supported by this configuration.
- Capacity and performance depend on the input-length distribution and
  concurrency. Re-run the long-context and accuracy gates for production
  deployments.
