# Deploy Kimi-K2.6 on Ascend NPUs

This guide explains how to deploy **Kimi-K2.6** (Moonshot AI's 555B MoE model with W4A16 quantization) on Huawei Ascend NPUs using vLLM-Ascend.

---

## Model Overview

Kimi-K2.6 is a large mixture-of-experts (MoE) language model with a vision encoder, developed by Moonshot AI. Key specifications:

| Property | Value |
|---|---|
| Architecture | KimiK25ForConditionalGeneration (DeepSeek-V3 MoE + MLA) |
| Total Parameters | ~555B (W4A16 quantized) |
| Experts | 384 MoE experts, 8 active per token |
| Layers | 61 transformer layers |
| Attention | Multi-head Latent Attention (MLA) |
| Vision Encoder | 27-layer ViT |
| Disk Size | ~555 GB (W4A16) |
| Recommended TP | 16 (full Ascend 910B cluster) |

---

## Prerequisites

- Ascend 910B cluster with **at least 16 NPUs** (TP=16 required)
- CANN toolkit installed
- vLLM-Ascend >= v0.18.0
- Model weights downloaded (e.g., from ModelScope):

```bash
pip install modelscope
modelscope download --model moonshotai/Kimi-K2.6 --local_dir /path/to/Kimi-K2.6
```

---

## Launch Command

```bash
# MoE models must use AIV expansion mode
export HCCL_OP_EXPANSION_MODE=AIV

python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Kimi-K2.6/moonshotai/Kimi-K2___6 \
    --tensor-parallel-size 16 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-acl-graph \
    --acl-graph-piecewise \
    --enforce-eager
```

### Key Parameters

- `--tensor-parallel-size 16`: Required for the 555B model (~40.5 GB/chip).
- `--trust-remote-code`: Required for custom tokenizer and model code.
- `--max-model-len 8192`: Adjust based on available KV cache memory.
- `--enable-acl-graph --acl-graph-piecewise`: Enables Ascend ACL Graph with piecewise capture for MoE optimization.
- `--enforce-eager`: Recommended for stability with MoE models.

---

## Verification

Send a test request after the server is ready:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/path/to/Kimi-K2.6/moonshotai/Kimi-K2___6",
        "messages": [{"role": "user", "content": "Hello, introduce yourself briefly."}],
        "max_tokens": 128
    }'
```

---

## Performance Reference

Benchmark results on Ascend 910B x16 (TP=16, W4A16, input=512, output=128, 64 concurrent requests):

| Metric | Value |
|---|---|
| Output Token Throughput | 46.58 tok/s |
| Mean TTFT | 691 ms |
| Mean TPOT | 83 ms |
| Peak Output Throughput | 52 tok/s |

> **Note**: Benchmark measured with `vllm bench serve` (input=512, output=128, 64 concurrent requests). TTFT (~691 ms) includes prefill on the 555B MoE model. TPOT (~83 ms) reflects the per-token decode speed.
