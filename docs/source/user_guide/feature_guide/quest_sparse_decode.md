# Quest Sparse Decode Attention

This document shows how to enable QUEST sparse decode attention in vLLM-Ascend, a query-aware KV-cache sparsity technique that accelerates long-context decoding.

## What is QUEST?

During autoregressive decoding, standard attention reads the **entire** KV cache for every generated token. As the context grows, this KV read dominates decode latency even though most tokens contribute very little to the attention output.

QUEST (Query-Aware Sparsity) reduces this cost by attending to only the **top-k most relevant KV pages** for each decode step instead of the whole sequence. It works in three stages:

1. **Metadata**: for every KV page (a block of 128 tokens) it maintains a per-channel *min* and *max* summary of the key vectors. These summaries are computed incrementally and only refreshed when a request crosses a page boundary.
2. **Selection**: for the current query it computes a cheap upper-bound relevance score per page from the min/max summaries, then selects the top-k pages. The first page (attention sink) and the most recent page are always kept as anchors.
3. **Sparse attention**: it runs flash attention over only the selected pages.

Because only a small fraction of pages is read per step, decode throughput improves substantially on long contexts, while the min/max upper bound and anchor pages keep the accuracy impact small.

QUEST is an **approximation** of full dense attention: it deliberately drops low-relevance pages, so outputs can differ slightly from dense attention. You should validate accuracy on your own workload.

## Quick Start

Enable QUEST via `--additional-config`. `topk_pages` is the number of KV pages (128 tokens each) attended per decode step and must be a positive multiple of 8:

```bash
vllm serve Qwen/Qwen3-8B \
  --block-size 128 \
  --additional-config '{"quest_decode_config": {"enable": true, "topk_pages": 32}}'
```

### Online Inference (Server Mode)

```bash
vllm serve Qwen/Qwen3-8B \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --additional-config '{"quest_decode_config": {"enable": true, "topk_pages": 32}}'
```

### Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-8B",
    block_size=128,
    compilation_config={"cudagraph_mode": "PIECEWISE"},
    additional_config={
        "quest_decode_config": {"enable": True, "topk_pages": 32},
    },
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=128)
outputs = llm.generate(["The future of AI is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

## Configuration

`quest_decode_config` accepts the following options:

| Name         | Type | Default | Description                                                                                     |
|--------------|------|---------|-------------------------------------------------------------------------------------------------|
| `enable`     | bool | `False` | Whether to enable QUEST sparse decode attention.                                                |
| `topk_pages` | int  | `None`  | Number of KV pages (128 tokens each) attended per decode step. Must be a positive multiple of 8 (`>= 8`). Required when `enable` is `true`. |

Choosing `topk_pages`: the effective attention window is `topk_pages * 128` tokens. Smaller values are faster but more approximate; larger values are more accurate but read more of the KV cache. A good starting point is a value that covers the portion of context that actually matters for your task, then tune for the accuracy/latency trade-off.

## Hardware Requirements

QUEST decode currently requires Ascend Atlas A2 (`ascend910b`) or A3 (`ascend910_93`) NPUs.

## Limitations

QUEST is a **decode-phase** optimization and is only used when it is both supported and worthwhile. When any condition below is not met, vLLM-Ascend transparently falls back to the standard dense Ascend attention path — QUEST never fails the request; unsupported configurations are disabled at startup with a warning in the log.

- **Decode only**: QUEST accelerates decode-only steps (query length 1). Prefill and chunked-prefill always use dense attention.
- **Sparsity gate**: if the selected pages would cover more than 50% of a request's pages, that batch falls back to dense attention, since dense is faster when little is pruned. Short sequences (fewer pages than `topk_pages`) therefore run as dense.
- **Fixed shapes**: requires a cache block size of 128 (`--block-size 128`) and an attention head size of 128.
- **Context length**: the per-request metadata table supports up to 6 metadata blocks, i.e. a `max_model_len` of at most `6 * 128 * 128 = 98304` tokens.
- **Standard v1 attention only**. QUEST is **not** supported together with, and will fall back to dense for:
    - Multi-head Latent Attention (MLA) models (e.g. DeepSeek).
    - Other sparse-attention models (models exposing `index_topk`).
    - Sliding-window attention or attention sinks.
    - Encoder-decoder / cross attention.
    - Context parallelism (prefill or decode CP).
    - PD disaggregation (`kv_transfer_config` set) — QUEST requires a local KV cache.
    - xLite graph mode.
- **Graph mode**: full-graph capture (`cudagraph_mode` with full cudagraphs) is not supported because QUEST switches between dense and sparse paths at runtime. Use `PIECEWISE` graph mode or eager execution. During graph capture QUEST automatically falls back to dense attention.
- **Approximation**: outputs may differ slightly from full dense attention. Validate accuracy for your workload before deploying.
