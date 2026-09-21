# BNSD long-prefill attention

!!! warning

    This optimization is experimental, disabled by default, and currently
    limited to Atlas A2 hardware.

## Overview

Cached prefill attention normally sends query tensors to Fused Infer
Attention (FIA) in TND layout. For one long prefill request, BNSD can use a
more efficient FIA tiling path while retaining the paged KV cache in its
native NHD layout.

When a scheduler step mixes decode requests with one eligible long prefill
request, vLLM Ascend keeps the short decode slice in TND and sends only the
prefill slice through BNSD. Unsupported shapes and modes continue to use the
existing TND path.

## Enable the optimization

Add the following option to `--additional-config`:

```json
{
  "enable_prefill_bnsd": true
}
```

For example:

```bash
vllm serve /path/to/model \
  --enable-chunked-prefill \
  --additional-config '{"enable_prefill_bnsd":true}'
```

## Eligibility and fallback

The BNSD path is selected only when all of the following are true:

- Atlas A2 hardware;
- BF16 query and paged KV cache;
- causal decoder attention without sliding-window attention or sinks;
- 256-dimensional attention heads;
- exactly one prefill request with at least 4096 query tokens in the current
  scheduler step; and
- cached prefill or chunked prefill using the standard 128-token KV block
  size.

All other cases automatically retain the existing TND implementation. The
optimization does not change the KV-cache layout.

## Reference benchmark

The following reference measurements were collected with Qwen3.8-27B on
Atlas A2 using tensor parallel size 2 and an experimental prototype based on
vLLM Ascend commit `f4a08bddd` and CANN 9.0.1. The profiled attention shape
had 12 query heads, 2 KV heads, and a head dimension of 256. An operator
benchmark compared an approximately 7680-token prefill query attending to a
100000-token paged KV cache:

| Path | Median latency |
| --- | ---: |
| TND query + NHD paged KV | 62.7079 ms |
| BNSD query + NHD paged KV | 54.4916 ms |

The BNSD query path reduced operator latency by 13.10%. A mixed batch with
query lengths `[7680, 4, 4, 4]`, including query reorder and output placement,
improved from 66.1986 ms to 55.4271 ms (16.27%).

Measure the effect with the target model and traffic distribution before
enabling this experimental option in production.
