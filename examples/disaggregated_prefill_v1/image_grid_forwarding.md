# Forwarding image grids to decode

The load-balancing proxy can replace original image URLs with processed image
grids returned by the Mooncake KV connector. Decode then constructs multimodal
placeholders from the grids, avoiding repeated image decoding and pixel
preprocessing.

The current proxy integration targets requests served as `qwen3.6-35b`, using
the Qwen3-VL image processor. The decode KV consumer must accept
`image_embeds` with `image_grid_thw` and omitted embedding tensors. Other
served names use the original payload.

## Deployment

Update vllm-ascend on the prefill instance and restart prefill and the proxy.
The implementation uses vLLM's existing connector completion callback and
`kv_transfer_params` response field. The vLLM source files need no edits.

This requires a vLLM version with metadata-only image embedding support on
KV consumers. The frontend capability was validated with vLLM revision
`4ba787ec9959e566d3b587a796dff07673324da5`. Check that capability before using
older versions.

The proxy sets `return_image_grids: true` in the prefill request's
`kv_transfer_params`. On successful prefill completion, the Mooncake connector
reads each image's `image_grid_thw` from `request.mm_features` and adds
`pd_image_grids` to the returned KV metadata. Each entry contains the image
hash and three positive integer grid dimensions, in feature order.

The engine's multimodal receiver cache restores cached input data before
scheduling. The connector reads available CPU tensors directly at completion.
Missing cache data, unresolved shared-memory handles, device tensors, and
unsupported modalities cause the connector to omit the hint. Metadata belongs
to the completed request and adds no connector-side cache.

The proxy validates grid count and dimensions, then creates the decode request.
It preserves original messages for prefill and recomputation. Missing or
malformed metadata and mixed non-image media use the original payload.
If decode rejects a rewritten request with HTTP 400, 422, 500, 502, or 503
before response streaming begins, the proxy performs one fresh prefill and
retries decode with the original images.

## Validation

Connector tests cover both completion callbacks, metadata ordering, missing
cache data, repeated reads of restored data, invalid dimensions, request opt-in,
failed completion, and non-CPU tensor fallback. Proxy HTTP tests cover message
ordering, original-message preservation, invalid metadata, and fresh prefill
after a decode rejection.

An isolated integration run also exercised the real renderer, input processor,
and LRU receiver cache with 20 dataset requests repeated twice and a cache
reset before request 11. All 40 comparisons matched prompt token IDs and
image-placeholder ranges, including 265 image items restored from cache.

```bash
python -m pytest --confcutdir=tests/ut/kv_offload/mooncake_v2 \
  tests/ut/kv_offload/mooncake_v2/test_image_grid.py
python -m pytest --confcutdir=tests/ut/proxy \
  tests/ut/proxy/test_load_balance_proxy_server.py
```

The earlier frontend prototype was evaluated on Ascend A5 with one TP1 prefill
instance and one TP1 decode instance using Qwen3.6 W8A8C8. All 499 decodable
requests from a 500-request multimodal dataset produced identical prompt
token IDs and image-placeholder ranges through the original-image and
metadata paths. One AVIF request failed decoding in both paths.

Excluding the first call and failed request, isolated frontend rendering
averaged 55.71 ms with original images and 6.08 ms with metadata. A full run at
a fixed 300 ms dispatch interval changed output throughput from 743.26 to
746.94 tokens/s and mean TTFT from 25.77 to 25.12 seconds.

Those measurements used frontend metadata extraction. The Connector
implementation requires a new end-to-end hardware run before claiming the
same performance. A3, other models and other deployment modes have not been
hardware-validated for this change.
