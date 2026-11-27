# Forwarding image grids to decode

The load-balancing proxy can replace image URLs with image-grid metadata
returned by a compatible prefill server. This avoids loading and preprocessing
the original images again on the decode server.

The current integration targets requests served as `qwen3.6-35b`, using the
Qwen3-VL image processor and a KV consumer that accepts `image_embeds` with
`image_grid_thw` and omitted embedding tensors. The model-name restriction
matches the validated deployment. Other served names use the original payload.

## Prefill dependency

The companion [vLLM patch](patches/pd_image_grid.patch) contains the prefill
implementation used for validation with vLLM revision
`4ba787ec9959e566d3b587a796dff07673324da5`. Apply it to a compatible vLLM
source checkout and restart the prefill server. Review the patch context when
using another revision.

```bash
git -C /path/to/vllm apply --check /path/to/vllm-ascend/examples/disaggregated_prefill_v1/patches/pd_image_grid.patch
git -C /path/to/vllm apply /path/to/vllm-ascend/examples/disaggregated_prefill_v1/patches/pd_image_grid.patch
```

The proxy requests metadata with the internal `x-pd-image-grid: 1` header.
A compatible prefill server adds `pd_image_grids` to its response's
`kv_transfer_params`. Each item contains the processed image's hash and a
three-element positive integer grid, in image order. The proxy removes this
extension before forwarding KV parameters.

The companion patch keeps an API-process-local bounded cache of 8192 grids to
handle processor-cache hits that omit tensors. Entries are accessed
synchronously on the API event loop and are lost on restart. If a needed grid
has been evicted or cannot be obtained, the prefill response omits the hint.

The proxy preserves the original request for prefill and recomputation.
Missing or malformed metadata, mixed non-image media, and image-count
mismatches use the original request. If the decode server rejects a rewritten
request with HTTP 400, 422, 500, 502, or 503 before response streaming begins,
the proxy performs one fresh prefill and retries decode with the original
images.

The companion vLLM interface and its long-term integration need upstream
review. This proxy change alone takes the original-image path when its
prefill server does not return the extension.

## Validation

The deployed prototype was tested on Ascend A5 with one TP1 prefill instance
and one TP1 decode instance, using Qwen3.6 W8A8C8 and a 500-request multimodal
workload averaging approximately 10K input tokens with 300 output tokens.
Of those requests, 499 had identical prompt token IDs and image-placeholder
ranges through the original-image and metadata paths. One AVIF image failed
to decode in the environment in both cases.

Excluding the first call and the failed request, isolated frontend rendering
averaged 55.71 ms with original images and 6.08 ms with metadata. A full run
at a fixed 300 ms dispatch interval changed output throughput from
743.26 to 746.94 tokens/s and mean TTFT from 25.77 to 25.12 seconds.
The throughput difference is small and comes from a single pair of runs.
A3 and other models have not been measured.

The HTTP tests cover metadata ordering, preserved prefill messages,
malformed metadata fallback, and fresh prefill after a decode rejection.

```bash
python -m pytest --confcutdir=tests/ut/proxy tests/ut/proxy/test_load_balance_proxy_server.py
```
