# DeepSeek V4.1 Long KV store

## Description

On Atlas A2 (910B), DeepSeek V4.1 can publish its main BF16 Long KV cache
from the same Triton launch that normalizes the compressed latent. C1 does
this after WKV projection. C2 does this inside token-aligned pooling for each
completed pair. The output latent is unrotated for the Indexer projection.

For a BF16 input row `x`, RMSNorm computes
`z = BF16(x * w * rsqrt(mean(x * x) + eps))`
in FP32. C2 rounds the pooled row to BF16 before normalization. The cache
store copies the first 448 dimensions of `z` and rotates the last 64
dimensions in adjacent pairs using the supplied cosine and sine. C1 uses
current-token RoPE. C2 uses the compression group's source RoPE.

## Parameters

| Parameter | Direction | Data type and layout | Purpose |
| --- | --- | --- | --- |
| `projected` | Input | BF16 contiguous `[T, 512]` | C1 WKV projection output |
| `cache_store.weight` | Input | BF16 contiguous `[512]` | Compressor RMSNorm weight |
| `cache_store.eps` | Attribute | Float | Compressor RMSNorm epsilon |
| `cache_store.cache` | In-place output | BF16 `[pages, page_size, 1, 512]` | Main Long KV cache with physical page and row strides |
| `cache_store.slots` | Input | Integer `[T, 2]` | Builder-provided physical page and row coordinates |
| `cache_store.cos`, `cache_store.sin` | Input | FP32 or BF16 token-aligned rows of width 64 | Interleaved source RoPE values |
| Return value | Output | BF16 `[T, 512]` | Unrotated normalized latent for the Indexer |

C1 uses `normalize_and_store_c1`. C2 passes the optional `cache_store`
to [`compressor_from_projected`](compressor.md) along with its usual
projection and ring inputs.

## Constraints

- Production dispatch selects A2 pure decode without PCP and with unbiased
  compressor RMSNorm and standard BF16 Long KV. Other inputs use the
  existing compressor and cache writer.
- Input tensors reside on the same NPU. Slot pairs and RoPE rows have a
  contiguous last dimension. Valid slots are in bounds and unique. A negative
  coordinate suppresses that row's cache write.
- C2 writes completed pairs and leaves incomplete and padded latent rows
  zero. Pooling reads prior ring entries before the separate ring update.
- ACL Graph callers keep tensor addresses stable and refresh their contents
  on replay. Token count and C2 request extents are runtime values. Layout
  strides, epsilon, and dimensions specialize the kernel. The C2 cache-writing
  body is a separate specialization from plain pooling.

## Origin and differences

The write-in-compression design follows [vLLM #67364](https://github.com/vllm-project/vllm/pull/67364).
The RMSNorm and interleaved RoPE math follows the Ascend SWA writer in
[#21183](https://github.com/vllm-project/vllm-ascend/pull/21183). Long KV writes
target A2's BF16 cache layout and also produce an unrotated dense latent for
the Indexer. C2 reuses the existing ring reads and updates.

## Test cases

The test compares against separate pooling, CANN RMSNorm, RoPE, and scatter.
It checks the latent and written slots within the repository's BF16 tolerance.
Untouched cache bytes and FP32 ring state are checked exactly. It covers
C1/C2, strided pages, skipped slots, odd-start completion, and graph replay.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_dsv41_long_kv_store.py
```
