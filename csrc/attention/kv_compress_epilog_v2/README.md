# KvCompressEpilogV2

The host, kernel, and input validation are ported from
[cann-recipes-infer at 0322b30fa0c31229d1aa917921ddc5acb09f6c2e](https://gitcode.com/cann/cann-recipes-infer/tree/0322b30fa0c31229d1aa917921ddc5acb09f6c2e/ops/ascendc/src/kv_compress_epilog_v2).
The CANN Open Software License notices are preserved; the agreement is
included in [LICENSE](LICENSE).

`torch.ops._C_ascend.kv_compress_epilog_v2` quantizes BF16 rows and writes
specified slots in a flat or paged cache in place. Negative and out-of-range
slots leave the cache unchanged. This entry point has its own cache ABI and
does not replace the existing `kv_compress_epilog` operator.

Supported modes are `mxfp8_bf16` with groups of 32 and `mxfp4_bf16` with
groups of 16 or 32. Each token stores its packed data followed by BF16 scales.
The production 512-element rows occupy 544 bytes for SWA MXFP8 and 320 bytes
for compressed MXFP4 with group size 16. FP4 conversion rounds ties to even;
it differs from the indexer's FP4/E8M0 format.

Paged caches have shape `[blocks, block_size, 1, row_capacity]`. Token
payloads are packed consecutively inside each physical block. The adapter
passes `cache.stride(0)` so non-contiguous block views remain supported.
`x_scale` is reserved and must equal 1.0. The public adapter validates the
mode, group size, dtype, dimensions, and read-only input contiguity.
