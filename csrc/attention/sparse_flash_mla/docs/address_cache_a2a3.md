# A2/A3 CSA address caching

DeepSeek V4.1 CSA can reuse logical-to-physical KV addresses across query heads
and query rows. The A2/A3 implementation computes the addresses in vector code
and retains a bounded per-AIV cache, avoiding repeated scalar block-table reads.
Attention arithmetic and the operator interface are unchanged.

## Selection and fallback

The host enables the cache only for CSA with TND queries, PA_BBND KV, at most
64 query heads, one KV head, sparse block size 1, and TopK 512. The compressed
page size must be a positive power of two, and the block-table row width must
be at most 1024 entries. All physical KV-pool element offsets must fit int32;
a narrow logical block table alone does not guarantee this bound.

The device additionally checks that it is processing the complete first
512-index slice and that its sparse indices are valid. Padded indices or an
unsupported slice retain the scalar path. Other architectures keep their
existing dispatch.

The table width is the number of block mappings allocated per request, rather
than the number of requests or the maximum physical block ID. For example,
`--max-model-len 65536 --block-size 128` gives a nominal width of 512, but actual
operator shapes and physical KV-pool allocation still determine eligibility.
Reducing scheduling concurrency alone does not reduce that row width.

## Validation

CPU checks cover physical-pool overflow boundaries, flag propagation and the
architecture-specific template matrix:

```bash
python -m pytest --noconftest -q \
  tests/ut/ops/test_sparse_flash_mla_address_cache.py \
  tests/ut/ops/test_aurora_tiling_keys.py
```

Run the numerical cases on an A2/A3 NPU with matching Python bindings and OPP:

```bash
pytest --noconftest -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_sparse_flash_mla.py \
  -k csa_v0_paged_address_cache_against_fp32
```

The reference computes attention independently in FP32. Cases include page
sizes 64/128, padded page strides, shuffled block mappings, duplicate and padded
indices, 64/128 heads, row widths 1024/1025, and NPU graph replay. A 256-valid
index case still supplies a supported TopK-512 tensor padded with `-1`.

Benchmark baseline and candidate with identical model, workload, cache budget
and software versions. Single-round end-to-end TTFT/TPOT differences do not
establish isolated kernel speedup. Record the native steady-stage sample count
and duration; short windows cannot establish stable gains. Production builds
contain no address-cache diagnostic printing.
