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

The existing architecture-specific template-matrix check covers the optimized
and fallback kernel selections:

```bash
python -m pytest --noconftest -q tests/ut/ops/test_aurora_tiling_keys.py
```

No new test scripts are included in this optimization. Numerical checks and
before/after A3 performance measurements were run separately; the PR records
the environment, results and limitations.

Benchmark baseline and candidate with identical model, workload, cache budget
and software versions. Single-round end-to-end TTFT/TPOT differences do not
establish isolated kernel speedup. Record the native steady-stage sample count
and duration; short windows cannot establish stable gains. Production builds
contain no address-cache diagnostic printing.
