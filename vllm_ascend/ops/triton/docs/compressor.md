# DeepSeek V4.1 Compressor

## Interface and integration

Use `compressor` for projection, pooling, normalization and state update, or
`compressor_from_projected` for the token-aligned PCP entry after projection.
The public wrappers select launch geometry and preserve pooling-before-update
ordering: both kernels can access the same ring slot. Callers must serialize
operations that mutate the same state pages.

`compressor` accepts padded or packed BF16 hidden states, BF16 projection and
normalization weights, FP32 state, request lengths, start positions and page
indices. It returns BF16 pooled rows and updates the trailing state in place.
`compressor_from_projected` accepts FP32 projected values/scores and packed
request metadata, with optional gathered predecessor rows. It returns
token-aligned BF16 rows; invalid page zero and unused token rows remain zero.
The general entry permits page zero. See the wrapper docstrings for layouts.

## Runtime extents and specialization

Request count, group/write slots per request, total slots and allocated cache
page count are runtime scalars in pooling and state update. These scheduling
extents vary with batching and cache allocation without changing tensor layout.
`do_not_specialize` prevents scalar-value/alignment specialization from creating
additional variants for those extents. Runtime masks still bound all accesses.
Cache page arithmetic remains INT64 before multiplication for large allocations.

Head width, compression ratio, ring size, chunk size, projection tiling, epsilon
and layout/algorithm flags remain constexpr because they change generated math
or memory layout. Projection autotuning remains keyed by token count. Changing
those parameters, tensor dtypes or target devices can still compile variants;
this change does not promise zero compilation for every workload.
Both entry points support the existing A2/A3/A5 dispatch. No warmup registration
or model configuration is introduced.

## Validation

The independent PyTorch reference and the nightly operator suite cover padded
and packed requests, ratios 2/3/8/9/16/128, projected widths 128/512, ring and
gathered predecessors, odd starts, changing graph inputs, empty ranks, invalid
pages and large page indices. Run on an Ascend device:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_compressor_triton.py
```

Reference comparisons check both output and mutated state. Recompilation
measurements must hold constexpr parameters/dtypes constant, vary only runtime
extents and separate cold compilation from warmed latency. Record the Triton
version and JIT artifacts with those measurements; private compiler hooks are
not part of the operator's runtime contract. Whole-model GPQA accuracy and
serving latency require separate model runs and are not implied by operator
reference checks.
