# GDN commit metadata staging

## Functionality

`stage_gdn_commit` copies and pads one cache group's integer request metadata
into persistent graph inputs. It replaces individual fill/copy operations
before replaying the recovery graph. It does not change cache layout, plan
semantics or state arithmetic.

## Interface and constraints

The production caller is `GDNCommitGraph.update`. `source` and `target` expose
indices, query offsets, an optional sampler mapping, and optional align block
tables/computed lengths. The other inputs are the sampler's accepted-count
vector and a persistent output vector. All tensors are on the same device.
`copy_accepted` defaults to true. The first group copies/pads that shared vector;
later groups pass false and leave it untouched. Each group still stages its own
page IDs, offsets and optional align metadata.

Sources may use non-unit positive strides. Destinations are contiguous,
separate allocations. Input/output aliasing is not supported. Integer inputs
use int32 or int64. The caller validates sampler and group capacities; mapping
values must select valid sampler rows. Table rows may include extra graph
padding, which is not copied beyond the actual sampler rows.

One program processes 256 elements. Query offsets are extended with the last
real offset; unused indices, mappings, accepted counts and table rows become
zero. Each group's page IDs are copied separately. `batch` and `sampled_rows`
are runtime scalars excluded from JIT specialization; bucket/capacity, integer
types, align/mapping/copy modes and source strides determine compiled variants.

## Correctness and tests

All outputs are integer/index data and require exact equality with the
reference. The test entry is:

```bash
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_stage_gdn_commit.py
```

Coverage includes int32/int64, B1/B3/B125/B192, mapped and direct rows,
normal/align, non-contiguous sources, padded table rows and preserving accepted
counts when copying is disabled. The same test
helper can run the actual Triton source through the CPU interpreter without
initializing NPU. CPU interpretation and offline binary compilation do not
prove device execution, latency or model accuracy.

## Performance validation

Compare full recovery and model steps at matching shapes/provider, including
the staging launch and graph replay. Count saved CPU launches and added device
work; kernel-only speed is insufficient. Warm actual graph buckets before the
formal run, then report MTP acceptance and no-profiler TTFT/TPOT/throughput.
