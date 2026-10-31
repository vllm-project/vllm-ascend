# Experimental candidate compaction for DeepSeek V4.1 on A2

This standalone prototype gathers candidate index K into a shorter paged cache
before invoking the existing native INT8 QLI consumer. It does not change model
execution or enable a new serving path.

**Status:** CPU selection semantics have been checked. Triton compilation, NPU
numerical agreement, ACL graph execution, and performance remain unvalidated.
No hardware speedup is claimed.

## Motivation and data flow

The current A2 consumer computes QK across the full history before penalizing
positions outside the candidate blocks. Layer 20 produces 2048 candidate blocks
of eight positions. Consumers 24, 28, 32, and 36 use the same index K and candidate
blocks, but each computes its own queries, head weights, scores, and Top512.

The experiment separates physical sparsity from changes to quantized arithmetic:

1. Sort candidate blocks and determine each query's causal visibility.
2. Gather only the required INT8 K and FP16 scales through the original page table.
3. Generate native QLI scheduling metadata for the compact lengths.
4. Run the unchanged native INT8 consumer on the compact paged cache.
5. Restore logical position IDs and use the existing chronological index preparation.

The four consumers can reuse one pack and schedule within a forward. They cannot
reuse their scores or selections. Scratch belongs to that forward and must not
be shared by concurrent forwards.

## Preserving candidate-shortage behavior

The native consumer gives outside-candidate positions a score of `-1e30`, rather
than deleting them. They can fill the result when fewer than TopK candidate
positions are visible.

The pack therefore contains the union of candidate blocks and the first
`ceil(TopK / 8)` blocks, in logical order. For Top512 this adds at most 64 blocks.
The added noncandidate positions remain noncandidates in the compact QLI call;
they retain the same penalty.

When at least 512 positions are visible, the union contains at least 512 visible
positions. Any omitted positions have the same outside-candidate penalty. When
fewer than 512 positions are visible, the prefix covers all of them. This preserves
the TopK score set for the normal finite INT8 score range above `-1e30`.

**Equal-score IDs may still differ** because compaction changes the native
sorting and merge partitions. Validation reports exact index agreement separately
from different indices with exactly equal recomputed quantized scores. Equal
scores do not establish identical downstream attention or model outputs.

The maximum compact input is 2112 blocks, or 16896 positions, for 2048 candidates
and Top512. At 128K history this is about 7.76 times fewer scoring positions,
**not a measured speedup**. Packed K/scales require about 2.09 MiB per query.
Gather reads, writes, scheduling, and restoration must all be included in timings.

## Files and contracts

| File | Purpose |
| --- | --- |
| `kernels.py` | Triton plan, paged gather, native QLI wrapper, and position restoration |
| `reference.py` | Torch-free dense and compact selection oracles |
| `test_reference.py` | CPU semantic comparisons, including partial blocks, shortages, ties, and long histories |
| `benchmark.py` | Native comparison, Hybrid cache checks, timing, graph refresh, and profiling |

Inputs follow the existing source contract: unique candidate block IDs with `-1`
padding, width-128 INT8 index K, FP16 scales, and contiguous cache rows. Actual
page strides are honored, with page addresses promoted to INT64 before stride
multiplication. FP32 block sorting covers the model's supported integer range.

Each query becomes an independent compact QLI request, retaining its original
request mapping and causal visibility. This changes native scheduling groups.
The intended first target is C1 decode with complete index K on each TP rank,
without DCP. Per-query packing is too expensive in memory and duplicate work to
unconditionally apply to long prefill.

## Validation and benchmark commands

Run the CPU checks from the repository root:

```bash
python3 -m unittest discover -s benchmarks/experimental/candidate_sparse_a2 -p test_reference.py -v
```

The five tests make 3340 dense-versus-compact selection comparisons. They do not
compile or execute Triton kernels and do not establish NPU correctness.

For hardware runs, enter this directory in an A2 environment with matching
vLLM-Ascend native QLI V3, torch_npu, and triton-ascend packages:

```bash
cd benchmarks/experimental/candidate_sparse_a2
python benchmark.py --lengths 513 16385 131072 --requests 1 --consumers 1 4 --samples 3 --repeats 5 --output smoke.json
```

The default C1 Hybrid cache uses a single backing allocation per page: 131072
latent-KV bytes, 16384 index-K bytes, and 256 scale bytes. Physical pages are
shuffled. Every packed key byte and scale is compared with its original page
before timing. Validation also checks request mapping, candidate remapping,
visibility, unique IDs, valid counts, chronological order, and padding.

Exercise shortages, ties, partial blocks, mixed requests, and multiple queries:

```bash
python benchmark.py --lengths 1 7 8 9 511 512 513 16385 --requests 1 3 --consumers 1 --patterns empty one prefix ties --samples 1 --repeats 1 --output edges.json
python benchmark.py --lengths 6145 32771 --requests 2 --queries 5 --consumers 1 --samples 1 --repeats 1 --output multi-query.json
```

Compare decode latency and ACL graph replay:

```bash
python benchmark.py --lengths 16384 32768 65536 131072 --requests 1 2 8 --consumers 1 4 --graph --output performance.json
```

Graph validation refreshes K, scales, page tables, candidates, queries, weights,
query scales, lengths, and positions in place. The reference schedule is rebuilt
for the refreshed lengths. Unsupported graph operations or validation failures
raise directly; the script does not silently switch execution paths.

| Timing | Included work |
| --- | --- |
| `dense_existing_metadata` | Full-history native QLI and chronological preparation, using the existing schedule |
| `sparse_full_including_pack_and_metadata` | One pack, one new schedule, then QLI, restoration, and preparation for one or four consumers |
| `sparse_pack_each_consumer` | Separate packing and scheduling for each consumer; an eager comparison for four consumers |
| `sparse_score_and_restore_only` | QLI, restoration, and preparation with packing and scheduling excluded; a diagnostic only |

Native schedule generation runs on AICPU. Compact lengths depend on device
candidate data, so the full sparse path intentionally includes this additional
dependency and scheduling cost. The dense reference uses its already available
schedule. Do not compare the score-only diagnostic as if it were the full path.

JSON records software versions, code hashes, exact-index agreement, all raw event
and wall-clock samples, and packed cache memory. Completed phases are also
appended immediately to a matching JSONL file. Measurement order rotates.
Four consecutive consumers are a microbenchmark without intervening Transformer
layers; cache behavior and end-to-end TPOT require actual model validation.

Collect a profile outside the timed measurements:

```bash
python benchmark.py --lengths 131072 --requests 1 --consumers 4 --samples 1 --repeats 1 --profile-dir profile --output profile-case.json
```

Optional C2 visibility and high-address checks:

```bash
python benchmark.py --lengths 6145 --requests 2 --queries 5 --ratio 2 --consumers 1 --samples 1 --repeats 1 --output c2.json
python benchmark.py --lengths 131072 --requests 32 --consumers 1 --samples 1 --repeats 1 --output high-address.json
```

The latter needs sufficient memory for 32 Hybrid caches and crosses INT32 byte
addressing. CPU tests cover empty history; mixed NPU cases include empty request
rows. The model already skips QLI for a completely empty batch, so that is not a
default benchmark case.

## Remaining work before serving integration

Validate A2 compilation and quantized numerical behavior, characterize tie IDs,
check refreshed graph inputs, and demonstrate full-path latency improvement.
Then validate model accuracy and profile the actual decode critical path before
enabling any model dispatch. Layer 20's initial full-history candidate generation
is outside this experiment.

If packing or AICPU scheduling consumes the gains, a subsequent native Cube
implementation can directly load candidate blocks, avoiding compact-cache GM
writes and rereads. This prototype first measures whether reducing scoring work
is worthwhile with unchanged native INT8 arithmetic.
