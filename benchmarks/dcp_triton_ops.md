# DCP operator measurements

## Grouped exact merge and fused preparation: 2026-10-30 follow-up

The frozen reference is PR40 at `04715a6e2`, after the first native-sort
optimization. The final production file has SHA256
`b3f903c3df00e0cb925000295ae740c6b9cde74944d45ac87374b2174454cc92`.
Measurements use Ascend910_9362 with 40 vector cores per logical device,
CANN 9.2.0, torch 2.10.0+cpu, torch_npu 2.10.0.post7, Triton 3.6.0 and
Python 3.11.10. Paired vLLM remains unchanged at
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

One Triton kernel prepares finite FP32 scores and exact FP32 ID keys before
the existing two stable native sorts. This removes separate validity, cast
and masking kernels. Candidate packing and the communication layout remain
unchanged.

Shape decisions use ordinary Python integer arithmetic. In the measured
Triton version, host calls to the constexpr `cdiv` wrapper cost about 2.7 us
each. Removing these wrappers resolves a small Q16/DCP8 eager regression:
four alternating-order rechecks improve from 281-294 us to 267-269 us,
while graph latency remains about 156 us versus 234 us. All JIT kernel ASTs
are unchanged by this host-only correction, and all 560 tests pass again.

Presorted candidate merge may first retain K winners from each group of four
ranks, then merge those sorted partial results. A discarded entry already has
K entries at least as good in its group, so discarding it cannot change the
global Top-K values or ID order. Equal score/ID pairs remain interchangeable.
The second launch uses the original DCP world for owner-local remapping;
the number of partial groups must never replace the owner coordinate system.
The two launches use the same stream, with no cross-program barrier or host
synchronization.

Grouping adds scratch traffic and a launch. Dispatch therefore requires at
least 10% fewer estimated merge waves, using query rows, DCP size and vector
core count. The Q1/16/64/128/129/256/512 sweep supports this margin: on the
measured device, Q128/DCP8 and DCP16 group, while Q256 and Q512 keep one pass.
There is no chip-model check or DCP whitelist. AllGather boundaries are
unchanged. Group scratch is `ceil(DCP/4) * Q * K * 8` bytes when enabled.

Production compute with fixed peer buffers replacing HCCL, Q128, K2048,
interleave768, FP32 scores; graph medians in microseconds:

| DCP | Previous PR40 | New, forward order | New, reverse order |
| ---: | ---: | ---: | ---: |
| 4 | 432.09 | 371.63 | 370.90 |
| 8 | 707.29 | 624.73 | 623.21 |
| 16 | 1259.99 | 1097.55 | 1099.87 |

The reverse-order reference medians are 433.62, 709.51 and 1257.87 us,
respectively. Q128/DCP16 compute latency decreases by 12.6-12.9%.
All 48 paired comparisons, spanning FP32/BF16 and the row/DCP grid above,
pass exact eager and graph checks. Every measured graph pair improves;
the reduction ranges from 2.9% to 53.2% across this grid. These are
shape-specific candidate-compute results, not model throughput.

Real single-node eight-rank HCCL, Q128, FP32 scores, K2048:

| Candidate path including AllGather | PR31 baseline | Previous PR40 | New |
| --- | ---: | ---: | ---: |
| Forward order | 916.67 us | 805.45 us | 720.38 us |
| Reverse order | 909.27 us | 811.25 us | 723.48 us |

Each cell is the median of eight per-rank graph medians, from nine samples
of 20 calls. Compilation, graph capture and profiling are excluded. The
reverse sweep does not profile. New latency is 10.6-10.8% lower than the
previous PR40 and 20.4-21.4% lower than PR31. All 48 rank results across
three versions and two sweeps match independent CPU Top-K oracles in eager
and graph mode. Peak extra PyTorch allocation is 68.0/35.5/30.0 MiB for
PR31/previous PR40/new; this excludes allocations outside the PyTorch allocator.

The final focused suite passes 560 tests, including the existing NPU merge
suite expanded to DCP16, exact FP32 boundaries and ties, strided inputs,
invalid rows, grouped owner mapping, and changed-input graph replay.
The old test that prohibited all native sort/gather calls now checks the
single collective and exact result; native local sorting was already part
of the previous PR40.

Compare a frozen previous PR40 checkout with the current source:

```bash
python benchmarks/dcp_candidate_merge_ab.py \
  --reference-root "$PREVIOUS_PR40" --expected-source-root "$CURRENT_PR40" \
  --expected-source-sha "$(sha256sum "$CURRENT_PR40/vllm_ascend/ops/triton/sharded_indexer.py" | cut -d' ' -f1)" \
  --rows 128 --world 16 --score-dtype fp32 --iterations 20 --repeats 9 \
  --only compute_reference compute_candidate --output compare.json
```

Select the intended source and paired vLLM before importing packages; the
driver rejects an unexpected candidate path or SHA256. Native profiling
uses the existing `dcp_four_point_ab.py` driver and its distributed command
below, with eight assigned devices and `--world 8` for this follow-up.

Limits: current A5 performance, real DCP16 HCCL, full-model accuracy and
serving throughput were not rerun for this follow-up. A separate
CANN9.1/Triton3.2 attempt failed exact merge checks in the frozen reference
as well, and is excluded from these results; that compatibility issue is
unresolved. One Q1/DCP8/BF16 eager grid pair measures 317.84 -> 332.64 us;
this outlier was not repeated, and an all-eager speedup is not claimed.
Runtime versions and original profile archives must accompany any comparison.

## Native local sorting and exact candidate merge: 2026-10-30

Measured on one Ascend910_9362 device with 40 vector cores, CANN 9.2.0,
torch 2.10.0+cpu, torch_npu 2.10.0.post7 and Triton 3.6.0. The fixed baseline
contains PR8 and PR31 (`b051d51a9`); original PR40 is `eff72887f`.
Paired vLLM is unmodified at `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

The candidate uses two native stable local sorts, reuses their FP32 ID values
instead of gathering INT64 permutations, and packs with Triton. Local-to-global
mapping reuses one integer quotient; the first sorted tile initializes merge
state directly. Scores remain unchanged, ties use ascending global IDs, and
transport remains INT32 IDs plus FP32 score bits. All ranks keep the existing
shape-based collective boundaries. No chip-model gate is added.

The table measures production candidate preparation with fixed peer buffers
in place of AllGather. Q=128, K=2048, interleave=768 and scores are FP32.
Graph medians use nine samples of 20 calls, excluding compilation and profiling.

| DCP data partitions | PR31 baseline | Original PR40 | Native local sort |
| ---: | ---: | ---: | ---: |
| 4 | 607.44 us | 909.07 us | 424.31 us |
| 8 | 805.18 us | 1183.48 us | 709.36 us |
| 16 | 1405.98 us | 1724.33 us | 1253.82 us |

All 18 Q128 comparisons across DCP4/8/16, FP32/BF16 scores and the three
implementations pass exact eager and graph output checks. With preloaded peer
buffers, native sorting uses more temporary memory than original PR40:
approximately 36.5 MiB versus 4 MiB at Q128/DCP16.

The same host also ran a real 16-rank HCCL comparison using all 16 logical
devices. Each cell below is the median of the 16 per-rank graph medians,
with Q128, FP32 scores and the same nine samples of 20 calls.

| Candidate path including AllGather | PR31 baseline | Original PR40 | Native local sort |
| --- | ---: | ---: | ---: |
| First sweep | 1626.97 us | 1924.33 us | 1465.26 us |
| Reverse-order sweep | 1657.43 us | 1960.20 us | 1470.89 us |

The first sweep's slowest-rank medians are 1631.72, 1929.97 and 1477.00 us,
respectively. The reverse-order sweep runs without profiling and confirms
10-11% lower latency than PR31 and 24-25% lower latency than original PR40.
All 16 ranks match independent CPU Top-K oracles in eager and graph mode in
both sweeps. The focused suite passes 441 CPU/NPU tests, including strided
inputs, FP32 ID boundaries, invalid scores and repeated graph replay.
Rank zero retains native profiling traces. Communication layout
and bytes are unchanged relative to original PR40; the PR31 baseline splits
Q128 into two collectives. Full-model execution and A5 regression remain
unverified for this follow-up.

Run the current driver against separately selected source roots:

```bash
PYTHONPATH="$CANDIDATE:$PAIRED_VLLM" python "$CANDIDATE/benchmarks/dcp_four_point_ab.py" \
  --expected-source-root "$CANDIDATE" --variant candidate \
  --rows 128 --world 16 --score-dtype fp32 --iterations 20 --repeats 9 \
  --only candidate_prepare_compute --profile candidate-profile --output candidate.json
```

Use `--variant baseline` for the fixed PR31 source. The driver rejects an
unexpected import root and reports failed cases with a nonzero exit status.
`--profile` captures one warmup step and three active steps in native
`*_ascend_pt` format. `--distributed` requires a real HCCL world matching
`--world`; direct batched LSE probes are restricted to single-device mode.
For the single-node HCCL comparison, use the assigned 16 logical devices and:

```bash
PYTHONPATH="$CANDIDATE:$PAIRED_VLLM" torchrun --standalone --local-addr=127.0.0.1 \
  --nnodes=1 --nproc-per-node=16 "$CANDIDATE/benchmarks/dcp_four_point_ab.py" \
  --expected-source-root "$CANDIDATE" --variant candidate --distributed --world 16 \
  --score-dtype fp32 --iterations 20 --repeats 9 --only distributed_topk_local \
  --profile candidate-hccl-profile --output 'candidate-rank{rank}.json'
```

The historical measurements below refer to their stated source revisions.

## Fused DCP candidate exchange and LSE combine: 2026-10-29

Measured on Ascend950DT_9581 with eight NPUs, 64 vector cores, CANN 9.2.0,
torch 2.10.0+cpu, torch_npu 2.10.0.post7 and Triton 3.6.0. The release baseline
is `4edde6e1ceea71426f8243676919a76cb62c2f9a`; paired vLLM remains unmodified
at `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

The workload uses 128 query rows, DCP8, K=2048 and interleave=768. Input
indices/scores are `[128,1,2048]` INT32/BF16, packed candidates are
`[128,2048,2]`, and the gathered buffer is `[1024,2048,2]`. The 128 dimension
counts query rows, not requests. LSE combine output is `[128,12,512]` BF16.

| Real eight-rank HCCL candidate path | Release | Fused path |
| --- | ---: | ---: |
| Graph latency, median rank medians | 364.63 us | 226.84 us |
| Peak temporary allocation | 83 MiB | 19 MiB |

Candidate-path latency decreases 37.8%. Measurements use nine samples of ten
calls per mode and rank, in a same-host release/previous/final/previous/release
sweep; compilation, graph capture and profiling are excluded. Individually
measured graph medians are 53.95 us for sorted pack, 90.95 us for global merge,
105.52 us for owner-local merge and 26.16 us for LSE combine. These component
measurements must not be added to the complete-path improvement; LSE combine
is outside the candidate path.

Profiling confirms one Triton pack/local-sort, one logical AllGather and one
Triton merge/finalize per call, without standalone cast, sort or gather
computation. Packing converts native BF16 comparison scores losslessly to FP32.
Merge preserves exact score-desc/ID-asc order, discards losing halves and keeps
only the K winners in UB. LSE fields are decoded once; normalization operates
on the small rank/row tile before output-feature accumulation.

The fused merge is selected by K=2048, without a chip-model check. Smaller K encountered a CANN
`consolidateToTail` compiler error and retain the existing merge implementation.
Shared shape-based byte-budget tiling includes the 256-row single-tile floor
from PR35. All ranks follow the same collective boundaries.

Validation passed 58 device transport/merge/graph cases, 104 device DCP pack/LSE
cases, 89 topology/reference cases and 251 CPU arithmetic cases. All 40 rank
results in the five HCCL sweep groups match independent CPU oracles. These
operator tests do not establish full-model accuracy, throughput, DSpark
acceptance or combined PCP+DCP model execution.
Two additional CPU dispatch tests verify that candidate fusion does not query
the chip model; removing those dispatch checks leaves the kernel bodies unchanged.

Use the benchmark driver from this revision with separately selected baseline
and candidate source roots. It rejects an unexpected imported checkout:

```bash
PYTHONPATH="$CANDIDATE:$PAIRED_VLLM" torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  "$CANDIDATE/benchmarks/dcp_four_point_ab.py" \
  --expected-source-root "$CANDIDATE" --variant candidate --distributed \
  --only distributed_topk_local --output 'candidate-rank{rank}.json'
```

For the release comparison, select `$BASELINE` in `PYTHONPATH` and
`--expected-source-root`, and use `--variant baseline` with the same driver.
An image's `.pth` must not override the selected source checkout.
The sections below retain measurements of earlier source revisions.

## Triton-only follow-up

Measured on 2026-10-27 on one user-approved Ascend950DT_9582 device with 64 vector
cores, CANN 9.2.0, torch 2.10.0+cpu, torch_npu 2.10.0.post7 and Triton 3.6.0. Paired vLLM:
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, unmodified. This is a different A5
variant from the historical measurements below; compare implementations within
each sweep, not absolute times across devices.

The final source passed 165 device tests and 110 related unit tests in the real
runtime. Two sweeps, with forward and reverse token order, each passed all 24
eager/graph validation cases before recording 48 timing comparisons. Each
comparison uses nine paired AB/BA samples of 20 calls, as described below.

The three references are: prior native tiled merge plus remap; native query
cat/transpose; and the prior Triton LSE encoder plus cat/transpose pack. Candidate
exchange packing and HCCL are excluded from the merge comparison. Each cell below
is **reference us → candidate us** in graph mode, from the forward sweep.

| Tokens | Tiled candidate merge | Query prep | Output pack vs hybrid |
| ---: | ---: | ---: | ---: |
| 1 | 73.37 → 53.58 | 1.90 → 1.88 | N/A |
| 16 | 91.04 → 60.96 | 5.18 → 3.34 | N/A |
| 17 | 106.22 → 76.80 | 5.18 → 3.46 | N/A |
| 64 | 180.57 → 132.03 | 7.70 → 3.22 | N/A |
| 65 | 261.49 → 188.16 | 7.74 → 3.25 | N/A |
| 127 | 363.53 → 264.82 | 9.12 → 3.62 | N/A |
| 128 | 360.99 → 264.86 | 9.14 → 3.61 | 19.93 → 9.26 |
| 129 | 425.85 → 316.82 | 9.14 → 3.73 | 19.52 → 10.83 |
| 256 | 698.46 → 517.50 | 12.13 → 4.43 | 31.63 → 15.62 |
| 512 | 1393.83 → 1046.43 | 10.18 → 6.13 | 78.07 → 42.76 |

Across both sweeps, graph merge improves 1.33–1.49x, query at T>=16 improves
1.49–2.80x, and pack improves 1.80–2.19x. T=1 query is effectively tied (1.013x
forward, 0.993x reverse). Eager merge and pack also improve, but **eager query
still regresses**: for example, T=512 is 22.05 → 39.55 us. These intervals include
host submission gaps; they are not kernel-only latency. The change must not be
described as faster for every mode or as an end-to-end model speedup.

Head-aligned query and pack tiles avoid per-row vector division and preserve
affine source/destination strides. Extra query address clamping was rejected:
it prevented efficient data movement and made the first T=512 candidate take
about 50 us in graph mode. Row/feature masks provide the required tail protection.
Global Top-K score precision, tie order and the packed communication layout are
unchanged. Full-model accuracy, HCCL behavior and PCP feature adaptation remain
outside this single-device validation.

Level-1 profiling supports the DMA explanation: median query Vector ratio
changes from 0.968 to 0.000, while MTE2/MTE3 ratios change from approximately zero
to 0.373/0.303. These counters overlap and are not additive. Profiled durations
are not used for the speedup table. A separate CPU cProfile attributes most of
the remaining eager call cost to Triton's runtime JIT/launcher; its instrumented
times are not latency measurements.

Final measured source SHA256 values under `vllm_ascend/ops/triton/`:

| Source | SHA256 |
| --- | --- |
| `sharded_indexer.py` | `a8cc5d59a6c01d4b634f6e34ffb50b67580844421122d7440e93366228ac52e6` |
| `sparse_index_remap.py` | `dd748e0b516224da4b067fa7d4cfd513271c53f37f267d3dc2acce8bf105b11c` |
| `query_gather_prep.py` | `5784fc57136119ba3960a3da7ece94793fda34bac62d09f3988796a2bfe3a079` |
| `dcp/dcp_a2a.py` | `fdd9641da971e74c4646f6ea27a8895314bf06942f6361504552a4148527668e` |
| `dcp/dcp_a2a_batched.py` | `e43a51b60013476a27a7d77d34dd1020b4f3a9b5e8e1bccfc627ea1cb1c29508` |

Boundary sizes support one token and uneven request lengths. Reproduce:

```bash
python benchmarks/dcp_triton_ops.py --tokens 1 16 17 64 65 127 128 129 256 512 --only candidate_merge_tiled_vs_native output_pack_vs_hybrid query_gather_prep --output triton-only.json
```

## Historical baseline

The remaining tables and conclusions record baseline revision
`9fbf2ad20633492099a6fd66ed0e534363e4c895`; they describe that earlier dispatch,
including its native alternatives, rather than the Triton-only follow-up.

Measured on 2026-10-27 using one managed Ascend950DT_9572 device (56 vector cores), CANN 9.2.1, torch 2.10.0+cpu with torch_npu 2.10.0.post7 and Triton 3.6.0. Paired vLLM: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

## Scope and method

- Tokens: 4, 8, 16, 32, 64, 128, 256, 512. Kimi shapes: DCP8, 96 attention heads, value dimension 512, top-k 2048, cache interleave 768. Received buffers for eight ranks are constructed on one device; HCCL is excluded.
- Candidate transport uses INT32 IDs and FP32 scores; attention output is BF16 and statistics are FP32 with head-major strides. Metadata covers B=min(T,16) and B=T. Sparse remapping also covers K=31/128/4096. Context/ID fixtures reach 1M positions.
- Each of 109 distinct cases passes three eager comparisons and ten graph replays before timing. There are 218 measurements across eager and graph modes. Integer results are exact; decoded LSE uses atol=rtol=2e-6 and BF16 combine uses atol=rtol=0.008.
- For each case, five warmup calls, then nine paired AB/BA samples of 20 calls. Graph mode also warms three replays. Tables use median NPU event duration divided by 20. Compilation and graph capture are excluded. Eager event intervals include host submission gaps; graph mode better represents the device execution path. Allocation requests remain inside the callable (graph replay reuses captured addresses).
- Reference is the equivalent PyTorch operator sequence on NPU, not a single ATen kernel. `candidate_merge` retains two stable native FP32 sorts and gathers. `output_pack` at T>=128 combines a Triton encoder with native concatenation and transpose. These are labeled stages, not standalone fused kernels.
- `output_pack_vs_prior` compares the same final packing implementation with externally computed `maximum + log(sum)` against accepting the two statistics directly. It is not a PyTorch-only reference or a comparison with an older Git revision.
- Three complete final sweeps were run, including reverse token order. The last sweep uses the final dispatch threshold, preserving the scalar combine path at T=32. All 218 final ratios are >=1.00. The first sweep had a 0.028 us difference at T=4 candidate selection (0.996x); later sweeps were 1.001x/1.003x, so this point is effectively tied.

Reproduce in an installed Ascend/vLLM environment with one visible device:

```bash
python benchmarks/dcp_triton_ops.py --output dcp-triton-ops.json
python benchmarks/dcp_triton_ops.py --tokens 512 --only candidate_pack candidate_unpack remap_k2048 lse_encode output_pack output_combine --output profile.json --profile profile
pytest -q tests/e2e/pull_request/one_card/test_dcp_pre_post.py
```

## Per-stage graph measurements

Each cell is **PyTorch/reference us → candidate us (speedup)**. `metadata_b16` means B=min(T,16); `metadata_bT` means B=T. Their identical entries at T<=16 reuse the same measurement.

| Stage | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `candidate_pack` | 27.34 → 7.56 (3.62x) | 29.42 → 7.76 (3.79x) | 32.30 → 7.76 (4.16x) | 32.59 → 7.72 (4.22x) | 34.94 → 7.74 (4.52x) | 40.03 → 7.72 (5.18x) | 50.77 → 9.54 (5.32x) | 64.47 → 16.20 (3.98x) |
| `candidate_unpack` | 18.98 → 7.60 (2.50x) | 20.10 → 7.73 (2.60x) | 22.11 → 7.78 (2.84x) | 24.92 → 7.74 (3.22x) | 30.95 → 7.75 (4.00x) | 45.25 → 7.84 (5.77x) | 82.82 → 12.31 (6.73x) | 195.06 → 31.80 (6.13x) |
| `candidate_select` | 7.79 → 7.76 (1.00x) | 8.74 → 7.71 (1.13x) | 9.63 → 7.75 (1.24x) | 10.04 → 7.72 (1.30x) | 10.73 → 7.71 (1.39x) | 12.05 → 7.73 (1.56x) | 12.79 → 7.76 (1.65x) | 15.36 → 7.78 (1.98x) |
| `candidate_merge` | 70.14 → 45.62 (1.54x) | 76.79 → 51.17 (1.50x) | 98.35 → 70.47 (1.40x) | 147.34 → 115.25 (1.28x) | 255.11 → 213.19 (1.20x) | 381.50 → 328.65 (1.16x) | 659.64 → 561.90 (1.17x) | 1295.68 → 1112.59 (1.16x) |
| `metadata_b16` | 21.64 → 7.73 (2.80x) | 21.62 → 7.71 (2.80x) | 21.84 → 7.80 (2.80x) | 22.56 → 7.75 (2.91x) | 22.72 → 7.74 (2.93x) | 23.32 → 7.77 (3.00x) | 25.11 → 7.71 (3.26x) | 26.97 → 7.76 (3.48x) |
| `metadata_bT` | 21.64 → 7.73 (2.80x) | 21.62 → 7.71 (2.80x) | 21.84 → 7.80 (2.80x) | 22.47 → 7.78 (2.89x) | 22.66 → 7.72 (2.94x) | 23.26 → 7.80 (2.98x) | 26.00 → 7.75 (3.35x) | 28.22 → 7.78 (3.63x) |
| `remap_k31` | 27.40 → 7.71 (3.56x) | 29.62 → 7.76 (3.82x) | 32.83 → 7.77 (4.22x) | 36.10 → 7.76 (4.65x) | 37.59 → 7.80 (4.82x) | 40.29 → 7.80 (5.16x) | 46.13 → 7.74 (5.96x) | 49.51 → 9.89 (5.00x) |
| `remap_k128` | 32.44 → 7.76 (4.18x) | 33.84 → 7.70 (4.40x) | 34.90 → 7.74 (4.51x) | 38.82 → 7.77 (4.99x) | 45.94 → 7.78 (5.90x) | 49.59 → 7.71 (6.43x) | 56.48 → 7.74 (7.30x) | 60.77 → 10.99 (5.53x) |
| `remap_k2048` | 45.80 → 9.48 (4.83x) | 49.20 → 9.57 (5.14x) | 55.44 → 9.74 (5.69x) | 59.90 → 10.15 (5.90x) | 68.07 → 18.23 (3.73x) | 79.77 → 26.13 (3.05x) | 104.11 → 41.68 (2.50x) | 139.41 → 80.35 (1.74x) |
| `remap_k4096` | 53.01 → 18.19 (2.91x) | 59.09 → 18.27 (3.23x) | 63.16 → 18.46 (3.42x) | 68.66 → 19.08 (3.60x) | 85.04 → 36.16 (2.35x) | 111.22 → 52.60 (2.11x) | 147.00 → 85.53 (1.72x) | 229.82 → 167.46 (1.37x) |
| `lse_encode` | 41.38 → 7.74 (5.35x) | 46.55 → 7.75 (6.00x) | 49.22 → 7.76 (6.35x) | 52.53 → 7.76 (6.77x) | 56.52 → 7.82 (7.23x) | 69.05 → 7.79 (8.87x) | 71.61 → 7.78 (9.20x) | 80.88 → 9.08 (8.91x) |
| `output_pack` | 49.23 → 9.77 (5.04x) | 55.46 → 9.64 (5.75x) | 60.16 → 13.02 (4.62x) | 66.94 → 21.58 (3.10x) | 70.21 → 34.58 (2.03x) | 89.88 → 23.99 (3.75x) | 107.08 → 35.07 (3.05x) | 176.73 → 81.41 (2.17x) |
| `output_combine` | 49.64 → 8.52 (5.83x) | 54.00 → 13.70 (3.94x) | 58.56 → 23.02 (2.54x) | 63.91 → 37.34 (1.71x) | 75.82 → 45.00 (1.68x) | 102.64 → 45.37 (2.26x) | 178.96 → 87.23 (2.05x) | 282.69 → 169.92 (1.66x) |
| `output_pack_vs_prior` | 11.84 → 9.73 (1.22x) | 12.47 → 9.73 (1.28x) | 15.83 → 13.06 (1.21x) | 24.92 → 21.53 (1.16x) | 38.01 → 34.60 (1.10x) | 27.43 → 24.05 (1.14x) | 39.59 → 35.12 (1.13x) | 85.07 → 81.46 (1.04x) |

## Per-stage eager measurements

Same units and timing boundary as above. Host dispatch gaps are included.

| Stage | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `candidate_pack` | 129.75 → 32.92 (3.94x) | 153.86 → 34.77 (4.43x) | 146.94 → 33.01 (4.45x) | 150.39 → 32.47 (4.63x) | 147.88 → 33.52 (4.41x) | 148.27 → 33.77 (4.39x) | 149.37 → 34.28 (4.36x) | 146.85 → 34.29 (4.28x) |
| `candidate_unpack` | 57.14 → 33.70 (1.70x) | 65.82 → 35.72 (1.84x) | 65.05 → 34.17 (1.90x) | 65.26 → 34.32 (1.90x) | 65.45 → 35.19 (1.86x) | 72.77 → 36.62 (1.99x) | 84.35 → 35.95 (2.35x) | 195.51 → 34.62 (5.65x) |
| `candidate_select` | 40.29 → 29.41 (1.37x) | 40.53 → 30.48 (1.33x) | 39.29 → 29.58 (1.33x) | 39.43 → 29.22 (1.35x) | 40.04 → 29.42 (1.36x) | 39.61 → 29.42 (1.35x) | 40.29 → 29.89 (1.35x) | 40.54 → 29.51 (1.37x) |
| `candidate_merge` | 166.00 → 125.71 (1.32x) | 167.88 → 125.03 (1.34x) | 162.58 → 121.90 (1.33x) | 161.38 → 122.70 (1.32x) | 258.07 → 217.53 (1.19x) | 390.00 → 330.06 (1.18x) | 659.47 → 563.82 (1.17x) | 1296.76 → 1115.39 (1.16x) |
| `metadata_b16` | 191.97 → 26.65 (7.20x) | 191.30 → 26.66 (7.18x) | 193.82 → 25.43 (7.62x) | 191.56 → 25.84 (7.41x) | 192.50 → 26.11 (7.37x) | 193.86 → 26.30 (7.37x) | 197.12 → 25.89 (7.61x) | 201.07 → 27.17 (7.40x) |
| `metadata_bT` | 191.97 → 26.65 (7.20x) | 191.30 → 26.66 (7.18x) | 193.82 → 25.43 (7.62x) | 188.60 → 25.87 (7.29x) | 192.33 → 24.31 (7.91x) | 189.17 → 25.38 (7.45x) | 191.40 → 25.14 (7.61x) | 190.38 → 25.75 (7.39x) |
| `remap_k31` | 232.93 → 36.46 (6.39x) | 235.04 → 36.36 (6.47x) | 229.33 → 34.35 (6.68x) | 239.88 → 35.72 (6.72x) | 236.27 → 34.55 (6.84x) | 236.72 → 35.67 (6.64x) | 237.86 → 34.76 (6.84x) | 236.15 → 36.42 (6.48x) |
| `remap_k128` | 234.75 → 36.68 (6.40x) | 232.71 → 34.96 (6.66x) | 231.96 → 34.37 (6.75x) | 237.50 → 35.44 (6.70x) | 231.85 → 34.31 (6.76x) | 236.26 → 35.14 (6.72x) | 232.05 → 35.73 (6.50x) | 232.96 → 37.35 (6.24x) |
| `remap_k2048` | 234.28 → 36.45 (6.43x) | 232.31 → 34.88 (6.66x) | 230.37 → 34.20 (6.74x) | 233.67 → 35.74 (6.54x) | 233.84 → 35.48 (6.59x) | 234.83 → 35.40 (6.63x) | 240.13 → 44.25 (5.43x) | 234.56 → 82.62 (2.84x) |
| `remap_k4096` | 234.51 → 35.58 (6.59x) | 231.13 → 34.81 (6.64x) | 230.37 → 34.98 (6.59x) | 231.93 → 35.87 (6.47x) | 230.19 → 38.59 (5.96x) | 230.73 → 54.51 (4.23x) | 231.68 → 87.73 (2.64x) | 238.20 → 169.39 (1.41x) |
| `lse_encode` | 312.05 → 33.82 (9.23x) | 316.18 → 34.24 (9.23x) | 303.17 → 32.59 (9.30x) | 310.85 → 33.56 (9.26x) | 312.49 → 32.39 (9.65x) | 310.68 → 32.74 (9.49x) | 315.91 → 33.05 (9.56x) | 308.54 → 34.24 (9.01x) |
| `output_pack` | 341.61 → 49.40 (6.92x) | 339.74 → 50.06 (6.79x) | 328.60 → 47.04 (6.99x) | 340.58 → 47.34 (7.19x) | 340.65 → 49.17 (6.93x) | 337.43 → 70.48 (4.79x) | 339.30 → 69.93 (4.85x) | 344.50 → 80.80 (4.26x) |
| `output_combine` | 206.59 → 43.10 (4.79x) | 207.53 → 42.96 (4.83x) | 205.67 → 42.37 (4.85x) | 201.58 → 43.34 (4.65x) | 206.22 → 48.75 (4.23x) | 207.81 → 48.98 (4.24x) | 209.43 → 90.36 (2.32x) | 294.00 → 174.94 (1.68x) |
| `output_pack_vs_prior` | 67.74 → 44.86 (1.51x) | 69.32 → 45.92 (1.51x) | 66.56 → 44.01 (1.51x) | 67.59 → 44.25 (1.53x) | 68.92 → 45.17 (1.53x) | 86.47 → 66.89 (1.29x) | 86.25 → 66.63 (1.29x) | 88.19 → 77.86 (1.13x) |

## Optimization evidence

The starting code for this sweep was `daca18dbddcec605c363d3c133743dc220c13853`. These are graph medians for T=512; they compare two code states, separately from the PyTorch comparison above.

| Stage | Starting code (us) | Final (us) | Speedup |
| --- | ---: | ---: | ---: |
| `candidate_pack` | 705.33 | 16.20 | 43.53x |
| `candidate_unpack` | 589.71 | 31.80 | 18.54x |
| `candidate_merge` | 1700.31 | 1112.59 | 1.53x |
| `remap_k2048` | 267.72 | 80.35 | 3.33x |
| `output_pack` | 358.21 | 81.41 | 4.40x |
| `output_combine` | 545.76 | 169.92 | 3.21x |

Candidate packing joins ID/score bits in UB and writes contiguous pairs; unpacking uses per-rank contiguous reads and UB splitting. Static strides simplify address generation. Sparse remapping sorts unique FP32 **position** keys and gathers INT32 payload bits, preserving original top-k order and IDs beyond FP32 integer precision.

Output packing uses 32-row tiles for D=512 below 128 tokens. At larger token counts, the small Triton encoder fuses max/log/sum and exact BF16 transport fields, while native kernels move the wide output. Received-only attention combine uses 32-row tiles and streams ranks without multibuffering to bound UB; the local-contribution path keeps eight-row tiles. Smaller batches retain their measured faster dispatch.

Level-1 profiling, T=512, medians over five calls (profiling overhead included; these durations are not used in the speedup tables):

| Stage / kernel | us | Vector ratio | Scalar ratio | MTE2 ratio | MTE3 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate_pack-t512 / `_pack_candidates` | 16.29 | 0.890 | 0.695 | 0.592 | 0.459 |
| candidate_unpack-t512 / `_unpack_candidates` | 33.84 | 0.350 | 0.274 | 0.862 | 0.600 |
| lse_encode-t512 / `_encode_dcp_lse_kernel` | 8.95 | 0.779 | 0.190 | 0.001 | 0.047 |
| output_combine-t512 / `_fused_dcp_lse_combine_batched_kernel` | 169.02 | 0.912 | 0.284 | 0.179 | 0.066 |
| output_pack-t512 / `_encode_dcp_lse_kernel` | 11.38 | 0.782 | 0.122 | 0.001 | 0.060 |
| output_pack-t512 / `aclnnCat_ConcatD_ConcatD` | 37.95 | 0.373 | 0.347 | 0.519 | 0.491 |
| output_pack-t512 / `aclnnInplaceCopy_TransposeAiCore_Transpose` | 31.33 | 0.000 | 0.114 | 0.806 | 0.533 |
| remap_k2048-t512 / `_remap_sort_a5` | 80.23 | 0.976 | 0.046 | 0.018 | 0.014 |

The original pack was MTE3-dominated (ratio 0.996); unpack was Vector-dominated (0.997). The old K=2048 compaction used a scan kernel plus a scalar-heavy gather. The final remap uses one FP32 position-sort kernel. The final output pack has three kernels per call. All profiled kernels report zero Cube utilization. Pipeline ratios can overlap and must not be summed as time shares.

The final profiler capture overlapped an external vLLM process that started after all three timed final sweeps had completed. Its pipeline counters are qualitative evidence only; no isolated profiler latency or overlap claim is made. The final timing JSON completed at 02:56:07 UTC (host clock), before the observed external worker started at 02:58:06 UTC.

These data support reducing scattered memory movement and scalar address work. They do not establish CV or communication overlap: LI/SFA, projections, HCCL, cross-rank skew and concurrent memory contention are outside this sweep. The packed `[P,H_local,T,516]` wire layout is unchanged; eliminating the remaining transpose would require a matching producer/consumer layout contract and a new end-to-end measurement.

## Correctness and limits

- 41 A5 kernel tests passed, including nonfinite scores, exact tie order, metadata graph mutation, noncontiguous tensors, remap integer payloads above 2^24, and output graph replay with all-invalid contributions and optional local FIA output.
- 101 existing single-card nightly DCP pack/combine cases and six deferred-combine unit tests also passed on the final code. The nightly dispatch assertions cover the encoding-plus-copy path.
- The full matrix validates the exact four source hashes below. Performance combine cases have no separate local FIA contribution; the local-contribution variants are covered by correctness tests.
- Candidate merge is benchmarked at all token counts as an isolated operation, but production selects packed candidate transport only for T<=16. Large-T merge numbers do not imply that the production routing was changed.
- Full-layer/dummy-model and HCCL measurements in the PR were collected for the earlier implementation. This operator sweep does not refresh those end-to-end results. A2/A3 hardware is not measured here.

## Source identity

| Source under `vllm_ascend/ops/triton/` | SHA256 |
| --- | --- |
| `sharded_indexer.py` | `c6fd8ecbec6cdba1d4d0308ef6151eecd8964895519c69a155439ab03b3e6a64` |
| `sparse_index_remap.py` | `1490ed4e74cb3c71b0c3ea7d2d54accb61492e106da6aeca3001d8e8b773f4ca` |
| `dcp/dcp_a2a.py` | `85d92b5128c8a6a8172724e2a7f7bf716594d8fc7f662104ebdff344a74ca5ed` |
| `dcp/dcp_a2a_batched.py` | `bcd1d80d5e5b84fc54749052eba6a74192bcc527b37c1ea227c125c054460fc3` |

## Owner-prefix follow-up (2026-10-31)

This follow-up uses release v0.30.0 plus PR53 `9acd68dd4` and PR55
`1be0186dc` as its frozen baseline. The tables above describe earlier code.
The new candidate path keeps FP32 comparator scores, descending score order,
ascending global-ID ties, INT32 ID transport and one AllGather per row tile.

For owner-local output, each rank needs only its own winners. Its candidate
at sorted position `m` has global rank `m + S(m)`, where `S(m)` counts all
strictly better peer entries. `S(m)` is monotone, so winners form an exact
prefix. The kernel locates that boundary with bounded search and contiguous
peer-tile counts. A tile containing any non-better entry terminates that
peer's scan. Interpolation changes probe positions only; comparison results
and monotone bounds establish correctness. All-own-winner and empty-owner
guards avoid unnecessary search. Equal-score owner interleave blocks permit
exact whole-block skipping because peer IDs cannot lie inside those blocks.

The owner contract is explicit (`owner_sharded=True` plus `presorted=True`);
generic callers retain the existing merge. The new merge writes every output
slot, including the invalid `-1` suffix. It uses neither approximate TopK nor
score quantization, host synchronization or cross-program scratch state.

At K2048, fusion also requires at least 75% as many rows as vector cores and
enough estimated serial merge levels to amortize a local sorting network.
This workload policy has no device-model whitelist. It selects one fused
Triton sort/pack, one AllGather and one fused owner merge for A3 T48/DCP16
and A5 T128/DCP8. Small or low-DCP workloads retain native sort preparation
and still use the new owner merge. Fused pack alone is not claimed to beat
native sorting; acceptance concerns the complete candidate chain.

Use separate source roots for the baseline and candidate. Run from an
experiment directory under the executor's user-specific data path, with
`src/baseline` containing the frozen baseline:

```bash
PYTHONPATH=src/candidate python src/candidate/benchmarks/dcp_candidate_pipeline_ab.py \
  --reference-root src/baseline --world 16 --rows 48 --dtype fp32 --output compute.json
PYTHONPATH=src/candidate python src/candidate/benchmarks/dcp_candidate_pipeline_ab.py \
  --reference-root src/baseline --world 8 --rows 128 --dtype bf16 --reverse --output reverse.json
```

The driver checks exact packed pairs and owner IDs in eager execution and
the measured graph. Its dependent `compute` section copies this rank's actual
pack output into fixed peer buffers before merge. **It excludes real HCCL**;
do not describe it as communication or full-model latency. Use `--pattern`
with `ties`, `owner_wins` and `peer_wins` to expose skew regressions.

For an allocated 16-rank group, the separate production-chain driver uses
real HCCL, validates every rank against an independent CPU full sort and can
export complete `ascend_pt` profiling:

```bash
PYTHONPATH=src/candidate torchrun --standalone --nproc-per-node 16 \
  src/candidate/benchmarks/dcp_candidate_hccl_ab.py --world 16 --rows 48 --dtype fp32 \
  --source-root src/candidate --output hccl-B --profile profiles/B
```

Run the same command against `src/baseline` before and after B. Both drivers
default to 15 graph samples of 100 calls, excluding compilation, capture and
profiling. Inspect raw samples and device/process occupancy before accepting
a comparison. A5 profiling configuration uses T128/DCP8/BF16, TopK2048 and
interleave768; these operator tests do not reproduce the full model workload.

## Exact digit-sort follow-up (2026-11-01)

The frozen reference is PR53 `8a221c58d` plus PR55 `1be0186dc`, integrated at
`94ca3953e`. This follow-up changes local preparation only. Every comparison
uses the same owner-prefix merge and the same dependent receive-buffer copy.
No new HCCL or full-model speedup is established by these measurements.

BF16 candidate scores now use an exact stable digit sort inside the existing
fused pack kernel. At K2048, a 13-bit digit and an 11-bit current-position
tag form a unique integer key below 2^24. FP32 represents every such key
exactly. Sorting these keys and gathering the original pairs gives a stable
least-significant-digit pass even if the native value sort is unstable.
The composite order is descending score, then ascending local ID. The local
map is monotone, so global-ID ties retain the same order. Original FP32 score
bits and INT32 mapped IDs are transported without encoding losses.

BF16-to-FP32 conversion has 16 meaningful score bits; this is a property of
the input dtype, not quantization of FP32 scores. Signed zeros share an
ordering key but keep their original payload bits. Nonfinite scores retain
the existing invalid-entry policy. The ID key includes the exact 2^24
boundary. An on-device range check can remove one pass, and the general path
handles larger keys without a host read, support gate or collective change.

The public FP32 path retains its previous paired network. An experimental
FP32 digit implementation substantially improved A3 but regressed on A5;
it is not enabled by default. No machine whitelist or DCP upper bound is
added. The existing workload occupancy policy still determines fusion.
The production chain remains fused Triton preparation, one AllGather, and
fused Triton merge for its selected workloads.

Use the new fixed-merge mode to compare native preparation, the frozen
fused pack, and this follow-up without mixing in merge improvements:

```bash
PYTHONPATH=src/candidate python src/candidate/benchmarks/dcp_candidate_pipeline_ab.py \
  --reference-root src/baseline --fixed-merge --world 8 --rows 128 --dtype bf16 \
  --output fixed-merge.json
```

Repeat with `--reverse` and with `--pattern ties`, `owner_wins` and
`peer_wins`. Each result retains all 15 graph samples of 100 calls. Profiling
and compilation are excluded from timing. Native preparation remains a
separate diagnostic comparator, including where it is faster than fusion.

The retained same-host results below use CANN9.2, Torch2.10.0,
torch_npu2.10.0.post7 and Triton3.6.0. A3 is Ascend910_9362 with 40 vector
cores; A5 is Ascend950DT_9581 with 64. They are not matched cross-hardware
performance comparisons. All entries use BF16 native comparator input.
Latency is in microseconds; `compute` means dependent pack/copy/merge.

| Platform / T / DCP | Frozen fused pack | Digit pack | Frozen compute | Digit compute | Native compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| A3 / 48 / 16 | 198.51 | 37.59 | 302.64 | 146.04 | 190.55 |
| A3 / 128 / 8 | 395.18 | 73.53 | 514.49 | 194.15 | 247.22 |
| A3 / 128 / 8, reverse | 395.13 | 73.49 | 517.20 | 194.83 | 249.22 |
| A5 / 48 / 16 | 26.65 | 24.12 | 74.89 | 72.64 | 71.83 |
| A5 / 128 / 8 | 51.68 | 46.94 | 103.65 | 99.13 | 86.52 |
| A5 / 128 / 8, reverse | 51.63 | 46.93 | 103.62 | 99.08 | 86.43 |

The BF16 T128/DCP8 compute reduction is 62.3% on A3 and 4.4% on A5 in both
orders. Native preparation remains faster than the digit fusion on A5.
The A3 FP32 deployment path is unchanged: T48/DCP16 compute is 298.78 versus
298.70 us; this is not an FP32 speedup. FP32 T128 and BF16 T1/T16 are also
unchanged paths. A3 T16 has an unrepeated 122.43-to-123.89-us compute
variation despite nearly identical pack timing; universal speedup is not
claimed. BF16 T129/DCP16 and T49/DCP17 improve in the completed matrix.

Validation includes 148 related checks on each platform (78 actual NPU
cases), plus 32 CPU digit-order cases with normal and random IEEE bit
patterns. Those counts overlap. Exact packed score bits, signed zero,
near-ULP values, nonfinite rows, mapped IDs above 2^24, strided tensors and
changing-input graph replay are covered. The local array-arithmetic and
dispatch suites separately pass 557 cases. This is operator validation,
not full-model accuracy validation.
