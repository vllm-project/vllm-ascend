# DCP decode port to v0.26

## 2026-09-15 larger-batch validation

O/LSE eligibility now follows `min(max_num_batched_tokens, 192)` instead of
the previous 12-token cap. The 192 cap is manually chosen, not a demonstrated
kernel capability limit: larger inputs remain untested and use the upstream
fallback. Query/store limits and kernel implementations are unchanged.

The same-image GLM-5.2 W4A8 synthetic decode experiment used 32 A3 NPUs in
one superpod, DP4/local-DP2/TP8/DCP8/EP32, MTP5 with zero synthetic acceptance,
DecodeBenchConnector with zero-filled KV, full decode graphs, concurrency32,
50 identical prompts (20K-40K input tokens), and exactly700 output tokens
per request. The scheduler budget was192. Actual vLLM revision: b5e8e2561.
Every fixed-output repetition completed50/50 requests with35000output tokens
and no recorded errors; input-length arrays were identical.

| Runtime composition | Per-run mean TPOT (ms) | Median of run TPOT (ms) | Per-run output throughput (tok/s) |
|---|---|---:|---|
| PR15970 default keep-view baseline |114.626,114.192|114.409|220.670,221.722|
| Previous combined port, O/LSE cap12 |114.339,114.006|114.172|221.009,222.032|
| Updated combined port, O/LSE cap192 |105.798,99.110,105.807|105.798|238.280,250.711,238.331|

Median-of-runs TPOT improves7.53% versus PR15970 and7.33% versus combined12;
median throughput improves7.75% and7.59%, respectively. All repetitions are
retained, including the faster99.110ms run. The candidate varies more than
the baseline; these small samples are not confidence intervals or guarantees.

The combined12/candidate source manifests differ only in the two O/LSE files.
This supports expanded fast-path eligibility as the cause of the added gain.
It does not increase the scheduled batch size or reduce collective count:
both paths use one AllToAll. The existing raw-bit packing, receiver views and
fused merge are now available to larger eligible calls. In separate component
AB/BA measurements, the complete T48 path took189.067/189.411us upstream versus
102.409/99.178us optimized. Those component percentages are not E2E gains.
T48 is a configuration-derived example, not a measured serving-shape histogram.

The serving arms were image-compatible Python overlays, not installations of
the entire worktree. PR15970's default behavior was adapted to the image's
older config API. Imported module paths/hashes were checked on32workers across
4DPx8TP before measurement. Warmups were excluded; no profiler or audit RPC ran
during included timings. Historical colleague source parity, model accuracy,
and broader workload transfer are not claimed. The known strict BF16 numerical
limitation described below remains unresolved; no tolerance was relaxed.

Additional direct-helper A3 checks covered T13/16/24/32/48/64/96/128/192,
including changing graph inputs (1152 rank-level checks). Registered-op graph
checks separately covered T48/T192 (128 rank-level checks). CPU routing tests
cover scheduler budgets, current-config changes and uninitialized-config
handling. Raw serving traces/requests contain private data and are not shipped.
The earlier sections below are historical validation checkpoints.

Fresh publication checks:95 focused CPU tests passed. Full `bash format.sh ci`
passed the other hooks, including Gitleaks and ShellCheck, but Actionlint's
macOS ShellCheck subprocesses were terminated. A full CI pass is not claimed.

## 2026-09-11 local refresh

Base: releases/v0.26.0rc at5f12bbf34a530cd6c87463c56404535246af6e94.
Branch: perf/dcp-decode-v026; historical port commit f2648bb0f07e35992398c2a25f3a9692285673f6.
Original donor: local82bcf6902a5d08cb736d941ea1a874d026d30238.
Current status: local source refresh and CPU checks complete; no new NPU run.

| Area | Current selection |
|---|---|
| Remap | Upstream two-kernel path plus DCP8/interleave128 integer and empty-input repairs |
| O/LSE | Existing ported raw-bit pack, one AllToAll, FP32 merge and Cast |
| Query | Existing head-major prepare and contiguous unpack |
| Indexer store | Existing native quantization plus fused scale conversion/key-scale stores |

Removed the old five-operation remap priority from the attention caller.
The old standalone sfa_dcp_remap helper remains as historical code, but no
production caller imports it. The ordinary upstream remap routing/fallbacks
now apply. The decode_only argument is retained for existing caller compatibility.
O/LSE/Q/store code and their existing guards are unchanged.

## Local verification

Final focused CPU suite: 50 tests passed (before documentation-only changes).
Run from this worktree:

```bash
python3 -m pytest --noconftest -q \
  tests/ut/attention/test_sfa_dcp_remap.py \
  tests/ut/attention/test_sparse_index_remap_port.py \
  tests/ut/attention/test_sfa_dcp_exchange_port.py \
  tests/ut/attention/test_sfa_dcp_query_port.py \
  tests/ut/attention/test_sfa_indexer_store.py
```

Single-card operator precision entries are in
`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/`:
`test_sfa_dcp_remap.py` and `test_sfa_indexer_store_triton.py`.
These are content-preserving relocations; numerical assertions and tolerances
are unchanged. Query gather and O/LSE exchange require an isolated DCP8 group
and remain distributed tests under `tests/ut/attention/a2/`.
Existing distributed O/LSE entry:
tests/ut/attention/a2/test_sfa_dcp_exchange_port.py.
The strict exchange oracle is preserved, including known historical failures.

## Reused experiment evidence (not new validation of these worktrees)

- [Pinned upstream component comparison](../../experiments/upstream_dcp_20260910/REPORT.md)
  tested release5f12bbf/f2648bb0f helpers on the pinned A3 environment. The
  pack/combine leaf implementations match local main base ac6405d9; main's
  additional PCP composition and full main serving were not measured.
- Original optimized raw-bit O/LSE versus upstream, complete path including
  communication, merge and BF16 Cast: T1 reduction30.5/30.7%, T6 23.6/28.4%,
  T12 35.1/33.6% in AB/BA runs. Both paths already use one AllToAll.
  This is not the later experimental VMM peer exchange and not a serving gain.
- Query+SFA T6 reduction12.3-13.6%, T12 16.5-19.2%; T1 is unchanged fallback.
  Native quant+store reduction73.9-75.9% for T1/6/12. Query/store NPU byte,
  consumer and changed-graph checks passed in the pinned helper experiments.
- [Remap follow-up](../../experiments/peer_olse_20260911/REMAP_RESULT.md):
  two-kernel path reduces whole-remap latency35.08-42.63% relative to our
  old five-operation path. The repaired module is copied byte-for-byte from
  the tested local working source, not its older82bcf commit.
- Integer regression: input16777343 belongs to owner0/local2097279; the
  observed vector division gave owner1/local2097151. DCP8/interleave128 now
  uses shifts/masks, including boundaries above2**24 and INT32_MAX.
  Other configurations retain upstream arithmetic, without a new exactness
  claim. Empty last dimension returns before division by TopK.
- Historical v0.26 remap test result38PASS/3FAIL preceded this repair.
  The former DCP_PORT claim of exact large-coordinate arithmetic was disproved
  on hardware. Those failures are not erased or relabeled as passes.
- Strict CPU-BF161e-3 tests fail for both original optimized and upstream
  exchange. Independent FP64 diagnostics bound pre-cast FP32 error below8e-7
  in the upstream comparison. These are not established donor-only regressions;
  no tolerance is relaxed and no complete numerical acceptance is claimed.
  The local9% TPOT result uses a different baseline and must not be claimed
  for this port or added to the component percentages.

## Scope boundary

No fetch, rebase, remote update, push, PR, runtime installation, pod operation,
NPU experiment, allocator/host-DRAM placement or experimental VMM peer code is
part of this refresh. Q/store kernels are unchanged and identical across the
two port worktrees and local donor. Local main always means ac6405d9, not
today's remote main. Changes remain uncommitted on the historical port commit.

The new NPU regression tests exercise the actual two-kernel remap against a
CPU int64 oracle, large coordinates and changed graph inputs. Their presence
does not mean they ran locally. A new full-worktree NPU/serving validation is
still required before making a full-port runtime acceptance claim.
