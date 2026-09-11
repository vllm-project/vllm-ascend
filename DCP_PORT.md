# DCP decode layout port: fixed local main

## 2026-09-11 local refresh

Base: local main ac6405d9eb12e1466d00a0fb4e67ab87c1a33061.
Branch: perf/dcp-decode-layouts; historical port commit3788d00d5ee808a785cee3a4cac2237bf53c9eea.
Original donor: local82bcf6902a5d08cb736d941ea1a874d026d30238.
Current status: local source refresh and CPU checks complete; no new NPU run.

| Area | Current selection |
|---|---|
| Remap | Retain upstream two-kernel path; add tested DCP8/interleave128 integer and empty-input repairs |
| O/LSE | Add v0.26 port's raw-bit pack, one AllToAll, FP32 merge and Cast |
| Query | Existing head-major prepare and contiguous unpack; unchanged |
| Indexer store | Existing native quantization plus fused stores in AscendSFAIndexerBackend; unchanged |

O/LSE reuses sfa_dcp_exchange.py and sfa_dcp_merge.py byte-for-byte from the
measured v0.26 port. This preserves its invalid-rank masking adaptation.
Routing stays inside the existing registered custom op and retains FakeTensor
behavior/signatures. It selects the new path only without a PCP group, for
scatter_size8/scatter_dim1 and eligible BF16 T1-12/H64/D512 + FP32 LSE.
The current native attention caller reaches this path from decode handling.
PCP composition, token-scatter DSA-CP, other group sizes, unsupported shapes
and scatter-size1 keep the original upstream pipeline. A missing scatter
group still raises before any communication.

## Local verification

Final focused CPU suite: 43 tests passed.
Tests cover old/new routing, PCP gather preservation (including no scatter),
DSA token scatter, dtype/shape fallback, FakeTensor output, invalid-shard
merge semantics, empty remap, query wait ordering and store routing.

```bash
python3 -m pytest --noconftest -q \
  tests/ut/attention/test_sparse_index_remap_port.py \
  tests/ut/attention/test_sfa_dcp_exchange_port.py \
  tests/ut/attention/test_sfa_dcp_query_port.py \
  tests/ut/attention/test_sfa_indexer_store_port.py
```

NPU entries: tests/ut/attention/a2/test_sfa_dcp_remap.py and
tests/ut/attention/a2/test_sfa_dcp_exchange_port.py. The exchange test retains
its strict oracle; it is not expected to waive the historical CPU-BF16 failures.
No main-PCP device result or full main serving speedup is asserted.

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
