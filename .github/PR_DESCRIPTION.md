# [Feature][KV Offload] Delayed batch D2H for sparse KV offload on Model Runner V2

## Summary

Replace the per-layer synchronous device-to-host KV copies in the sparse KV
offload path with a delayed batch pipeline: full blocks are detected after
input preparation, an asynchronous D2H batch is launched on a dedicated
stream that overlaps the current forward pass, and the copied blocks are
retired at the next iteration boundary before their stable prefixes are
published to the attention metadata.

This is a V2-only implementation on top of current main, adapted from
F.Liu's delayed Nano D2H design (design doc:
`docs/source/developer_guide/Design_Documents/delayed_nano_tail_d2h_adaptation.md`
on the source branch).

## Key design points

- **Single length truth**: `num_computed_tokens` is the sole full-block
  trigger — no parallel committed_length or PENDING state.
- **One INFLIGHT batch**: at most one D2H batch in flight; no PENDING
  queues, no second staging bank, no Python background threads.
- **Slot generations**: monotonic per-slot generation counters prevent
  stale completions from advancing reused slots' state.
- **stable_prefix**: published only after D2H completion and the TP
  completion collective; attention metadata consumes it instead of
  recomputing geometry.
- **Four hooks** at V2 runner call sites: retire (before input prep),
  plan+launch (after attention metadata, overlapping the forward),
  source_ready (after the last KV writer), and finished-request
  delayed-free.
- **PD coordination**: request-owned rows stay reserved until every
  delayed batch that references them retires.

## What is NOT touched

- No speculative-decode logic (input preparation, draft/verify path,
  proposer internals — zero lines).
- No `_310p` code.
- No V1 runner files.
- No monkey patches or unconditional module rebinding.

## Commits

| # | Hash | Subject |
|---|------|---------|
| 1 | 9f14ab6ee | feat(kv_offload): delayed batch D2H planner and manager core |
| 2 | 4c0868fa3 | feat(kv_offload): publish stable prefixes into attention metadata |
| 3 | 641243ea4 | feat(kv_offload): wire the four delayed D2H hooks into Model Runner V2 |
| 4 | 8e6b5ea02 | feat(kv_offload): PD delayed-free coordination in SfaRemoteD2H |
| 5-9 | | test adaptation and regression suites |

## Diff summary

- 16 files changed, **+982 source / -11**, +888 tests
- Core: `sparse_kv_offload_manager.py` (+645)
- V2 hooks: `worker/v2/model_runner.py` (+69), `model_states/default.py` (+71)
- Attention: `sfa_kv_offload.py` (metadata builder consumes stable_prefix)
- PD: `sfa_pd_rd2h/{protocol,scheduler,worker}.py` (delayed-free handshake)

## Test plan

- [x] 205 unit tests pass (manager, copy-op, connector, attention metadata)
- [x] Cross-host PD disaggregation (TP16 prefill + DP8×TP2 decode, MTP3):
      coherent long-context outputs, 32-request load zero failures
- [x] MTP acceptance rate preserved (~2.9, matching the stock path)
- [ ] Full benchmark opt vs base (in progress)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
