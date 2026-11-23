# Shared EC cache statistics implementation plan

> For agentic workers: use the executing-plans skill to implement this plan task by task.

**Goal:** Compare DP=1 and multi-DP cache decisions with identical count and rate definitions.

**Approved design:** Keep scheduler-local exact-hit, similarity-hit and miss counters. Publish cumulative snapshots through the existing shared index RPC at each nonempty connector batch and shutdown. The index stores one snapshot per scheduler instance, retains completed instances, and sums counters. Repeated or older snapshots cannot increase the total. A new instance gets a new ID; an index restart gets a new statistics epoch. Reporting errors must not fail inference. Local mode needs no RPC.

**Rates:** For E exact hits, S similarity hits and M misses, total T=E+S+M. Overall hit rate is (E+S)/T; similarity contribution is S/T; similarity lookup hit rate is S/(S+M). Empty denominators yield zero. Count final scheduler decisions, not requests, unique hashes or successful worker loads.

**Scope:** Production edits only in ec_transfer. No root initializer, environment variable, extra service, timer thread or scheduler collective changes. Preserve the existing local hit_rate property.

**Files:** Add cache_statistics.py for immutable snapshots and a locked accumulator. Extend similarity_index.py and similarity_index_service.py with one statistics RPC. Integrate batch reporting and scoped logs in ec_memcache_connector.py. Extend CPU index tests, connector tests, user documentation and the pending release note.

**Failure semantics:** A timed-out RPC may already have applied the snapshot; replay is safe. The next batch or shutdown retries the latest cumulative counters. Global output includes an epoch and reporting-scheduler count, and is a snapshot of reports received so far, not a distributed barrier. Index restart loses stopped schedulers; live schedulers restore their totals on their next report. Compare fresh deployments or deltas within one uninterrupted epoch.

- [x] Add focused aggregation and real cross-process regression tests; observe failure.
- [x] Implement immutable statistics, atomic aggregation and RPC serialization.
- [x] Report from the connector without changing cache actions; add scoped rates and failure tests.
- [x] Document reporting, formulas and comparison limitations.
- [x] Run CPU tests, changed-file lint/format and Markdown checks; attempt format.sh ci.
- [x] Review the implementation before publication.
- [ ] Commit with sign-off and push the existing shared-index branch.

Full matcher/connector tests require torch/vLLM; NPU throughput and DP end-to-end validation remain hardware checks.

Validation: 19 CPU index/statistics tests passed; changed-file Ruff, Markdown, Python syntax and YAML checks passed. Implementation review found no blocking issues. Full format.sh ci stopped because pre-commit is absent; connector runtime tests are not run because torch/vLLM are absent.
