# Workflow Checklist

## Baseline

1. Create `work_dir` if missing.
2. Check whether `ledger.md` already exists.
3. If resuming, read the most recent completed verdict and continue from the next unfinished lever.
4. Freeze the benchmark contract after baseline:
   - concurrency levels
   - requests per level
   - input tokens
   - output tokens
   - target metric

## Per iteration

1. Pick one lever from `tuning-knobs-reference.md`.
2. State the hypothesis.
3. Launch one managed server for that exact config.
4. Run warm-up.
5. Run benchmark with the frozen workload.
6. Decide the verdict.
7. Append to `ledger.md` immediately.
8. Run scoped cleanup with `scripts/cleanup_managed_server.sh`.

## Exit conditions

Stop only when one of these is true:

1. `max_iterations` reached
2. user explicitly stops
3. all planned phases are exhausted
4. the run is blocked before any safe further attempt can be made

Baseline-only output is not a completed tuning campaign.
