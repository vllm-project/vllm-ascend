# Ledger And Report

## Ledger requirements

Append one section per iteration:

```markdown
## Iteration N — <phase>.<lever_name>

**Hypothesis**: ...
**Change**: `...`
**Verdict**: WIN | LOSS | NEUTRAL | STARTUP_FAIL | BENCHMARK_FAIL | SKIPPED_CONFLICT
**Carry forward**: YES | NO
**Notes**: ...
```

For successful benchmarked attempts, include the per-concurrency metric table.

For failed or skipped attempts, include:

- `Failure stage`
- `Reason`
- `Evidence`

## Verdict rules

- `WIN`: target metric improves enough and guard metrics do not regress beyond the allowed threshold
- `NEUTRAL`: no material win, or gain is cancelled by guard regression
- `LOSS`: target metric regresses materially
- `STARTUP_FAIL`: server never became ready
- `BENCHMARK_FAIL`: server started but benchmark did not complete correctly
- `SKIPPED_CONFLICT`: lever is known to be incompatible before launch

## Final report

`scripts/generate_tuning_report.py` must:

1. populate per-iteration metrics from result dirs when available
2. preserve failure verdicts instead of collapsing them into neutral output
3. compute improvement with the correct direction:
   - lower latency metrics are better when the number goes down
   - throughput metrics are better when the number goes up
