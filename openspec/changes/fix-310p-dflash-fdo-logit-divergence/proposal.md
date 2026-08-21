## Why

The initial 310P DFlash `FULL_DECODE_ONLY` matrix proved genuine target/draft graph replay but exposed a reproducible correctness cluster. Boundary-first hardware tracing located a plugin-side draft RoPE extent defect: when the live context was larger than the query graph descriptor, FDO prepared only the query-sized context range. After the scoped repair, all request, graph, acceptance, and throughput gates pass. Residual wording branches were traced to target-model numerical drift whose Eager-to-Eager magnitude is at least as large as Eager-to-FDO; they do not justify an additional FDO runtime change.

## What Changes

- Add a default-off, plugin-owned diagnostic path that compares Eager and FDO at identical speculative iterations without modifying upstream vLLM or any operator.
- Locate the first divergence in stages: scheduler/model inputs, target versus draft boundary outputs, target/draft final hidden states and logits, then one selected transformer layer at a time.
- Capture only active logical lanes and deterministic summaries/top-k evidence; never let inactive descriptor padding create a false mismatch.
- Make probes graph-replay-aware by copying a selected tensor into persistent device storage inside the captured graph and exporting it only after replay has completed.
- Test one root-cause hypothesis at a time; the proven correction preserves the full live context extent while preparing stable draft RoPE buffers.
- Implement only the smallest plugin-side correction supported by first-divergence evidence, keeping Eager, Piecewise, non-DFlash, other graph modes, upstream vLLM, and operators unchanged.
- Re-run the exact 35B divergent prompt fast gate before the complete 9B/35B acceptance matrix; preserve graph authenticity, acceptance length, throughput, memory, and concurrency-10 safety.

## Capabilities

### New Capabilities

- `310p-dflash-fdo-numerical-equivalence`: Defines replay-safe numerical probes, first-divergence localization, hypothesis isolation, and deterministic eager/FDO correctness acceptance for 310P DFlash FDO.

### Modified Capabilities

None.

## Impact

- Implementation areas are a new 310P FDO numerical-probe module, narrow default-off hooks in the target runner and DFlash proposer, the scoped draft RoPE extent correction, focused tests, and the paired acceptance driver.
- The diagnostic capability remains default-off and retained as DEBUG functionality after repair.
- The existing staged FDO implementation remains the baseline; no rollback, commit, or push is performed during diagnosis.
- No C++, AscendC, custom operator, upstream vLLM, model-weight, or public serving API changes are authorized.
