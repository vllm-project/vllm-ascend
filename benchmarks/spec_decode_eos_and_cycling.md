# Comparing speculative acceptance with EOS and dataset cycling

This protocol prepares the four comparisons requested in [roadmap #18827, item 8](https://github.com/vllm-project/vllm-ascend/issues/18827).
It defines an experiment; it does not contain measured NPU results.
Use a supported target/draft pair and an existing serving workload runner.

## Freeze the workload

Record the target and draft revisions, tokenizer revision, dataset revision and
ordered prompt IDs, vLLM/vLLM-Ascend commits, CANN/torch_npu versions, hardware,
parallel configuration, sampling parameters, speculative method and draft length.
Keep request concurrency, request count, output limit and prefix-cache settings
identical across cells. Record whether model loading, compilation, tokenization
and warmup are excluded from the reported timer.

| Cell | Prompt selection | EOS handling |
| ---- | ---------------- | ------------ |
| A | Distinct prompts, without cycling | Honor EOS |
| B | The same distinct prompts as A | Ignore EOS |
| C | Cycle a fixed subset of prompts | Honor EOS |
| D | The same cycled prompt order as C | Ignore EOS |

For N measured requests, A/B need N distinct prompt IDs. C/D use a declared
subset of size M < N, repeating its original order until N requests are produced.
Cycling repeats requests; it does not concatenate prompts or carry conversation
state between requests. Save M, N and the exact order. Do not silently substitute
cycling when a dataset contains fewer than N distinct prompts.

Honor/ignore EOS refers to the generation setting, not adding or removing EOS
from the input text. Keep other stop conditions fixed and record them. With EOS
ignored, generation can continue to the output limit; the longer outputs are
part of the comparison, not evidence of an inference speedup by themselves.

## Separate warmup from measurement

Use a fresh server for each cell, with identical warmup, then capture the metrics
baseline **after** warmup and **before** measured requests. This avoids cross-cell
prefix-cache reuse and keeps warmup draft tokens out of acceptance statistics.
Keep unrelated clients off that server while measuring.

The existing `tools.spec_decode_metrics` helpers can wrap a workload runner's
`server` fixture (which supplies `url_for("metrics")`):

```python
from tools.spec_decode_metrics import capture_baseline, measure_acceptance_rate

# warmup() and run_measured_requests() belong to the chosen workload runner.
baseline = capture_baseline(server, num_speculative_tokens, warmup_fn=warmup)
run_measured_requests()
pos0_rate, per_position_rates = measure_acceptance_rate(
    server, num_speculative_tokens, baseline
)
```

Wait for all measured requests to finish before the final scrape. Save both raw
`/metrics` responses and the runner's request-level timings and token counts.
The helper rejects decreasing counters and mismatched baseline position counts.
It cannot detect every restart (new counters may already exceed the baseline),
so also record server identity and reject any interval spanning a restart or a
change in the participating engines. Start the cell again with a fresh baseline.

## Report the quantities separately

- Per-position acceptance is `delta(accepted_at_position) / delta(num_drafts)`.
  Sum counts across engines before division; do not average per-engine rates.
  Position zero acceptance is not the overall accepted/proposed-token ratio.
- A window with no new drafts returns zero rates for compatibility with the
  helper's callers. Mark it as **no speculative observations** in a comparison;
  it does not demonstrate measured zero acceptance.
- Report completed/failed requests, actual output tokens, finish reasons, EOS
  setting, TTFT and request latency. Report aggregate output tokens per measured
  wall-clock second separately from individual request latency.
- Preserve the full per-position vector. Dynamic draft lengths change which
  positions are proposed, so compare like-for-like draft policies.
- Repeat cells in independently started processes and preserve individual
  samples before making variability or tail-latency claims. Do not present
  correctness or quality preservation as a consequence of acceptance alone.

Before publishing results, run the matrix on Ascend hardware and attach the
recorded environment, commands, raw counters, workload order and timing boundary.
CPU tests of the helpers validate aggregation only.
