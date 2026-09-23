# rms_norm_cast Timeline Traces

Chrome-trace timelines backing the pipelining comparison in
`RMS_NORM_CAST_OPT_SUMMARY.md` (Round 2 section). Both were captured with
`msprof --ai-core=on --aic-metrics=PipeUtilization --task-time=on` on the
same workload: `benchmarks/rms_norm_cast_msprof.py 2048`
(2048x7168 bfloat16, 10 warmup + 30 iterations, 40 kernel instances each).

| File | Kernel version | Median kernel duration |
|---|---|---|
| `rms_norm_cast_2048x7168_bf16_round1_before_pipeline.json` | Round 1 (before row pipelining) | 79.9 µs |
| `rms_norm_cast_2048x7168_bf16_round3_after_pipeline.json` | Round 3 (current, after pipelining) | 53.6 µs |

Viewing:

- Any Chrome-trace viewer: drag the file into `chrome://tracing` or
  [Perfetto UI](https://ui.perfetto.dev), then search for `RmsNormCast`.
- These files are the timeline export only. For the full pipe-level
  breakdown (vec/MTE2/MTE3 utilization bands) re-run the msprof command
  above and open the resulting `PROF_*` directory in MindStudio Profiler.
