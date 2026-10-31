# Engram WKV overlap

DeepSeek-V4.1-Flash can prepare Engram WKV projections on a separate NPU stream
after an earlier MoE router finishes. This overlaps the original projection with
dispatch and expert work before the Engram layer consumes it. The projection
backend, weights and gate calculation are unchanged.

The option is experimental and disabled by default. Add these fields to the
existing `--additional-config` JSON:

```json
{
  "multistream_engram_overlap": true,
  "multistream_engram_wkv_overlap": true
}
```

This requires sequence parallelism, TP greater than one, PP equal to one, and
FULL ACL Graph execution (for example `FULL_DECODE_ONLY`). LoRA, DBO and compiler
tracing are excluded. Other execution modes use the original projection at the
consumer. The optimization does not require VMM or cross-instance shared tables.

For Engram layers 1 and 14, router 0 submits the first projection and router 1
submits the second. Each submission waits for that table's lookup-ready event,
shards rows using the model's TP rank, then calls the original WKV backend. The
consumer waits for its projection before applying the gate. A missing router
notification retains the original lookup/projection path.

The model retains one projected-result buffer per eligible Engram layer. Its
capacity is the maximum captured token count divided by TP, rounded up. For two
25600-wide BF16 projections, capture size 192 and TP4, these buffers total
4.6875 MiB per rank, excluding temporary outputs and backend workspace.

Measure the full workload before enabling this option: WKV competes for compute
and memory bandwidth with MoE. A3 validation showed positive end-to-end results
with visible baseline variation; this is not a guaranteed speedup. A5 execution
and production P/D workloads have not been validated for this option.

## Measure an existing service

Start the model with `--served-model-name engram-wkv --port 8089` and the desired
deployment options, then run:

```bash
python benchmarks/benchmark_engram_wkv.py \
    --input-words 256 --concurrency 32 --output-tokens 512 --output baseline.json
```

Repeat with only `multistream_engram_wkv_overlap` changed and a new output path.
The client uses deterministic random-word prompts, two waves of 32-token warmup
requests and three measured waves. Its input length argument counts words, not
tokens; actual prompt and output token counts are recorded. It checks all
requests completed with the requested output length and saves payload digests,
streaming timing and speculative-decoding counter deltas. It is a performance
workload, not an output-equivalence or model-quality test. Keep graph sizes,
KV-cache budget, lookup/shared-expert overlap and speculative decoding identical
between runs, and report repeats rather than only the fastest result.
