# GLM-5.3 A3 16K combined Nightly

This is a **hardware calibration candidate**, not a validated replacement for
run43574755884. One full-checkpoint deployment on four A3 hosts (16 logical
NPUs each) runs P, PD E2E, two Decode latency profiles, and GPQA once.
The previous concurrency16 throughput baselines are deliberately removed.

## Serving and capture budgets

D uses nodes0-1, DP16/TP2, eight instances per host. P uses nodes2-3,
PP2/PCP16/TP1, DP1, partition42,36. Retain MRV2, INT8 main/indexer KV,
MooncakeConnectorV2, P eager MTP1 and D MTP5 with eager draft.
This redesign retains the existing serving recipe, not a v0.30.0 port.

P uses maxseq64/batched16384. P can queue incoming requests; it does not
need to schedule all340 Decode clients simultaneously. Check that cached
prefill and KV transfer do not bottleneck Decode measurements.
D uses maxseq32/batched192 with Static/Super Kernel and npugraph_ex.
For concurrency C, the sizing average is ceil(C/16) sequences, with six
query tokens per sequence (one target plus five draft tokens).

| Client concurrency | Average rounded up per D | Query token capture |
| ---: | ---: | ---: |
| 7 | 1 | 6 |
| 32 | 2 | 12 |
| 40 | 3 | 18 |
| 80 | 5 | 30 |
| 340 | 22 | 132 |

Capture candidates are [6,12,18,24,30,36,48,72,96,132,192], maximum192.
The32-sequence budget leaves skew margin above22; this does not guarantee
uniform routing. Intermediate loads can pad to the next capture size.
Verify actual runner capture sizes, padding, per-rank running/waiting and
startup duration on hardware. Do not equate a CPU loader test with capture
success or claim that fewer configured sizes guarantees faster startup.
Keep max-model-len131072 for GPQA output65536 and its original prompts.

## Workloads and boundaries

All performance input comes from
`vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8`.
Check raw token lengths; the dataset name is a nominal16K profile.

| Formal phase | Requests | Concurrency | Max output | Rate | Warmup requests/concurrency |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill | 28 | 7 | 1 | 0 | 14/7 |
| PD E2E | 160 | 40 | 1024 | 1.8 | 80/40 |
| Decode20ms | 320 | 80 | 1024 | 0 | 64/32 |
| Decode50ms | 1360 | 340 | 1024 | 0 | 64/32 |

Rate0 means no configured rate limit. These are historical RP3 search seeds,
not new measured capacity points. Historical concurrency80 averaged20.2ms,
so it does not establish compliance with a20ms ceiling.

P and PD warm up, reset P prefix cache, then run formally. Prepare a local
Decode dataset by repeating the first16K prompt1360 times. Each Decode
profile independently resets, warms this shared prefix, then runs formally.
Cache hits must be measured, not assumed100%; Decode is not device-only.
After performance, reset once more and run GPQA at concurrency32, output65536,
temperature1, top-p0.95, accuracy floor86.92%. Do not pad GPQA to16K.

All five resets POST node2:7100/reset_prefix_cache and assert JSON success=true.
Development mode is enabled only for this controlled CI service. Reset does
not restart servers or change graphs. Inspect AISBench internal warmups too.

## Calibration before accepting regression baselines

1. Record actual Ascend/vLLM revisions, driver/CANN/torch_npu, ready time,
   capture duration and all actual runtime flags.
2. Search P concurrency near the4s mean TTFT target, D concurrency near20/50ms
   mean TPOT, and E2E request rate near4s TTFT/20ms TPOT. Check client saturation,
   per-D skew, scheduler caps, cache hits and KV transfer. If32 sequences caps
   a D instance, reassess routing/budgets rather than calling it model capacity.
3. Use bounded increasing trials then refine the passing/failing interval;
   independently confirm each selected point. A failed or incomplete run is
   not a baseline sample. Preserve the raw JSON/CSV and all request outcomes.
4. Freeze selected loads for Nightly. Run each formal profile twice with the
   same configuration and cache preparation. Set its baseline to the lower
   valid throughput, threshold0.95. Prefer input_throughput_baseline and
   input_throughput_threshold for P; use output baseline/threshold for PD/D.
   Never put input throughput in the runner's output baseline field.

Throughput baseline/threshold fields are absent during calibration, so a
passing candidate is **not a throughput regression pass**. Current runner
supports mean TPOT ceilings (20ms PD/D20,50ms D50); the4s TTFT target requires
raw-result review and is not an automated gate. GPQA retains its accuracy gate.
No startup/runtime estimate is promised before measuring this larger graph.

## Dispatch and checks

```text
/nightly GLM-5.3-W8A8C8-A3-ALL-COMBINED
```

The case moves from Weekly to Nightly, without duplicate scheduling.

```bash
python -m unittest tests.ut.tools.test_glm53_full_model_cases
bash format.sh ci
```

CPU checks cover topology resolution, cache boundaries, sparse graph budget
coverage, both Decode profiles and Nightly matrix dispatch. NPU startup,
performance, cache hits and GPQA must be revalidated for this configuration.
