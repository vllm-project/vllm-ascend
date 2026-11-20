# GLM-5.3 A3 16K combined Nightly

One full-checkpoint deployment on four A3 hosts (16 logical NPUs each) runs
P, PD E2E, the Decode20ms profile, and GPQA once. The 16K workload passed in
run44659395560. Throughput gates use the measurements below, not the previous
concurrency16 baselines. Decode20ms has a provisional single-sample baseline.

## Serving and capture budgets

D uses nodes0-1, DP16/TP2, eight instances per host. P uses nodes2-3,
PP2/PCP16/TP1, DP1, partition42,36. Retain MRV2, INT8 main/indexer KV,
MooncakeConnectorV2, P eager MTP1 and D MTP5 with eager draft.
This redesign retains the existing serving recipe, not a v0.30.0 port.

P uses maxseq64/batched16384. P can queue incoming requests; it does not
need to schedule all80 Decode clients simultaneously. Check that cached
prefill and KV transfer do not bottleneck Decode measurements.
D uses maxseq32/batched192 with Static/Super Kernel and npugraph_ex.
For concurrency C, the sizing average is ceil(C/16) sequences, with six
query tokens per sequence (one target plus five draft tokens).

| Client concurrency | Average rounded up per D | Query token capture |
| ---: | ---: | ---: |
| 7 | 1 | 6 |
| 32 | 2 | 12 |
| 40 | 3 | 30 (padded from 18) |
| 80 | 5 | 30 |

Capture candidates are [6,12,30,48,132,192], maximum192.
The32-sequence budget and six capture tiers are retained from the previous
configuration while removing D50; they are not retuned for this smaller load.
The sizing average of five sequences at concurrency80 does not guarantee
uniform routing. Intermediate loads can pad to the next capture size.
Verify actual runner capture sizes, padding, per-rank running/waiting and
startup duration on hardware. Do not equate a CPU loader test with capture
success or claim that fewer configured sizes guarantees faster startup.
The six tiers retain capture headroom for uneven routing.
Intermediate loads, especially between48 and132, incur padding and must be
checked for performance impact. Run44286450331 completed only3/11 captures
in38 minutes before the old3600-second readiness budget expired.
This case now allows14400 seconds for readiness and21600 seconds for worker
waiting; the workflow stream budget is360 minutes and job budget420 minutes.
Other cases retain their defaults. AOP PR runs retain their longer budgets.
These are calibration limits, not a completion-time or performance guarantee.

Keep max-model-len131072 for GPQA output65536 and its original prompts.

## Workloads and boundaries

P and PD performance input comes from
`vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8`.
Check raw token lengths; the dataset name is a nominal16K profile.

| Formal phase | Requests | Concurrency | Max output | Rate | Warmup requests/concurrency |
| --- | ---: | ---: | ---: | ---: | --- |
| Prefill | 28 | 7 | 1 | 0 | 14/7 |
| PD E2E | 160 | 40 | 1024 | 1.8 | 80/40 |
| Decode20ms | 320 | 80 | 1024 | 0 | 64/32 |

Rate0 means no configured rate limit. These loads originated from RP3 search
seeds; they are validated workload points, not a maximum-capacity search.
Run44659395560 averaged18.5ms TPOT at Decode concurrency80 (P90:20.6ms).
The automated20ms gate applies to mean TPOT, not P90.

P and PD warm up, reset P prefix cache, then run formally. Decode uses
`vllm-ascend/GSM8K_prefix100_in16384_bs500_GLM5_3_flash_w8a8` directly.
The downloaded dataset contains1000 identical prompts, enough for the320
formal requests. CPU validation with the full GLM-5.3 tokenizer found16378
raw tokens and16390 tokens with its default chat template and generation
prompt. Verify actual server input lengths on hardware; the Flash dataset
name does not establish full-model performance or accuracy.
D20 resets, warms this shared prefix, then runs formally.
Cache hits must be measured, not assumed100%; Decode is not device-only.
After performance, reset once more and run GPQA at concurrency32, output65536,
temperature1, top-p0.95, accuracy floor86.92%. Do not pad GPQA to16K.

All four resets POST node2:7100/reset_prefix_cache and assert JSON success=true.
Development mode is enabled only for this controlled CI service. Reset does
not restart servers or change graphs. Inspect AISBench internal warmups too.

## Calibration before accepting regression baselines

1. Record actual Ascend/vLLM revisions, driver/CANN/torch_npu, ready time,
   capture duration and all actual runtime flags.
2. Search P concurrency near the4s mean TTFT target, D concurrency near20ms
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

## Measured throughput baselines (2026-11-20)

| Formal phase | Nov 19 | Nov 20 | Baseline | Threshold |
| --- | ---: | ---: | ---: | ---: |
| P input token/s | 39802.5547 | 39643.0557 | 39643.0557 | 0.95 |
| PD output token/s | 1546.4292 | 1554.8294 | 1546.4292 | 0.95 |
| D20 output token/s | Not executed | 3787.2478 | 3787.2478 | 0.95 |

Sources: [Nov 19 model job](https://github.com/vllm-project/vllm-ascend/actions/runs/44440873203/job/135285671057)
and [Nov 20 model job](https://github.com/vllm-project/vllm-ascend/actions/runs/44659395560/job/136034231445).
P and PD use the lower of two successful formal phases with the same serving
configuration, load and cache reset preparation. Nov 19's overall job failed
later while preparing the Decode dataset; it is not a complete passing run.
Its P28/28 and PD160/160 phases completed with zero failed requests.
Nov 20 completed P28/28, PD160/160 and D320/320 with zero failed requests,
and GPQA89.90%. D50 removal and the D dataset fix do not change preceding P/PD.

D20 explicitly uses a provisional single valid sample at the owner's request;
it does not satisfy the two-sample calibration rule yet. Replace it with the
lower of two valid comparable samples once a second result is available.
Warmups have no throughput gates. The4s mean TTFT target still requires raw
result review; PD/D20 retain their20ms mean TPOT ceilings, and GPQA retains
its86.92% accuracy floor. These measurements do not establish device-only
Decode performance or100% cache hits.

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
coverage, the Decode20ms profile and Nightly matrix dispatch. Future NPU runs
check throughput against these baselines; cache-hit claims need separate evidence.
