# GLM-5.3 A3 Weekly PD coverage

Only the hardware-validated A3 W8A8C8 `ALL-COMBINED` case is included.
It runs Prefill, PD, Decode replay and GPQA on one full-checkpoint deployment.
All 78 layers and real weights are retained. Unvalidated A5 configurations and
their new Weekly workflow/dispatch changes are excluded from this PR.

## Serving recipe

Four A3 hosts each expose 16 logical devices. D runs on nodes0-1 with DP16/TP2
(eight instances per host); P runs on nodes2-3 with PP2/PCP16/TP1 and DP1.
The P pipeline partition is `42,36`. All client concurrency ceilings are16;
16/DP16 is a sizing average, not observed uniform per-instance concurrency.

Both roles use MRV2, INT8 main/indexer KV and MooncakeConnectorV2. P uses eager
MTP1, maxseq16/batched16384, prefix caching, FlashComm1 and fusedMC2 mode1.
D uses MTP5, maxseq1/batched6, FULL_DECODE_ONLY capture[6]/max6, npugraph_ex,
Static/Super Kernel, recompute scheduling and shared-expert overlap.
Maximum context is131072. Check actual runtime flags rather than inferring
execution from configuration alone.

## Workloads and cache boundaries

All performance phases use
`vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8` at concurrency16.

| Formal phase | Requests | Max output | Request rate | Warmup requests |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 28 | 1 | 0 | 32 |
| PD | 160 | 1024 | 1.8 | 80 |
| Decode replay | 320 | 1024 | 0 | 320 |

Prefill and PD reset P prefix cache after their warmups. Decode resets before
warming the320-prompt set, then replays it without another reset. After all
performance phases, reset again and run GPQA once: output65536, temperature1,
top-p0.95, concurrency16, accuracy floor86.92%.

All four resets target the non-headless P API on node2:7100 and assert JSON
`success=true`; HTTP200 alone is insufficient. P servers enable upstream
`VLLM_SERVER_DEV_MODE=1`, which also exposes other development endpoints and
is confined to this controlled CI service. Benchmark placeholders are resolved
by the common config loader. Resets do not restart servers or change graphs.
AISBench internal warmups/cache retention still require inspection before
calling a workload cold or fully cached.

## Dispatch and validation

```text
/weekly GLM-5.3-W8A8C8-A3-ALL-COMBINED
```

```bash
python -m unittest tests.ut.tools.test_glm53_full_model_cases
bash format.sh ci
```

## A3 calibrated performance gates

The successful [combined run 43574755884](https://github.com/vllm-project/vllm-ascend/actions/runs/43574755884/job/132388142386)
completed all 11 phases, four verified cache resets and all formal requests:
Prefill 28/28, PD 160/160 and Decode 320/320 with zero failures. GPQA was
90.4040%, above the retained 86.92% accuracy floor. Actual versions were
Ascend `13c66ee` and vLLM `4ba787ec9959`; all four nodes reported npu-smi 25.5.2.

| Formal phase | Output token/s baseline | Threshold | Minimum output token/s |
| --- | ---: | ---: | ---: |
| Prefill | 2.4064 | 0.95 | 2.28608 |
| PD | 936.7987 | 0.95 | 889.958765 |
| Decode replay | 1001.9522 | 0.95 | 951.85459 |

The existing runner gates aggregate **output** token throughput. Prefill emits
one token per request, so its output baseline equals request throughput; its
measured aggregate input throughput was 39455.3918 token/s. Do not put that
input metric in the output-baseline field. Warmups have no calibrated gate.

Compared with the archived isolated P/PD/D runs, combined throughput differed
by -0.43%, -1.90% and -0.28%, respectively. Different hosts and single runs do
not prove statistical equivalence. Decode replay is not a device-only benchmark
or evidence of 100% cache hits.
