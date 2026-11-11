# GLM-5.3 Weekly PD coverage

Three full-checkpoint Weekly cases cover A3 W8A8C8, A5 W8A8C8 and A5 W4A4C8.
Each `ALL-E2E` entry starts one fixed PD deployment and sequentially runs
Prefill, end-to-end, decode replay and GPQA benchmarks. Separate `ALL-PREFILL`
and `ALL-DECODE` entries are removed. All 78 layers and real weights are retained;
nightly, reduced-layer cases and 1M are outside this scope.

## One deployment per hardware and quantization family

This follows the structure of
[MiniMax's combined P/PD/decode case](../../MiniMax/MiniMax-M3-w8a8-PP-prefix-PD.yaml):
keep serving topology and optimization flags fixed while changing request loads.
The GLM cases retain their own datasets, request counts and TPOT targets rather
than copying MiniMax performance baselines or claiming identical prefix ratios.

| Platform | Allocation | P allocation | D allocation | KV format | Max context |
| --- | --- | --- | --- | --- | --- |
| A3 W8A8C8 | 4 x 16 logical devices | Nodes2-3: PP2, PCP16, TP1 | Nodes0-1: DP16, TP2 | INT8 | 131072 |
| A5 W8A8C8 / W4A4C8 | 4 x 8 devices | Nodes0-1: PP2, PCP8, TP1 | Nodes2-3: DP16, TP1 | FP8 | 135000 |

Each family uses one workload-sized serving configuration throughout all phases,
with P prefix caching enabled and D prefix caching disabled. Both roles use
Model Runner V2 and MooncakeConnectorV2, with DSA CP disabled. P retains eager
MTP1, PP partition `42,36`, FlashComm1 and fused MC2 mode 1. Shared-expert DP
is explicitly disabled on P because TP1 disables it in the config validator.
D retains
MTP5, FULL_DECODE_ONLY, recompute scheduling and shared-expert overlap. A3 retains
npugraph_ex with static and super kernels explicitly enabled;
A5 retains its previous static/super-disabled, MLAPO
and LM-head TP8 settings. Flags are not switched between benchmarks.

The recipes originate from the
[ordinary PD tutorial revision](https://github.com/vllm-project/vllm-ascend/blob/bee583c95f7e3222fd79a4deba206a6b2d9694b3/docs/source/tutorials/models/GLM5.2.md)
and the P PCP adaptation in [PR20228](https://github.com/vllm-project/vllm-ascend/pull/20228).
Capability checks can disable requested optimizations, including shared-expert DP
at TP1. Check effective runtime flags instead of assuming the YAML proves execution.

## Scheduler, capture and context budgets

`max-num-seqs` is a scheduling ceiling **per DP rank**, not a measurement of
live concurrency. The two D hosts together contain DP16 (8 instances per host),
so divide client concurrency by 16, not by the local DP8 count. PP/PCP/TP do not
multiply request-serving replicas. P is one DP1 pipeline across both P hosts.

| Family / phase | Client concurrency ceiling | Average D load at full client concurrency |
| --- | --- | --- |
| A3 Prefill | 16 | 1 |
| A3 PD | 16 | 1 |
| A3 Decode replay | 16 | 1 |
| A3 GPQA | 16 | 1 |
| A5 short 20 ms / long 50 ms | 16 | 1 |
| A5 short 50 ms / GPQA | 32 | 2 |
| A5 long 30 ms | 8 | 0.5 |

These are sizing averages, not observed running-request counts. Request lengths,
arrival rate (A3 PD remains 1.8 requests/s), prompt completion and token-aware
proxy routing can make live per-rank loads uneven or lower. Collect each D
engine's running/waiting metrics during measurement before reporting actual
concurrency. A3 now allows one scheduled sequence per D instance; uneven routing
can queue requests at an instance. This cap does not enforce even proxy routing.
When clients are fewer than DP ranks, some ranks may be idle.

| Parameter | A3 P / D | A5 P / D | Reason |
| --- | --- | --- | --- |
| `max-num-seqs` | 16 / 1 | 32 / 4 | A3 D matches16 clients/DP16; A5 D retains margin above average2. |
| `max-num-batched-tokens` | 16384 / 6 | 16384 / 24 | P retains chunked-prefill budget; D covers `seqs * (MTP5 + 1)`. |
| D capture token sizes | 6 | 6,12,18,24 | Exact uniform MTP5 batches for every supported request count. |
| D max capture token size | 6 | 24 | Avoid capturing batches far above tested client loads. |
| `max-model-len` | 131072 | 135000 | Total input plus output; A3 includes GPQA output65536; A5 retains128K input plus1024 output/headroom. |
| GPU/NPU memory utilization | 0.92 / 0.92 | 0.90 / 0.95 | Keep tutorial memory reservations; cache capacity still needs measurement. |
| Worker execution timeout | 3000 seconds | 3000 seconds | A per-step timeout, reduced from30000; not a request or service-start budget. |

A3 restores static **and** super kernels; A5 keeps both disabled. Each D keeps
MRV2, MTP5, FULL_DECODE_ONLY and npugraph_ex. Smaller captures reduce startup work
but do not prove startup fits the local3600-second readiness budget. The common
runner's global5400-second readiness/master-stop budget is unchanged. A YAML
server environment cannot extend those parent test-runner deadlines.

Use nested `scheduler_config.recompute_scheduler_enable` instead of the deprecated
top-level key. This keeps D steps decode-shaped after remote KV transfer. Fused
MC2 is explicitly1, matching current Ascend's automatic rollback of requested2.
P's eager mode, PP42/36 partition, FlashComm1, INT8/FP8 KV and indexer KV, and A5's
MLAPO/LM-head TP8/CPU binding remain recipe-specific choices, not knobs inferred
from client concurrency. MTP draft eager settings and optimization flags do not
change between phases. HCCL buffers (P2560MiB/D1024MiB), AIV/resource settings,
connect/execute timeouts1800s, transfer timeout, allocator settings, seed,
quantization, parsers, prefetch and Mooncake ports/topology are retained. Reducing
scheduler load alone does not establish a safe smaller collective buffer.

Definitions and implementation references:

- [vLLM0.33 serve arguments](https://docs.vllm.ai/en/v0.33.0/cli/serve/)
- [DP rank sizing and load balancing](https://docs.vllm.ai/en/v0.33.0/serving/data_parallel_deployment/)
- [Chunked-prefill token budgets](https://docs.vllm.ai/en/v0.33.0/configuration/optimization/)
- [Ascend speculative decoding capture sizes](../../../../../../../docs/source/user_guide/feature_guide/speculative_decoding.md)
- [Ascend config schema and effective optimization constraints](../../../../../../../vllm_ascend/ascend_config.py)

## Benchmark and cache sequence

For each performance profile, one service lifecycle runs these steps:

1. Prefill warmup, then reset P prefix cache, then measured Prefill with output 1.
2. PD warmup, then reset P prefix cache, then measured PD with output 1024.
3. Reset P prefix cache, warm the decode prompt set, then replay the identical
   prompt set with output 1024 and no intervening reset.
4. After all performance profiles, reset P prefix cache and run GPQA once.

Cache resets use the existing `curl_cmd` benchmark type and the non-headless P API at
port7100 (`/reset_prefix_cache`); they do not restart the service or
change compilation settings. HTTP errors fail the benchmark via `curl --fail`.
A5 keeps P on node0 and uses the local endpoint. A3 puts D on nodes0-1 and
P on nodes2-3; its resets target `${NODE_2_IP}:7100`, resolved by the config
loader before execution. Node indices are logical allocation positions, not
fixed physical hosts. The role swap does not fix dependency installation or
establish that the earlier SuperKernel failure is resolved.

The explicit resets prevent earlier full-prompt replays from warming the measured
P/PD phases. AISBench can still perform its own internal warmup; inspect actual
cache counters and prompt identities before calling a phase cold or fully cached.
Decode replay is a decode-dominated request benchmark, not a device-only decode
microbenchmark: API, KV transfer and residual prefill costs remain possible.
Check retained cache capacity and measured hit rates; repeated prompts alone do
not prove 100% hits. A5's 128K dataset has a 90% shared prefix and is not a
replacement for MiniMax's 99% dataset.

## Workloads

A3 retains the 16K datasets and measured counts adapted from release rows
104/124/135, with all client concurrency ceilings reduced to16 (including GPQA).
Prefill warmup uses32 prompts so it can fill16 client slots. They share
one serving recipe and therefore are no longer exact reproductions of the three
separately tuned release deployments.

| A3 phase | Measured requests | Concurrency | Request rate | Max output | Explicit warmup requests | TPOT limit |
| --- | --- | --- | --- | --- | --- | --- |
| Prefill | 28 | 16 | 0 | 1 | 32 | None |
| PD | 160 | 16 | 1.8 | 1024 | 80 | 20 ms |
| Decode replay | 320 | 16 | 0 | 1024 | 320 | 20 ms |

A3 uses `vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8`.
The earlier [successful A3 run](https://github.com/vllm-project/vllm-ascend/actions/runs/40859196273/job/123400043943)
validates dataset usability with different topology/load settings, not this new
combined deployment or its performance baseline.

A5 keeps four profiles per phase, using the same 16K dataset and
`vllm-ascend/GSM8K_prefix90_in131072_bs100_glm` for 128K:

| A5 profile | Requests per phase | Concurrency | PD/decode TPOT limit |
| --- | --- | --- | --- |
| Short 20 ms | 32 | 16 | 20 ms |
| Short 50 ms | 64 | 32 | 50 ms |
| Long 30 ms | 16 | 8 | 30 ms |
| Long 50 ms | 32 | 16 | 50 ms |

Each A5 warmup uses the same request count as its measured phase. All performance
phases produce separate results. Throughput baselines remain unset until hardware
calibration; provisional TPOT limits also require validation. GPQA remains last:
output 65536, temperature 1.0, top-p 0.95, concurrency16 on A3 /32 on A5,
provisional floor86.92.
That floor is not a measured A5 baseline. No deepSwe task is added without its
dataset and scoring protocol.

## Dispatch and validation

```text
/weekly GLM-5.3-W8A8C8-A3-ALL-E2E
/weekly GLM-5.3-W8A8C8-A5-ALL-E2E
/weekly GLM-5.3-W4A4C8-A5-ALL-E2E
```

A3 uses the existing Weekly matrix; A5 uses its multi-node Weekly workflow and
runs the two quantization families sequentially. No new timer is added.

```bash
python -m unittest tests.ut.tools.test_glm53_full_model_cases
bash format.sh ci
```

The previous PR20575 A3 runs failed before effective performance validation:
Prefill encountered a missing MegaMoE weight attribute, Decode did not become
ready, and E2E hit a SuperKernel graph optimization error. Combining deployments
does not fix those implementation or environment problems. Hardware startup,
cache resets, cache capacity/hits, KV transfer, performance and GPQA still require
NPU validation. Record actual commits, image digest, CANN/driver versions and
raw AISBench results before treating these draft cases as regression coverage.
