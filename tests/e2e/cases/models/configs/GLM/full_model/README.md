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

| Platform | Allocation | P: nodes 0-1 | D: nodes 2-3 | KV format | Max context |
| --- | --- | --- | --- | --- | --- |
| A3 W8A8C8 | 4 x 16 logical devices | PP2, PCP16, TP1 | DP16, TP2 | INT8 | 200000 |
| A5 W8A8C8 / W4A4C8 | 4 x 8 devices | PP2, PCP8, TP1 | DP16, TP1 | FP8 | 135000 |

Each family uses its previous E2E serving configuration throughout all phases,
with P prefix caching enabled and D prefix caching disabled. Both roles use
Model Runner V2 and MooncakeConnectorV2, with DSA CP disabled. P retains eager
MTP1, PP partition `42,36`, FlashComm1 and requested fused MC2 mode 2. D retains
MTP5, FULL_DECODE_ONLY, recompute scheduling and shared-expert overlap. A3 retains
npugraph_ex with static kernels disabled (super kernels inherit this setting);
A5 retains its previous static/super-disabled, MLAPO
and LM-head TP8 settings. Flags are not switched between benchmarks.

The recipes originate from the
[ordinary PD tutorial revision](https://github.com/vllm-project/vllm-ascend/blob/bee583c95f7e3222fd79a4deba206a6b2d9694b3/docs/source/tutorials/models/GLM5.2.md)
and the P PCP adaptation in [PR20228](https://github.com/vllm-project/vllm-ascend/pull/20228).
Capability checks can disable requested optimizations, including shared-expert DP
at TP1. Check effective runtime flags instead of assuming the YAML proves execution.

## Benchmark and cache sequence

For each performance profile, one service lifecycle runs these steps:

1. Prefill warmup, then reset P prefix cache, then measured Prefill with output 1.
2. PD warmup, then reset P prefix cache, then measured PD with output 1024.
3. Reset P prefix cache, warm the decode prompt set, then replay the identical
   prompt set with output 1024 and no intervening reset.
4. After all performance profiles, reset P prefix cache and run GPQA once.

Cache resets use the existing `curl_cmd` benchmark type and the master P API at
`http://127.0.0.1:7100/reset_prefix_cache`; they do not restart the service or
change compilation settings. HTTP errors fail the benchmark via `curl --fail`.
The public test endpoint on node 0 must remain port 7100 if these commands are reused.

The explicit resets prevent earlier full-prompt replays from warming the measured
P/PD phases. AISBench can still perform its own internal warmup; inspect actual
cache counters and prompt identities before calling a phase cold or fully cached.
Decode replay is a decode-dominated request benchmark, not a device-only decode
microbenchmark: API, KV transfer and residual prefill costs remain possible.
Check retained cache capacity and measured hit rates; repeated prompts alone do
not prove 100% hits. A5's 128K dataset has a 90% shared prefix and is not a
replacement for MiniMax's 99% dataset.

## Workloads

A3 retains the 16K profiles adapted from release rows 104/124/135. They now share
one serving recipe and therefore are no longer exact reproductions of the three
separately tuned release deployments.

| A3 phase | Measured requests | Concurrency | Request rate | Max output | Explicit warmup requests | TPOT limit |
| --- | --- | --- | --- | --- | --- | --- |
| Prefill | 28 | 7 | 0 | 1 | 14 | None |
| PD | 160 | 40 | 1.8 | 1024 | 80 | 20 ms |
| Decode replay | 320 | 80 | 0 | 1024 | 320 | 20 ms |

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
output 65536, temperature 1.0, top-p 0.95, concurrency 32, provisional floor 86.92.
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
