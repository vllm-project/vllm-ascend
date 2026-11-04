# GLM-5.3 Weekly PD coverage

These 12 full-checkpoint cases implement the first Weekly coverage batch on
main: A3 W8A8C8, A5 W8A8C8 and A5 W4A4C8, each with `ALL-PREFILL`,
`ALL-DECODE`, `ALL-E2E` and `ACCURACY`. They retain all 78 layers and real
weights. Nightly, reduced-layer cases and **1M are outside this scope**.

## Recipe sources and migration

The starting point is the [official GLM-5.3 ordinary PD tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.2.html),
specifically the A3 and 950DT tabs, accuracy records and workload inventory.
The [source revision used for this adaptation](https://github.com/vllm-project/vllm-ascend/blob/bee583c95f7e3222fd79a4deba206a6b2d9694b3/docs/source/tutorials/models/GLM5.2.md)
distinguishes GLM-5.3 from older GLM-5.2 recipes. The P PCP migration follows
[PR #20228](https://github.com/vllm-project/vllm-ascend/pull/20228).
These are main-branch adaptations; the tutorial's older runner/DSA CP settings
are not copied unchanged.

| Platform | Allocation | P: nodes 0–1 | D: nodes 2–3 | Cache | Max context |
| --- | --- | --- | --- | --- | --- |
| A3 W8A8C8 | 4 × 16 logical devices | PP2, PCP16, TP1 | DP16, TP2 | INT8 | 200000 |
| A5 W8A8C8 / W4A4C8 | 4 × 8 devices | PP2, PCP8, TP1 | DP16, TP1 | FP8 | 135000 |

The A3 runner counts logical devices (two dies per physical card), giving
32 physical cards. A5 uses 32 devices. Both are one P instance spanning two
nodes and one D group spanning two nodes. The official A5 example has two P
instances; this batch retains one to match the four-node 1P1D allocation.
A5 PCP8 is an adaptation of the A3 PCP recipe and needs hardware validation.

- Both roles set `VLLM_USE_V2_MODEL_RUNNER=1`, use `MooncakeConnectorV2`,
  and omit DSA CP. Connector metadata agrees with the actual PP/TP/DP topology.
- P uses the `42,36` PP partition, eager execution, MTP1, FlashComm1 and
  fused MC2 mode 2. Its HCCL buffer is 2560 MiB, following the PCP reference.
  `enable_shared_expert_dp=true` is retained from that reference, but main
  disables it for TP1; it is not claimed as an effective optimization.
- D uses MTP5, recompute scheduling and `FULL_DECODE_ONLY`. For the
  PREFILL-dominated case it uses fused MC2 mode 1, following the RP3 script.
  DECODE/E2E/ACCURACY instead use shared-expert overlap and `npugraph_ex`.
  A3 enables static kernels; A5 disables static/super kernels and uses MLAPO
  and LM-head TP8, following the platform-specific tutorial settings.
- A3 retains the PCP reference's scheduler settings: P batch tokens 16384,
  D 256, 64 sequences and memory utilization 0.92. A5 follows the ordinary
  PD tutorial: P 16384/500/0.9 and D 60/10/0.95, respectively. A5 uses local
  communication resource version 1.3 and a 1200 MiB D HCCL buffer.
- P prefix caching is enabled only for DECODE. D prefix caching is disabled
  to keep each decode request's KV-transfer path active. This mode-specific
  choice follows the RP3 reference rather than the tutorial's general caching
  defaults. Separate YAMLs start separate server lifecycles.

Fused MC2 mode 2 can fall back or be disabled by main's capability checks.
Verify effective flags in startup logs and traces; a requested flag alone
does not prove that an optimization executed. W4A4 PD and graph/MTP behavior
remain unmeasured; the tutorial's W4A4 accuracy record uses colocated GSM8K,
not this PD GPQA stack.

## A3 release comparison: one 16K profile per mode

The A3 performance cases use the ordinary PD rows 104, 124 and 135 from the
`A3_发布` sheet in `测试-推理解决方案26.2.0性能一览表 (2).xlsx`.
Row 124 selects the low-latency DECODE point. Each YAML contains one warmup
and one measured workload. Existing `ALL-*` dispatch names are retained;
for A3 they now cover only the selected 16K comparison point, not all lengths.
4K, 64K, 128K, pooling, dedicated datasets and 1M are outside this A3 batch.

| A3 mode | Input / output tokens | Concurrency | Measured requests | Request rate | P / D prefix caching | Average TPOT limit |
| --- | --- | --- | --- | --- | --- | --- |
| PREFILL (row 104) | 16384 / 1 | 7 | 28 | 0 (closed loop) | Off / Off | Not applicable |
| DECODE (row 124) | 16384 / 1024 | 80 | 320 | 0 (closed loop) | On / Off | 20 ms |
| E2E (row 135) | 16384 / 1024 | 40 | 160 | 1.8 req/s | Off / Off | 20 ms |

All three reuse `vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8`.
The earlier [successful A3 job](https://github.com/vllm-project/vllm-ascend/actions/runs/40859196273/job/123400043943)
checked out `d74a7178421715e3d1166965258be5578dab14ea`; its
[configuration](https://github.com/vllm-project/vllm-ascend/blob/d74a7178421715e3d1166965258be5578dab14ea/tests/e2e/cases/features/pp/GLM-5.3-W8A8C8-A3-200k-PP-PD.yaml)
uses this dataset. The log records 48/48 successful warmup requests and
96/96 successful measured requests, zero failures and 1122.6822 output tokens/s.
That run used concurrency 24 and 1.4 req/s; its throughput is evidence for
dataset usability, not a comparable baseline for the new release workloads.
The release spreadsheet does not identify its original dataset files, so
this change aligns load values and cache policy without claiming identical
prompt content to the release run.

PREFILL warms 14 requests and E2E warms 80 before their separate measured
phases. These warmup counts are commissioning choices, not values specified
by the release sheet. DECODE first warms all 320 measured prompts with P
caching enabled, then replays the same workload. The dataset itself has
distinct prompts (prefix0); warmup/replay supplies the intended 100% P hit
scenario. Verify actual prompt reuse, cache hits and retained KV capacity
before comparing DECODE results. D caching stays off, matching the release
script. Each YAML starts a separate server lifecycle.

## A5 initial workloads and common gates

A5 retains its initial 16K/128K profiles below; A3 release values are not
treated as A5 capacity measurements. Concurrency and request counts are
commissioning values. TPOT limits require hardware calibration.

| Profile | Input tokens (dataset label) | Concurrency | Requests per phase | D / E2E average TPOT limit |
| --- | --- | --- | --- | --- |
| Short 20 ms | 16384 | 16 | 32 | 20 ms |
| Short 50 ms | 16384 | 32 | 64 | 50 ms |
| Long 30 ms | 131072 | 8 | 16 | 30 ms |
| Long 50 ms | 131072 | 16 | 32 | 50 ms |

The existing datasets are `vllm-ascend/GSM8K_prefix0_in16384_bs500_GLM5_3_w8a8c8`
and `vllm-ascend/GSM8K_prefix90_in131072_bs100_glm`. The long dataset has a
shared prefix; only DECODE enables its reuse. Record actual tokenizer lengths
and generated lengths rather than assuming dataset labels or output maxima
equal observed lengths.

Each A5 profile warms up and measures the same prompt set with closed-loop
traffic (`request_rate: 0`). Profiles within a YAML share the warmed service.
Confirm cache-hit metrics and retained cache capacity before interpreting
DECODE as a 100%-prefix-hit workload. PREFILL remains a prefill-dominated
request through PD; its one output token cannot measure TPOT. Throughput is
collected without a regression gate until measured baselines are available.

ACCURACY runs GPQA with maximum output 65536, temperature 1.0, top-p 0.95
and concurrency 32. The provisional floor 86.92 is the existing A3 reference
91.92 minus five points; it is not an official tolerance or an A5 measurement.
Calibrate each quantization before enabling this as a regression gate.
The scope's tentative deepSwe ten-question item has no supplied dataset or
scoring protocol and is not implemented by these GPQA cases.

## Dispatch and commissioning

The A3 cases join the existing Weekly matrix. A dedicated Weekly-A5 workflow
runs the eight A5 cases sequentially and uses `KUBECONFIG_A5_B64`. The workflow
supports manual dispatch and the existing authorized `/weekly` route, without
activating legacy A5 single-node cases. It must land on main before that route
can dispatch it by filename. This change adds no new timer schedule.

Start with E2E and GPQA for A3 W8A8C8, then A5 W8A8C8 and A5 W4A4C8. After
startup, memory, transfer and accuracy checks pass, run PREFILL and DECODE
for each platform. Authorized maintainers can select an individual entry:

```text
/weekly GLM-5.3-W8A8C8-A3-ALL-E2E
/weekly GLM-5.3-W8A8C8-A5-ACCURACY
/weekly GLM-5.3-W4A4C8-A5-ALL-DECODE
```

Run CPU configuration, command-rendering and dispatch checks before NPU runs:

```bash
python -m unittest tests.ut.tools.test_glm53_full_model_cases
bash format.sh ci
```

Capture vLLM/vLLM-Ascend commits, image digest, CANN/driver and checkpoint
versions, resolved commands, effective flags, memory use, cache-hit metrics
and AISBench artifacts. Confirm all 78 layers, PCP, MRV2 and connector V2,
and the absence of DSA CP. Hardware execution and calibrated throughput and
accuracy thresholds are required before merging this draft as regression
coverage; CPU checks cannot establish NPU correctness.
