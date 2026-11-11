# FastEEP on Ascend (experimental)

This port adapts the retained-state design in
[vLLM FastEEP #44994](https://github.com/vllm-project/vllm/pull/44994)
to Ascend HCCL and CANN memory pools. Start all workers at the maximum
EP size and select an active prefix of ranks. Sleeping ranks release weights,
KV allocations, captured graph storage, and unused native HCCL communication
buffers. Worker processes and CPU rendezvous groups remain available. Returning
ranks recover the same virtual addresses, refill dense weights from a live peer,
and receive individual experts according to the new placement.

Current Ascend main uses Model Runner V2. FastEEP requires BF16 weights, TP=PP=PCP=DCP=1,
one Python API process, internal DP routing, and AllGather or AllToAll MoE
communication. Nodes use a mutually reachable CPU rendezvous network and HCCL
network. Every rank has equal expert-slot capacity; the active ranks together
must have enough slots to cover all logical experts. Complete shard replicas
are not required. Continuous EPLB and prefix caching are disabled.

Eager and PIECEWISE graph execution are supported by the implementation.
PIECEWISE captures compute while MoE communication stays outside capture, so
native communicators can be recreated without retaining stale graph handles.
Graph storage is copied to CPU before physical memory is released, then restored
to the original addresses before replay. FULL graphs containing communication
handles are rejected. This configuration contract is distinct from the hardware
qualification results described below.

## Launch

Build the native extension with CANN headers providing `HcclCreateSubCommConfig`.
The C++ shim uses the installed `HcclCommConfig` ABI. Use Model Runner V2 on
current main; the V1 commands in the matched comparison apply to its pinned compatibility runtime.

For Qwen3-30B-A3B-Base on four NPUs, create a placement JSON with 48 entries
in `layer_list`. Each entry has a `device_list` of four objects whose
`device_expert` arrays are respectively 0–63, 32–95, 64–127, and 0–31 plus 96–127.
Set `moe_layer_count` to 48. This provides 128 redundant experts and supports
active prefixes of two, three, or four ranks. The full example is in
`tests/e2e/pull_request/four_card/test_fasteep_serving.py`.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-30B-A3B-Base \
  --data-parallel-size 4 --data-parallel-size-local 4 \
  --enable-expert-parallel --api-server-count 1 --dtype bfloat16 \
  --enforce-eager --enable-sleep-mode --no-enable-prefix-caching \
  --worker-extension-cls vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension \
  --middleware vllm_ascend.distributed.fasteep.frontend.FastEEPMiddleware \
  --additional-config '{"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":128,"expert_map_path":"placement.json"}}'
```

`--enable-elastic-ep` is left unset: the retained worker lifecycle is managed
by the FastEEP worker extension. The bundled stateless EEP platform changes
originate in [Ascend #19961](https://github.com/vllm-project/vllm-ascend/pull/19961).

Set `additional_config.fasteep.communication` to `"alltoall"` to use AllToAll;
the default is `"allgather"`. Optional `verify_migration: true` checks destination
weights immediately after each transfer and reports the checked tensor count.
The checks synchronize each layer and are intended for validation.

For captured execution, remove `--enforce-eager` and add:

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
```

## Resize

```bash
curl -X POST http://localhost:8000/fasteep_scale_ep \
  -H 'Content-Type: application/json' \
  -d '{"new_data_parallel_size":2,"drain_timeout":120}'
```

The endpoint uses the server's configured API-key authentication. New inference
requests wait while existing responses (including SSE streams) finish. The
engine drains and clears its caches before the worker collective changes the
topology. The response is sent after every startup worker reports the committed
state. A second concurrent resize returns 409. Disconnecting the resize client
does not cancel an in-progress mutation. A failure after pausing keeps admission
closed and requires restarting the instance. A timeout while waiting for HTTP
responses reopens admission without changing topology. Close streaming-input
sessions before resizing.

The native `/scale_elastic_ep`, sleep/wake, pause/resume and collective-RPC
control endpoints are disabled in this mode. Worker
processes and CPU rendezvous groups remain resident until server shutdown.

Layers whose expert slots already match the target placement skip migration
collectives. The migration communicator is initialized only for peer transfers
or joining-worker weight refill. Equal layer placements reuse routing-table
construction within each resize while retaining independent layer metadata.
After migration channels close, inactive workers reclaim their pools while
surviving workers initialize the serving communicator. Every startup rank
participates in the final topology commit barrier.
Native migration communicators exist only during transitions. Active serving
communicators are rebuilt for the new group, and replaced startup device
communicators are destroyed. This releases their communication buffers while
retaining the process and device context.

## Validation

- CPU tests exercise placement validation, admission, streaming drain, routing,
  authentication, timeouts, cancellation and worker-state receipts.
- `test_fasteep_hccl.py` exercises actual four-NPU retained communicator handles,
  collective values, excluded ranks and peer copies.
- `test_fasteep_memory.py` exercises pooled weight/KV discard, peer refill,
  rank-local metadata restoration and address stability.
- `test_fasteep_serving.py` requires a compatible current-main build and the
  Qwen3 checkpoint; it checks repeated 4→3→2→4 transitions, deterministic outputs,
  streaming drain, admission during resize, and output after final restoration.
- `test_fasteep_expert_recovery.py` combines heterogeneous expert placement,
  actual native communicator destruction, zero-split AllToAll, captured graph
  release/restore/replay, and stable parameter addresses.

Run the targeted NPU tests on four idle devices. Component results and full
serving qualification must be reported separately for the exact tested source.

### Earlier full-model functional qualification

Qwen3-30B-A3B-Base BF16 was validated on three Ascend 910B3 NPUs across
two nodes, using Model Runner V2, PIECEWISE graphs, AllToAll, and
TP=PP=PCP=DCP=1. Repeated DP=EP 3→2→3 transitions, streaming drain,
arrival admission, and final restoration produced 16/16 exact output-token
and finish-reason pairs, including 128 streaming chunks.

Each rank retained the same 388 captured graph objects. Inactive ranks
offloaded and restored 24 MiB of graph storage at the original addresses.
Observed inactive native communication reclamation was 145,805,312,
307,748,864, and 307,576,832 bytes across shrink transitions. Each rejoin
verified 6,144 expert tensors after migration. Initial MC2 handle cleanup
executed, and owned processes and reservations were released afterward.
These observations precede the shrink-overhead follow-up and establish functionality for that configuration.

Single-node checks also compared V1 AllGather graph execution against eager
execution at each matched topology and an independent EP2 control. BF16
reduction grouping can change output across different EP topologies;
graph/eager parity is evaluated at the same topology. Qualification used
a normalization compatibility shim corresponding to
[Ascend #8384](https://github.com/vllm-project/vllm-ascend/pull/8384).

Performance of the earlier retained-communicator implementation is separate
from the reclaimable graph/communication implementation described here.

## Matched eager resize comparison (2026-11-11)

Three independent method pairs were measured at each offered rate, with process order Native/FastEEP, FastEEP/Native, Native/FastEEP. Each method block starts a fresh server and completes three 3→2→3 warmup cycles before measured arrivals. The six completed measurement blocks passed: 156/156 HTTP requests per method (144 load requests and 12 streaming single-request controls). Ownership checks cover all six logical chips of three reserved physical cards; only registered task processes are cleaned up. The third pair resumed in a fresh controller after an ownership guard stopped its earlier server start before measurement; the first two pairs were retained. Processes and reservations were released and checked after the study.

Qwen3-30B-A3B-Base BF16; three Ascend 910 A3 cards, one model chip per card; TP=1, DP=EP3↔2, V1 eager, AllGather and Ray. Both methods use the same pinned vLLM 0.21 compatibility runtime, full Ascend Native Elastic EP port, normalization shim, checkpoint hashes and four-shot prompts. Both disable continuous EPLB. The measured FastEEP modules match commit `834f1dcfad842c2e19eef1e0511b92fb68fb1b07`; the shared legacy allocator/framework adapters remain pinned separately. Current-main V2 functional qualification is a separate configuration.

KV budget is 2 GiB per worker; actual block-rounded backing is 2,139,095,040 bytes (2,040 MiB). Maximum model length 2,048, maximum sequences 16, maximum batched tokens 512, maximum output 64, temperature 0, seed 51027. Actual capacity and placement match across methods at every measured transition: 128 expert slots per layer at EP2, 192 at EP3, across 48 layers. Initial per-layer placement is experts 0–63,64–127,0–63. Every direction schedules arrivals for 20 seconds and starts resize at 8 seconds, followed by a four-second post-transition arrival phase. A common frontend drains complete responses and queues new arrivals during resize.

### Resize latency

Values below are medians of three calls per method and condition. API latency includes response drain; endpoint time is the server transaction duration after subtracting that drain. Column medians are computed independently and need not sum. Three pairs do not establish statistical significance.

|Rate(req/s)|Scale|Native API(s)|FastEEP API(s)|Fast−Native API(s)|Native endpoint(s)|FastEEP endpoint(s)|Native drain(s)|FastEEP drain(s)|
|---|---|---|---|---|---|---|---|---|
|0.25|3→2|11.347|6.081|-5.267|3.313|1.937|7.994|3.929|
|0.25|2→3|47.070|15.533|-31.538|43.187|7.794|3.883|7.737|
|0.75|3→2|10.112|8.421|-1.690|3.009|1.552|6.923|6.917|
|0.75|2→3|49.985|16.897|-33.088|43.010|8.970|7.734|8.177|

Individual measured calls (API / endpoint after drain, seconds); [full-precision resize data](fasteep_resize_measurements.csv) and [per-block serving data](fasteep_serving_measurements.csv) accompany these tables:

|Rate(req/s)|Pair|Native 3→2|FastEEP 3→2|Native 2→3|FastEEP 2→3|
|---|---|---|---|---|---|
|0.25|1|11.347 / 3.351|6.110 / 1.918|52.350 / 44.912|11.325 / 7.842|
|0.25|2|11.412 / 3.313|6.081 / 2.150|47.070 / 43.187|15.533 / 7.794|
|0.25|3|7.010 / 3.306|5.830 / 1.937|46.248 / 42.724|15.783 / 7.628|
|0.75|1|10.403 / 2.957|8.629 / 1.710|50.745 / 43.010|17.641 / 8.970|
|0.75|2|10.112 / 3.228|8.421 / 1.417|49.985 / 42.127|16.541 / 9.613|
|0.75|3|9.932 / 3.009|8.336 / 1.552|49.598 / 43.131|16.897 / 8.718|

### Serving and memory

Each p95 is calculated over requests within one method block and rate; the table reports the median of those three p95 values. Throughput is the median of three whole-interval measurements.

|Rate(req/s)|Method|Load requests(total)|TTFT p95(s)|TPOT p95(ms)|Response p95(s)|Output(token/s)|
|---|---|---|---|---|---|---|
|0.25|Native|36|47.265|132.764|55.043|7.042|
|0.25|FastEEP|36|9.715|124.490|17.350|10.786|
|0.75|Native|108|49.033|128.649|57.044|19.651|
|0.75|FastEEP|108|14.273|128.937|22.630|28.017|

Load completion tokens / elapsed first shrink load-phase start to final expansion post-phase completion, including drain, resize, controls and admission waits; sentinel tokens excluded. This measures completed work under the offered load.

|Rate(req/s)|After|Native HBM(driverMB)|FastEEP HBM(driverMB)|Fast−Native(driverMB)|Samples Native/FastEEP|
|---|---|---|---|---|---|
|0.25|EP2|87959.0|91840.5|+3881.5|12 / 12|
|0.25|EP3|123139.2|127131.0|+3991.8|11 / 12|
|0.75|EP2|88004.2|91941.7|+3937.5|17 / 17|
|0.75|EP3|123150.7|127088.5|+3937.8|17 / 18|

HBM is the median of the three block means over post-transition arrival phases, summed over all six chips of three physical cards, in the driver's reported MB. Actual monitor sample spacing within controller segments: median 1.908 s, maximum 3.783 s. Sampled peaks are lower bounds. Torch allocated/reserved counters are separate; virtual allocations may remain counted after physical unmapping. FastEEP's additional HBM remains an observed cost in this configuration.

Output checks: all 144 paired load prompt hashes and all measured active topologies/placement digests match. Fixed controls match token IDs and finish reason in 33/33 pairs (initial generation, six warmup probes and four measured sentinels per pair). Load token/finish parity is 112/144.
Native: 4/144 explicit final answers correct, 0 incorrect explicit answers, 140 missing final answers.
FastEEP: 4/144 explicit final answers correct, 1 incorrect explicit answer, 139 missing final answers.
Explicit #### numeric final answer only; exact Decimal comparison. 64 output-token cap; reused questions; descriptive, not full GSM8K accuracy.

The original CUDA FastEEP paper describes retaining execution metadata and offloading inactive data buffers (2–4 GB, excluding KV) to CPU memory. That CPU figure does not quantify a Native-versus-FastEEP HBM difference. See [FastEEP §6](https://arxiv.org/html/2610.18451v1#S6); the Ascend HBM values above are measured separately.

### Isolated optimization and migration checks

An instrumented no-arrival check in the same compatibility stack compared the previous FastEEP core with this change. Three successive shrink API calls in one fresh process per implementation were 4.855586 / 3.518592 / 3.459938 s before and 3.709730 / 2.626814 / 2.735158 s after. The median fell from 3.518592 to 2.735158 s (22.3%). This diagnostic is distinct from the uninstrumented Native comparison above.

The change skips empty migration work, creates peer communication only when transfers/refill require it, shares routing-table construction across equal layer placements, and overlaps inactive pool reclamation with active communicator setup. Before pool unmapping, migration and old serving communicators are closed; all ranks synchronize before publishing readiness.

A separate full-model heterogeneous-placement check completed two 3→2→3 cycles with migration readback enabled. Its initial shards were 0–63,32–95,64–127. The first shrink checked 3,072 destination tensors; each expansion checked 6,144. All post-transition generation checks passed. Inactive native-communication reclamation was 146,296,832 and 148,332,544 bytes, accounted before weight/KV pool reclamation. Performance measurements above disable readback and profiling hooks.

### Recorded commands and sources

Set `PAIR` to 0, 1 or 2. Start a fresh server for every method block, perform three 3→2→3 warmup cycles with a post-transition generation check, then execute the corresponding load command. Run Native/FastEEP for pair 0, FastEEP/Native for pair 1, and Native/FastEEP for pair 2.

Set `MODEL` to the locally acquired, SHA-verified checkpoint and `MODEL_CHIPS` to three exclusively reserved chips on three physical cards (the measured two-chip-per-card mapping used logical chips 0,2,4). Monitor/reserve the companion chips as well. The explicit KV byte budget takes precedence over the utilization fraction; it does not cap model weights or native workspaces.

The shared compatibility environment uses vLLM commit ad7125a431e176d4161099480a66f0169609a690 and Ascend base commit 80610e4438dba05011b05f89fc45d91e96992671 with the full legacy Native Elastic EP port and common framework adapters. The measured FastEEP core matches commit 834f1dcfad842c2e19eef1e0511b92fb68fb1b07.

Python 3.12.9, PyTorch 2.10.0+cpu, torch-npu 2.10.0, Ray 2.48.0, transformers 5.5.4, CANN 8.5.1, NNAL 8.5.1 and triton-ascend 3.2.1. These are the recorded benchmark-harness commands, requiring the shared compatibility port, prepared server wrapper, observer/admission modules and frozen inputs.

**Native Elastic EP**

```bash
ASCEND_RT_VISIBLE_DEVICES="$MODEL_CHIPS" FASTEEP_LEASED_NPUS="$MODEL_CHIPS" \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:False HCCL_DETERMINISTIC=true \
HCCL_BUFFSIZE=64 HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 OMP_NUM_THREADS=8 \
python prepared-serve.py \
  --model "${MODEL}" \
  --served-model-name fasteep-qwen \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 3 \
  --data-parallel-backend ray \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --enable-sleep-mode \
  --no-enable-prefix-caching \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.80 \
  --additional-config '{"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":64,"expert_map_path":"frozen-expert-map.json"},"enable_cpu_binding":false}' \
  --worker-extension-cls benchmark_observer.MemoryExtension \
  --enable-elastic-ep \
  --middleware benchmark_observer.Observer \
  --middleware serving_admission_queue.ServingAdmissionQueue \
  --safetensors-load-strategy eager
```

```bash
python load_client.py --url http://localhost:8000 \
  --requests frozen-requests.json --prompt-policy prompt_policy.json \
  --output requests.jsonl --action evaluate --mode elastic \
  --rates 0.25 0.75 --cycles 1 --cycle-offset "$PAIR" \
  --seconds 20 --post-seconds 4 --max-tokens 64
```

**FastEEP**

```bash
ASCEND_RT_VISIBLE_DEVICES="$MODEL_CHIPS" FASTEEP_LEASED_NPUS="$MODEL_CHIPS" \
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTORCH_NPU_ALLOC_CONF=expandable_segments:False HCCL_DETERMINISTIC=true \
HCCL_BUFFSIZE=64 HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 OMP_NUM_THREADS=8 \
python prepared-serve.py \
  --model "${MODEL}" \
  --served-model-name fasteep-qwen \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-size-local 3 \
  --data-parallel-backend ray \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --enforce-eager \
  --enable-sleep-mode \
  --no-enable-prefix-caching \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 2147483648 \
  --gpu-memory-utilization 0.80 \
  --additional-config '{"eplb_config":{"dynamic_eplb":false,"num_redundant_experts":64,"expert_map_path":"frozen-expert-map.json"},"enable_cpu_binding":false}' \
  --worker-extension-cls vllm_ascend.distributed.fasteep.runtime.FastEEPWorkerExtension \
  --middleware vllm_ascend.distributed.fasteep.frontend.FastEEPMiddleware \
  --middleware benchmark_observer.Observer \
  --middleware serving_admission_queue.ServingAdmissionQueue \
  --safetensors-load-strategy eager
```

```bash
python load_client.py --url http://localhost:8000 \
  --requests frozen-requests.json --prompt-policy prompt_policy.json \
  --output requests.jsonl --action evaluate --mode reuse \
  --rates 0.25 0.75 --cycles 1 --cycle-offset "$PAIR" \
  --seconds 20 --post-seconds 4 --max-tokens 64
```

Frozen input and harness SHA-256 values:

|File|SHA-256|
|---|---|
|run_prepared_comparison.py|91339c56c744566a1fe177c10442b73b3af721789d34a4460856ecc95de046a3|
|Third-pair resumed controller (pair offset only)|c3d6b9fe4b0f8f501ab0864922ae52c0a37d07d169bc2a6439491c2b7e910d62|
|load_client.py|0ee23c339959f0d5c165aa9c2405bc0b4454ffa3af8e26b2f6265aade14f0bef|
|frozen-requests.json|cd709866888661a97b20684c17e3bf57b926ffba7357821e967745e25f7c2057|
|frozen-expert-map.json|6cf98cab8c3287ea56715b4c556ffa4c8c71496c061c72f7270018b59c6c15a6|
|prompt_policy.json|b49c6d464df5f4cf28ef6121d8ac628f3c780c2d945d9553b12799ace79ff3f6|
|candidate-current-head-core-manifest.json|a8682ddcdd488b3cb023d478992c69d422c9c24f8da926982ec97141a6712713|
|Shared compatibility differences: vllm-legacy-common.patch|5b925ccd87287c7b5e6dd520e0a1d686190ac5e90ba714e4cf2e843c77f40962|
|Shared compatibility differences: vllm-ascend-legacy-common.patch|90daafeb9bbc036924fcb6a80cc850d97bc43e295cd3637393169155cc54da88|

The retained measurement record preserves full precision, per-rank Torch snapshots, source/model hashes and paired output counts. The [checkpoint manifest](fasteep_checkpoint_manifest.json) records all 16 shard SHA-256 values and acquisition revisions. Its canonical sorted-key compact JSON SHA-256 is `f7168f67afd0fbdcb662ac757bf05a8d29c3f40d6aa818938b14487a8e33f69a`.
