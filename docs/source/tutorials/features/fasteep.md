# FastEEP on Ascend (experimental)

This port adapts the retained-state design in
[vLLM FastEEP #44994](https://github.com/vllm-project/vllm/pull/44994)
to Ascend HCCL and CANN memory pools. Start all workers at the maximum
EP size and select an active prefix of ranks. Sleeping ranks release weights,
KV allocations, captured graph storage, and unused native HCCL communication
buffers. Worker processes and CPU rendezvous groups remain available. Returning
ranks recover the same virtual addresses, refill dense weights from a live peer,
and receive individual experts according to the new placement.

The runtime accepts Model Runner V1 or V2, BF16 weights, TP=PP=PCP=DCP=1,
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
The C++ shim uses the installed `HcclCommConfig` ABI. Select V1 or V2 with
the existing `VLLM_USE_V2_MODEL_RUNNER` setting.

For Qwen3-30B-A3B-Base on four NPUs, create a placement JSON with 48 entries
in `layer_list`. Each entry has a `device_list` of four objects whose
`device_expert` arrays are respectively 0–63, 32–95, 64–127, and 0–31 plus 96–127.
Set `moe_layer_count` to 48. This provides 128 redundant experts and supports
active prefixes of two, three, or four ranks. The full example is in
`tests/e2e/pull_request/four_card/test_fasteep_serving.py`.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve Qwen/Qwen3-30B-A3B-Base \
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

### Full-model functional qualification

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
These observations establish functionality for that configuration.

Single-node checks also compared V1 AllGather graph execution against eager
execution at each matched topology and an independent EP2 control. BF16
reduction grouping can change output across different EP topologies;
graph/eager parity is evaluated at the same topology. Qualification used
a normalization compatibility shim corresponding to
[Ascend #8384](https://github.com/vllm-project/vllm-ascend/pull/8384).

Performance of the earlier retained-communicator implementation is separate
from the reclaimable graph/communication implementation described here.

## Matched eager pilot

One method pair was measured at each offered rate, in fixed Native-then-FastEEP order. Each method completed 52/52 measured HTTP requests (48 load requests and four single-request controls), after real generation and 3→2→3 warmup. Process ownership was recorded on all six logical chips of the three reserved physical cards; no unregistered process was observed there during either method. Task-owned processes and reservations were released and checked independently.

Qwen3-30B-A3B-Base BF16; three Ascend 910 A3 cards, one model chip per card; TP=1, DP=EP3↔2, V1 eager, AllGather and Ray. Both methods use the same vLLM 0.21 compatibility runtime, full Ascend Native Elastic EP port, normalization shim, checkpoint hashes and four-shot prompts. Both disable continuous EPLB; the Native port performs placement changes during explicit resizes. This comparison measures the current FastEEP core in that common compatibility runtime. The current-stack PIECEWISE/AllToAll multi-node qualification above establishes functionality separately.

KV budget is 2 GiB per worker; actual block-rounded backing is 2,139,095,040 bytes (2,040 MiB). Maximum model length 2,048, maximum sequences 16, maximum batched tokens 512, maximum output 64, temperature 0, seed 51027. Actual physical capacity matches at 128 expert slots per layer for EP2 and 192 for EP3, with identical placement digests across all 48 layers. Initial placement per layer is experts0–63,64–127,0–63; heterogeneous-placement functionality is qualified separately above. Every direction uses 20 seconds of scheduled arrivals and a 4-second post-transition arrival phase; resize starts at 8 seconds. The same frontend waits for full responses to drain and queues new arrivals.

|Rate(req/s)|Scale|Native API(s)|FastEEP API(s)|Fast−Native(s)|Native drain(s)|FastEEP drain(s)|
|---|---|---:|---:|---:|---:|---:|
|0.25|3→2|8.159|12.949|+4.790|4.946|5.336|
|0.25|2→3|54.182|22.422|-31.760|4.515|8.943|
|0.75|3→2|10.848|20.491|+9.643|7.936|9.428|
|0.75|2→3|57.398|19.502|-37.896|7.845|8.153|

API includes the response drain. The accompanying data separates drain and endpoint time after drain. Each row has one observed call per method; there is no run-to-run confidence interval.

|Rate(req/s)|Method|Load requests|TTFT p95(s)|TPOT p95(ms)|Response p95(s)|Output(token/s)|
|---|---|---:|---:|---:|---:|---:|
|0.25|Native|12|54.266|146.78|63.514|6.580|
|0.25|FastEEP|12|16.698|138.57|25.306|8.836|
|0.75|Native|36|57.472|144.19|66.371|17.810|
|0.75|FastEEP|36|18.741|143.50|27.601|23.358|

At 0.25 requests/s, whole-interval output throughput changes by +2.257 token/s (+34.30%). At 0.75 requests/s, whole-interval output throughput changes by +5.549 token/s (+31.16%).

Output throughput counts load output tokens over the entire rate interval, including drain, transition, controls and admission waiting; control tokens are excluded from the numerator. This is completed work under the offered load, not peak serving capacity. TTFT and TPOT are measured from nonempty streamed chunks; p95 summarizes requests, not repeated runs.

|Condition|Native HBM mean(driverMB)|FastEEP HBM mean(driverMB)|Fast−Native(driverMB)|
|---|---:|---:|---:|
|r0.25-c0-post2|87916.2|91907.2|+3991.0|
|r0.25-c0-post3|123157.7|127218.3|+4060.7|
|r0.75-c0-post2|88011.5|92035.2|+4023.7|
|r0.75-c0-post3|123227.0|127180.5|+3953.5|

HBM covers all six chips of the three physical cards after each transition. Driver MB is the driver’s reported unit. Actual sample spacing: median 2.334s, maximum 3.427s; sampled peaks are lower bounds. Torch current allocated/reserved bytes are reported separately from driver HBM. Torch peak counters are per-process lifetime values; retained virtual allocations can remain counted after physical discard.

Output validation: 7/7 fixed single-request control pairs match token IDs and finish reason; load token parity is 39/48. Native: 2/48 explicit final answers correct, 46 missing final answers; FastEEP: 2/48 explicit final answers correct, 46 missing final answers. Load scoring requires the explicit final-answer marker and exact numeric comparison. The 64-token limit and reused questions make these descriptive observations, not full GSM8K accuracy.

The earlier retained-communicator study is preserved separately and is not used for these numbers. Software/model source hashes, absolute values, paired deltas, Torch snapshots, output counts and definitions accompany the measurement record.

### Recorded commands and sources

Set `MODEL` to the locally acquired, SHA-verified checkpoint and `MODEL_CHIPS` to three exclusively reserved chips on three physical cards (the measured two-chip-per-card mapping used logical chips 0,2,4). Monitor/reserve the companion chips as well. The explicit KV byte budget takes precedence over the utilization fraction; it does not cap model weights or native workspaces.

The shared compatibility environment uses vLLM commit ad7125a431e176d4161099480a66f0169609a690 and Ascend base commit 80610e4438dba05011b05f89fc45d91e96992671 with the full legacy Native Elastic EP port and common framework adapters. The measured FastEEP core matches commit 789752924ae45d6f7018945a31d6b8d38afaf7ae.

Python 3.12.9, PyTorch 2.10.0+cpu, torch-npu 2.10.0, Ray 2.48.0, transformers 5.5.4, CANN 9.2.0 and NNAL 9.0.0. These are the recorded benchmark-harness commands, requiring the shared compatibility port, prepared server wrapper, observer/admission modules and frozen inputs.

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
  --rates 0.25 0.75 --cycles 1 --cycle-offset 0 \
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
  --rates 0.25 0.75 --cycles 1 --cycle-offset 0 \
  --seconds 20 --post-seconds 4 --max-tokens 64
```

Frozen input and harness SHA-256 values:

|File|SHA-256|
|---|---|
|run_prepared_comparison.py|c615879f31ca941d4823b401b23d5ed1267215ec6dfdeeaa6b4834abcf11a532|
|load_client.py|0ee23c339959f0d5c165aa9c2405bc0b4454ffa3af8e26b2f6265aade14f0bef|
|frozen-requests.json|cd709866888661a97b20684c17e3bf57b926ffba7357821e967745e25f7c2057|
|frozen-expert-map.json|6cf98cab8c3287ea56715b4c556ffa4c8c71496c061c72f7270018b59c6c15a6|
|prompt_policy.json|b49c6d464df5f4cf28ef6121d8ac628f3c780c2d945d9553b12799ace79ff3f6|
|candidate-current-head-core-manifest.json|df9ee804df8552529e0fa3f9fb87dce16acb6513ba0a0b4fb020699ffa17c5b5|
|Shared compatibility differences: vllm-legacy-common.patch|5b925ccd87287c7b5e6dd520e0a1d686190ac5e90ba714e4cf2e843c77f40962|
|Shared compatibility differences: vllm-ascend-legacy-common.patch|90daafeb9bbc036924fcb6a80cc850d97bc43e295cd3637393169155cc54da88|

The generated measurement JSON and CSV preserve full precision, per-rank Torch snapshots, source/model hashes and paired output counts.
