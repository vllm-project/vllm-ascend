# FastEEP on Ascend (experimental)

This port adapts the retained-state design in
[vLLM FastEEP #44994](https://github.com/vllm-project/vllm/pull/44994)
to Ascend HCCL and CANN memory pools. Start all workers at the maximum
EP size, retain their processes and communication state, and select an active
prefix of ranks. Sleeping ranks discard weights and KV allocations. On return,
they restore the same virtual addresses and refill weights from a live replica.

The implementation uses eager Model Runner V1, BF16 weights, TP=PP=PCP=DCP=1,
one node, one Python API process, internal DP routing, static expert placement,
and AllGather MoE communication. Each sleeping rank must have a complete shard
replica on one active rank across all layers. Prefix caching is disabled.

## Launch

Build the native extension with CANN headers providing `HcclCreateSubCommConfig`.
The C++ shim uses the installed `HcclCommConfig` ABI. Set
`VLLM_USE_V2_MODEL_RUNNER=0` through the existing runner selector.

For Qwen3-30B-A3B-Base on four NPUs, create a placement JSON with 48 entries
in `layer_list`. Each entry has a `device_list` of four objects whose
`device_expert` arrays are respectively 0–63, 64–127, 0–63, and 64–127.
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
processes, parent communicators and child communicators remain resident until
server shutdown. Released weight/KV storage is therefore distinct from releasing
all per-process NPU resources.

## Validation

- CPU tests exercise placement validation, admission, streaming drain, routing,
  authentication, timeouts, cancellation and worker-state receipts.
- `test_fasteep_hccl.py` exercises actual four-NPU retained communicator handles,
  collective values, excluded ranks and peer copies.
- `test_fasteep_memory.py` exercises pooled weight/KV discard, peer refill,
  rank-local metadata restoration and address stability.
- `test_fasteep_serving.py` requires a compatible current-main build and the
  Qwen3 checkpoint; it checks repeated 4↔2 transitions, deterministic outputs,
  streaming drain, admission during resize, and output after final restoration.

Run the targeted NPU tests on four idle devices. Component results and full
serving qualification must be reported separately for the exact tested source.
