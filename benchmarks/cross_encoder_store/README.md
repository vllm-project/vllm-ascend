# Cross-Encoder Store benchmark runner

This runner compares the same 2E2PD deployment with the shared Encoder-output
Store disabled (`p2p`) and enabled (`store`). It does not use the one-byte
forced-fallback configuration.

The benchmark dataset is the prepared MuirBench subset described by its own
`metadata.json`: 512 requests and 2,275 non-repeating images. Each request is
kept intact on one Encoder. Population routes are balanced at 256 requests per
Encoder (1,138/1,137 images), and measurement swaps every request to the other
Encoder.

## Phases

Each case starts fresh 2E2PD services and runs:

1. 16 disjoint warmup requests at concurrency 16 across 8 client processes;
2. 512 population requests at concurrency 1;
3. for `store`, wait until all 2,275 successful PUT markers are present;
4. 512 owner-swapped requests at closed-loop concurrency 96 across 8 client
   processes (12 concurrent requests per process);
5. for `store`, require exactly 2,275 loaded Store outputs.

`population.json` records the Store cold-miss + asynchronous-PUT cost. Its
default concurrency is deliberately 1 so every object can be published before
the warm pass; treat it as a cold-path latency reference, not as a throughput
comparison with the concurrency-96 measurement pass. The primary A/B result is
`p2p/measurement.json` versus `store/measurement.json`.

The serving defaults follow the final 2E2PD Qwen3.6 benchmark configuration:
`max_model_len=131072`, `max_num_seqs=96`,
`max_num_batched_tokens=98304`, chunked prefill enabled, graph mode enabled,
`direct_rpc` multimodal IPC, and a 32-image request limit. Encoder and PD
`gpu_memory_utilization` default to 0.90 and 0.93 respectively. The client uses
streaming, `temperature=0`, `max_tokens=32`, and disables thinking. It does
not request token IDs because large multimodal prompt-token arrays can exceed
the HTTP client's per-line SSE limit and are not used by this benchmark.

Local Encoder-output-cache hits are not part of this runner. That cache is
capacity bounded, so replaying all 512 requests would not guarantee a local-hit
path. A valid optional local-cache test should use a small working set that fits,
prime it once on each Encoder, and immediately replay it while verifying that
neither encoding nor Store GET occurred.

The runner uses four lightweight proxy processes, one for each E/PD route.
They do not use NPUs. Giving each proxy exactly one Encoder and one PD prevents
the general EPD proxy's random/round-robin routing from overriding the manifest.

## Required inputs

Extract the uploaded dataset archive and point `DATA_DIR` at its `prepared`
directory. Point `VLLM_ROOT` at the matching vLLM checkout that contains the
EPD proxy with `--ec-consumer-zmq-addrs` support.

```bash
export DATA_DIR=/path/to/muirbench_benchmark_512/prepared
export VLLM_ROOT=/path/to/vllm
export MODEL=/path/to/Eco-Tech/Qwen3.6-35B-A3B-w8a8
export RESULT_ROOT=/path/to/results
```

The default logical NPU assignment is:

```text
E0=NPU 0, E1=NPU 1, PD0=NPU 2, PD1=NPU 3
```

Override it with `NPU_E0`, `NPU_E1`, `NPU_PD0`, and `NPU_PD1`.

## Run

One Store-disabled case:

```bash
CASE=p2p ROUND=1 bash benchmarks/cross_encoder_store/run_2e2pd_case.sh
```

One Store-enabled case:

```bash
CASE=store ROUND=1 bash benchmarks/cross_encoder_store/run_2e2pd_case.sh
```

One Store-enabled case with an independently launched Store owner:

```bash
CASE=store STORE_MODE=standalone-store ROUND=1 \
  bash benchmarks/cross_encoder_store/run_2e2pd_case.sh
```

Three alternating rounds:

```bash
bash benchmarks/cross_encoder_store/run_matrix.sh
```

The `store` case defaults to an embedded 16 GiB nominal pool: each Encoder
process contributes an 8 GiB `global_segment_size`. Override
`STORE_GLOBAL_SEGMENT_SIZE` when the host-memory budget requires a different
value. `local_buffer_size` defaults to 512 MiB per Encoder and is not part of
the Store data capacity.

With `STORE_MODE=standalone-store`, the runner starts one additional
`mooncake_client` process. It sees the same four NPUs, contributes one 16 GiB
pool by default, and the Encoder requesters contribute zero bytes. This is a
fifth process, not a fifth NPU. Override its capacity with
`STORE_STANDALONE_GLOBAL_SEGMENT_SIZE`; its HIXL listener defaults to port
16667 so it does not reuse the embedded-client default port 16666. The runner
records its output in `mooncake_client.log`.

On the single-node A3 benchmark host, the runner defaults
`ASCEND_ENABLE_USE_FABRIC_MEM=1` for every Encoder and PD process. FabricMem
makes the Store's Host allocation directly addressable by the peer Encoder;
it is separate from the 3 GiB EC producer/consumer buffers and the 512 MiB
Store local buffer. The deprecated `ASCEND_BUFFER_POOL` setting is explicitly
removed from the child environment. The same environment is used for both
`p2p` and `store`, and the selected transport mode is printed as
`[CONFIG][ASCEND]` at the start of each run. Disabling FabricMem for `store`
is rejected unless `HCCL_INTRA_ROCE_ENABLE=1` explicitly selects the RoCE
path instead.

## Outputs

Each fresh run directory contains service/proxy logs and:

- `warmup.json`;
- `population.json`;
- `measurement.json`;

Every JSON report contains per-request route, status, TTFT, end-to-end latency,
TPOT, output tokens and any error, plus phase-level p50/p90/p95 and throughput.
The client performs closed-loop dispatch; it does not generate an open-loop
arrival trace and it does not retry measured requests.

The `store` run fails if PUT/GET counts do not equal the dataset's 2,275 unique
images or if a Store failure marker appears. Server readiness polling is only
startup synchronization; there is no separate environment-precheck or capacity
calibration phase.

## Long-lived dual-engine staging probe

Before changing the production ownership model, the narrow probe below tests
whether two independent TransferEngines can keep the exact same complete,
2 MiB-aligned NPU staging slab registered for many waves. In every wave, both
engines concurrently read the same immutable 4 MiB source region: the EC engine
writes to the peer NPU slab and the second engine writes to the peer pinned-Host
range. The target verifies both destinations before the source is reused.

Run both cleanup orders in separate output directories. Increase `--timeout`
with the iteration count because the timeout covers the complete case:

```bash
python3 benchmarks/cross_encoder_store/probe_non_fabric_host_memory.py \
  --host <local_service_ip> --npu-a 0 --npu-b 1 --link-mode roce \
  --engine-topology two --cases dual-engine-long-lived-staging \
  --iterations 100 --timeout 600 --cleanup-order store-first \
  --output-dir /tmp/dual-engine-staging-store-first

python3 benchmarks/cross_encoder_store/probe_non_fabric_host_memory.py \
  --host <local_service_ip> --npu-a 0 --npu-b 1 --link-mode roce \
  --engine-topology two --cases dual-engine-long-lived-staging \
  --iterations 100 --timeout 600 --cleanup-order ec-first \
  --output-dir /tmp/dual-engine-staging-ec-first
```

This case deliberately exercises raw engine I/O rather than Store object
metadata. `summary.json` reports transfer verification and cleanup separately;
a clean transfer result with a failed unregister is evidence only for
process-lifetime registration, not proof that per-wave dual registration is
safe. The optional `--store-buffer-pool` argument is probe-only and must not be
treated as enabling that configuration in production.

If the raw case transfers correctly, run the real Store PUT/GET probe with the
same two-engine, process-lifetime registration topology:

```bash
python3 benchmarks/cross_encoder_store/probe_shared_staging_store.py \
  --host <local_service_ip> --npu-a 0 --npu-b 1 --link-mode roce \
  --engine-topology two --iterations 7 --timeout 600 \
  --cleanup-order store-first \
  --output-dir /tmp/dual-engine-real-store-store-first

python3 benchmarks/cross_encoder_store/probe_shared_staging_store.py \
  --host <local_service_ip> --npu-a 0 --npu-b 1 --link-mode roce \
  --engine-topology two --iterations 7 --timeout 600 \
  --cleanup-order ec-first \
  --output-dir /tmp/dual-engine-real-store-ec-first
```

This second probe binds MooncakeDistributedStore to the second engine, performs
concurrent EC writes and real Store PUTs from the shared slab, overwrites the
source after completion, then verifies Store existence, placement, batched GET,
and returned bytes. Do not add `--store-buffer-pool` for the first pass.
