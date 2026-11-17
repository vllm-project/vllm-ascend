# HiSparse on Ascend

## Feature Introduction

HiSparse keeps a Host mirror of sparse MLA KV rows and refills only the missing
rows selected by attention. This reduces the NPU cache capacity needed for long
contexts. The experimental Ascend backend requires a SIMT-capable CANN 9.2
toolchain on Ascend 950.

### Working Principle

HiSparse keeps recent KV rows in the NPU Resident cache and uses the NPU Hot
cache for rows refilled from Host memory. The resolver checks the selected rows
against both NPU caches. Resident and Hot
hits can be read directly. For misses, HiSparse refills only the missing rows
from the Host KV mirror into Hot before attention runs. With speculative decoding,
the resolver considers the selected rows for all query tokens of a request
together before evicting Hot rows.

### Usage Scenarios

Use HiSparse with Model Runner V2 for sparse RoPE MLA/SFA models serving long
contexts. BF16 KV and native Ascend 950 C8 (E4M3FN values with FP32 scales) are
supported, with independent Main and Indexer KV dtype selection.

### Constraints and Limitations

| Category | Constraint |
| --- | --- |
| Hardware and software | Ascend 950 with a SIMT-capable CANN 9.2 toolchain. |
| Cache formats | C4, TurboQuant and INT8-format C8 are outside the current support scope. |
| Parallelism and transport | Context parallelism is unsupported. For P/D transport, use `MooncakeConnectorV2` with `mooncake_transport=memfabric` on P and D and matching TP sizes. |
| Host storage | Eligible local TP configurations share one Offload Host pool; others keep private pools. Keep eager Host mirroring enabled. |
| P/D Host import | MemFabric owns D's Host Source allocation; D reads main KV directly into it. Indexer is read into D NPU. |
| TopK | The existing SFA requirement of divisibility by 128 applies. |
| Feature mutual exclusion | HiSparse and `additional_config.sparse_kv_offload_config.enabled` cannot be enabled together. |

## Feature Usage

### Usage Example

Set `MODEL_PATH` to a compatible sparse MLA model and start the server with
Model Runner V2:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve "$MODEL_PATH" \
  --block-size 128 \
--kv-cache-dtype bfloat16 \
--attention-config '{"indexer_kv_dtype":"bf16","hisparse_config":{"eager_host_mirror":true}}' \
--kv-transfer-config '{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":4}}'
```

For native A5 C8, replace the two dtype options with:

```bash
--kv-cache-dtype fp8 \
--attention-config '{"indexer_kv_dtype":"fp8","hisparse_config":{"eager_host_mirror":true}}'
```

### P/D with Mooncake TE and MemFabric Transport

Run a Prefill instance without HiSparse and a Decode instance with HiSparse.
Use the Pull P/D proxy: Prefill publishes its completed KV and Decode reads it.

Host allocation uses vendor-provided `memfabric_hybrid`
`1.5.0.dev20261115` with a matching A5 Offload extension. Obtain the SDK through
its vendor distribution and build its Offload kernel for the target SoC before
starting the workers (`mfcli kernel install --soc-version A5`). If the wheel
omits the HYBM kernel installer, use a complete SDK build produced in a valid
CANN environment; rebuilding packaging resources does not require SDK logic
changes. The SDK source pairing is `4c0b301537bed6606954084ba2376d99bf7bc543`.
Use the unmodified verified vLLM Core commit
[`ba15685ab7f4f5fbed8630bbb7441f8837e76fbe`](https://github.com/vllm-project/vllm/commit/ba15685ab7f4f5fbed8630bbb7441f8837e76fbe).
Each Decode TP rank uses its own Offload Host pool and the existing Core
HiSparseConnector worker. No shared-pool IPC, custom HiSparse connector module,
companion Core source change or additional runtime patch is required.
This is a fixed experimental pairing, not a claim of compatibility with arbitrary
vLLM or SDK releases.

Install a Mooncake build with TENT and MemFabric support, such as the
`mooncake-transfer-engine-memfabric` package. Its native MemFabric Transport
must match the SDK's `smem_trans_batch_status_query` ABI (the current upstream
adapter requires the four-argument API with an execution-result output).
The previously tested direct SDK TransferEngine path does not validate this
Mooncake/SDK pairing. See the [Mooncake MemFabric Transport guide](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/transport/memfabric_transport.html).

Configure both P and D worker environments:

```bash
export MC_USE_TENT=1
export MF_PROTOCOL=device_urma
cat > memfabric-tent.json <<'JSON'
{"transports":{"memfabric":{"enable":true}}}
JSON
export MC_TENT_CONF="$PWD/memfabric-tent.json"
MF_LIB_DIR=$(python -c 'import pathlib,memfabric_hybrid; print(pathlib.Path(memfabric_hybrid.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$MF_LIB_DIR:$LD_LIBRARY_PATH"
```

`mooncake_transport` defaults to `ascend`, preserving existing Mooncake users.
Selecting `memfabric` requires `MC_USE_TENT=1` and `MF_PROTOCOL=device_urma`;
invalid settings fail before engine construction. Reads explicitly pin the
`memfabric` transport, with no SDMA fallback. The consumer's HiSparse transfer
groups select Host Source versus NPU Indexer destinations automatically.

Prefill `--kv-transfer-config`:

```json
{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_connector_extra_config":{"mooncake_transport":"memfabric"}}
```

Decode `--kv-transfer-config`:

```json
{"kv_connector":"MultiConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"connectors":[{"kv_connector":"HiSparseConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"host_pool_gib":4}},{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_consumer","kv_connector_extra_config":{"mooncake_transport":"memfabric"}}]}}
```

Set distinct `kv_port` values for P and D. Use the Pull proxy described in
[disaggregated prefill](../../developer_guide/Design_Documents/disaggregated_prefill.md).
P finishes Prefill and retains its registered NPU KV until D finishes reading.
D allocates Host Source through MemFabric Offload and calls
Mooncake `TransferEngine.batch_transfer_sync_read()` with local Host destinations for
main KV and local NPU destinations for Indexer. Both P source regions and D Host/Indexer destination regions are registered
with TransferEngine for `device_urma`. This differs from the SFA SDMA path,
where the D-side local destinations are not explicitly registered. MLA NoPE/RoPE components are read into
token-interleaved Host rows. There is no separate Host staging pool.

Every D TP rank allocates an independent MemFabric Offload `Scene.LOCAL` Host
Source, imports its own main KV and Indexer, and writes its own Host mirror.
Resident and Hot remain on each rank's NPU. Core's private-pool events order
writes and reads locally; there are no cross-rank Host events or shared-memory
metadata. The existing Core connector and worker are used directly.

This deliberately trades replicated Host capacity and main-KV traffic for a
smaller integration: TP2 holds approximately two copies of MLA Host KV, TP4 four.
`host_pool_gib` is per rank. LOCAL `GIANT_PAGE` allocations round up to GiB
boundaries, including alignment padding; provision HugePages for every D rank.
Register each complete physical allocation once. Host and device addresses may
differ; HiSparse uses the SDK's `get_dva()` mapping plus view offsets for AIV.
Build the SDK's A5 Offload kernel before starting workers. No custom SDK source
patch is used.

This is an alternative to shared-Host integration. Previous shared-pool model
results do not qualify this per-rank candidate. Real A5 Host/AIV/PD transfer,
eager output, graph replay, long contexts and performance remain pending.
Engine recreation in one worker is unsupported; drain transfers before teardown.
BM groups and `L2GH` are not used. The connector requires matching
TP sizes and PCP=DCP=1. After all requested groups finish, Core HiSparse imports
the prefix and restores the last page to Resident. This first implementation uses
bulk Pull; layerwise overlap remains a separate adaptation.

### Verifying the Feature

Send a request and check the server's metrics endpoint for the HiSparse miss
counter. The counter can stay at zero when selected rows are already on the NPU.

```bash
curl -fsS http://127.0.0.1:8000/metrics | grep '^vllm:hisparse_cache_misses_total'
```

## Configuration Parameters

| Parameter | Meaning |
| --- | --- |
| `hisparse_config` | Enables Core HiSparse for the supported sparse Attention backend. |
| `eager_host_mirror` | Keep `true` for this foundation; Core maintains the Host KV mirror. |
| `host_pool_gib` | Logical Host pool capacity in GiB per Decode TP rank. Allow for page padding and SDK allocation rounding on every rank. |
| `device_buffer_size` | Per-request Hot capacity in KV rows. It must hold the deduplicated union of rows selected during speculative verification and fit the resolver's on-chip UB budget. |
| `kv_cache_dtype` / `indexer_kv_dtype` | Select main and Indexer KV formats independently from model-weight quantization. |
