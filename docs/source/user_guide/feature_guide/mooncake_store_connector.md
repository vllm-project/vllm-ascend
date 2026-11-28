# Upstream MooncakeStoreConnector

vLLM Ascend provides an optional compatibility patch for using upstream vLLM's
`MooncakeStoreConnector` with Ascend Fabric memory on supported HDK versions before
26.1. This connector stores KV caches in Mooncake Store. It is separate from
`AscendStoreConnector` and the direct P/D `MooncakeConnectorV1`.

## Prerequisites

- Use vLLM with the common Store changes in
  [vllm-project/vllm#66271](https://github.com/vllm-project/vllm/pull/66271).
  The patch requires the `MooncakeStoreWorker._setup_store` hook and extensible
  config loading from that change; it does not install without the hook.
- Install a compatible NPU Mooncake client and start a matching Mooncake master.
  See the [Mooncake backend prerequisites](kv_cache_pool_ascend_store.md#scenario-1-mooncake-backend)
  for the underlying transport setup. Use the upstream connector configuration
  below rather than the AscendStore connector options in that guide.
- HDK 26.1 or later can leave this compatibility patch disabled. The patch does
  not detect the HDK version automatically.

## Configuration

Set `MOONCAKE_CONFIG_PATH` to a Mooncake Store JSON file on each worker. For
example, an embedded Store client contributing 4 GiB uses:

```json
{
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "<master-host>:50051",
  "protocol": "ascend",
  "mode": "embedded",
  "global_segment_size": "4GB"
}
```

Enable Fabric memory and select the file before starting vLLM:

```bash
export ASCEND_ENABLE_USE_FABRIC_MEM=1
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

Pass the following JSON as `--kv-transfer-config`:

```json
{
  "kv_connector": "MooncakeStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "enable_ascend_fabric_patch": true
  }
}
```

`enable_ascend_fabric_patch` defaults to `false` and must be a JSON boolean.
It controls the entire compatibility patch, including transfer-thread device
binding. In a `MultiConnector`, set it on each `MooncakeStoreConnector` child's
`kv_connector_extra_config`; all Store children in one worker must use the same
value. Restart the workers when changing this option.

When enabled with Fabric memory, the patch defers Store setup until serving,
leaves KV-cache memory registration to Fabric, and returns a cache miss for
lookups before initialization. It also binds transfer threads to their worker's
NPU before handling a request. Fabric memory requires `protocol="ascend"`.
Without Fabric memory, Store setup remains immediate.

For other Store options, see the
[upstream MooncakeStoreConnector guide](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/).
Portable KV layout handling and event synchronization remain in upstream vLLM.
