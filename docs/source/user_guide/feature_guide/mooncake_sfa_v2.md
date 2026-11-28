# Mooncake V2 with SFA KV Offload

## Feature Introduction

This feature uses `MooncakeConnectorV2` to transfer sparse-attention KV from
Prefill (P) into the Decode (D) node's Host KV cache in a PD-disaggregated deployment.
It lets Mooncake PD deployments use SFA Decode offload: a bounded HBM buffer
stages transfers into the existing Host pool, reducing persistent Decode HBM use.

Use a vLLM-Ascend build containing this feature and its matching vLLM version;
see [Installation](../../getting_started/installation.md). Both nodes require
ModelRunner V2. The integration has been validated with vLLM v0.34.0.

### Constraints and Limitations

| Category | Requirement |
| --- | --- |
| Hardware | Atlas A3. **950 DT&950PR products are not supported**; enabling sparse KV offload on these products fails at startup. |
| Deployment | PD disaggregation; enable sparse KV offload only on D. Use a direct `MooncakeConnectorV2` on both nodes. |
| Model | Sparse-attention models with `index_topk`; compressed-cache models with `compress_ratios` are not supported. Validated model: GLM-5.2 W4A8. |
| KV format | BF16 main KV; packed main KV formats such as C8, TQ4 and HIF8 are not supported. Device-resident INT8 indexer KV is supported separately. |
| Parallelism | Decode supports TP and DP. Context parallelism and pipeline parallelism are not supported for sparse offload. |
| Transport | Pull mode only: keep `use_layerwise=false`. `MooncakePullConnector`, `MooncakeLayerwiseConnector` and wrapping this feature in `MultiConnector` are not supported. |
| Attention | Use LIM + Copy-SFA with `use_fused_overlap=false`. DSpark and `PreemptOffloadConnector` cannot be combined with this feature. |
| Transfer failure | D requires `kv_load_failure_policy="fail"`. Recovery by recomputing failed transfers locally is not supported. |

## Feature Usage

### Environment Preparation

- Prepare the Mooncake transfer dependencies and networking using the
  [Mooncake PD deployment guide](../../tutorials/features/pd_disaggregation_mooncake_single_node.md).
- Use matching model weights, tokenizer and compatible KV layouts on P and D.
  Keep the model, quantization and parallelism options from the model's deployment guide.
- Reserve the connector control ports and D's Host memory before starting services.

### Usage Example

Append the following options to the corresponding `vllm serve` commands.
Merge the JSON into any existing `--additional-config` instead of passing the
option twice.

**1. Configure Prefill.** Keep sparse KV offload disabled on P:

```bash
--kv-transfer-config '{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_producer",
  "kv_port": 27300,
  "kv_connector_extra_config": {"use_layerwise": false}
}'
```

**2. Configure Decode.** Enable sparse offload and set its Host memory budget:

```bash
--kv-transfer-config '{
  "kv_connector": "MooncakeConnectorV2",
  "kv_role": "kv_consumer",
  "kv_port": 27400,
  "kv_load_failure_policy": "fail",
  "kv_connector_extra_config": {"use_layerwise": false}
}' \
--additional-config '{
  "sparse_kv_offload_config": {
    "enabled": true,
    "topk_buffer_size": -1,
    "dram_size_per_dp_GB": 16,
    "use_fused_overlap": false
  },
  "mooncake_staging_config": {
    "window_chunks": 16,
    "bank_count": 4
  }
}'
```

`dram_size_per_dp_GB=16` is an example budget, not the default. Size it for the
available Host memory and workload. Leave `topk_buffer_size=-1` to calculate
the hot-buffer capacity from the speculative-token count.

**3. Configure request routing.** The proxy must follow this feature's prompt protocol:

- Send the same original prompt and tokenization to P and D.
- Discard P's first output token; do not append it to D's prompt.
- Return D's complete output. D replays the final prompt token before decoding.

### Verifying the Feature

Send a request through the PD proxy, then check D's log:

```bash
grep -F "Staged DSA PD mooncake receive timing:" /path/to/decode.log
```

After a Host-bound transfer completes, the log reports `mode`, `bank_workers`
and `banks`. The default staging configuration uses `mode=pipeline` and
`banks=4`; `bank_count=1` selects `mode=sync`. Confirm that the proxy returns
the generated response without a KV-load failure.

## Configuration Parameters

### KV Transfer Settings

Set these fields in `--kv-transfer-config`. The last two fields belong inside
`kv_connector_extra_config`.

| Parameter | Type | Default | Required | Values for this feature |
| --- | --- | --- | --- | --- |
| `kv_connector` | string | `null` | Yes, P and D | `"MooncakeConnectorV2"`. |
| `kv_role` | string | `null` | Yes, P and D | `"kv_producer"` on P; `"kv_consumer"` on D. |
| `kv_port` | integer | `14579` | No | Connector control-port base (`1` to `65535`). Reserve available TCP ports for the instance and its TP workers; use distinct bases for instances sharing a network namespace. |
| `kv_load_failure_policy` | string | `"fail"` | No | Keep `"fail"` on D; `"recompute"` is rejected. |
| `use_layerwise` | boolean | `false` | No | Keep `false` on both nodes to select Pull. |
| `pull_transfer_mode` | string | `"request"` | No | `"request"` reports completion per request; `"batch"` uses a batch barrier. |

### Sparse KV Offload Settings

Set these fields under `sparse_kv_offload_config` in `--additional-config` on D.

| Parameter | Type | Default | Required | Range / behavior |
| --- | --- | --- | --- | --- |
| `enabled` | boolean | `false` | Yes, D | Set `true` on D; leave disabled on P. |
| `topk_buffer_size` | integer | `-1` | No | Hot-buffer token capacity. `-1` resolves to `(1 + num_speculative_tokens) * 2048`; otherwise use a positive value. The resolved value must be at least the model's `index_topk`. |
| `dram_size_per_dp_GB` | integer | `110` | No | Positive Host KV memory budget in GiB per DP group. This is separate from the HBM staging window. |
| `use_fused_overlap` | boolean | `false` | No | Keep `false` to use LIM + Copy-SFA; `true` is rejected for this integration. |
| `keep_device_kv_cache` | boolean | `false` | No | Keep `false` for normal offload. `true` retains device KV for debugging and does not provide the HBM capacity benefit. |
| `store_port_base` | integer | `8500` | No | Host-store TCP base port, from `1` to `65535`. DP group `i` binds `store_port_base + i`; reserve that range and avoid conflicts with other instances. |

### Staging Settings

Set these fields under `mooncake_staging_config` in `--additional-config` on D.
They accept JSON integers, not strings, booleans or fractional numbers.

| Parameter | Type | Default | Required | Range / behavior |
| --- | --- | --- | --- | --- |
| `window_chunks` | integer | `16` | No | Total reusable HBM staging chunks. Values below `1` become `1`. Each chunk is sized for the KV layout and aligned to 2 MiB. |
| `bank_count` | integer | `4` | No | Values below `1` become `1`; the effective count is capped by `window_chunks`. One bank uses serial READ/landing; multiple banks pipeline the work. |

The default window has four banks of four chunks. Increasing `bank_count`
does not increase total staging capacity. Each bank receives
`floor(window_chunks / effective_bank_count)` chunks; any remainder is unused.

For older experimental configurations, remove `dsa_sparse_attention_config`;
nonempty values are rejected. Configure staging through the two fields above
instead of the retired staging environment variables.
