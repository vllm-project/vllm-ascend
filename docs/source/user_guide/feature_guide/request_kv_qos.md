# Request-level KV transfer priority

`--ai-qos` configures KV transfer priority independently of inference scheduling
priority. The initial backend is Mooncake; additional backend support can use
the same request-level interface.

## Configuration

Add this option to an existing KV transfer deployment:

```bash
--ai-qos '{"kv_transfer":{"enabled":true,"request_priority":true,"default_priority":"low","log":false}}'
```

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` within an explicit `kv_transfer` section | Enable KV priority lanes. Omitting `--ai-qos` keeps existing behavior. |
| `request_priority` | `false` | Honor each request's `kv_priority`; otherwise use the configured default for every request. |
| `default_priority` | `low` | Priority for requests without a priority, or all requests in fixed-priority mode. |
| `log` | `false` | Emit request-to-priority mapping logs; disabling logs does not disable QoS. |

Levels `low`, `medium`, and `high` map to channel QoS values `0`, `3`, and `7`.
Labels are case-insensitive; canonical numeric values `0`, `3`, and `7` are
also accepted. Unknown configuration fields, duplicate CLI options, and
conflicting legacy `kv_qos` mappings are rejected.

For request-priority mode, add the following field to a Completion or Chat
Completion request:

```json
{"kv_transfer_params": {"kv_priority": "high"}}
```

The existing PD example proxies preserve this metadata in the P-to-D handoff.
Invalid priorities are rejected before scheduling. In fixed-priority mode,
the default takes precedence over a supplied priority.

## Supported transfer paths

| Connector | Data path |
| --- | --- |
| `AscendStoreConnector`, `backend=mooncake`, `use_layerwise=false` | Store GET/PUT |
| `MooncakeConnectorV1` | Decode-side PD READ |
| `MooncakeLayerwiseConnector` | Prefill-side layerwise PD WRITE |

Both peers must use compatible priority-lane implementations. Mooncake must
provide `TransferEngine.initialize_with_ascend_resource_config`,
`mooncake.qos_lane`, and `mooncake.qos_pd_lane`, built with
`USE_ASCEND_DIRECT=ON` and a matching CANN/HIXL SDK. Each lane owns an engine
with immutable resource configuration; requests select a lane without changing
process-wide environment variables. Store lanes share the original key space.

The current Store path requires one KV cache group and equal Store TP sizes;
SSD offload and fabric-memory mode are rejected. Connector topology restrictions
still apply, including no decode-side PCP for ordinary PD. This option controls
KV transfers; collective communication and operator submission sections are
reserved and rejected until their implementations are available.

Do not combine request-level QoS with explicit connector `qos_priority`.
That existing setting remains available when request-level QoS is disabled.
Channel tagging requires compatible device/network QoS configuration to affect
traffic arbitration; it does not change the inference scheduler's request order.

## Regression checks

The focused tests can run from the repository root with the supported vLLM
installation available:

```bash
python -m pytest -q --timeout=60 \
  tests/ut/distributed/kv_transfer/test_kv_qos_*.py \
  tests/ut/distributed/kv_transfer/kv_p2p/test_qos_pd.py \
  tests/ut/distributed/kv_transfer/kv_pool/ascend_store/test_qos.py \
  tests/ut/kv_offload/test_kv_qos_adaptation.py
```
