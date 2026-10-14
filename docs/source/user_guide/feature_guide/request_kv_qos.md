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

Both peers must use compatible priority-lane implementations. The PD and Store
pools live in vLLM-Ascend and use upstream
`TransferEngine.initialize(host, "P2PHANDSHAKE", "ascend", device_name)`.
Mooncake must be built with TENT and `USE_ASCEND_DIRECT=ON`, using a matching
CANN/HIXL SDK.
No Mooncake Python pool modules or custom initialization binding are required.

During worker startup a shared lock serializes PD and Store engine construction
and initialization. Each lane receives its own `ASCEND_GLOBAL_RESOURCE_CONFIG`
and derived `MC_TENT_CONF` snapshot; original environment values are restored on
success and failure, and the original TENT file is never modified. Derived files
remain available for the worker lifetime. Initialize other native engines before
or after this scope, not concurrently in another thread. Requests only select
existing lanes; they do not mutate environment variables. Store lanes share the
original key space, and only the default lane contributes Store capacity.

The A5 regression used a locally built Mooncake candidate based on
`07e658f0db89576fd95f6c610df53c20a0bc25cd` with CANN 9.1.0. Results apply to
that tested build and environment; other package builds require their own
compatibility validation.

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

The migration passed 39 supported A5 test groups: native PD and Store pools,
Store cold restart, installed adaptation tests, disabled controls, fixed and
request priorities, sync/async Store loading, chunked prefill, async scheduling,
layerwise PD, and supported TP2/PP2/PCP2 configurations. Checks included actual
HTTP inference with dummy weights, transfer data and completion, channel QoS,
and generated-token consistency. These are functional checks, not pretrained
model accuracy or throughput measurements.

The PCP2 Store cases enabled `save_decode_cache=true` to exercise suffix-cache
saving and readback. Ordinary PD with PCP2 on both peers remains excluded by
the pinned upstream decode-side topology restriction. Cross-host, TP4/TP8 and
DP>1 configurations are outside this validation scope.

The focused tests can run from the repository root with the supported vLLM
installation available:

```bash
python -m pytest -q --timeout=60 \
  tests/ut/distributed/kv_transfer/test_kv_qos_*.py \
  tests/ut/distributed/kv_transfer/kv_p2p/test_qos_pd.py \
  tests/ut/distributed/kv_transfer/kv_pool/ascend_store/test_qos.py \
  tests/ut/kv_offload/test_kv_qos_adaptation.py
```
