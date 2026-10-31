# Layerwise pool PD validation

These templates configure the independent `LayerwisePoolConnector`. P writes
each completed layer to the pool; D restores the whole request before decode.
They are development examples, not a validated GLM deployment recipe. Full
GLM accuracy, graph replay and heterogeneous topology acceptance remain pending.
See the [design and compatibility boundaries](../../../docs/source/developer_guide/Design_Documents/layerwise_pool_connector.md).

## Prepare the existing pool

Install the compatible MemCache/MemFabric runtime and prepare the pool using
the [MemCache backend guide](../../../docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend).
P and D must reach the same MetaService and each other's configured pool data
endpoints. Pool capacity, device selection and transport configuration belong
to that deployment; these examples do not allocate them.

Set `MMC_LOCAL_CONFIG_PATH` to each engine's actual local configuration file.
The P control endpoint defaults to `get_ip()` and an automatically bound port.
Both its scheduler and the D schedulers must reach that address and port. When
automatic address selection is unsuitable, set `control_host` in P's
`kv_connector_extra_config` to the reachable local interface address. This is
a worker control endpoint, separate from the HTTP service and pool endpoints.

Choose pool capacity ownership separately from P/D inference roles. If D must
restart independently while P retains handoffs, validate D as a pool client
with no contributed storage. In D's MemCache local configuration:

```ini
ock.mmc.local_service.dram.size=0
ock.mmc.local_service.hbm.size=0
ock.mmc.local_service.storage.enabled=false
```

Keep the cluster's `max.dram.size`, `max.hbm.size`, membership and transport
settings consistent with its actual storage providers. The settings above do
not replace the complete MemCache configuration. Other members must provide
enough pool capacity. A three-client MemCache 1.3.0 `device_urma` test verified
correct reads and D shutdown while P retained its objects in this arrangement;
validate the selected SDK and transport before relying on it in deployment.

With nonzero pool capacity, D is also a storage provider and may hold P's
objects. Closing D can then wait for foreign leases even after D's requests
have drained. Drain those users before removing its capacity, or use a storage
service whose lifetime is independent of the inference workers. The connector
does not relocate retained objects or forcibly revoke another client's leases.

## Start P and D

Run from the repository root, in the corresponding prepared P/D environments.
Set `MODEL_PATH`, `SERVED_MODEL_NAME`, `TP_SIZE` and `MMC_LOCAL_CONFIG_PATH`
before running each command. Select devices with the usual deployment launcher.
P and D may use different TP sizes only where their actual cache layouts are
supported by the restore planner. Use the same model revision, dtype and
compatible cache format on both sides.

P:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONHASHSEED=0
vllm serve "${MODEL_PATH:?Set MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE:?Set TP_SIZE}" \
  --host 0.0.0.0 --port 8001 \
  --enforce-eager --shutdown-timeout 30 \
  --kv-transfer-config "$(cat examples/disaggregated_prefill_v1/layerwise_pool/prefill.json.example)"
```

D:

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export PYTHONHASHSEED=0
vllm serve "${MODEL_PATH:?Set MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE:?Set TP_SIZE}" \
  --host 0.0.0.0 --port 8002 \
  --enforce-eager --shutdown-timeout 30 \
  --kv-transfer-config "$(cat examples/disaggregated_prefill_v1/layerwise_pool/decode.json.example)"
```

These commands show connector wiring. Retain the model-specific options from
the working Mooncake V2 baseline, including MTP and Mamba cache mode. Configure
D's speculative decoding locally, using the same strategy as V2. No D MTP
requirements are sent through P or the proxy. RecoverSSM retains its existing
PP=PCP=DCP=1 and prefix-cache restrictions. Start validation in eager mode;
P requires PIECEWISE for graph execution so layer-write callbacks run each
step. D bulk restore happens outside forward and preserves the configured
graph mode, including FULL_DECODE_ONLY. The Ascend platform derives this
distinction from `kv_role`; no extra user setting is needed. Actual graph
capture/replay and MTP correctness still require NPU acceptance.

For DCP, retain the existing PD requirement that
`--cp-kv-cache-interleave-size` equals the effective KV block size. For example,
use `--cp-kv-cache-interleave-size 128` with 128-token blocks. Hybrid GLM cache
alignment can increase this block size, and changing TP or MTP can change it
again. Use the effective block size reported for the D configuration; do not
copy 128 from another model or parallel layout. See the
[context parallel guide](../../../docs/source/user_guide/feature_guide/context_parallel.md)
for the model's supported PCP/DCP combinations.
GLM-5.3's KPool indexer currently rejects DCP>1 during model construction,
including when RecoverSSM is disabled. This is an existing model restriction;
choosing the correct interleave or changing the connector does not remove it.

Stop the API process first and let vLLM stop its engine and workers. The example
reserves 30 seconds with `--shutdown-timeout`; choose a deadline that covers the
deployment's request drain and resource cleanup. A zero API exit code alone
does not establish graceful worker cleanup: inspect engine/worker logs for
forced termination and confirm pool operations and native resources have drained.

## Use the original CPD proxy

Set `P_HOST` and `D_HOST` to the HTTP service hosts, then run:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts "${P_HOST:?Set P_HOST}" --prefiller-ports 8001 \
  --decoder-hosts "${D_HOST:?Set D_HOST}" --decoder-ports 8002
```

Send normal requests to the proxy. Its existing `kv_transfer_params` forwarding
carries the connector's nested `pool_transfer` payload without a proxy change.
Do not manufacture this metadata in client requests: it describes a committed
P attempt and its source worker identities.

## Configuration and acceptance

| Extra option | Default | Meaning |
| --- | --- | --- |
| `backend` | `memcache` | Existing pool backend; requires GVA or key-range operations |
| `pool_namespace` | Model path basename | Source key namespace; use an explicit model/revision namespace |
| `control_host` | `get_ip()` | P worker's bind and advertised control address |
| `lease_ttl_ms` | `600000` | Native GVA lease duration; not an unlimited write timeout |
| `retention_seconds` | `300` | P manifest and pool retention; positive and shorter than the native lease |

The namespace is not a model compatibility check. In MemCache GVA shared mode,
ordinary layerwise pooling and pooled PD use the same format-qualified
`pool-full-v1` and `pool-component-v1` objects. Unqualified legacy keys are not
assumed compatible. Non-GVA private attempt objects remain separate. P can
release HBM after its own writes drain; D completion releases source pool
retention.

Shared prefix keys describe accepted tokens. D's MTP runs locally and does not
roll back P's prompt objects. Ordinary rotating-HBM offload can additionally
save unaccepted working data under `pool-step-v1` private IDs; those snapshots
are excluded from shared prefix lookup.

Compare deterministic outputs against the same model's V2 baseline before
changing topology. Cover block-boundary prompts, chunked prefill, local prefix
hits, KDA checkpoints, circular/tail state and each intended MTP configuration.
Then test heterogeneous TP/PP/CP only within the model's existing restrictions,
followed by cancellation, failed copies, retention expiry and PIECEWISE replay.
An incompatible Mamba component length or ring shape must fail before restore;
this connector does not add a second MTP conversion policy.

Preserve the first error and the exact source/image/topology for each case.
Successful CPU byte-copy tests or small SDK copies do not establish end-to-end
model correctness, native teardown correctness or production readiness.

### Compare running services

The reusable probe is `tests/e2e/common/kv_pool/pool_pd.py`. Prepare an isolated
V2 baseline and a candidate using the original CPD proxy. Keep model revision,
sampling, cache dtype, topology and MTP settings matched between the baseline
and candidate. Stop other test traffic for the comparison. The probe starts no
services and changes no server configuration.

Prepare a JSONL file of completion or chat requests, for example:

```json
{"id":"short","prompt":"Explain the role of a KV cache in one sentence.","max_tokens":32}
{"id":"followup","prompt":"Explain the role of a KV cache in three numbered sentences.","max_tokens":64}
```

For multimodal models, use `messages` with the model's normal OpenAI-compatible
image/video content parts instead of `prompt`. The probe selects
`/v1/chat/completions` and sends the same content to both endpoints. Include
cases with identical text and different media, as well as repeated media. The
same exact-token and actual-transfer checks apply; generated text alone cannot
establish a successful pool restore. Enable the tested modalities on both engines.

Add token-ID prompts at the target model's block and chunk boundaries, including
one token below and above each boundary. Each row may include normal completion
sampling fields; the probe fixes temperature=0, seed=0, n=1 and requests output
token IDs. It compares the exact generated IDs and finish/stop reasons.

```bash
python -m tests.e2e.common.kv_pool.pool_pd \
  --model "${SERVED_MODEL_NAME:?Set SERVED_MODEL_NAME}" \
  --baseline-url "${BASELINE_PROXY_URL:?Set BASELINE_PROXY_URL}" \
  --candidate-url "${POOL_PROXY_URL:?Set POOL_PROXY_URL}" \
  --prefill-metrics-url "${P_METRICS_URL:?Set the full P metrics URL}" \
  --decode-metrics-url "${D_METRICS_URL:?Set the full D metrics URL}" \
  --cases "${CASES_JSONL:?Set CASES_JSONL}" \
  --output "${RESULT_JSON:?Set a new result filename}"
```

Repeat each metrics URL option for every P/D API engine. The output parent
directory must exist; an existing result file is never overwritten. Preserve
the source SHA, image digest and effective launch configurations with the result.

The candidate must increase `vllm:pool_pd_save_bytes_total` and
`vllm:pool_pd_restore_bytes_total` with successful operation counters and no
failure/cancellation increments. Missing counters, counter resets, restore
fallback, or mismatching tokens fail the run. Counts aggregate worker operations,
not unique HTTP requests; zero-byte KVPP owners alone cannot satisfy the check.
These counters measure successful operations, not total network traffic.
Exclusive traffic and complete endpoint coverage are required because counters
are engine-wide. A passing probe establishes these cases only; graph replay,
fault injection, the full topology matrix and long-running performance remain
separate acceptance work.

### Four-node GLM-5.2 acceptance configuration

`tests/e2e/cases/features/kv_pool/GLM-5.2-W8A8C8-A3-LayerwisePool-PD.yaml`
is a pending hardware acceptance case derived from the existing GLM-5.2
MemCache PD case. It uses four A3 nodes with 16 devices each: two P replicas
with TP=1/PCP=16 and two D replicas with TP=16. Both roles use V2 Runner and
the standalone `LayerwisePoolConnector`. The original CPD proxy is unchanged.
P retains MTP=1 and D MTP=5; these settings follow the existing V2 test case.

The external-DP runner recognizes `test_content: [pool_pd_transfer]` and brackets
AISBench with metrics from every P/D API engine. It writes
`pool_pd_transfer.json` alongside rank logs. A run with no completed write and
restore, or any reported transfer failure/cancellation, fails this check.
Benchmark exceptions remain the first failure. Counters establish that some
real transfer occurred; they do not prove that every request avoided fallback.
Use the isolated per-case token comparison above for targeted correctness.

This YAML has not passed a full hardware run and is not registered in the
automatic weekly matrix yet. Its inherited accuracy targets are test criteria,
not measured results. It is not an A5 single-node launch configuration.
