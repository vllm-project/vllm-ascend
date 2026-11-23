# Qwen3.8 Next cache pooling

## Layout and supported paths

On the unified runner, QSA K/V, indexer, GDN, PLE and hidden-state owners
use separate contiguous banks inside one backing allocation. Group banks
must not overlap: different logical block IDs do not isolate physical bytes
when groups use different block strides. Pool bytes per block are the sum
of all owner page sizes; this reduces capacity compared with unsafe aliasing.
The legacy runner retains its existing slab layout. Views may have a physical block stride
larger than their payload. Transfers must preserve that stride and leave other
owners, unused blocks and padding unchanged.

Use `AscendStoreConnector`. The adaptation supports bulk
transfers and the existing Mooncake block-key/Memcache GVA layerwise paths.
Other connectors retain their existing allocation path. Layerwise QSA opens
its transfer window after both indexer and main-KV updates; PLE waits before
reading its convolution state. Owners sharing a physical layer wait for one
load, then copy their own state rows. GDN has two rows per layer; PLE has one.

Private raw-indexer state is reconstructed by replay. Lookup must first select
a restorable common boundary. With transfer granularity 1536, a 9216-token
prompt queries at 7680, while 9217 can query at 9216. Layouts whose necessary
state is all externally cacheable retain the existing lookup behavior.

MTP defers save finalization until after draft generation.
`MambaCopyKVConnector.finalize_step()` forwards to the wrapped connector so
that deferred saves complete and metadata is cleared.

## Configuration

Install matching vLLM/vLLM-Ascend builds, native extensions and backend runtime.
Configure the existing connector:

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "backend": "memcache",
    "use_layerwise": false,
    "load_async": false
  }
}
```

Bulk MTP validation used FULL_DECODE_ONLY; layerwise requires PIECEWISE.
`load_async=true` uses the existing receive-completion and per-layer wait
paths. Asynchronous graph-mode restoration and arbitrary layer reuse/offload
settings require separate validation. The adaptation does not add recovery
for incomplete external reads.

For Mooncake, configure `backend="mooncake"`, matching master/client versions
and the deployment's metadata, master and transport settings. One tested
Ascend setup required explicit device RoCE:

```bash
unset ASCEND_ENABLE_USE_FABRIC_MEM
export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["roce:device"],"store.comm_resource_config.protocol_desc":["roce:device"]}'
export MOONCAKE_CONFIG_PATH=/path/to/mooncake.json
```

That setup used matching 0.3.14rc1 master/client; it does not establish that
HCCS fails on other machines. Size layerwise read leases for the complete
transfer/compute session; a lease timeout is not a cache-layout failure.

## Regression tests

Run the focused CPU suite:

```bash
python -m pytest -q \
  tests/ut/distributed/ascend_store/test_qsa_basic_pooling.py \
  tests/ut/distributed/ascend_store/test_ascend_store_connector.py \
  tests/ut/distributed/ascend_store/test_mooncake_hybrid.py \
  tests/ut/models/qwen4_exp/test_qsa_layerwise_hooks.py \
  tests/ut/models/qwen4_exp/test_ple_internal_format.py \
  tests/ut/worker/test_mamba_kv_connector.py \
  tests/ut/worker/test_model_runner_v2_mamba.py
```

Coverage includes actual packed shared-page views, both backend adapters,
dense/padded strides, partial/full fragments, reordered block IDs, overwritten
source/destination bytes and whole-backing sentinels. Scheduler tests cover
restoration boundaries; model and wrapper tests cover load/update/save order,
shared-layer deduplication, per-owner state rows and deferred MTP finalization.

CPU mocks do not establish native cross-process transfer correctness or
model-level restoration. Those checks require deployment-specific hardware
validation; this PR does not add a standalone NPU E2E test.

## Validation boundary

Earlier model results used a previous dependency/layout stack. They do not
validate the current packed-page layout. Report hardware and GPQA results
against the exact tested source revision and runtime in the PR. Distinguish
external restoration from recomputation using zero local hits, actual backend
reads and matching output. For SSD, require SSD hits and rewarm bytes after
eviction; configuration or SSD writes alone do not prove a readback.

## Unified model dispatch

`allocate_kv_cache_main` uses an explicit `model_type` allowlist for
`qwen4_exp`, `qwen4_exp_text` and `qwen4_exp_mtp`, matching the
implemented Qwen4Exp architecture used by Qwen3.8-Flash-Next.
Unverified Qwen3.8/Qwen4-Max aliases are not added to the allowlist.
It reads the text config first, falling back to the outer HF config only
when the text model type is absent. Unlisted suffixes do not match.
These types select `unified_allocate_kv_cache`;
the actual cache layout still comes from specs and backend runtime views.
Repository/display names are not used to infer architecture.

This is allocator integration, not a Qwen4-Max model implementation. A
Qwen4-Max checkpoint requires an upstream model/config registration, correct
cache specs and compatible Ascend runtime backends. Unknown attention
backends without runtime shape/dtype methods raise an explicit error.
Real Qwen4-Max inference has not been validated by this change.

Dependency-free dispatch and byte-isolation checks can be run with:

```bash
python -m unittest discover -s tests/ut/worker -p test_qwen_unified_allocation_contract.py
```
