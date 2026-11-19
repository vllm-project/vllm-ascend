# Install and configure Mooncake cache namespaces on Atlas A2

`cache_prefix` is a deployment setting for the existing `AscendStoreConnector`
with `backend="mooncake"`. It applies to bulk and layerwise KV pooling.

## Install matching components

Follow the [source installation guide](../../getting_started/installation/install_vllm_ascend.inc.md)
for your A2 host architecture and driver/CANN environment. Use the vLLM revision
specified by `.github/vllm-main-verified.commit` when developing against main;
do not pair the checkout with an arbitrary vLLM version from an existing image.

Use the Python environment that will run the service. A virtual environment is
optional: it isolates Python packages, but not the NPU, driver or CANN. An
ordinary virtual environment does not inherit system-installed Mooncake or
torch packages.

After preparing the matching dependencies, rebuild an updated vLLM-Ascend
checkout in that environment:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

Let the build detect the actual A2 `ascend910b*` chip. Do not reuse A3 build
outputs or disable custom-kernel compilation for NPU serving. Install the
A2 variant of optional operator packages when required by the model, following
the source installation guide.

Install a compatible Mooncake Ascend SDK and master. Bulk pooling requires
at least `mooncake-transfer-engine-npu==0.3.11.post1`; layerwise requires at least
0.3.14 and the connector's session/range APIs. A cache namespace itself does
not require a new Mooncake storage protocol. Prepare `MOONCAKE_CONFIG_PATH`
using the [Mooncake backend configuration](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#configuration-examples).

For the A2 RoCE path, configure reachable communication addresses/interfaces
according to that guide. Do not enable the A3 Fabric Memory setting
`ASCEND_ENABLE_USE_FABRIC_MEM` on A2. Keep `PYTHONHASHSEED` consistent across
instances sharing KV entries.

## Configure a deployment namespace

Add this transfer configuration to the normal model-serving command:

```json
{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "backend": "mooncake",
        "lookup_rpc_port": "0",
        "cache_prefix": "model-release-v1"
    }
}
```

Both instances may use `kv_both`; both can query, load and write the pool.
Give co-located instances distinct `lookup_rpc_port` values, such as `"0"`
and `"1"`. They must use the same Mooncake master/tenant, namespace and
compatible model weights, cache layouts and parallel topology to share KV.
For layerwise pooling, add `"use_layerwise": true` to the same extra config
on each participating instance and use the compatible SDK APIs noted above.

For automatic namespaces, set `"cache_prefix": "auto"`. Supply an immutable
`VLLM_IMAGE_TAG`, or a known `VLLM_BUILD_COMMIT` when the image identity is
unavailable. These are existing upstream vLLM settings. Automatic mode combines
that identity with model/cache configuration hashes and the resolved HF commit.
It does not hash live weight files or independently identify Ascend source
changes; choose a new explicit prefix when those change without an identity
change. See [namespace semantics](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md#deployment-cache-namespaces)
for defaults, upstream value conversion and automatic-mode limitations.

For a rolling upgrade, keep current instances on `model-release-v1` and start
the new release on `model-release-v2`. Warm the new release before shifting
traffic. A namespace change does not require a full reset; account for both
releases occupying the shared pool during the overlap.

**External full reset is not restricted by `cache_prefix`.** It can remove all
prefixes in the client's effective Mooncake tenant namespace. Coordinate all
writers in that domain before cleanup. Layerwise external reset remains
unsupported. Nonempty prefixes on Memcache or Yuanrong are rejected; their
omitted/empty-prefix behavior is unchanged.
