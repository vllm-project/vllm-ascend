# KV Cache Pool

## Why KV Cache Pool?

Prefix caching is an important feature in LLM inference that can reduce prefill computation time drastically.

However, the performance gain from prefix caching is highly dependent on the cache hit rate, while the cache hit rate can be limited if one only uses HBM for KV cache storage.

Hence, KV Cache Pool is proposed to utilize various types of storage including HBM, DRAM, and SSD, making a pool for KV Cache storage while making the prefix of requests visible across all nodes, increasing the cache hit rate for all requests.

vLLM Ascend currently supports [MooncakeStore](https://github.com/kvcache-ai/Mooncake), one of the most recognized KV Cache storage engines.

While one can utilize Mooncake Store in vLLM V1 engine by setting it as a remote backend of LMCache with GPU (see [Tutorial](https://github.com/LMCache/LMCache/blob/dev/examples/kv_cache_reuse/remote_backends/mooncakestore/README.md)), we find it would be better to integrate a connector that directly supports Mooncake Store and can utilize the data transfer strategy that best fits Huawei NPU hardware.

Hence, we propose to integrate Mooncake Store with a brand new **MooncakeStoreConnectorV1**, which is indeed largely inspired by **LMCacheConnectorV1** (see the `How is MooncakeStoreConnectorV1 Implemented?` section).

## Usage

vLLM Ascend currently supports Mooncake Store for KV Cache Pool. To enable Mooncake Store, one needs to configure `kv-transfer-config` and choose `MooncakeStoreConnector` as the KV Connector.

For step-by-step deployment and configuration, please refer to the [KV Pool User Guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/kv_pool.html).

## How it works?

The KV Cache Pool integrates multiple memory tiers (HBM, DRAM, SSD, etc.) through a connector-based architecture.

Each connector implements a unified interface for storing, retrieving, and transferring KV blocks between tiers, depending on access frequency and hardware bandwidth.

When combined with vLLM’s Prefix Caching mechanism, the pool enables efficient caching both locally (in HBM) and globally (via Mooncake), ensuring that frequently used prefixes remain hot while less frequently accessed KV data can spill over to lower-cost memory.

### 1. Combining KV Cache Pool with HBM Prefix Caching

Prefix Caching with HBM is already supported by the vLLM V1 Engine.
By introducing KV Connector V1, users can seamlessly combine HBM-based Prefix Caching with Mooncake-backed KV Pool.

 The user can enable both features simply by enabling Prefix Caching, which is enabled by default in vLLM V1 unless the `--no_enable_prefix_caching` flag is set, and setting up the KV Connector for KV Pool (e.g., the MooncakeStoreConnector).

**Workflow**:

1. The engine first checks for prefix hits in the HBM cache.

2. After getting the number of hit tokens on HBM, it queries the KV Pool via the connector. If there are additional hits in the KV Pool, we get the **additional blocks only** from the KV Pool, and get the rest of the blocks directly from HBM to minimize the data transfer latency.

3. After the KV Caches in the KV Pool are loaded into HBM, the remaining process is the same as Prefix Caching in HBM.

### 2. Combining KV Cache Pool with Mooncake PD Disaggregation

When used together with Mooncake PD (Prefill-Decode) Disaggregation, the KV Cache Pool can further decouple prefill and decode stages across devices or nodes.

Currently, we only perform put and get operations of KV Pool for **Prefill Nodes**, and Decode Nodes get their KV Cache from Mooncake P2P KV Connector, i.e., MooncakeConnector.

The key benefit of doing this is that we can keep the gain in performance by computing less with Prefix Caching from HBM and KV Pool for Prefill Nodes, while not sacrificing the data transfer efficiency between Prefill and Decode nodes with P2P KV Connector that transfers KV Caches between NPU devices directly.

To enable this feature, we need to set up both Mooncake Connector and Mooncake Store Connector with a Multi Connector, which is a KV Connector class provided by vLLM that can call multiple KV Connectors in a specific order.

For details, please also refer to the Mooncake Connector Store Deployment Guide.

## How is MooncakeStoreConnectorV1 Implemented?

**MooncakeStoreConnectorV1** inherits the KV Connector V1 class in vLLM V1: through implementing the required methods defined in the KV connector V1 base class, one can integrate a third-party KV cache transfer/storage backend into the vLLM framework.

MooncakeStoreConnectorV1 is also largely inspired by LMCacheConnectorV1 in terms of the `Lookup Engine`/`Lookup Client` design for looking up KV cache keys, and the `ChunkedTokenDatabase` class for processing tokens into prefix-aware hashes as well as other hashing related designs. On top of this, we have also added our own design including `KVTransferThread` that allows async `get` and `put` of KV caches with multi-threading, and NPU-related data transfer optimization such as removing the `LocalBuffer` in LMCache to remove redundant data transfer.

The KV Connector methods that need to be implemented can be categorized into scheduler-side methods that are called in V1 scheduler and worker-side methods that are called in V1 worker, namely:

### KV Connector Scheduler-Side Methods

`get_num_new_matched_tokens`: Get prefix cache hit in number of tokens through looking up into the KV pool.  
`update_states_after_alloc`:  Update KVConnector state after temporary buffer alloc.  
`build_connector_meta`: Attach the connector metadata to the request object.  
`request_finished`: Once a request is finished, determine whether request blocks should be freed now or will be sent asynchronously and freed later.

### Connector Worker-Side Methods

`register_kv_caches`: Register KV cache buffers needed for KV cache transfer.
`start_load_kv`: Perform KV cache load operation that transfers KV cache from storage to device.
`wait_for_layer_load`: Optional; Wait for layer load in layerwise + async KV load scenario.
`save_kv_layer`: Optional; Do layerwise KV cache put into KV Pool.
`wait_for_save`: Wait for KV Save to finish if async KV cache save/put.
`get_finished`: Get request that finished KV transfer, `done_sending` if `put` finished, `done_receiving` if `get` finished.

## DFX

1. When looking up a key in KV Pool, if we cannot find the key, there is no Cache Hit for this specific block; we return no hit for this block and do not look up further blocks for the current request.
2. Similarly, when we are trying to put a block into KV Pool and it fails, we do not put further blocks (subject to change).

## Limitations

1. Currently, Mooncake Store for vLLM-Ascend only supports DRAM as the storage for KV Cache pool.

2. For now, if we successfully looked up a key and found it exists, but failed to get it when calling KV Pool's get function, we just output a log indicating the get operation failed and keep going; hence, the accuracy of that specific request may be affected. We will handle this situation by falling back the request and re-compute everything assuming there's no prefix cache hit (or even better, revert only one block and keep using the Prefix Caches before that).
