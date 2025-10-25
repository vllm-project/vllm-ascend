# Disaggregated Prefill

## Why Disaggregated Prefill?

The Disaggregated Prefill feature is designed to address the coupling issue between the preprocessing stage and the inference computation stage in the vllm-ascend inference system. In traditional architectures, data preprocessing and model inference are tightly coupled within the same execution context, leading to:
• Unbalanced resource utilization: CPU-intensive preprocessing and NPU-intensive computation block each other.
• Inefficient pipeline performance: Preprocessing becomes a bottleneck for inference performance.
• Limited scalability: Inability to independently scale preprocessing and inference resources.
This feature decouples the preprocessing and inference execution processes, achieving:
• Asynchronous parallel execution, improving overall throughput.
• Resource isolation and independent scalability.
• Reduced end-to-end inference latency.

## Usage

vLLM Ascend Currently supports Mooncake Store for KV Cache Pool. To enable Mooncake Store, one needs to config kv-transfer-config and choose MooncakeStoreConnector as KV Connector. 

For step-by-step deployment and configuration, please refer to the guide: https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/mooncake_connector_store_deployment_guide.md

## How it works?
The KV Cache Pool integrates multiple memory tiers (HBM, DRAM, SSD) through a connector-based architecture. Each connector implements a unified interface for storing, retrieving, and transferring KV blocks between tiers, depending on access frequency and hardware bandwidth.

When combined with vLLM’s Prefix Caching mechanism, the pool enables efficient caching both locally (in HBM) and globally (via Mooncake), ensuring that frequently used prefixes remain hot while less frequently accessed KV data can spill over to lower-cost memory.


### Combining KV Cache Pool with HBM Prefix Caching 
Prefix Caching with HBM is already supported by the vLLM V1 Engine.
By introducing KV Connector V1, users can seamlessly combine HBM-based Prefix Caching with Mooncake-backed KV Pool. The user can enable both features simply by enabling 

Workflow:

1. The engine first checks for prefix hits in the HBM cache.

2. After getting the number of hit tokens on HBM, it queries the KV Pool via the connector, if there is additional hits in KV Pool, we get the **additional blocks only** from KV Pool, and get the rest of the blocks directly from HBM to minimize the data transfer latency.

3. After the KV Caches in KV Pool is load into HBM, the remaining process is the same as Prefix Caching in HBM.

### Combining KV Cache Pool with Mooncake PD Disaggregation 

When used together with Mooncake PD (Prefill-Decode) Disaggregation, the KV Cache Pool can further decouple prefill and decode stages across devices or nodes. Currently, we only perform put and get operation of KV Pool for **Prefiil Nodes**, and Decode Nodes get their KV Cache from Mooncake P2P KV Connector, i.e. MooncakeConnector.

 The key benefit of doing this is that we can keep the gain in performance by computing less with Prefix Caching from HBM and KV Pool for Prefill Nodes while not sacrificing the data transfer efficiency between Prefill and Decode nodes with P2P KV Connector that transfer KV Caches between NPU devices directly. 

To Enable this feature, we need to setup both Mooncake Connector and Mooncake Store connector with a Multi Connector, which is a KV Connector class proviced by vLLM that can call multiple KV Connectors in specific order; For details, please also refer to the Mooncake Connector Store Deployment Guide.

## DFX
1. When looking up a key in KV Pool, if we cannot find the key, there is no Cache Hit for this specific block; we return no hit for this block and do not look up further blocks for current request.
2. Similaly, when we are trying to put a block into KV Pool and failed, we do not put furhter blocks (subject to change).



## Limitation

1. Currently, Mooncake Store for vLLM-Ascend only supports DRAM as the storage for KV Cache pool.

2. For now, if we successfully looked up a key and found it exists, but failed to get it when calling KV Pool's get function, we just output a log indicating the get operation failed and keep going; hence, the accuracy of that specific request may be affected. We will handle this situation by falling back the request and re-compute everything assuming there's no prefix cache hit (or even better, revert only one block and keep using the Prefix Caches before that).
