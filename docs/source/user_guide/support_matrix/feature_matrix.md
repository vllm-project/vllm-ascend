# feature_matrix

The table below shows mutually exclusive features and the support on Ascend hardware, extended from the [vLLM table](https://docs.vllm.ai/en/latest/features/#feature-x-feature).

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

| Feature | [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | Async Scheduling | [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | [Prefill Context Parallel](../feature_guide/context_parallel.md#prefill-context-parallel) | [Decode Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | [CPU Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | [<abbr title="Data Parallel">DP</abbr>](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) | [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | [Eagle3](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/) | [EPLB](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/expert_parallelism_load_balancer.html) | [<abbr title="Expert-Parallel">EP</abbr>](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) | [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html) | Lmhead TP | MLAPO | [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | Multistream MoE | Shared Expert DP | [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | <abbr title="Tensor Parallel">TP</abbr> | Weight nz | [KVPP (Experimental)](../feature_guide/kvpp.md) | [Pipeline Parallel](../feature_guide/pipeline_parallel.md) | [Disaggregated Encoder (EPD)](../feature_guide/epd_disaggregation.md) |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ❌ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Async Scheduling | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Prefill Context Parallel](../feature_guide/context_parallel.md#prefill-context-parallel) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Decode Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [CPU Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Data Parallel">DP</abbr>](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Eagle3](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [EPLB](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/expert_parallelism_load_balancer.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Expert-Parallel">EP</abbr>](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Lmhead TP | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ❔ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |
| MLAPO | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>2</sup> | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | 🟠 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |
| Multistream MoE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |
| Shared Expert DP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |  |  |  |
| [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ✅ | ✅ | ❔ | ❌ | ❔ | ❔ | ✅ |  |  |  |  |  |  |  |
| [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | ❌ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |  |
| [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |  |  |  |
| <abbr title="Tensor Parallel">TP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |
| Weight nz | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟠 | ✅ | ✅ | ✅ |  |  |  |
| [KVPP (Experimental)](../feature_guide/kvpp.md) | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ❔ | 🟠 | ✅ | ❔ | ✅ | 🟠 | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ | ✅ |  |  |
| [Pipeline Parallel](../feature_guide/pipeline_parallel.md) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>3</sup> | 🟠<sup>4</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| [Disaggregated Encoder (EPD)](../feature_guide/epd_disaggregation.md) | 🟠<sup>6</sup> | 🟠<sup>6</sup> | ❔<sup>8</sup> | ✅ | ✅ | ❔<sup>8</sup> | ❔<sup>8</sup> | ✅ | ❔<sup>8</sup> | ✅ | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ✅ | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>8</sup> | ❔<sup>7</sup> | ✅<sup>5</sup> |

- <sup>1</sup> Lmhead TP is only enabled in the pure DP scenarios.
- <sup>2</sup> MLAPO is only supported on the decode stage.
- <sup>3</sup> Pipeline Parallel is supported on the prefill node; the decode node must use `PP1`.
- <sup>4</sup> Eagle3 with Pipeline Parallel supports MiniMax-M3 sparse models only.
- **Mooncake backend:** KV Cache Pool with Decode Context Parallel is supported for non-layerwise complete-block transfers and for block-key layerwise pooling with `prefill_context_parallel_size=1`, subject to the cache-group constraints in the [layerwise pooling guide](../feature_guide/mooncake_layerwise_adaptation_and_optimization.md#decode-context-parallelism-dcp). Hybrid DCP support is experimental. The Memcache backend supports DCP as before.
- <sup>5</sup> Disaggregated Encoder (EPD) is supported on A2/A3 with Model Runner V2 by default; set `VLLM_USE_V2_MODEL_RUNNER=0` to force MRV1. EPD is not supported on 310P.
- <sup>6</sup> EPD with VL encoder graph (`--compilation-config '{"cudagraph_mm_encoder": true}'`) is not supported on Model Runner V2 and falls back to MRV1.
- <sup>7</sup> EPD combined with Pipeline Parallel (EC + PP) is unverified (❔). `ECCPUConnector` works on a single node only; `ECMooncakeConnector` requires a shared Mooncake cluster; `ECExampleConnector` requires a shared filesystem path visible to all E/PD instances.
- <sup>8</sup> These EPD combinations have not been validated yet (❔): async scheduling, context parallelism, data parallelism, speculative decoding, EPLB, KV Cache Pool, TP > 1, quantization, MLAPO, Lmhead TP, KVPP.

See the detailed PP feature-stacking matrix in [Pipeline Parallel](../feature_guide/pipeline_parallel.md).
