# feature_matrix

The table below shows mutually exclusive features and the support on Ascend hardware, extended from the [vLLM table](https://docs.vllm.ai/en/latest/features/#feature-x-feature).

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

| Feature | [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | Async Scheduling | [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | [Decode Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | [CPU Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | [<abbr title="Data Parallel">DP</abbr>](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) | [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | [Eagle3](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/) | [EPLB](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/model_runner_v1_eplb.html) | [<abbr title="Expert-Parallel">EP</abbr>](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) | [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html) | Lmhead TP | MLAPO | [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | Multistream MoE | Shared Expert DP | [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | <abbr title="Tensor Parallel">TP</abbr> | Weight nz | [KVPP (Experimental)](../feature_guide/kvpp.md) |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ❌ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Async Scheduling | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Decode Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [CPU Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Data Parallel">DP</abbr>](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [Eagle3](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [EPLB](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/model_runner_v1_eplb.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |
| [<abbr title="Expert-Parallel">EP</abbr>](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |
| [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |
| Lmhead TP | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | 🟠<sup>2</sup> | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |  |  |  |  |  |
| MLAPO | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>3</sup> | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |
| [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |
| Multistream MoE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |
| Shared Expert DP | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |  |  |
| [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ✅ | ✅ | ❔ | ❌ | ❔ | ❔ | ✅ |  |  |  |  |  |
| [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | ❌ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |
| [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |  |
| <abbr title="Tensor Parallel">TP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |
| Weight nz | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟠 | ✅ | ✅ | ✅ |  |
| [KVPP (Experimental)](../feature_guide/kvpp.md) | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❔ | ❔ | 🟠 | ❌ | ❔ | ✅ | 🟠 | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ | ✅ |
| [Pipeline Parallel](../feature_guide/pipeline_parallel.md) | 🟠<sup>4</sup> | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>5</sup> | 🟠<sup>6</sup> | ✅ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ✅ | ✅ | ❔ | ✅ |
|  | See the detailed PP feature-stacking matrix in [Pipeline Parallelism](../feature_guide/pipeline_parallel.md). |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

- <sup>1</sup> Only dcp supports dp while pcp does not support dp.
- <sup>2</sup> Lmhead TP is only enabled in the pure dp scenarios.
- <sup>3</sup> MLAPO is only supported on the decode stage.
- <sup>4</sup> Pipeline Parallel supports graph mode with `FULL_DECODE_ONLY` only.
- <sup>5</sup> Pipeline Parallel is supported on the prefill node; the decode node must use `PP1`.
- <sup>6</sup> Eagle3 with Pipeline Parallel supports MiniMax-M3 sparse models only.
