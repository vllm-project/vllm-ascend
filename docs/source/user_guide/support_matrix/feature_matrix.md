# Feature × Feature

The tables below show mutually exclusive features and the support on Ascend hardware, extended from [vLLM table](https://docs.vllm.ai/en/latest/features/#feature-x-feature).

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

| Feature | [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | Async Scheduling | [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | [Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | [Cpu Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | <abbr title="Data Parallel">DP</abbr> | [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | <abbr title="Speculative Decoding">Dflash</abbr> | Eagle3 | [Eplb](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/eplb_swift_balancer.html) | <abbr title="Expert-Parallel">EP</abbr> | Flashcomm1 | [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html)  | Layer Sharding | Lmhead TP | [Lora](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html) | Mlapo | [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | Multistream Moe | Shared Expert DP | [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | [<abbr title="Speculative Decoding">suffix</abbr>](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/suffix_speculative_decoding.html) | <abbr title="Tensor Parallel">TP</abbr> | Weight nz |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [ACLGraph Full_Decode_Only](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ✅ | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| [ACLGraph Piecewise](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ❌ | ✅ | | | | | | | | | | | | | | | | | | | | | | | | | | |
| Async Scheduling | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | | | | | | |
| [<abbr title="Automatic Prefix Caching">APC</abbr>](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | | | | | |
| [Chunked Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | | | | |
| [Context Parallel](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/context_parallel.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | | | |
| [Cpu Binding](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/cpu_binding.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | | |
| <abbr title="Data Parallel">DP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ | | | | | | | | | | | | | | | | | | | | |
| [Disaggregated Prefill](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/disaggregated_prefill.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | | | | |
| <abbr title="Speculative Decoding">Dflash</abbr> | ✅ | ✅ | ✅ | ❔ | ✅ | ❌ | ✅ | ❔ | ❌ | ✅ | | | | | | | | | | | | | | | | | | |
| Eagle3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | | | | | | | | | | | | | | | | | |
| [Eplb](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/eplb_swift_balancer.html) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | | |
| <abbr title="Expert-Parallel">EP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | | | |
| Flashcomm1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>2</sup> | ❔ | ✅ | ✅ | ✅ | ✅ |  | | | | | | | | | | | | | |
| [KV Cache Pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/KV_Cache_Pool_Guide.html)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | | | | | | |
| Layer Sharding | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | 🟠<sup>3</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | | | | | | | | | | | | |
| Lmhead TP | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | 🟠<sup>4</sup> | ✅ | ❔ | ✅ | ✅ | ✅ | ❌ | ❔ | ✅ | ✅ | | | | | | | | | | | |
| [Lora](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html)<sup>5</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ | ✅ | | | | | | | | | | |
| Mlapo | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>6</sup> | ✅ | ✅ | ✅ | ✅ | ❌ | ❔ | ❌ | ✅ | ❔ | ✅ | | | | | | | | | |
| [<abbr title="Multimodal Inputs">mm</abbr>](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠 | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | | | | | | | | |
| Multistream Moe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | |
| Shared Expert DP | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ❔ | ✅ | ✅ | ❔ | ✅ | | | | | | |
| [Quantization W4A4](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❔ | ❔ | ✅ | ❔ | ✅ | ✅ | ❔ | ❌ | ❔ | ❔ | ✅ | | | | | |
| [Quantization W4A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ❌ | ✅ | ✅ | ❔ | ✅ | | | | |
| [Quantization W8A8](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/quantization.html#) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | | | |
| [<abbr title="Speculative Decoding">suffix</abbr>](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/suffix_speculative_decoding.html) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❔ | ❌ | ❌ | ✅ | ✅ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ✅ | ✅ | ❔ | ❔ | ❔ | ✅ | | |
| <abbr title="Tensor Parallel">TP</abbr> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |
| Weight nz | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟠 | ✅ | ✅ | ✅ | ✅ |

- <sup>1</sup> Only dcp supports dp while pcp does not support dp.

- <sup>2</sup> Falshcomm is only enabled on the prefill stage.

- <sup>3</sup> Layer sharding is only enabled on the prefill stage.

- <sup>4</sup> Lmhead TP is only enabled in the pure dp scenarios.

- <sup>5</sup> LoRA applies only to the language backbone of multimodal models (upstream).

- <sup>6</sup> MLAPO is only supported on the decode stage.
