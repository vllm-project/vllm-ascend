# Feature × Feature

The tables below show mutually exclusive features and the support on Ascend hardware, extended from [vLLM table](https://docs.vllm.ai/en/latest/features/#feature-x-feature).

The symbols used have the following meanings:

- ✅ = Full compatibility
- 🟠 = Partial compatibility
- ❌ = No compatibility
- ❔ = Unknown or TBD

<style>
td:not(:first-child) {
  text-align: center !important;
}
td {
  padding: 0.5rem !important;
  white-space: nowrap;
}

th {
  padding: 0.5rem !important;
  min-width: 0 !important;
}

th:not(:first-child) {
  writing-mode: vertical-lr;
  transform: rotate(180deg)
}
</style>

| Feature | [ACLGraph-FULL_DECODE_ONLY](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | [ACLGraph-PIECEWISE](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | async-scheduling | [automatic_prefix_caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | [Chunked-Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | **Context-Parallel** | **Cpu-Binding** | **Data-Parallel** | **Disaggregated-prefill** | **dflash** | **EAGLE3** | **EPLB** | **Expert-Parallel** | **Flashcomm1** | **KV-Cache-Pool**  | **layer_sharding** | **lmhead_tensor_parallel_size** | [LoRA](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html) | **MLAPO** | [mm](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | **multistream_moe** | **shared_expert_dp** | **Quantization-W4A4** | **Quantization-W4A8** | **Quantization-W8A8** | [suffix](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/suffix_speculative_decoding.html) | **Tensor-Parallel** | **weight_nz** |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| [ACLGraph-FULL_DECODE_ONLY](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ✅ |
| [ACLGraph-PIECEWISE](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) | ❌ | ✅ |
| **async-scheduling** | ✅ | ✅ | ✅ |
| [automatic_prefix_caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | ✅ | ✅ | ✅ | ✅ |
| [Chunked-Prefill](https://docs.vllm.ai/en/stable/configuration/optimization/#chunked-prefill) | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Context-Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Cpu-Binding** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Data-Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | 🟠<sup>1</sup> | ✅ | ✅ |
| **Disaggregated-prefill** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **dflash** | ✅ | ✅ | ✅ | ❔ | ✅ | ❌ | ✅ | ❔ | ❌ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **EAGLE3** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **EPLB** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **Expert-Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |
| **Flashcomm1**<sup>2</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **KV-Cache-Pool**  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ |  |  |  |  |  |  |  |  |  |  |  |  |  |
| **layer_sharding** | ✅ | ✅ |  |  |  | 🟠 |  |  |  | ✅ | ❔ | ❔ | ✅ | ✅ | ❔ |
| **lmhead_tensor_parallel**<sup>3</sup> | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ❌ | ❔ |
| [LoRA](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html) | ✅ | ✅ |  |  |  | ❌ |  |  |  | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ |
| **MLAPO**<sup>5</sup> | ✅ | ✅ |  |  |  | ✅ |  |  |  | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ |
| [mm](https://docs.vllm.ai/en/latest/features/multimodal_inputs/) | ✅ | ✅ |  |  |  | 🟠 |  |  |  | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **multistream_moe** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **shared-expert-dp** | ✅ | ✅ |  |  |  | 🟠<sup>1</sup> |  |  |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Quantization-W4A4** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❔ | ❔ | ✅ | ❔ | ✅ | ✅ | ❔ | ❔ | ❔ | ❔ | ✅ |  |  |  |  |  |
| **Quantization-W4A8** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ❔ | ✅ | ✅ | ❔ | ✅ |  |  |  |  |
| **Quantization-W8A8** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ❔ | ✅ | ✅ |  |  |  |
| [suffix](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/features/suffix_speculative_decoding.html) | ❌ | ✅ |  |  |  | ❌ |  |  |  | ❌ | ❌ | ✅ | ✅ | ✅ | ❔ |  |  |  |  |  |  |  | ❔ | ❔ | ❔ |
| **Tensor-Parallel** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |  |
| **weight-nz** | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟠 | ✅ | ✅ | ✅ | ✅ |

<sup>1</sup> Only dcp supports dp while pcp does not support dp.
<sup>2</sup> Falshcomm is only enabled on the prefill stage.
<sup>3</sup> lmhead_tensor_parallel is only enabled in the pure dp scenarios.
<sup>4</sup> LoRA applies only to the language backbone of multimodal models (upstream).
<sup>5</sup> only decode stage supports MLAPO.