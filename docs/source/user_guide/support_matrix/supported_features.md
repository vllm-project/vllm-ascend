# Supported Features

The feature support principle of vLLM Ascend is: **aligned with the vLLM**. We are also actively collaborating with the community to accelerate support.

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       |      Status    | Next Step                                                              |
|-------------------------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🟢 Functional  | Functional, see detailed note: [Chunked Prefill][cp]                   |
| Automatic Prefix Caching      | 🟢 Functional  | Functional, see detailed note: [Automatic Prefix Caching][apc]         |
| LoRA                          | 🟢 Functional  | Functional, see detailed note: [LoRA][lora]                            |
| Speculative decoding          | 🟢 Functional  | Basic support                                                          |
| Pooling                       | 🟢 Functional  | CI needed to adapt to more models;                                     |
| Enc-dec                       | 🟡 Planned     | vLLM should support this feature first.                                |
| Multi Modality                | 🟢 Functional  | [Tutorial][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional  | CI needed                                                              |
| Prompt logProbs               | 🟢 Functional  | CI needed                                                              |
| Async output                  | 🟢 Functional  | CI needed                                                              |
| Beam search                   | 🟢 Functional  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional  | See detailed note: [Structured Output Guide][guided_decoding]          |
| Tensor Parallel               | 🟢 Functional  | Make TP >4 work with graph mode.                                       |
| Pipeline Parallel             | 🟡 Planned     | Broken in this version, will fix in next release.                      |
| Expert Parallel               | 🟢 Functional  | See detailed note: [Expert Load Balance (EPLB)][graph_mode]            |
| Data Parallel                 | 🟢 Functional  | Data Parallel support for Qwen3 MoE.                                   |
| Prefill Decode Disaggregation | 🟢 Functional  | Functional, xPyD is supported.                                         |
| Quantization                  | 🟢 Functional  | See detailed note: [Quantization Guide][qaunt]                         |
| Graph Mode                    | 🟢 Functional  | See detailed note: [Graph Mode Guide][graph_mode]                      |
| Sleep Mode                    | 🟢 Functional  | See detailed note: [Sleep Mode][sleep]                                 |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan/Deprecated: No plan or deprecated by vLLM.

[v1_user_guide]: https://docs.vllm.ai/en/latest/usage/v1_guide/
[multimodal]: https://vllm-ascend.readthedocs.io/en/latest/tutorials/single_npu_multimodal.html
[guided_decoding]: https://docs.vllm.ai/projects/ascend/en/v0.11.0-dev/user_guide/feature_guide/structured_output.html
[lora]: https://docs.vllm.ai/en/stable/features/lora/
[graph_mode]: https://docs.vllm.ai/projects/ascend/en/v0.11.0-dev/user_guide/feature_guide/graph_mode.html
[apc]: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
[cp]: https://docs.vllm.ai/en/stable/performance/optimization.html#chunked-prefill
[1P1D]: https://github.com/vllm-project/vllm-ascend/pull/950
[ray]: https://github.com/vllm-project/vllm-ascend/issues/1751
[sleep]:https://docs.vllm.ai/projects/ascend/en/v0.11.0-dev/user_guide/feature_guide/sleep_mode.html
[quant]:https://docs.vllm.ai/projects/ascend/en/v0.11.0-dev/user_guide/feature_guide/quantization.html
