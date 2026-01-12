# Supported Features

The feature support principle of vLLM Ascend is: **aligned with the vLLM**. We are also actively collaborating with the community to accelerate support.

Functional call: https://docs.vllm.ai/en/latest/features/tool_calling/

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       |      Status    | Next Step                                                              |
|-------------------------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🟢 Functional  | Functional, see detailed note: [Chunked Prefill][cp]                     |
| Automatic Prefix Caching      | 🟢 Functional  | Functional, see detailed note: [vllm-ascend#732][apc]                    |
| LoRA                          | 🟢 Functional  | [vllm-ascend#396][multilora], [vllm-ascend#893][v1 multilora]          |
| Speculative decoding          | 🟢 Functional  | Basic support                                                          |
| Pooling                       | 🟢 Functional  | CI needed to adapt to more models; V1 support relies on vLLM support.   |
| Enc-dec                       | 🟡 Planned     | vLLM should support this feature first.                                |
| Multi Modality                | 🟢 Functional  | [Tutorial][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional  | CI needed                                                              |
| Prompt logProbs               | 🟢 Functional  | CI needed                                                              |
| Async output                  | 🟢 Functional  | CI needed                                                              |
| Beam search                   | 🟢 Functional  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional  | [vllm-ascend#177][guided_decoding]                                     |
| Tensor Parallel               | 🟢 Functional  | Make TP >4 work with graph mode.                                        |
| Pipeline Parallel             | 🟢 Functional  | Write official guide and tutorial.                                     |
| Expert Parallel               | 🟢 Functional  | Support dynamic EPLB.                                                  |
| Data Parallel                 | 🟢 Functional  | Data Parallel support for Qwen3 MoE.                                   |
| Prefill Decode Disaggregation | 🟢 Functional  | Functional, xPyD is supported.                                         |
| Quantization                  | 🟢 Functional  | W8A8 available; working on more quantization method support (W4A8, etc) |
| Graph Mode                    | 🔵 Experimental| Experimental, see detailed note: [vllm-ascend#767][graph_mode]           |
| Sleep Mode                    | 🟢 Functional  |                                                                        |
| Context Parallel              | 🔵 Experimental|                                                                        |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan/Deprecated: No plan or deprecated by vLLM.

[v1_user_guide]: https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html
[multimodal]: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/single_npu_multimodal.html
[guided_decoding]: https://github.com/vllm-project/vllm-ascend/issues/177
[multilora]: https://github.com/vllm-project/vllm-ascend/issues/396
[v1 multilora]: https://github.com/vllm-project/vllm-ascend/pull/893
[graph_mode]: https://github.com/vllm-project/vllm-ascend/issues/767
[apc]: https://github.com/vllm-project/vllm-ascend/issues/732
[cp]: https://docs.vllm.ai/en/stable/performance/optimization.html#chunked-prefill
[1P1D]: https://github.com/vllm-project/vllm-ascend/pull/950
[ray]: https://github.com/vllm-project/vllm-ascend/issues/1751
