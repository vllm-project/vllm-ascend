# Feature Support

The feature support principle of vLLM Ascend is: **aligned with the vLLM**. We are also actively collaborating with the community to accelerate support.

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       |      Status    | Next Step                                                              |
|-------------------------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🟢 Functional  | Functional, see detail note: [Chunked Prefill][cp]                     |
| Automatic Prefix Caching      | 🟢 Functional  | Functional, see detail note: [vllm-ascend#732][apc]                    |
| LoRA                          | 🟢 Functional  | [vllm-ascend#396][multilora], [vllm-ascend#893][v1 multilora]          |
| Speculative decoding          | 🟢 Functional  | Basic support                                                          |
| Pooling                       | 🟡 Planned     | v0.9.1 doesn't support pooling. Use higher version instead  |
| Enc-dec                       | 🟡 Planned     | vLLM should support this feature first.                                |
| Multi Modality                | 🟢 Functional  | [Tutorial][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional  | CI needed                                                              |
| Prompt logProbs               | 🟢 Functional  | CI needed                                                              |
| Async output                  | 🟢 Functional  | CI needed                                                              |
| Beam search                   | 🟢 Functional  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional  | [vllm-ascend#177][guided_decoding]                                     |
| Tensor Parallel               | 🟢 Functional  | Make TP >4 work with graph mode                                        |
| Pipeline Parallel             | 🟢 Functional  | Write official guide and tutorial.                                     |
| Expert Parallel               | 🟢 Functional  | Dynamic EPLB support.                                                  |
| Data Parallel                 | 🟢 Functional  | Data Parallel support for Qwen3 MoE.                                   |
| Prefill Decode Disaggregation | 🟢 Functional  | xPyD works on v1 engine                                                |
| Quantization                  | 🟢 Functional  | W8A8 available; working on more quantization method support(W4A8, etc) |
| Graph Mode                    | 🔵 Experimental| Experimental, see detail note: [vllm-ascend#767][graph_mode]           |
| Sleep Mode                    | 🟢 Functional  |                                                                        |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan / Deprecated: No plan or deprecated by vLLM.

[v1_user_guide]: https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html
[multimodal]: https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/single_npu_multimodal.html
[best_of]: https://github.com/vllm-project/vllm/issues/13361
[guided_decoding]: https://github.com/vllm-project/vllm-ascend/issues/177
[v1_scheduler]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
[v1_rfc]: https://github.com/vllm-project/vllm/issues/8779
[multilora]: https://github.com/vllm-project/vllm-ascend/issues/396
[v1 multilora]: https://github.com/vllm-project/vllm-ascend/pull/893
[graph_mode]: https://github.com/vllm-project/vllm-ascend/issues/767
[apc]: https://github.com/vllm-project/vllm-ascend/issues/732
[cp]: https://docs.vllm.ai/en/stable/performance/optimization.html#chunked-prefill
[1P1D]: https://github.com/vllm-project/vllm-ascend/pull/950
[ray]: https://github.com/vllm-project/vllm-ascend/issues/1751
