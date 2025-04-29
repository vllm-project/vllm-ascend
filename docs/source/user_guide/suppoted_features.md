# Feature Support

The feature support principle of vLLM Ascend is: **aligned with the vLLM**. We are also actively collaborating with the community to accelerate support.

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       | vLLM V0 Engine | vLLM V1 Engine | Next Step                                                              |
|-------------------------------|----------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🚧 WIP         | 🚧 WIP         | Functional, waiting for CANN 8.1 nnal package release                  |
| Automatic Prefix Caching      | 🚧 WIP         | 🚧 WIP         | Functional, waiting for CANN 8.1 nnal package release                  |
| LoRA                          | 🟢 Functional  | 🚧 WIP         | [vllm-ascend#396][multilora], CI needed, working on V1 support         |
| Prompt adapter                | No plan        | 🟡 Planned     | Plan in 2025.06.30                                                     |
| Speculative decoding          | 🟢 Functional  | 🚧 WIP         | CI needed; working on V1 support                                       |
| Pooling                       | 🟢 Functional  | 🟢 Functional  | CI needed and adapting more models; V1 support rely on vLLM support.   |
| Enc-dec                       | 🔴 NO plan     | 🟡 Planned     | Plan in 2025.06.30                                                     |
| Multi Modality                | 🟢 Functional  | 🟢 Functional  | [Tutorial][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Prompt logProbs               | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Async output                  | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Multi step scheduler          | 🟢 Functional  | 🔴 Deprecated  | [vllm#8779][v1_rfc], replaced by [vLLM V1 Scheduler][v1_scheduler])    | 
| Best of                       | 🟢 Functional  | 🔴 Deprecated  | [vllm#13361][best_of], CI needed                                       |
| Beam search                   | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional  | 🟢 Functional  | [vllm-ascend#177][guided_decoding]                                     |
| Tensor Parallel               | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Pipeline Parallel             | 🟢 Functional  | 🟢 Functional  | CI needed                                                              |
| Expert Parallel               | 🔴 NO plan     | 🟢 Functional  | CI needed; No plan on V0 support                                       |
| Data Parallel                 | 🔴 NO plan     | 🟢 Functional  | CI needed;  No plan on V0 support                                      |
| Prefill Decode Disaggregation | 🟢 Functional  | 🟢 Functional  | 1P1D available, working on xPyD and V1 support.                        |
| Quantization                  | 🟢 Functional  | 🟢 Functional  | W8A8 available, CI needed; working on more quantization method support |
| Graph Mode                    | 🔴 NO plan     | 🟢 Functional  | Functional, waiting for CANN 8.1 nnal package release                  |
| Sleep Mode                    | 🟢 Functional  | 🟢 Functional  | level=1 available, CI needed, working on V1 support                    |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🚧 WIP: Under active development
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan / Deprecated: No plan for V0 or deprecated by vLLM v1.

[v1_user_guide]: https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html
[multimodal]: https://vllm-ascend.readthedocs.io/en/latest/tutorials/single_npu_multimodal.html
[best_of]: https://github.com/vllm-project/vllm/issues/13361
[guided_decoding]: https://github.com/vllm-project/vllm-ascend/issues/177
[v1_scheduler]: https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py
[v1_rfc]: https://github.com/vllm-project/vllm/issues/8779
[multilora]: https://github.com/vllm-project/vllm-ascend/issues/396
