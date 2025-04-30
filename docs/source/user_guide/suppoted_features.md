# Feature Support

The feature support principle of vLLM Ascend is: **aligned with the vLLM**. We are also actively collaborating with the community to accelerate support.

vLLM Ascend offers the overall functional support of the most features in vLLM, and the usage keep the same with vLLM except for some limits.

| Feature                       | vLLM Ascend    | vLLM Ascend (+ MindIE Turbo)    | Notes                                                                  |
|-------------------------------|----------------|-----------------|------------------------------------------------------------------------|
| V1Engine                      | 🔵 Experimental| 🔵 Experimental| Will enhance in v0.8.x                                                 |
| Chunked Prefill               | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Automatic Prefix Caching      | 🟢 Functional  | 🟢 Functional  | [Usage Limits][#732](https://github.com/vllm-project/vllm-ascend/issues/732) |
| LoRA                          | 🟢 Functional  | 🟡 Planned     | /                                                                      |
| Prompt adapter                | 🟡 Planned     | 🟡 Planned     | /                                                                      |
| Speculative decoding          | 🟢 Functional  | 🟢 Functional  | [Usage Limits][#734](https://github.com/vllm-project/vllm-ascend/issues/734) |
| Pooling                       | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Enc-dec                       | 🟡 Planned     | 🟡 Planned     | /                                                                      |
| Multi Modality                | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| LogProbs                      | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Prompt logProbs               | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Async output                  | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Multi step scheduler          | 🟢 Functional  | 🟢 Functional  | /                                                                      | 
| Best of                       | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Beam search                   | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Guided Decoding               | 🟢 Functional  | 🟢 Functional  | /                                                                      |
| Tensor Parallel               | 🟢 Functional  | ⚡Optimized     | /                                                                       |
| Pipeline Parallel             | 🟢 Functional  | ⚡Optimized     | /                                                                       |
| Expert Parallel               | 🟡 Planned     | 🟡 Planned     | Will support in v0.8.x                                                 |
| Data Parallel                 | 🟡 Planned     | 🟡 Planned     | Will support in v0.8.x                                                 |
| Prefill Decode Disaggregation | 🟡 Planned     | 🟡 Planned     | Will support in v0.8.x                                                 |
| Quantization                  | 🟡 Planned     | 🟢 Functional  | Will support in v0.8.x                                                 |
| Graph Mode                    | 🟡 Planned     | 🟡 Planned     | Will support in v0.8.x                                                 |
| Sleep Mode                    | 🟢 Functional  | 🟢 Functional  | [Usage Limits][#733](https://github.com/vllm-project/vllm-ascend/issues/733) |
| MTP                           | 🟢 Functional  | 🟢 Functional  | [Usage Limits][#734](https://github.com/vllm-project/vllm-ascend/issues/734) |
| Custom Scheduler              | 🟢 Functional  | 🟢 Functional  | [Usage Limits][#788](https://github.com/vllm-project/vllm-ascend/issues/788) |


*MindIE Turbo is an LLM inference engine acceleration plug-in library on Ascend hardware. Find more information [here](https://www.hiascend.com/document/detail/zh/mindie/20RC1/AcceleratePlugin/turbodev/mindie-turbo-0001.html).*

- 🟢 Functional: Fully operational, with ongoing optimizations.

- 🔵 Experimental: Experimental support, interfaces and functions may change.

- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
