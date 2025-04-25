# Feature Support

|           Feature        | V0 Engine |  V1 Engine | CI Coverage | Guidance Document |     Current Status        |    Next Step       |
|--------------------------|-----------|------------|-------------|-------------------|---------------------------|--------------------|
|      Chunked Prefill     |     ❌    |     ❌     |             |                   |            NA             | Rely on CANN 8.1 nnal package release |
| Automatic Prefix Caching |     ❌    |     ❌     |             |                   |            NA             | Rely on CANN 8.1 nnal package release |
|          LoRA            |     ✅    |     ❌     |             |                   | Basic functions available | Need fully test and performance improvement; working on V1 support |
|      Prompt adapter      |     ❌    |     ❌     |             |                   |            NA             | Plan in 2025.06.30 |
|    Speculative decoding  |     ✅    |     ❌     |      ✅     |                   | Basic functions available | Need fully test; working on V1 support  |
|        Pooling           |     ✅    |     ❌     |             |                   | Basic functions available(Bert) | Need fully test and add more models support; V1 support rely on vLLM support.|
|        Enc-dec           |     ❌    |     ❌     |             |                   |            NA             | Plan in 2025.06.30|
|      Multi Modality      |     ✅    |     ✅     |             |         ✅        | Basic functions available(LLaVA/Qwen2-vl/Qwen2-audio/internVL)| Improve performance, and add more models support |
|        LogProbs          |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|     Prompt logProbs      |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|       Async output       |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|   Multi step scheduler   |     ✅    |     N/A    |             |                   | Basic functions available | Need fully test, Find more details at [<u> Blog </u>](https://blog.vllm.ai/2024/09/05/perf-update.html#batch-scheduling-multiple-steps-a head-pr-7000), [<u> RFC </u>](https://github.com/vllm-project/vllm/issues/6854) and [<u>issue</u>](https://github.com/vllm-project/vllm/pull/7000)  |
|          Best of         |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|        Beam search       |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|      Guided Decoding     |     ✅    |     ✅     |             |                   | Basic functions available | Find more details at the [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues/177) |
|      Tensor Parallel     |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|     Pipeline Parallel    |     ✅    |     ✅     |             |                   | Basic functions available | Need fully test  |
|      Expert Parallel     |     ❌    |     ✅     |             |                   | Basic functions available | Need fully test; No plan on V0 support  |
|      Data Parallel       |     ❌    |     ✅     |             |                   | Basic functions available | Need fully test;  No plan on V0 support |
| Prefill Decode Disaggregation | ✅   |     ❌     |             |                   |      1P1D available       | working on xPyD and V1 support. |
|       Quantization       |     ✅    |     ✅     |             |                   |      W8A8 available       | Need fully test; working on more quantization method support |
|        Graph Mode        |     ❌    |     ✅     |             |                   |      ACLGraph avaiable    | Need fully test; Rely on CANN 8.1 and torch-npu new version release |
|        Sleep Mode        |     ✅    |     ❌     |             |                   |      level=1 avaiable    | Need fully test; working on V1 support |
