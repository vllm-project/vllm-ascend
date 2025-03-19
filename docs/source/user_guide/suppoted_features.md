# Feature Support

|           Feature        | Supported | CI Coverage | Guidance Document |     Current Status        |    Next Step       |
|--------------------------|-----------|-------------|-------------------|---------------------------|--------------------|
| Chunked Prefill          |     ❌    |             |                   |          NA               | Plan in 2025.03.30 |
| Automatic Prefix Caching |     ❌    |             |                   |          NA               | Plan in 2025.03.30 |
|          LoRA            |     ❌    |             |                   |          NA               | Plan in 2025.06.30 |
|      Prompt adapter      |     ❌    |             |                   |          NA               | Plan in 2025.06.30 |
|    Speculative decoding  |     ✅    |             |                   | Basic functions available |   Need fully test  |
|        Pooling           |     ✅    |             |                   | Basic functions available(Bert) | Need fully test and add more models support|
|        Enc-dec           |     ❌    |             |                   |          NA               | Plan in 2025.06.30|
|      Multi Modality      |     ✅    |             |         ✅        | Basic functions available(LLaVA/Qwen2-vl/Qwen2-audio/internVL)| Improve perforamance, and add more models support |
|        LogProbs          |     ✅    |             |                   | Basic functions available |   Need fully test  |
|     Prompt logProbs      |     ✅    |             |                   | Basic functions available |   Need fully test  |
|       Async output       |     ✅    |             |                   | Basic functions available |   Need fully test  |
|   Multi step scheduler   |     ✅    |             |                   | Basic functions available |   Need fully test, Find more details at [<u> Blog </u>](https://blog.vllm.ai/2024/09/05/perf-update.html#batch-scheduling-multiple-steps-ahead-pr-7000), [<u> RFC </u>](https://github.com/vllm-project/vllm/issues/6854) and [<u>issue</u>](https://github.com/vllm-project/vllm/pull/7000)  |
|          Best of         |     ✅    |             |                   | Basic functions available |   Need fully test  |
|        Beam search       |     ✅    |             |                   | Basic functions available |   Need fully test  |
|      Guided Decoding     |     ✅    |             |                   | Basic functions available | Find more details at the [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues/177) |
|      Tensor Parallel     |     ✅    |             |                   | Basic functions available |   Need fully test  |
|     Pipeline Parallel    |     ✅    |             |                   | Basic functions available |   Need fully test  |
