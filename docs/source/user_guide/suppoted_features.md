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
|      Multi Modality      |     ✅    |             |         ✅        | Basic functions available(LLaVA/Qwen2-vl/Qwen2-audio/internVL)| Improve performance, and add more models support |
|        LogProbs          |     ✅    |             |                   | Basic functions available |   Need fully test  |
|     Prompt logProbs      |     ✅    |             |                   | Basic functions available |   Need fully test  |
|       Async output       |     ✅    |             |                   | Basic functions available |   Need fully test  |
|   Multi step scheduler   |     ✅    |             |                   | Basic functions available |   Need fully test  |
|          Best of         |     ✅    |             |                   | Basic functions available |   Need fully test  |
|        Beam search       |     ✅    |             |                   | Basic functions available |   Need fully test  |
|      Guided Decoding     |     ✅    |             |                   | Basic functions available | Find more details at the [<u>issue</u>](https://github.com/vllm-project/vllm-ascend/issues/177) |
|      Tensor Parallel     |     ✅    |             |                   | Basic functions available |   Need fully test  |
|     Pipeline Parallel    |     ✅    |             |                   | Basic functions available |   Need fully test  |
