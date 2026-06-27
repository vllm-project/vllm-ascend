# Supported Models

Get the latest info here: <https://github.com/vllm-project/vllm-ascend/issues/1608>

**Legend Description**:

- ✅ = Supported model/feature
- 🔵 = Experimental supported model/feature
- ❌ = Not supported model/feature
- 🟡 = Not tested or verified

## Text-Only Language Models

### Generative Models

#### Core Supported Models

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| DeepSeek V4-Flash               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ |✅|| ✅ |✅| ✅ || ✅ | ✅ | ✅ || ✅ | 1M || [DeepSeek-V4](../../tutorials/models/DeepSeek-V4-Flash.md) |
| DeepSeek V4-Pro               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ |✅|| ✅ |✅| ✅ || ✅ | ✅ | ✅ || ✅ | 1M || [DeepSeek-V4](../../tutorials/models/DeepSeek-V4-Pro.md) |
| DeepSeek V3/3.1               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k || [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
| DeepSeek V3.2                 | 🔵        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md) |
| DeepSeek R1                   | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k || [DeepSeek-R1](../../tutorials/models/DeepSeek-R1.md) |
| Qwen3                         | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ ||| ✅ || ✅ | ✅ | 128k | ✅ | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |
| Qwen3-Coder                   | ✅        |                                                                      | ✅ | A2/A3 ||✅|✅|✅|||✅|✅|✅|✅||||||[Qwen3-Coder-30B-A3B tutorial](../../tutorials/models/Qwen3-Coder-30B-A3B.md)|
| Qwen3-Moe                     | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | 256k || [Qwen3-235B-A22B](../../tutorials/models/Qwen3-235B-A22B.md) |
| Qwen3-Next                    | 🔵        |                                                                      | ✅ | A2/A3 | ✅ |||||| ✅ ||| ✅ || ✅ | ✅ ||| [Qwen3-Next](../../tutorials/models/Qwen3-Next.md) |
| GLM-4.x                       | 🔵        |                                                                      || A2/A3 |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|198k||[GLM-4.x](../../tutorials/models/GLM4.x.md)|
| GLM-5/5.1                         | 🔵        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k || [GLM-5](../../tutorials/models/GLM5.md) |
| GLM-5.2                         | 🔵        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k || [GLM-5](../../tutorials/models/GLM5.2.md) |
| Kimi-K2-Thinking              | 🔵        |                                                                      || A2/A3 |||||||||||||||| [Kimi-K2-Thinking](../../tutorials/models/Kimi-K2-Thinking.md) |
| DeepseekOCR2                  | ✅        |                                                                      | ✅ | A2/A3 ||✅||||✅|||||||||| [DeepSeekOCR2](../../tutorials/models/DeepSeekOCR2.md) |
| MiniMax-M2.5/2.7                  | ✅        |                                                                      | ✅ | A2/A3/Ascend950 (Ascend950 experimental) |✅|✅|✅|❌|✅|✅|✅|🟡|✅|✅|✅|🟡|✅|200k|🟡| [MiniMax-M2](../../tutorials/models/MiniMax-M2.md) |
| Qwen2.5-Math-RM-72B           | ✅        | vllm-rm, tensor_parallel_size=4, max_model_len=4096 | ✅ | A2 | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 4096 | 🟡 | [Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md) |

#### Extended Compatible Models

| Model                         | Support   | Note                                                                 | Supported Hardware |
|-------------------------------|-----------|----------------------------------------------------------------------|--------------------|
| DeepSeek Distill (Qwen/Llama) | ✅        |                                                                      | A2/A3 |
| Qwen3-based                   | ✅        |                                                                      | A2/A3 |
| Qwen2                         | ✅        |                                                                      | A2/A3 |
| Qwen2.5                       | ✅        |                                                                      | A2/A3 |
| Qwen2-based                   | ✅        |                                                                      | A2/A3 |
| QwQ-32B                       | ✅        |                                                                      | A2/A3 |
| Llama2/3/3.1/3.2              | ✅        |                                                                      | A2/A3 |
| Internlm                      | 🔵        | [#1962](https://github.com/vllm-project/vllm-ascend/issues/1962)     | A2/A3 |
| Baichuan                      | 🔵        |                                                                      | A2/A3 |
| Baichuan2                     | 🔵        |                                                                      | A2/A3 |
| Phi-4-mini                    | 🔵        |                                                                      | A2/A3 |
| MiniCPM                       | 🔵        |                                                                      | A2/A3 |
| MiniCPM3                      | 🔵        |                                                                      | A2/A3 |
| Ernie4.5                      | 🔵        |                                                                      | A2/A3 |
| Ernie4.5-Moe                  | 🔵        |                                                                      | A2/A3 |
| Gemma-2                       | 🔵        |                                                                      | A2/A3 |
| Gemma-3                       | 🔵        |                                                                      | A2/A3 |
| Phi-3/4                       | 🔵        |                                                                      | A2/A3 |
| Mistral/Mistral-Instruct      | 🔵        |                                                                      | A2/A3 |
| Hy3-preview                   | 🔵        |                                                                      | A3    |
| DeepSeek V2.5                 | 🟡        | Need test                                                            |       |
| Mllama                        | 🟡        | Need test                                                            |       |
| MiniMax-Text                  | 🟡        | Need test                                                            |       |

### Pooling Models

| Model                         | Support   | Note                                                                 |    Supported Hardware    |  Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|--------------------------|------|
| Qwen3-Embedding               | 🔵        |                                                                      |         A2/A3            | [Qwen3_embedding](../../tutorials/models/Qwen3_embedding.md)|
| Qwen3-VL-Embedding            | 🔵        |                                                                      |         A2/A3            | [Qwen3-VL-Embedding](../../tutorials/models/Qwen3-VL-Embedding.md)|
| Qwen3-Reranker                | 🔵        |                                                                      |         A2/A3            | [Qwen3_reranker](../../tutorials/models/Qwen3_reranker.md)|
| Qwen3-VL-Reranker             | 🔵        |                                                                      |         A2/A3            | [Qwen3-VL-Reranker](../../tutorials/models/Qwen3-VL-Reranker.md)|
| Molmo                         | 🔵        | [1942](https://github.com/vllm-project/vllm-ascend/issues/1942)      |         A2/A3            |      |
| XLM-RoBERTa-based             | 🔵        |                                                                      |         A2/A3            |      |
| Bert                          | 🔵        |                                                                      |         A2/A3            |      |
| Qwen2.5-Math-RM-72B           | ✅        | Reward Model, gsm8k_correctness accuracy=0.80 | A2 | [Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md) |

## Multimodal Language Models

### Generative Models

#### Core Supported Models

| Model                          | Support       | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|--------------------------------|---------------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| Qwen3-VL                       | ✅            |                                                                      ||A2/A3|||||||✅|||||✅|✅||| [Qwen-VL-Dense](../../tutorials/models/Qwen-VL-Dense.md) |
| Qwen3-VL-MOE                   | ✅            |                                                                      | ✅ | A2/A3||✅|✅|||✅|✅|✅|✅|✅|✅|✅|✅|256k||[Qwen3-VL-MOE](../../tutorials/models/Qwen3-VL-235B-A22B-Instruct.md)|
| Qwen3.5-397B-A17B              | ✅            |                                                                      |✅| A2/A3 |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|1010000|| [Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md) |
| Qwen3.5-27B / Qwen3.6-27B                    | ✅            |                                                                      |✅| A2/A3 |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|262144|| [Qwen3.5-27B / Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) |
| Qwen3.6-35B-A3B                | 🔵            |                                                                      |✅| A2/A3 |✅|✅|✅||🔵|✅|✅||✅|✅|❌|✅|✅|262144|| [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) |
| Qwen3-Omni-30B-A3B-Thinking    | 🔵            |                                                                      ||A2/A3|||||||✅||✅|||||||[Qwen3-Omni-30B-A3B-Thinking](../../tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md)|
| Kimi-K2.5/Kimi-K2.6    | ✅            |                                                                      ||A2/A3||✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|262144||[Kimi-K2.5](../../tutorials/models/Kimi-K2.5.md)/[Kimi-K2.6](../../tutorials/models/Kimi-K2.6.md)|

#### Extended Compatible Models

| Model                          | Support       | Note                                                                 | Supported Hardware |
|--------------------------------|---------------|----------------------------------------------------------------------|--------------------|
| Qwen2-VL                       | ✅            |                                                                      | A2/A3 |
| Qwen3-Omni                     | 🔵            |                                                                      | A2/A3 |
| QVQ                            | 🔵            |                                                                      | A2/A3 |
| Qwen2-Audio                    | 🔵            |                                                                      | A2/A3 |
| Aria                           | 🔵            |                                                                      | A2/A3 |
| LLaVA-Next                     | 🔵            |                                                                      | A2/A3 |
| LLaVA-Next-Video               | 🔵            |                                                                      | A2/A3 |
| MiniCPM-V                      | 🔵            |                                                                      | A2/A3 |
| Mistral3                       | 🔵            |                                                                      | A2/A3 |
| Phi-3-Vision/Phi-3.5-Vision    | 🔵            |                                                                      | A2/A3 |
| Gemma3                         | 🔵            |                                                                      | A2/A3 |
| Llama3.2                       | 🔵            |                                                                      | A2/A3 |
| PaddleOCR-VL                   | 🔵            |                                                                      | A2/A3 |
| Llama4                         | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |       |
| Keye-VL-8B-Preview             | ❌            | [1961](https://github.com/vllm-project/vllm-ascend/issues/1961)      |       |
| Florence-2                     | ❌            | [2259](https://github.com/vllm-project/vllm-ascend/issues/2259)      |       |
| GLM-4V                         | ❌            | [2260](https://github.com/vllm-project/vllm-ascend/issues/2260)      |       |
| InternVL2.0/2.5/3.0<br>InternVideo2.5/Mono-InternVL | ❌ | [2064](https://github.com/vllm-project/vllm-ascend/issues/2064) |  |
| Whisper                        | ❌            | [2262](https://github.com/vllm-project/vllm-ascend/issues/2262)      |       |
| Ultravox                       | 🟡            | Need test                                                            |       |
