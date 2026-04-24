# Supported Models

Get the latest info here: https://github.com/vllm-project/vllm-ascend/issues/1608

## Text-Only Language Models

### Generative Models

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| DeepSeek V4               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ ||| ✅ |✅| ✅ || ✅ | ✅ | ✅ || ✅ | 1M || [DeepSeek-V4](../../tutorials/DeepSeek-V4.md) |
| DeepSeek V3/3.1               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k || [DeepSeek-V3.1](../../tutorials/DeepSeek-V3.1.md) |
| DeepSeek V3.2                 | 🔵        | Experimental                                                         | ✅ | A2/A3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/DeepSeek-V3.2.md) |
| DeepSeek R1                   | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k || [DeepSeek-R1](../../tutorials/DeepSeek-R1.md) |
| DeepSeek Distill (Qwen/Llama) | ✅        |                                                                      || A2/A3 |||||||||||||||||
| Qwen3                         | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ ||| ✅ || ✅ | ✅ | 128k | ✅ | [Qwen3-Dense](../../tutorials/Qwen3-Dense.md) |
| Qwen3-based                   | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Qwen3-Coder                   | ✅        |                                                                      | ✅ | A2/A3 ||✅|✅|✅|||✅|✅|✅|✅||||||[Qwen3-Coder-30B-A3B tutorial](../../tutorials/Qwen3-Coder-30B-A3B.md)|
| Qwen3-Moe                     | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | 256k || [Qwen3-235B-A22B](../../tutorials/Qwen3-235B-A22B.md) |
| Qwen3-Next                    | 🔵        | Experimental                                                         | ✅ | A2/A3 | ✅ |||||| ✅ ||| ✅ || ✅ | ✅ ||| [Qwen3-Next](../../tutorials/Qwen3-Next.md) |
| Qwen2.5                       | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ |||| ✅ ||| ✅ |||||| [Qwen2.5-7B](../../tutorials/Qwen2.5-7B.md) |
| Qwen2                         | ✅        |                                                                      || A2/A3 |||||||||||||||||
| Qwen2-based                   | ✅        |                                                                      || A2/A3 |||||||||||||||||
| QwQ-32B                       | ✅        |                                                                      || A2/A3 |||||||||||||||||
| Llama2/3/3.1/3.2              | ✅        |                                                                      || A2/A3 |||||||||||||||||
| Internlm                      | 🔵        | [#1962](https://github.com/vllm-project/vllm-ascend/issues/1962)     || A2/A3 |||||||||||||||||
| Baichuan                      | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Baichuan2                     | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Phi-4-mini                    | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| MiniCPM                       | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| MiniCPM3                      | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Ernie4.5                      | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Ernie4.5-Moe                  | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Gemma-2                       | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Gemma-3                       | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Phi-3/4                       | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Mistral/Mistral-Instruct      | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| GLM-4.x                       | 🔵        | Experimental                                                         || A2/A3 |✅|✅|✅||✅|✅|✅|||✅||✅|✅|128k||../../tutorials/GLM4.x.md|
| Kimi-K2-Thinking              | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||| [Kimi-K2-Thinking](../../tutorials/Kimi-K2-Thinking.md) |
| GLM-4                         | ❌        | [#2255](https://github.com/vllm-project/vllm-ascend/issues/2255)     |||||||||||||||||||
| GLM-4-0414                    | ❌        | [#2258](https://github.com/vllm-project/vllm-ascend/issues/2258)     |||||||||||||||||||
| ChatGLM                       | ❌        | [#554](https://github.com/vllm-project/vllm-ascend/issues/554)       |||||||||||||||||||
| DeepSeek V2.5                 | 🟡        | Need test                                                            |||||||||||||||||||
| Mllama                        | 🟡        | Need test                                                            |||||||||||||||||||
| MiniMax-Text                  | 🟡        | Need test                                                            |||||||||||||||||||

### Pooling Models

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| Qwen3-Embedding               | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||| [Qwen3_embedding](../../tutorials/Qwen3_embedding.md)|
| Qwen3-Reranker                | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||| [Qwen3_reranker](../../tutorials/Qwen3_reranker.md)|
| Molmo                         | 🔵        | [1942](https://github.com/vllm-project/vllm-ascend/issues/1942)      || A2/A3 |||||||||||||||||
| XLM-RoBERTa-based             | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||
| Bert                          | 🔵        | Experimental                                                         || A2/A3 |||||||||||||||||

## Multimodal Language Models

### Generative Models

| Model                          | Support       | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|--------------------------------|---------------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| Qwen2-VL                       | ✅            |                                                                      || A2/A3 |||||||||||||||||
| Qwen2.5-VL                     | ✅            |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ |||| ✅ | ✅ | ✅ | 30k || [Qwen-VL-Dense](../../tutorials/Qwen-VL-Dense.md) |
| Qwen3-VL                       | ✅            |                                                                      ||A2/A3|||||||✅|||||✅|✅||| [Qwen-VL-Dense](../../tutorials/Qwen-VL-Dense.md) |
| Qwen3-VL-MOE                   | ✅            |                                                                      | ✅ | A2/A3||✅|✅|||✅|✅|✅|✅|✅|✅|✅|✅|256k||[Qwen3-VL-MOE](../../tutorials/Qwen3-VL-235B-A22B-Instruct.md)|
| Qwen3-Omni-30B-A3B-Thinking    | 🔵            | Experimental                                                         ||A2/A3|||||||✅||✅|||||||[Qwen3-Omni-30B-A3B-Thinking](../../tutorials/Qwen3-Omni-30B-A3B-Thinking.md)|
| Qwen2.5-Omni                   | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||| [Qwen2.5-Omni](../../tutorials/Qwen2.5-Omni.md) |
| Qwen3-Omni                     | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| QVQ                            | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Qwen2-Audio                    | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Aria                           | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| LLaVA-Next                     | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| LLaVA-Next-Video               | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| MiniCPM-V                      | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Mistral3                       | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Phi-3-Vision/Phi-3.5-Vision    | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Gemma3                         | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Llama3.2                       | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| PaddleOCR-VL                   | 🔵            | Experimental                                                         || A2/A3 |||||||||||||||||
| Llama4                         | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |||||||||||||||||||
| Keye-VL-8B-Preview             | ❌            | [1963](https://github.com/vllm-project/vllm-ascend/issues/1963)      |||||||||||||||||||
| Florence-2                     | ❌            | [2259](https://github.com/vllm-project/vllm-ascend/issues/2259)      |||||||||||||||||||
| GLM-4V                         | ❌            | [2260](https://github.com/vllm-project/vllm-ascend/issues/2260)      |||||||||||||||||||
| InternVL2.0/2.5/3.0<br>InternVideo2.5/Mono-InternVL | ❌ | [2064](https://github.com/vllm-project/vllm-ascend/issues/2064) |||||||||||||||||||
| Whisper                        | ❌            | [2262](https://github.com/vllm-project/vllm-ascend/issues/2262)      |||||||||||||||||||
| Ultravox                       | 🟡            | Need test                                                            |||||||||||||||||||
