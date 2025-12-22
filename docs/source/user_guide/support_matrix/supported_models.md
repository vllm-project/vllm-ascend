# Supported Models

Get the latest info here: https://github.com/vllm-project/vllm-ascend/issues/1608

## Text-Only Language Models

### Generative Models

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| DeepSeek V3/3.1               | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k || [DeepSeek-V3.1](../../tutorials/DeepSeek-V3.1.md) |
| DeepSeek V3.2                 | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | ✅ | [DeepSeek-V3.2](../../tutorials/DeepSeek-V3.2.md) |
| DeepSeek R1                   | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ || ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k || [DeepSeek-R1](../../tutorials/DeepSeek-R1.md) |
| DeepSeek Distill (Qwen/Llama) | ✅        |                                                                      |||||||||||||||||||
| Qwen3                         | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ ||| ✅ || ✅ | ✅ | 128k | ✅ | [Qwen3-Dense](../../tutorials/Qwen3-Dense.md) |
| Qwen3-based                   | ✅        |                                                                      |||||||||||||||||||
| Qwen3-Coder                   | ✅        |                                                                      | ✅ | A2/A3 ||✅|✅|✅|||✅|✅|✅|✅||||||[Qwen3-Coder-30B-A3B tutorial](../../tutorials/Qwen3-Coder-30B-A3B.md)|
| Qwen3-Moe                     | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ || ✅ | ✅ | ✅ | ✅ | ✅ | 256k || [Qwen3-235B-A22B](../../tutorials/Qwen3-235B-A22B.md) |
| Qwen3-Next                    | ✅        |                                                                      | ✅ | A2/A3 | ✅ |||||| ✅ ||| ✅ || ✅ | ✅ ||| [Qwen3-Next](../../tutorials/Qwen3-Next.md) |
| Qwen2.5                       | ✅        |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ |||| ✅ ||| ✅ |||||| [Qwen2.5-7B](../../tutorials/Qwen2.5-7B.md) |
| Qwen2                         | ✅        |                                                                      |||||||||||||||||||
| Qwen2-based                   | ✅        |                                                                      |||||||||||||||||||
| QwQ-32B                       | ✅        |                                                                      |||||||||||||||||||
| Llama2/3/3.1                  | ✅        |                                                                      |||||||||||||||||||
| Internlm                      | ✅        | [#1962](https://github.com/vllm-project/vllm-ascend/issues/1962)     |||||||||||||||||||
| Baichuan                      | ✅        |                                                                      |||||||||||||||||||
| Baichuan2                     | ✅        |                                                                      |||||||||||||||||||
| Phi-4-mini                    | ✅        |                                                                      |||||||||||||||||||
| MiniCPM                       | ✅        |                                                                      |||||||||||||||||||
| MiniCPM3                      | ✅        |                                                                      |||||||||||||||||||
| Ernie4.5                      | ✅        |                                                                      |||||||||||||||||||
| Ernie4.5-Moe                  | ✅        |                                                                      |||||||||||||||||||
| Gemma-2                       | ✅        |                                                                      |||||||||||||||||||
| Gemma-3                       | ✅        |                                                                      |||||||||||||||||||
| Phi-3/4                       | ✅        |                                                                      |||||||||||||||||||
| Mistral/Mistral-Instruct      | ✅        |                                                                      |||||||||||||||||||
| GLM-4.5                       | ✅        |                                                                      |||||||||||||||||||
| GLM-4                         | ❌        | [#2255](https://github.com/vllm-project/vllm-ascend/issues/2255)     |||||||||||||||||||
| GLM-4-0414                    | ❌        | [#2258](https://github.com/vllm-project/vllm-ascend/issues/2258)     |||||||||||||||||||
| ChatGLM                       | ❌        | [#554](https://github.com/vllm-project/vllm-ascend/issues/554)       |||||||||||||||||||
| DeepSeek V2.5                 | 🟡        | Need test                                                            |||||||||||||||||||
| Mllama                        | 🟡        | Need test                                                            |||||||||||||||||||
| MiniMax-Text                  | 🟡        | Need test                                                            |||||||||||||||||||

### Pooling Models

| Model                         | Support   | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|-------------------------------|-----------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| Qwen3-Embedding               | ✅        |                                                                      |||||||||||||||||||
| Qwen3-Reranker                | ✅        |                                                                      |||||||||||||||||||
| Molmo                         | ✅        | [1942](https://github.com/vllm-project/vllm-ascend/issues/1942)      |||||||||||||||||||
| XLM-RoBERTa-based             | ✅        |                                                                      |||||||||||||||||||
| Bert                          | ✅        |                                                                      |||||||||||||||||||

## Multimodal Language Models

### Generative Models

| Model                          | Support       | Note                                                                 | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
|--------------------------------|---------------|----------------------------------------------------------------------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------------|-----|
| Qwen2-VL                       | ✅            |                                                                      |||||||||||||||||||
| Qwen2.5-VL                     | ✅            |                                                                      | ✅ | A2/A3 | ✅ | ✅ | ✅ ||| ✅ | ✅ |||| ✅ | ✅ | ✅ | 30k || [Qwen-VL-Dense](../../tutorials/Qwen-VL-Dense.md) |
| Qwen3-VL                       | ✅            |                                                                      ||A2/A3|||||||✅|||||✅|✅||| [Qwen-VL-Dense](../../tutorials/Qwen-VL-Dense.md) |
| Qwen3-VL-MOE                   | ✅            |                                                                      | ✅ |A2/A3||✅|✅|||✅|✅|✅|✅|✅|✅|✅|✅|256k||[Qwen3-VL-235B-A22B-Instruct](../../tutorials/Qwen3-VL-235B-A22B-Instruct.md)|
| Qwen2.5-Omni                   | ✅            |                                                                      | ✅ | A2/A3 ||||||| ✅ ||||| ✅ | ✅ ||| [Qwen2.5-Omni](../../tutorials/Qwen2.5-Omni.md) |
| QVQ                            | ✅            |                                                                      |||||||||||||||||||
| Qwen2-Audio                    | ✅            |                                                                      |||||||||||||||||||
| Aria                           | ✅            |                                                                      |||||||||||||||||||
| LLaVA-Next                     | ✅            |                                                                      |||||||||||||||||||
| LLaVA-Next-Video               | ✅            |                                                                      |||||||||||||||||||
| MiniCPM-V                      | ✅            |                                                                      |||||||||||||||||||
| Mistral3                       | ✅            |                                                                      |||||||||||||||||||
| Phi-3-Vision/Phi-3.5-Vision      | ✅            |                                                                      |||||||||||||||||||
| Gemma3                         | ✅            |                                                                      |||||||||||||||||||
| Llama4                         | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |||||||||||||||||||
| Llama3.2                       | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |||||||||||||||||||
| Keye-VL-8B-Preview             | ❌            | [1963](https://github.com/vllm-project/vllm-ascend/issues/1963)      |||||||||||||||||||
| Florence-2                     | ❌            | [2259](https://github.com/vllm-project/vllm-ascend/issues/2259)      |||||||||||||||||||
| GLM-4V                         | ❌            | [2260](https://github.com/vllm-project/vllm-ascend/issues/2260)      |||||||||||||||||||
| InternVL2.0/2.5/3.0<br>InternVideo2.5/Mono-InternVL | ❌ | [2064](https://github.com/vllm-project/vllm-ascend/issues/2064) |||||||||||||||||||
| Whisper                        | ❌            | [2262](https://github.com/vllm-project/vllm-ascend/issues/2262)      |||||||||||||||||||
| Ultravox                       | 🟡            | Need test                                                            |||||||||||||||||||
