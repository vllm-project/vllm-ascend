# Model Support

Get the newest info here: https://github.com/vllm-project/vllm-ascend/issues/1608

## Text-only Language Models

### Generative Models

| Model                         | Supported | Note                                                                 |
|-------------------------------|-----------|----------------------------------------------------------------------|
| DeepSeek v3                   | ✅        |                                                                      |
| DeepSeek R1                   | ✅        |                                                                      |
| DeepSeek Distill (Qwen/LLama) | ✅        |                                                                      |
| Qwen3                         | ✅        |                                                                      |
| Qwen3-based                   | ✅        |                                                                      |
| Qwen3-Coder                   | ✅        |                                                                      |
| Qwen3-Moe                     | ✅        |                                                                      |
| Qwen2.5                       | ✅        |                                                                      |
| Qwen2                         | ✅        |                                                                      |
| Qwen2-based                   | ✅        |                                                                      |
| QwQ-32B                       | ✅        |                                                                      |
| LLama2/3/3.1                  | ✅        |                                                                      |
| Internlm                      | ✅        | [#1962](https://github.com/vllm-project/vllm-ascend/issues/1962)     |
| Baichuan                      | ✅        |                                                                      |
| Baichuan2                     | ✅        |                                                                      |
| Phi-4-mini                    | ✅        |                                                                      |
| MiniCPM                       | ✅        |                                                                      |
| MiniCPM3                      | ✅        |                                                                      |
| Ernie4.5                      | ✅        |                                                                      |
| Ernie4.5-Moe                  | ✅        |                                                                      |
| Gemma-2                       | ✅        |                                                                      |
| Gemma-3                       | ✅        |                                                                      |
| Phi-3/4                       | ✅        |                                                                      |
| Mistral/Mistral-Instruct      | ✅        |                                                                      |
| GLM-4.5                       | ✅            |                                                                  |
| GLM-4                         | ❌        | [#2255](https://github.com/vllm-project/vllm-ascend/issues/2255)     |
| GLM-4-0414                    | ❌        | [#2258](https://github.com/vllm-project/vllm-ascend/issues/2258)     |
| ChatGLM                       | ❌        | [#554](https://github.com/vllm-project/vllm-ascend/issues/554)       |
| DeepSeek v2.5                 | 🟡        | Need test                                                            |
| Mllama                        | 🟡        | Need test                                                            |
| MiniMax-Text                  | 🟡        | Need test                                                            |

### Pooling Models

| Model                         | Supported | Note                                                                 |
|-------------------------------|-----------|----------------------------------------------------------------------|
| Qwen3-Embedding               | ✅        |                                                                      |
| Molmo                         | ✅        | [1942](https://github.com/vllm-project/vllm-ascend/issues/1942)      |
| XLM-RoBERTa-based             | ❌        | [1960](https://github.com/vllm-project/vllm-ascend/issues/1960)      |

## Multimodal Language Models

### Generative Models

| Model                          | Supported     | Note                                                                 |
|--------------------------------|---------------|----------------------------------------------------------------------|
| Qwen2-VL                       | ✅            |                                                                      |
| Qwen2.5-VL                     | ✅            |                                                                      |
| Qwen2.5-Omni                   | ✅            | [1760](https://github.com/vllm-project/vllm-ascend/issues/1760)      |
| QVQ                            | ✅            |                                                                      |
| LLaVA 1.5/1.6                  | ✅            | [1962](https://github.com/vllm-project/vllm-ascend/issues/1962)      |
| InternVL2                      | ✅            |                                                                      |
| InternVL2.5                    | ✅            |                                                                      |
| Qwen2-Audio                    | ✅            |                                                                      |
| Aria                           | ✅            |                                                                      |
| LLaVA-Next                     | ✅            |                                                                      |
| LLaVA-Next-Video               | ✅            |                                                                      |
| MiniCPM-V                      | ✅            |                                                                      |
| Mistral3                       | ✅            |                                                                      |
| Phi-3-Vison/Phi-3.5-Vison      | ✅            |                                                                      |
| Gemma3                         | ✅            |                                                                      |
| LLama4                         | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |
| LLama3.2                       | ❌            | [1972](https://github.com/vllm-project/vllm-ascend/issues/1972)      |
| Keye-VL-8B-Preview             | ❌            | [1963](https://github.com/vllm-project/vllm-ascend/issues/1963)      |
| Florence-2                     | ❌            | [2259](https://github.com/vllm-project/vllm-ascend/issues/2259)      |
| GLM-4V                         | ❌            | [2260](https://github.com/vllm-project/vllm-ascend/issues/2260)      |
| InternVL2.0/2.5/3.0<br>InternVideo2.5/Mono-InternVL | ❌ | [2064](https://github.com/vllm-project/vllm-ascend/issues/2064) |
| Whisper                        | ❌            | [2262](https://github.com/vllm-project/vllm-ascend/issues/2262)      |
| Ultravox                       | 🟡 Need test  |                                                                      |
