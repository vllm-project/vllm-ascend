# A2 PR 测试模型下载清单

## Hugging Face 下载（存入 `~/.cache/huggingface/hub`）

| 模型 | 说明 |
|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | gated 仓库，需要 HF token |
| `jeeejeee/llama32-3b-text2sql-spider` | LoRA 适配器 |

## ModelScope 下载（存入 `~/.cache/modelscope/hub`）

以下仓库 ID 与 ModelScope 一致，直接按 ID 下载即可。

**基础模型**

| 模型 |
|---|
| `Qwen/Qwen3-0.6B` |
| `Qwen/Qwen3-1.7B` |
| `Qwen/Qwen3-8B` |
| `Qwen/Qwen3.5-0.8B` |
| `Qwen/Qwen3.5-4B` |
| `Qwen/Qwen3.5-35B-A3B` |
| `zai-org/GLM-5.1` |
| `vllm-ascend/ilama-3.2-1B` |
| `LLM-Research/Meta-Llama-3.1-8B-Instruct` |
| `allenai/OLMoE-1B-7B-0125-Instruct` |
| `openbmb/MiniCPM-2B-sft-bf16` |
| `OpenBMB/MiniCPM4-0.5B` |

**量化 / 投机解码变体**

| 模型 |
|---|
| `vllm-ascend/Qwen3-8B-W8A8` |
| `vllm-ascend/DeepSeek-V2-Lite-W8A8` |
| `RedHatAI/Qwen3-8B-speculator.eagle3` |
| `vllm-ascend/EAGLE-LLaMA3.1-Instruct-8B` |
| `vllm-ascend/EAGLE3-LLaMA3.1-Instruct-8B` |
| `amd/PARD-Llama-3.2-1B` |
| `z-lab/Qwen3-8B-DFlash-b16` |
| `deepseek-ai/dspark_qwen3_8b_block7` |
| `wemaster/deepseek_mtp_main_random_bf16` |
| `MNN/Qwen3-VL-8B-Instruct-Eagle3` |

**多模态模型**

| 模型 |
|---|---|
| `Qwen/Qwen3-VL-8B-Instruct` |
| `Qwen/Qwen3-VL-4B-Instruct` |
| `Qwen/Qwen2.5-VL-3B-Instruct` |
| `Qwen/Qwen2-VL-2B-Instruct` |
| `Qwen/Qwen2-Audio-7B-Instruct` |
| `openai-mirror/whisper-large-v3-turbo` |

**Pooling 模型**

| 模型 |
|---|
| `Qwen/Qwen3-Embedding-0.6B` |
| `Qwen/Qwen3-Reranker-0.6B` |
| `BAAI/bge-m3` |
| `BAAI/bge-reranker-v2-m3` |
| `intfloat/multilingual-e5-small` |
| `dengcao/ms-marco-MiniLM-L6-v2` |
| `sentence-transformers/all-MiniLM-L12-v2` |
| `Howeee/Qwen2.5-1.5B-apeach` |

**LoRA 适配器**

| 模型 |
|---|
| `vllm-ascend/ilama-text2sql-spider` |
| `vllm-ascend/qwen35-4b-text-only-sql-lora` |
| `vllm-ascend/olmoe-instruct-text2sql-spider` |
| `vllm-ascend/qwen-linear-algebra-coder` |
| `charent/self_cognition_Alice` |
| `charent/self_cognition_Bob` |
| `vllm-ascend/qwen2-vl-lora-pokemon` |
| `vllm-ascend/qwen25-vl-lora-pokemon` |
| `vllm-ascend/qwen2.5-3b-vl-lora-vision-connector` |
| `vllm-ascend/qwen3-4b-vl-lora-vision-connector` |
| `vllm-ascend/qwen2vl-flickr-lora-language` |
| `vllm-ascend/qwen2vl-flickr-lora-tower-connector` |
| `vllm-ascend/qwen2vl-flickr-lora-tower` |
