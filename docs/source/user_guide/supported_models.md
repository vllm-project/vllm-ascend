# Supported Models

| Model | Supported | Note |
|---------|-----------|------|
| DeepSeek v3 | ✅|||
| DeepSeek R1 | ✅|||
| DeepSeek Distill (Qwen/LLama) |✅||
| Qwen3 | ✅ ||
| Qwen3-Moe | ✅ ||
| Qwen2-VL | ✅ ||
| Qwen2-Audio | ✅ ||
| Qwen2.5 | ✅ ||
| Qwen2.5-VL | ✅ ||
| QwQ-32B | ✅ ||
| MiniCPM |✅| |
| LLama3.1/3.2 | ✅ ||
| Internlm | ✅ ||
| InternVL2 | ✅ ||
| InternVL2.5 | ✅ ||
| Molmo | ✅ ||
| LLaVA 1.5 | ✅ ||
| LLaVA 1.6 | ✅ |[#553](https://github.com/vllm-project/vllm-ascend/issues/553)|
| Baichuan | ✅ ||
| Phi-4-mini | ✅ ||
| XLM-RoBERTa-based | ✅ ||
| MiniCPM3 | ✅ ||
| Gemma-3 | ❌ |[#496](https://github.com/vllm-project/vllm-ascend/issues/496)|
| ChatGLM | ❌ | [#554](https://github.com/vllm-project/vllm-ascend/issues/554)|
| LLama4 | ❌ |[#471](https://github.com/vllm-project/vllm-ascend/issues/471)|
| Mllama |  |Need test|
| LLaVA-Next |  |Need test|
| LLaVA-Next-Video |  |Need test|
| Phi-3-Vison/Phi-3.5-Vison |  |Need test|
| Ultravox |  |Need test|
| Mistral |  | Need test |
| DeepSeek v2.5 | |Need test |
| Gemma-2 |  |Need test|
| GLM-4v |  |Need test|

## List of Text-only Language Models

### Generative Models

#### Text Generation

Specified using `--task generate`.

:::{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `AquilaForCausalLM`
  * Aquila, Aquila2
  * `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.
  * 
  * 
- * `ArcticForCausalLM`
  * Arctic
  * `Snowflake/snowflake-arctic-base`, `Snowflake/snowflake-arctic-instruct`, etc.
  *
  * 
- * `BaiChuanForCausalLM`
  * Baichuan2, Baichuan
  * `baichuan-inc/Baichuan2-13B-Chat`, `baichuan-inc/Baichuan-7B`, etc.
  * ✅
  * 
- * `BambaForCausalLM`
  * Bamba
  * `ibm-ai-platform/Bamba-9B-fp8`, `ibm-ai-platform/Bamba-9B`
  *
  *
- * `BloomForCausalLM`
  * BLOOM, BLOOMZ, BLOOMChat
  * `bigscience/bloom`, `bigscience/bloomz`, etc.
  *
  * 
- * `BartForConditionalGeneration`
  * BART
  * `facebook/bart-base`, `facebook/bart-large-cnn`, etc.
  *
  *
- * `ChatGLMModel`, `ChatGLMForConditionalGeneration`
  * ChatGLM
  * `THUDM/chatglm2-6b`, `THUDM/chatglm3-6b`, `ShieldLM-6B-chatglm3`, etc.
  * ❌
  * [#554](https://github.com/vllm-project/vllm-ascend/issues/554) 
- * `CohereForCausalLM`, `Cohere2ForCausalLM`
  * Command-R
  * `CohereForAI/c4ai-command-r-v01`, `CohereForAI/c4ai-command-r7b-12-2024`, etc.
  * 
  * 
- * `DbrxForCausalLM`
  * DBRX
  * `databricks/dbrx-base`, `databricks/dbrx-instruct`, etc.
  *
  * 
- * `DeciLMForCausalLM`
  * DeciLM
  * `nvidia/Llama-3_3-Nemotron-Super-49B-v1`, etc.
  *
  * 
- * `DeepseekForCausalLM`
  * DeepSeek
  * `deepseek-ai/deepseek-llm-67b-base`, `deepseek-ai/deepseek-llm-7b-chat` etc.
  *
  * 
- * `DeepseekV2ForCausalLM`
  * DeepSeek-V2
  * `deepseek-ai/DeepSeek-V2`, `deepseek-ai/DeepSeek-V2-Chat` etc.
  *
  * 
- * `DeepseekV3ForCausalLM`
  * DeepSeek-V3
  * `deepseek-ai/DeepSeek-V3-Base`, `deepseek-ai/DeepSeek-V3` etc.
  * ✅
  * 
- * `ExaoneForCausalLM`
  * EXAONE-3
  * `LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct`, etc.
  * 
  * 
- * `FalconForCausalLM`
  * Falcon
  * `tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.
  *
  * 
- * `FalconMambaForCausalLM`
  * FalconMamba
  * `tiiuae/falcon-mamba-7b`, `tiiuae/falcon-mamba-7b-instruct`, etc.
  * 
  * 
- * `GemmaForCausalLM`
  * Gemma
  * `google/gemma-2b`, `google/gemma-1.1-2b-it`, etc.
  * 
  * 
- * `Gemma2ForCausalLM`
  * Gemma 2
  * `google/gemma-2-9b`, `google/gemma-2-27b`, etc.
  * 
  * Need test
- * `Gemma3ForCausalLM`
  * Gemma 3
  * `google/gemma-3-1b-it`, etc.
  * ❌
  * [#496](https://github.com/vllm-project/vllm-ascend/issues/496)
- * `GlmForCausalLM`
  * GLM-4
  * `THUDM/glm-4-9b-chat-hf`, etc.
  * 
  * Need test
- * `Glm4ForCausalLM`
  * GLM-4-0414
  * `THUDM/GLM-4-32B-0414`, etc.
  * 
  * 
- * `GPT2LMHeadModel`
  * GPT-2
  * `gpt2`, `gpt2-xl`, etc.
  *
  * 
- * `GPTBigCodeForCausalLM`
  * StarCoder, SantaCoder, WizardCoder
  * `bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, `WizardLM/WizardCoder-15B-V1.0`, etc.
  * 
  * 
- * `GPTJForCausalLM`
  * GPT-J
  * `EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.
  *
  * 
- * `GPTNeoXForCausalLM`
  * GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
  * `EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, `OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.
  *
  * 
- * `GraniteForCausalLM`
  * Granite 3.0, Granite 3.1, PowerLM
  * `ibm-granite/granite-3.0-2b-base`, `ibm-granite/granite-3.1-8b-instruct`, `ibm/PowerLM-3b`, etc.
  * 
  * 
- * `GraniteMoeForCausalLM`
  * Granite 3.0 MoE, PowerMoE
  * `ibm-granite/granite-3.0-1b-a400m-base`, `ibm-granite/granite-3.0-3b-a800m-instruct`, `ibm/PowerMoE-3b`, etc.
  * 
  * 
- * `GraniteMoeHybridForCausalLM`
  * Granite 4.0 MoE Hybrid
  * `ibm-granite/granite-4.0-tiny-preview`, etc.
  * 
  * 
- * `GraniteMoeSharedForCausalLM`
  * Granite MoE Shared
  * `ibm-research/moe-7b-1b-active-shared-experts` (test model)
  * 
  * 
- * `GritLM`
  * GritLM
  * `parasail-ai/GritLM-7B-vllm`.
  * 
  * 
- * `Grok1ModelForCausalLM`
  * Grok1
  * `hpcai-tech/grok-1`
  * 
  *
- * `InternLMForCausalLM`
  * InternLM
  * `internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.
  * ✅
  * 
- * `InternLM2ForCausalLM`
  * InternLM2
  * `internlm/internlm2-7b`, `internlm/internlm2-chat-7b`, etc.
  * 
  * 
- * `InternLM3ForCausalLM`
  * InternLM3
  * `internlm/internlm3-8b-instruct`, etc.
  * 
  * 
- * `JAISLMHeadModel`
  * Jais
  * `inceptionai/jais-13b`, `inceptionai/jais-13b-chat`, `inceptionai/jais-30b-v3`, `inceptionai/jais-30b-chat-v3`, etc.
  *
  * 
- * `JambaForCausalLM`
  * Jamba
  * `ai21labs/AI21-Jamba-1.5-Large`, `ai21labs/AI21-Jamba-1.5-Mini`, `ai21labs/Jamba-v0.1`, etc.
  * 
  * 
- * `LlamaForCausalLM`
  * Llama 3.1, Llama 3, Llama 2, LLaMA, Yi
  * `meta-llama/Meta-Llama-3.1-405B-Instruct`, `meta-llama/Meta-Llama-3.1-70B`, `meta-llama/Meta-Llama-3-70B-Instruct`, `meta-llama/Llama-2-70b-hf`, `01-ai/Yi-34B`, etc.
  * ✅
  * 
- * `MambaForCausalLM`
  * Mamba
  * `state-spaces/mamba-130m-hf`, `state-spaces/mamba-790m-hf`, `state-spaces/mamba-2.8b-hf`, etc.
  *
  * 
- * `MiniCPMForCausalLM`
  * MiniCPM
  * `openbmb/MiniCPM-2B-sft-bf16`, `openbmb/MiniCPM-2B-dpo-bf16`, `openbmb/MiniCPM-S-1B-sft`, etc.
  * ✅
  * 
- * `MiniCPM3ForCausalLM`
  * MiniCPM3
  * `openbmb/MiniCPM3-4B`, etc.
  * ✅ 
  * 
- * `MistralForCausalLM`
  * Mistral, Mistral-Instruct
  * `mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.
  * 
  * Need test
- * `MixtralForCausalLM`
  * Mixtral-8x7B, Mixtral-8x7B-Instruct
  * `mistralai/Mixtral-8x7B-v0.1`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistral-community/Mixtral-8x22B-v0.1`, etc.
  * 
  * 
- * `MPTForCausalLM`
  * MPT, MPT-Instruct, MPT-Chat, MPT-StoryWriter
  * `mosaicml/mpt-7b`, `mosaicml/mpt-7b-storywriter`, `mosaicml/mpt-30b`, etc.
  *
  * 
- * `NemotronForCausalLM`
  * Nemotron-3, Nemotron-4, Minitron
  * `nvidia/Minitron-8B-Base`, `mgoin/Nemotron-4-340B-Base-hf-FP8`, etc.
  * 
  * 
- * `OLMoForCausalLM`
  * OLMo
  * `allenai/OLMo-1B-hf`, `allenai/OLMo-7B-hf`, etc.
  *
  * 
- * `OLMo2ForCausalLM`
  * OLMo2
  * `allenai/OLMo2-7B-1124`, etc.
  *
  * 
- * `OLMoEForCausalLM`
  * OLMoE
  * `allenai/OLMoE-1B-7B-0924`, `allenai/OLMoE-1B-7B-0924-Instruct`, etc.
  * 
  * 
- * `OPTForCausalLM`
  * OPT, OPT-IML
  * `facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.
  *
  * 
- * `OrionForCausalLM`
  * Orion
  * `OrionStarAI/Orion-14B-Base`, `OrionStarAI/Orion-14B-Chat`, etc.
  *
  * 
- * `PhiForCausalLM`
  * Phi
  * `microsoft/phi-1_5`, `microsoft/phi-2`, etc.
  * 
  * 
- * `Phi3ForCausalLM`
  * Phi-4, Phi-3
  * `microsoft/Phi-4-mini-instruct`, `microsoft/Phi-4`, `microsoft/Phi-3-mini-4k-instruct`, `microsoft/Phi-3-mini-128k-instruct`, `microsoft/Phi-3-medium-128k-instruct`, etc.
  * ✅
  * 
- * `Phi3SmallForCausalLM`
  * Phi-3-Small
  * `microsoft/Phi-3-small-8k-instruct`, `microsoft/Phi-3-small-128k-instruct`, etc.
  *
  * 
- * `PhiMoEForCausalLM`
  * Phi-3.5-MoE
  * `microsoft/Phi-3.5-MoE-instruct`, etc.
  * 
  * 
- * `PersimmonForCausalLM`
  * Persimmon
  * `adept/persimmon-8b-base`, `adept/persimmon-8b-chat`, etc.
  *
  * 
- * `Plamo2ForCausalLM`
  * PLaMo2
  * `pfnet/plamo-2-1b`, `pfnet/plamo-2-8b`, etc.
  *
  *
- * `QWenLMHeadModel`
  * Qwen
  * `Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.
  * 
  * 
- * `Qwen2ForCausalLM`
  * QwQ, Qwen2
  * `Qwen/QwQ-32B-Preview`, `Qwen/Qwen2-7B-Instruct`, `Qwen/Qwen2-7B`, etc.
  * 
  * 
- * `Qwen2MoeForCausalLM`
  * Qwen2MoE
  * `Qwen/Qwen1.5-MoE-A2.7B`, `Qwen/Qwen1.5-MoE-A2.7B-Chat`, etc.
  *
  * 
- * `Qwen3ForCausalLM`
  * Qwen3
  * `Qwen/Qwen3-8B`, etc.
  * 
  * 
- * `Qwen3MoeForCausalLM`
  * Qwen3MoE
  * `Qwen/Qwen3-30B-A3B`, etc.
  *
  * 
- * `StableLmForCausalLM`
  * StableLM
  * `stabilityai/stablelm-3b-4e1t`, `stabilityai/stablelm-base-alpha-7b-v2`, etc.
  *
  * 
- * `Starcoder2ForCausalLM`
  * Starcoder2
  * `bigcode/starcoder2-3b`, `bigcode/starcoder2-7b`, `bigcode/starcoder2-15b`, etc.
  *
  * 
- * `SolarForCausalLM`
  * Solar Pro
  * `upstage/solar-pro-preview-instruct`, etc.
  * 
  * 
- * `TeleChat2ForCausalLM`
  * TeleChat2
  * `Tele-AI/TeleChat2-3B`, `Tele-AI/TeleChat2-7B`, `Tele-AI/TeleChat2-35B`, etc.
  * 
  * 
- * `TeleFLMForCausalLM`
  * TeleFLM
  * `CofeAI/FLM-2-52B-Instruct-2407`, `CofeAI/Tele-FLM`, etc.
  * 
  * 
- * `XverseForCausalLM`
  * XVERSE
  * `xverse/XVERSE-7B-Chat`, `xverse/XVERSE-13B-Chat`, `xverse/XVERSE-65B-Chat`, etc.
  * 
  * 
- * `MiniMaxText01ForCausalLM`
  * MiniMax-Text
  * `MiniMaxAI/MiniMax-Text-01`, etc.
  *
  * 
- * `Zamba2ForCausalLM`
  * Zamba2
  * `Zyphra/Zamba2-7B-instruct`, `Zyphra/Zamba2-2.7B-instruct`, `Zyphra/Zamba2-1.2B-instruct`, etc.
  *
  *
:::

### Pooling Models

:::{important}
Since some model architectures support both generative and pooling tasks,
you should explicitly specify the task type to ensure that the model is used in pooling mode instead of generative mode.
:::

#### Text Embedding

Specified using `--task embed`.

:::{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `BertModel`
  * BERT-based
  * `BAAI/bge-base-en-v1.5`, `Snowflake/snowflake-arctic-embed-xs`, etc.
  *
  *
- * `Gemma2Model`
  * Gemma 2-based
  * `BAAI/bge-multilingual-gemma2`, etc.
  *
  * 
- * `GritLM`
  * GritLM
  * `parasail-ai/GritLM-7B-vllm`.
  * 
  * 
- * `GteModel`
  * GteModel
  * `Snowflake/snowflake-arctic-embed-m-v2.0`.
  *
  * ︎
- * `NomicBertModel`
  * NomicBertModel
  * `nomic-ai/nomic-embed-text-v1`, `nomic-ai/nomic-embed-text-v2-moe`, `Snowflake/snowflake-arctic-embed-m-long`, etc.
  * ︎
  * ︎
- * `LlamaModel`, `LlamaForCausalLM`, `MistralModel`, etc.
  * Llama-based
  * `intfloat/e5-mistral-7b-instruct`, etc.
  * 
  * 
- * `Qwen2Model`, `Qwen2ForCausalLM`
  * Qwen2-based
  * `ssmits/Qwen2-7B-Instruct-embed-base` (see note), `Alibaba-NLP/gte-Qwen2-7B-instruct` (see note), etc.
  * 
  * 
- * `RobertaModel`, `RobertaForMaskedLM`
  * RoBERTa-based
  * `sentence-transformers/all-roberta-large-v1`, etc.
  *
  *
- * `XLMRobertaModel`
  * XLM-RoBERTa-based
  * `intfloat/multilingual-e5-large`, `jinaai/jina-reranker-v2-base-multilingual`, `Snowflake/snowflake-arctic-embed-l-v2.0`, `jinaai/jina-embeddings-v3`(see note), etc.
  * ✅
  *
:::


#### Reward Modeling

Specified using `--task reward`.

:::{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `InternLM2ForRewardModel`
  * InternLM2-based
  * `internlm/internlm2-1_8b-reward`, `internlm/internlm2-7b-reward`, etc.
  * 
  * 
- * `LlamaForCausalLM`
  * Llama-based
  * `peiyi9979/math-shepherd-mistral-7b-prm`, etc.
  * 
  * 
- * `Qwen2ForRewardModel`
  * Qwen2-based
  * `Qwen/Qwen2.5-Math-RM-72B`, etc.
  * 
  * 
- * `Qwen2ForProcessRewardModel`
  * Qwen2-based
  * `Qwen/Qwen2.5-Math-PRM-7B`, `Qwen/Qwen2.5-Math-PRM-72B`, etc.
  * 
  * 
:::

#### Classification

Specified using `--task classify`.

:::{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `JambaForSequenceClassification`
  * Jamba
  * `ai21labs/Jamba-tiny-reward-dev`, etc.
  * 
  * 
- * `Qwen2ForSequenceClassification`
  * Qwen2-based
  * `jason9693/Qwen2.5-1.5B-apeach`, etc.
  * 
  * 
:::

#### Sentence Pair Scoring

Specified using `--task score`.

:::{list-table}
:widths: 25 25 50 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `BertForSequenceClassification`
  * BERT-based
  * `cross-encoder/ms-marco-MiniLM-L-6-v2`, etc.
  *
  *
- * `RobertaForSequenceClassification`
  * RoBERTa-based
  * `cross-encoder/quora-roberta-base`, etc.
  *
  *
- * `XLMRobertaForSequenceClassification`
  * XLM-RoBERTa-based
  * `BAAI/bge-reranker-v2-m3`, etc.
  * ✅
  *
- * `ModernBertForSequenceClassification`
  * ModernBert-based
  * `Alibaba-NLP/gte-reranker-modernbert-base`, etc.
  *
  *
:::

## List of Multimodal Language Models

The following modalities are supported depending on the model:

- **T**ext
- **I**mage
- **V**ideo
- **A**udio

### Generative Models
#### Text Generation

Specified using `--task generate`.

:::{list-table}
:widths: 25 25 15 20 5 5
:header-rows: 1

- * Architecture
  * Models
  * Inputs
  * Example HF Models
  * Supported
  * Note
- * `AriaForConditionalGeneration`
  * Aria
  * T + I<sup>+</sup>
  * `rhymes-ai/Aria`
  *
  * 
- * `AyaVisionForConditionalGeneration`
  * Aya Vision
  * T + I<sup>+</sup>
  * `CohereForAI/aya-vision-8b`, `CohereForAI/aya-vision-32b`, etc.
  *
  * 
- * `Blip2ForConditionalGeneration`
  * BLIP-2
  * T + I<sup>E</sup>
  * `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-opt-6.7b`, etc.
  *
  * 
- * `ChameleonForConditionalGeneration`
  * Chameleon
  * T + I
  * `facebook/chameleon-7b` etc.
  *
  * 
- * `DeepseekVLV2ForCausalLM`<sup>^</sup>
  * DeepSeek-VL2
  * T + I<sup>+</sup>
  * `deepseek-ai/deepseek-vl2-tiny`, `deepseek-ai/deepseek-vl2-small`, `deepseek-ai/deepseek-vl2` etc.
  *
  * 
- * `Florence2ForConditionalGeneration`
  * Florence-2
  * T + I
  * `microsoft/Florence-2-base`, `microsoft/Florence-2-large` etc.
  *
  *
- * `FuyuForCausalLM`
  * Fuyu
  * T + I
  * `adept/fuyu-8b` etc.
  *
  * 
- * `Gemma3ForConditionalGeneration`
  * Gemma 3
  * T + I<sup>+</sup>
  * `google/gemma-3-4b-it`, `google/gemma-3-27b-it`, etc.
  * 
  * 
- * `GLM4VForCausalLM`<sup>^</sup>
  * GLM-4V
  * T + I
  * `THUDM/glm-4v-9b`, `THUDM/cogagent-9b-20241220` etc.
  * 
  * 
- * `GraniteSpeechForConditionalGeneration`
  * Granite Speech
  * T + A
  * `ibm-granite/granite-speech-3.3-8b`
  * 
  * 
- * `H2OVLChatModel`
  * H2OVL
  * T + I<sup>E+</sup>
  * `h2oai/h2ovl-mississippi-800m`, `h2oai/h2ovl-mississippi-2b`, etc.
  *
  * 
- * `Idefics3ForConditionalGeneration`
  * Idefics3
  * T + I
  * `HuggingFaceM4/Idefics3-8B-Llama3` etc.
  * 
  *
- * `InternVLChatModel`
  * InternVL 3.0, InternVideo 2.5, InternVL 2.5, Mono-InternVL, InternVL 2.0
  * T + I<sup>E+</sup>
  * `OpenGVLab/InternVL3-9B`, `OpenGVLab/InternVideo2_5_Chat_8B`, `OpenGVLab/InternVL2_5-4B`, `OpenGVLab/Mono-InternVL-2B`, `OpenGVLab/InternVL2-4B`, etc.
  * ✅
  * 
- * `KimiVLForConditionalGeneration`
  * Kimi-VL-A3B-Instruct, Kimi-VL-A3B-Thinking
  * T + I<sup>+</sup>
  * `moonshotai/Kimi-VL-A3B-Instruct`, `moonshotai/Kimi-VL-A3B-Thinking`
  *
  *
- * `Llama4ForConditionalGeneration`
  * Llama 4
  * T + I<sup>+</sup>
  * `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`, `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, etc.
  * ❌
  * [#471](https://github.com/vllm-project/vllm-ascend/issues/471)
- * `LlavaForConditionalGeneration`
  * LLaVA-1.5
  * T + I<sup>E+</sup>
  * `llava-hf/llava-1.5-7b-hf`, `TIGER-Lab/Mantis-8B-siglip-llama3` (see note), etc.
  * ✅
  * 
- * `LlavaNextForConditionalGeneration`
  * LLaVA-NeXT
  * T + I<sup>E+</sup>
  * `llava-hf/llava-v1.6-mistral-7b-hf`, `llava-hf/llava-v1.6-vicuna-7b-hf`, etc.
  * 
  * Need test
- * `LlavaNextVideoForConditionalGeneration`
  * LLaVA-NeXT-Video
  * T + V
  * `llava-hf/LLaVA-NeXT-Video-7B-hf`, etc.
  *
  * Need test
- * `LlavaOnevisionForConditionalGeneration`
  * LLaVA-Onevision
  * T + I<sup>+</sup> + V<sup>+</sup>
  * `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `llava-hf/llava-onevision-qwen2-0.5b-ov-hf`, etc.
  *
  * 
- * `MiniCPMO`
  * MiniCPM-O
  * T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>E+</sup>
  * `openbmb/MiniCPM-o-2_6`, etc.
  * 
  * 
- * `MiniCPMV`
  * MiniCPM-V
  * T + I<sup>E+</sup> + V<sup>E+</sup>
  * `openbmb/MiniCPM-V-2` (see note), `openbmb/MiniCPM-Llama3-V-2_5`, `openbmb/MiniCPM-V-2_6`, etc.
  * 
  * 
- * `MiniMaxVL01ForConditionalGeneration`
  * MiniMax-VL
  * T + I<sup>E+</sup>
  * `MiniMaxAI/MiniMax-VL-01`, etc.
  *
  * 
- * `Mistral3ForConditionalGeneration`
  * Mistral3
  * T + I<sup>+</sup>
  * `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, etc.
  * 
  * 
- * `MllamaForConditionalGeneration`
  * Llama 3.2
  * T + I<sup>+</sup>
  * `meta-llama/Llama-3.2-90B-Vision-Instruct`, `meta-llama/Llama-3.2-11B-Vision`, etc.
  * ✅
  *
- * `MolmoForCausalLM`
  * Molmo
  * T + I<sup>+</sup>
  * `allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`, etc.
  * ✅
  * 
- * `NVLM_D_Model`
  * NVLM-D 1.0
  * T + I<sup>+</sup>
  * `nvidia/NVLM-D-72B`, etc.
  *
  * 
- * `Ovis`
  * Ovis2, Ovis1.6
  * T + I<sup>+</sup>
  * `AIDC-AI/Ovis2-1B`, `AIDC-AI/Ovis1.6-Llama3.2-3B`, etc.
  *
  *
- * `PaliGemmaForConditionalGeneration`
  * PaliGemma, PaliGemma 2
  * T + I<sup>E</sup>
  * `google/paligemma-3b-pt-224`, `google/paligemma-3b-mix-224`, `google/paligemma2-3b-ft-docci-448`, etc.
  *
  * 
- * `Phi3VForCausalLM`
  * Phi-3-Vision, Phi-3.5-Vision
  * T + I<sup>E+</sup>
  * `microsoft/Phi-3-vision-128k-instruct`, `microsoft/Phi-3.5-vision-instruct`, etc.
  *
  * 
- * `Phi4MMForCausalLM`
  * Phi-4-multimodal
  * T + I<sup>+</sup> / T + A<sup>+</sup> / I<sup>+</sup> + A<sup>+</sup>
  * `microsoft/Phi-4-multimodal-instruct`, etc.
  * 
  *
- * `PixtralForConditionalGeneration`
  * Pixtral
  * T + I<sup>+</sup>
  * `mistralai/Mistral-Small-3.1-24B-Instruct-2503`, `mistral-community/pixtral-12b`, etc.
  *
  * 
- * `QwenVLForConditionalGeneration`<sup>^</sup>
  * Qwen-VL
  * T + I<sup>E+</sup>
  * `Qwen/Qwen-VL`, `Qwen/Qwen-VL-Chat`, etc.
  * 
  * 
- * `Qwen2AudioForConditionalGeneration`
  * Qwen2-Audio
  * T + A<sup>+</sup>
  * `Qwen/Qwen2-Audio-7B-Instruct`
  * ✅
  * 
- * `Qwen2VLForConditionalGeneration`
  * QVQ, Qwen2-VL
  * T + I<sup>E+</sup> + V<sup>E+</sup>
  * `Qwen/QVQ-72B-Preview`, `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, etc.
  * ✅
  * 
- * `Qwen2_5_VLForConditionalGeneration`
  * Qwen2.5-VL
  * T + I<sup>E+</sup> + V<sup>E+</sup>
  * `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, etc.
  * ✅
  * 
- * `Qwen2_5OmniThinkerForConditionalGeneration`
  * Qwen2.5-Omni
  * T + I<sup>E+</sup> + V<sup>E+</sup> + A<sup>+</sup>
  * `Qwen/Qwen2.5-Omni-7B`
  *
  * 
- * `SkyworkR1VChatModel`
  * Skywork-R1V-38B
  * T + I
  * `Skywork/Skywork-R1V-38B`
  *
  * 
- * `SmolVLMForConditionalGeneration`
  * SmolVLM2
  * T + I
  * `SmolVLM2-2.2B-Instruct`
  *
  * 
- * `UltravoxModel`
  * Ultravox
  * T + A<sup>E+</sup>
  * `fixie-ai/ultravox-v0_5-llama-3_2-1b`
  * 
  * Need test
:::

### Pooling Models
#### Text Embedding

Specified using `--task embed`.

:::{list-table}
:widths: 25 25 15 25 5 5
:header-rows: 1

- * Architecture
  * Models
  * Inputs
  * Example HF Models
  * Supported
  * Note
- * `LlavaNextForConditionalGeneration`
  * LLaVA-NeXT-based
  * T / I
  * `royokong/e5-v`
  *
  * 
- * `Phi3VForCausalLM`
  * Phi-3-Vision-based
  * T + I
  * `TIGER-Lab/VLM2Vec-Full`
  * 
  * 
- * `Qwen2VLForConditionalGeneration`
  * Qwen2-VL-based
  * T + I
  * `MrLight/dse-qwen2-2b-mrl-v1`
  *
  * 
:::

#### Transcription

Specified using `--task transcription`.

:::{list-table}
:widths: 25 25 25 5 5
:header-rows: 1

- * Architecture
  * Models
  * Example HF Models
  * Supported
  * Note
- * `Whisper`
  * Whisper-based
  * `openai/whisper-large-v3-turbo`
  * 
  * 
:::