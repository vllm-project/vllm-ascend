# vllm_ascend/spec_decode：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **17 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `vllm_ascend/spec_decode/`

草稿生成器、候选组织及投机解码辅助。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../vllm_ascend/spec_decode/__init__.py) | Python 包入口，提供本子模块的初始化、注册或导出接口；定义见右栏。 | 定义：`get_spec_decode_method` |
| [dflash2_proposer.py](../../../vllm_ascend/spec_decode/dflash2_proposer.py) | 候选生成、DFlash 草稿 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendDflash2Proposer`, `is_dflash2_draft`, `greedy_select_path` |
| [dflash_proposer.py](../../../vllm_ascend/spec_decode/dflash_proposer.py) | 候选生成、DFlash 草稿 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendDflashProposer` |
| [draft_proposer.py](../../../vllm_ascend/spec_decode/draft_proposer.py) | 候选生成、草稿模型 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendDraftModelProposer` |
| [dspark_proposer.py](../../../vllm_ascend/spec_decode/dspark_proposer.py) | 候选生成、DSpark 草稿 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendDSparkProposer` |
| [dspark_utils.py](../../../vllm_ascend/spec_decode/dspark_utils.py) | DSpark 草稿、辅助函数 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 注释：Shared DSpark target auxiliary-layer configuration, independent of PD roles.；定义：`get_dspark_aux_layer_ids` |
| [eagle_proposer.py](../../../vllm_ascend/spec_decode/eagle_proposer.py) | 候选生成、Eagle 投机 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendEagleProposer` |
| [extract_hidden_states_proposer.py](../../../vllm_ascend/spec_decode/extract_hidden_states_proposer.py) | 候选生成、状态 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 注释：Ascend adaptation of ExtractHiddenStatesProposer for extracting and caching；定义：`AscendExtractHiddenStatesProposer` |
| [gemma4_proposer.py](../../../vllm_ascend/spec_decode/gemma4_proposer.py) | 候选生成、Gemma 模型 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 注释：Gemma4 MTP proposer for Ascend NPUs.；定义：`AscendGemma4Proposer` |
| [llm_base_proposer.py](../../../vllm_ascend/spec_decode/llm_base_proposer.py) | 候选生成 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`_HiddenStateDrafter`, `AscendSpecDecodeBaseProposer`, `greedy_sample` |
| [medusa_proposer.py](../../../vllm_ascend/spec_decode/medusa_proposer.py) | 候选生成、Medusa 候选 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendMedusaProposer` |
| [multi_kv_cache_group_proposer.py](../../../vllm_ascend/spec_decode/multi_kv_cache_group_proposer.py) | 候选生成、缓存、KV 缓存 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendMultiKVCacheGroupMTPProposer`, `is_multi_kv_cache_group_mtp` |
| [ngram_proposer.py](../../../vllm_ascend/spec_decode/ngram_proposer.py) | 候选生成、n-gram 候选 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendNgramProposer` |
| [ngram_proposer_npu.py](../../../vllm_ascend/spec_decode/ngram_proposer_npu.py) | 候选生成、n-gram 候选 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendNgramProposerNPU` |
| [step3p5.py](../../../vllm_ascend/spec_decode/step3p5.py) | step3p5 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendStep3p5MTPProposer` |
| [suffix_proposer.py](../../../vllm_ascend/spec_decode/suffix_proposer.py) | 候选生成、后缀候选 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`AscendSuffixDecodingProposer` |
| [utils.py](../../../vllm_ascend/spec_decode/utils.py) | 辅助函数 的 NPU 候选生成、配置或执行辅助；具体入口见右栏。 | 定义：`SlidingWindowAdapter`, `DynamicSpecScheduler`, `compact_mtp_topk_indices`, `update_num_computed_tokens_for_batch_change`, `correct_optimistic_seq_lens_cpu` |
