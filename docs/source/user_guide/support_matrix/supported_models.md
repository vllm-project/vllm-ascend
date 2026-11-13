# Supported Models

!!! note

    This list is based on the vLLM main community model list and categorizes the support status of vllm-ascend with labels such as Supported, Experimental, and Unmaintained. Models not listed may be verified independently.

**Legend Description**:

|Legend|Description|Usage|Recommendation|
|----------|----------|-----------|------------|
|✅|Supported|Fully implemented and verified, meeting both functional and performance standards.|Can be used in production environments.|
|🔵|Experimental|Implemented and basically functional, but still under improvement. May have instability or performance fluctuations.|Can be used for testing/pre-research, production use requires risk assessment.|
|🟡|Unverified|Code has been merged but not yet subjected to system testing. Behavior is uncertain and cannot be evaluated.|Users assume the risk of use.|
|🔴|Unmaintained|Previously supported, but now discontinued and no longer maintained. Issues will not be fixed.|Do not use in new projects.|
|❌|Unsupported|Not implemented in the current version and cannot be used.|Not available.|

## Text-Only Language Models

### Generative Models

#### Core Supported Models

=== "950PR&950DT Products"

    | Model | Support | Hardware | [Graph Mode](../../user_guide/feature_guide/graph_mode.md) |||| [Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex) || [CP](../../user_guide/feature_guide/context_parallel.md) || [CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md) | [KVPP(only D node)](../../user_guide/feature_guide/kvpp.md) | TP | EP | DP | MLAPO | [EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md) | Multistream MoE(only D node) | Shared Expert DP | Fused MC2 | [Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md) |||| [Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md) || Prefix Cache | Weight nz | Async Scheduling | [CPU Binding](../../user_guide/feature_guide/cpu_binding.md) | flashcomml(only P node) | [Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md) ||||
    | ^^ | ^^ | ^^ | PIECEWISE | FULL_DECODE_ONLY | FULL | FULL_AND_PIECEWISE | Static kernel | Super kernel | PCP | DCP | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | MTP | eagle3 | DSpark | DFlash | Layerwise | Non-layerwise | SSD | Layerwise | Non-layerwise | SSD | Layerwise | Sparse | ^^ | ^^ | ^^ | ^^ | ^^ | o_proj | embedding | MLP | LM_head |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | [DeepSeek V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md) | ✅ | 950DT Products | 🟡 | ✅ | ❌ | 🟡 | 🟡 | 🟡 | ✅ | ❌ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | 🟡 | 5 | ❌ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | 🟡 |
    | [DeepSeek V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | ✅ | ❌ | 🟡 | ❌ | 🟡 | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | 🟡 | 5 | ❌ | 🟡 | 🟡 | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ |
    | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) | 🔴 | 950DT Products | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [GLM-5.1](../../tutorials/models/GLM5.md) | ✅ | 950DT Products | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [GLM-5.2](../../tutorials/models/GLM5.2.md) | ✅ | 950DT Products | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [GLM-5.3](../../tutorials/models/GLM5.2.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ❌ | ✅ | 🟡 | 5 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ |

=== "A2/A3"

    !!! note

        The feature support in the table below applies only to the A3 hardware scenario. For the A2 scenario, please refer to the specific configuration in the [Model Tutorials](../../tutorials/models/index.md).

    | Model | Support | Hardware | [Graph Mode](../../user_guide/feature_guide/graph_mode.md) |||| [Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex) || [CP](../../user_guide/feature_guide/context_parallel.md) || [CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md) | [KVPP(only D node)](../../user_guide/feature_guide/kvpp.md) | TP | EP | DP | MLAPO | [EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md) | Multistream MoE(only D node) | Shared Expert DP | Fused MC2 | [Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md) |||| [Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md) || Prefix Cache | Weight nz | Async Scheduling | [CPU Binding](../../user_guide/feature_guide/cpu_binding.md) | flashcomml(only P node) | [Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md) ||||
    | ^^ | ^^ | ^^ | PIECEWISE | FULL_DECODE_ONLY | FULL | FULL_AND_PIECEWISE | Static kernel | Super kernel | PCP | DCP | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | MTP | eagle3 | DSpark | DFlash | Layerwise | Non-layerwise | SSD | Layerwise | Non-layerwise | SSD | Layerwise | Sparse | ^^ | ^^ | ^^ | ^^ | ^^ | o_proj | embedding | MLP | LM_head |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | [DeepSeek V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md) | ✅ | A2/A3 | ❌ | ✅ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | 🟡 | ✅ | 🟡 | ❌ | 5 | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | 🟡 |
    | [DeepSeek V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md) | ✅ | A2/A3 | ❌ | ✅ | ❌ | ❌ | 🟡 | 🟡 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | 🟡 | ✅ | 🟡 | ❌ | 5 | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ❌ | 🟡 |
    | [DeepSeek V3/3.1](../../tutorials/models/DeepSeek-V3.1.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [DeepSeek V3.2](../../tutorials/models/DeepSeek-V3.2.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [DeepSeek R1](../../tutorials/models/DeepSeek-R1.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [DeepSeekOCR2](../../tutorials/models/DeepSeekOCR2.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [GLM-4.x](../../tutorials/models/GLM4.x.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [GLM-5](../../tutorials/models/GLM5.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [GLM-5.1](../../tutorials/models/GLM5.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [GLM-5.2](../../tutorials/models/GLM5.2.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [GLM-5.3](../../tutorials/models/GLM5.2.md) | ✅ | A2/A3 | 🟡 | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ✅ | 5 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | 🟡 | 🟡 | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ | 🟡 |
    | [Gemma4](../../tutorials/models/Gemma4.md) | 🔵 | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Hy3-preview](../../tutorials/models/Hy3-preview.md) | 🔴 | A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Hy4-preview (Experimental) ](../../tutorials/models/Hy4-preview.md) | 🔵 | A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K2-Thinking](../../tutorials/models/Kimi-K2-Thinking.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [MiniMax-M2.5/2.7](../../tutorials/models/MiniMax-M2.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3-30B-A3B](../../tutorials/models/Qwen3-30B-A3B.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3-Coder-30B-A3B](../../tutorials/models/Qwen3-Coder-30B-A3B.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Qwen3-235B-A22B](../../tutorials/models/Qwen3-235B-A22B.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Qwen3-Next](../../tutorials/models/Qwen3-Next.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md) | 🔴 | A2 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

=== "Atlas 300I DUO"

    | Model | Support | Note | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | Doc |
    |-------|---------|------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------------------|--------------------|--------------------|---------------|-----|
    | Qwen3-Dense | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 20k | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |
    | Qwen3-30B-A3B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 16k | [Qwen3-30B-A3B](../../tutorials/models/Qwen3-30B-A3B.md) |

### Pooling Models

=== "950PR&950DT Products"

    | Model                | Support | Supported Hardware              | Doc |
    |----------------------|---------|----------------------------------|-----|
    | Qwen3-Embedding       | 🔴      | Ascend 950DT&950PR Products      | [Qwen3-Embedding](../../tutorials/models/Qwen3-Embedding.md) |
    | Qwen3-VL-Embedding    | 🔴      | Ascend 950DT&950PR Products      | [Qwen3-VL-Embedding](../../tutorials/models/Qwen3-VL-Embedding.md) |
    | Qwen3-Reranker        | 🔴      | Ascend 950DT&950PR Products      | [Qwen3-Reranker](../../tutorials/models/Qwen3-Reranker.md) |
    | Qwen3-VL-Reranker     | 🔴      | Ascend 950DT&950PR Products      | [Qwen3-VL-Reranker](../../tutorials/models/Qwen3-VL-Reranker.md) |

=== "A2/A3"

    | Model                         | Support   | Note                                                                 |    Supported Hardware | W8A8   |  Doc |
    |-------------------------------|-----------|----------------------------------------------------------------------|-----------------------|--------|------|
    | Qwen3-Embedding               | 🔴        |                                                                      |         A2/A3        | 🟡 | [Qwen3-Embedding](../../tutorials/models/Qwen3-Embedding.md)|
    | Qwen3-VL-Embedding            | 🔴        |                                                                      |         A2/A3        | 🔵 | [Qwen3-VL-Embedding](../../tutorials/models/Qwen3-VL-Embedding.md)|
    | Qwen3-Reranker                | 🔴        |                                                                      |         A2/A3        | 🟡 | [Qwen3-Reranker](../../tutorials/models/Qwen3-Reranker.md)|
    | Qwen3-VL-Reranker             | 🔴        |                                                                      |         A2/A3        | 🔵 | [Qwen3-VL-Reranker](../../tutorials/models/Qwen3-VL-Reranker.md)|
    | Qwen2.5-Math-RM-72B           | 🔴        | Reward Model, gsm8k_correctness accuracy=0.80 | A2 | [Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md) |

=== "Atlas 300I DUO"

    | Model | Support | Note | Supported Hardware | W8A8|Doc |
    |-------|---------|------|--------------------|-----|------|
    | Qwen3-Embedding | 🔴 | FP16 | Atlas 300I DUO |🟡| [Qwen3_Embedding](../../tutorials/models/Qwen3-Embedding.md) |
    | Qwen3-VL-Embedding | 🔴 | FP16 | Atlas 300I DUO |🔵| [Qwen3_VL_Embedding](../../tutorials/models/Qwen3-VL-Embedding.md) |
    | SigLIP2 | ✅ | FP16; text/image embedding | Atlas 300I DUO |🔵| [SigLIP2](../../tutorials/models/SigLIP2.md) |
    | Qwen3-Reranker  | 🔴 | FP16 | Atlas 300I DUO |🟡| [Qwen3_Reranker](../../tutorials/models/Qwen3-Reranker.md) |
    | Qwen3-VL-Reranker | 🔴 | FP16 | Atlas 300I DUO |🔵| [Qwen3_VL_Reranker](../../tutorials/models/Qwen3-VL-Reranker.md) |

## Multimodal Language Models

=== "950PR&950DT Products"

    | Model | Support | Hardware | [Graph Mode](../../user_guide/feature_guide/graph_mode.md) |||| [Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex) || [CP](../../user_guide/feature_guide/context_parallel.md) || [CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md) | [KVPP(only D node)](../../user_guide/feature_guide/kvpp.md) | TP | EP | DP | MLAPO | [EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md) | Multistream MoE(only D node) | Shared Expert DP | Fused MC2 | [Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md) |||| [Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md) || Prefix Cache | Weight nz | Async Scheduling | [CPU Binding](../../user_guide/feature_guide/cpu_binding.md) | flashcomml(only P node) | [Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md) ||||
    | ^^ | ^^ | ^^ | PIECEWISE | FULL_DECODE_ONLY | FULL | FULL_AND_PIECEWISE | Static kernel | Super kernel | PCP | DCP | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | MTP | eagle3 | DSpark | DFlash | Layerwise | Non-layerwise | SSD | Layerwise | Non-layerwise | SSD | Layerwise | Sparse | ^^ | ^^ | ^^ | ^^ | ^^ | o_proj | embedding | MLP | LM_head |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | [DeepSeek V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md) | ✅ | 950DT Products | 🟡 | ✅ | ❌ | 🟡 | ❌ | ❌ | 🟡 | ❌ | 🟡 | ❌ | ❌ | ✅ | ✅ | ❌ | 🟡 | ✅ | 🟡 | ❌ | 🟡 | 🟡 | 5 | ❌ | ❌ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ❌ | ❌ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ | ❌ | 🟡 |
    | [GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md) | ✅ | 950DT Products | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 5 | ❌ | ❌ | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K3](../../tutorials/models/Kimi-K3.md) | ✅ | 950DT Products | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | ❌ | ❌ | 3 | ❌ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K3.1-Pro](../../tutorials/models/Kimi-K3.1-Pro.md) | 🔵 | 950DT Products | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [MiniMax-M3](../../tutorials/models/MiniMax-M3.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | 3 | ❌ | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ | 7 | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
    | [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) | ✅ | 950DT Products | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | 🟡 | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ |
    | [Qwen3.8-Flash-Next](../../tutorials/models/Qwen3.8-Flash-Next.md) | ✅ | 950DT Products | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | / | ❌ | ✅ | ✅ | 🟡 | 3 | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
    | [Step-5-Preview](../../tutorials/models/Step-5-Preview.md) | 🔵 | 950DT Products | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    
=== "A2/A3"

    !!! note

        The feature support in the table below applies only to the A3 hardware scenario. For the A2 scenario, please refer to the specific configuration in the [Model Tutorials](../../tutorials/models/index.md).

    | Model | Support | Hardware | [Graph Mode](../../user_guide/feature_guide/graph_mode.md) |||| [Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex) || [CP](../../user_guide/feature_guide/context_parallel.md) || [CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md) | [KVPP(only D node)](../../user_guide/feature_guide/kvpp.md) | TP | EP | DP | MLAPO | [EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md) | Multistream MoE(only D node) | Shared Expert DP | Fused MC2 | [Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md) |||| [Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md) ||| [KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md) || Prefix Cache | Weight nz | Async Scheduling | [CPU Binding](../../user_guide/feature_guide/cpu_binding.md) | flashcomml(only P node) | [Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md) ||||
    | ^^ | ^^ | ^^ | PIECEWISE | FULL_DECODE_ONLY | FULL | FULL_AND_PIECEWISE | Static kernel | Super kernel | PCP | DCP | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | ^^ | MTP | eagle3 | DSpark | DFlash | Layerwise | Non-layerwise | SSD | Layerwise | Non-layerwise | SSD | Layerwise | Sparse | ^^ | ^^ | ^^ | ^^ | ^^ | o_proj | embedding | MLP | LM_head |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | [Cohere-Transcribe](../../tutorials/models/Cohere-Transcribe.md) | 🔵 | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [DeepSeek-V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md) | ✅ | A2/A3 | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ✅ | 🟡 | 🟡 | 5 | ❌ | ❌ | 🟡 | 🟡 | ❌ | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | 🟡 |
    | [GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md) | ✅ | A2/A3 | 🟡 | ✅ | ❌ | 🟡 | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | 3 | ❌ | ❌ | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K2.5](../../tutorials/models/Kimi-K2.5.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Kimi-K2.6](../../tutorials/models/Kimi-K2.6.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K3](../../tutorials/models/Kimi-K3.md) | ✅ | A3 | 🟡 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ | 7 | ❌ | ❌ | 🟡 | 🟡 | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Kimi-K3.1-Pro](../../tutorials/models/Kimi-K3.1-Pro.md) | 🔵 | A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [MiniMax-M3](../../tutorials/models/MiniMax-M3.md) | ✅ | A2/A3 | 🟡 | ✅ | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | 3 | ❌ | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md) | ✅ | A2/A3 | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | 🟡 | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | 🟡 | ❌ | ❌ | 7 | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3.8-Flash-Next](../../tutorials/models/Qwen3.8-Flash-Next.md) | ✅ | A2/A3 | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | / | ❌ | ✅ | ✅ | ❌ | 3 | ❌ | ❌ | ❌ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | 🟡 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
    | [Qwen3-VL](../../tutorials/models/Qwen-VL-Dense.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3-VL-30B-A3B](../../tutorials/models/Qwen3-VL-30B-A3B-Instruct.md) | ✅ | A2/A3 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3-VL-235B-A22B](../../tutorials/models/Qwen3-VL-235B-A22B-Instruct.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md) | ✅ | A2/A3 | 🟡 | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 🟡 | 🟡 | ✅ | ✅ | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
    | [Qwen3.5-27B/Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) | ✅ | A2/A3 | 🟡 | ✅ | 🟡 | 🟡 | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) | ✅ | A2/A3 | 🟡 | ✅ | 🟡 | 🟡 | ✅ | 🟡 | ❌ | ❌ | ❌ | ❌ | 🟡 | 🟡 | 🟡 | ❌ | 🟡 | 🟡 | 🟡 | 🟡 | 3 | 🟡 | 🟡 | 🟡 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 |
    | [Qwen3-Omni-30B-A3B-Thinking](../../tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    | [PaddleOCR-VL](../../tutorials/models/PaddleOCR-VL.md) | 🔴 | A2/A3 | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

=== "Atlas 300I DUO"

    | Model | Support | Note | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | Doc |
    |-------|---------|------|------|--------------------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------------------|--------------------|--------------------|---------------|-----|
    | Qwen3-VL | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | 🟡 | ✅ | ✅ | ❌ | ✅ | ✅ | 16k | [Qwen-VL-Dense](../../tutorials/models/Qwen-VL-Dense.md) |
    | Qwen3.5-Dense | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.5-Dense](../../tutorials/models/Qwen3.5-Dense.md) |
    | Qwen3.5-35B-A3B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.5-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) |
    | Qwen3.5-27B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.5-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) |
    | Qwen3.6-27B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) |
    | Qwen3.6-35B-A3B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) |
    | Qwen3.8-27B | ✅ |  | ❌ | Atlas 300I DUO | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | 256k | [Qwen3.8-27B](../../tutorials/models/Qwen3.8-27B.md) |
    | PaddleOCR-VL | 🔴 |  | ❌ | Atlas 300I DUO | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | 16k | [PaddleOCR-VL](../../tutorials/models/PaddleOCR-VL.md) |
    | Qwen3-ASR | 🔴 |  | ❌ | Atlas 300I DUO | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | 🟡 | ❌ | ✅ | ✅ | 4096 | [Qwen3-ASR-1.7B](../../tutorials/models/Qwen3-ASR-1.7B.md) |
