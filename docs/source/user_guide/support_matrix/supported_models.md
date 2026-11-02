# Supported Models

Get the latest info here: <https://github.com/vllm-project/vllm-ascend/issues/1608>

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

    | Model   | Support   | Note   | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallelism | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
    |-------|--------|--------|------|------|------|---------|-------|------|------|--------|-------|--------|--------|-------|-------|--------|----------|---------|----------|-----|
    |DeepSeek V4-Flash|✅|Native mixed MXFP8/MXFP4 weights||950DT Products|✅|✅|✅||✅|✅||✅|✅|✅|✅||✅|1M||[DeepSeek V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md)|
    |DeepSeek V4-Pro|✅|Native mixed MXFP8/MXFP4 weights||950DT Products|✅|✅|✅||✅|✅||✅|✅|✅|✅||✅|1M||[DeepSeek V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md)|
    |DeepSeek-V3.1|🔴|  |✅| 950DT Products |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|240k|| [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
    | GLM-5.1 | ✅ |  | ✅ | 950DT Products | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | 200k |  | [GLM-5.1](../../tutorials/models/GLM5.md) |
    | GLM-5.2 | ✅ | W8A8C8 MXFP8 / W4A4C8 MXFP4 | 🟡 | 950DT Products | ✅ | 🟡 | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 🟡 |  | [GLM-5.2](../../tutorials/models/GLM5.2.md) |
    | GLM-5.3 | ✅ | W8A8C8 MXFP8 / W4A4C8 MXFP4 | 🟡 | 950DT Products | ✅ | 🟡 | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 1M |  | [GLM-5.3](../../tutorials/models/GLM5.2.md) |

=== "A2/A3"

    |       Model         | Support | Note | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallelism | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | Doc |
    | ------------------- | ------- | ---- | ---- | ------------------ | ---- | --------------- | ---------------------- | ------ | ---------------------- | ------------------ | ----------------- | ------------------- | ----------------- | --------------- | ------------------------------- | -------------------- | -------------------- | --------------- | ----- |
    | DeepSeek V4-Flash   |   ✅    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | 1M | [DeepSeek-V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md) |
    | DeepSeek V4-Pro     |   ✅    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | 1M | [DeepSeek-V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md) |
    | DeepSeek V3/3.1     |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 240k | [DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md) |
    | DeepSeek V3.2       |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 160k | [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md) |
    | DeepSeek R1         |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 128k | [DeepSeek-R1](../../tutorials/models/DeepSeek-R1.md) |
    | Qwen3-Dense         |   ✅    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  |  | ✅ |  | ✅ | ✅ | 128k | [Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md) |
    | Qwen3-30B-A3B       |   ✅    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ |  | ✅ | ✅ |  | [Qwen3-30B-A3B](../../tutorials/models/Qwen3-30B-A3B.md) |
    | Qwen3-Coder-30B-A3B |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ |  | ✅ | ✅ |  | [Qwen3-Coder-30B-A3B](../../tutorials/models/Qwen3-Coder-30B-A3B.md) |
    | Qwen3-235B-A22B     |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 256k | [Qwen3-235B-A22B](../../tutorials/models/Qwen3-235B-A22B.md) |
    | Qwen3-Next          |   🔴    |      |  ✅ |       A2/A3        | ✅ |  |  |  |  |  | ✅ |  |  | ✅ |  | ✅ | ✅ |  | [Qwen3-Next](../../tutorials/models/Qwen3-Next.md) |
    | GLM-4.x             |   🔴    |      |      |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 198k | [GLM-4.x](../../tutorials/models/GLM4.x.md) |
    | GLM-5               |   🔴    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k | [GLM-5](../../tutorials/models/GLM5.md) |
    | GLM-5.1             |   ✅    |      |  ✅ |       A2/A3        | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k | [GLM-5.1](../../tutorials/models/GLM5.md) |
    | GLM-5.2 | ✅ | W8A8C8 | ✅ | A2/A3 | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k | [GLM-5.2](../../tutorials/models/GLM5.2.md) |
    | GLM-5.3 | ✅ | W8A8C8 | ✅ | A2/A3 | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 135k (A2) / 1M (A3) | [GLM-5.3](../../tutorials/models/GLM5.2.md) |
    | Gemma4              |   🔵    |      |  ✅ |   A2/A3/950PR&950DT Products  |     | ✅ | ✅ |  |  | ✅ | ✅ |  |  | ✅ |  | ✅ | ✅ |  | [Gemma4](../../tutorials/models/Gemma4.md) |
    | Kimi-K2-Thinking    |   🔴     |      |      |       A2/A3       |      |  |  |  |  |  |  |  |  |  |  |  |  |  | [Kimi-K2-Thinking](../../tutorials/models/Kimi-K2-Thinking.md) |
    | DeepSeekOCR2        |   🔴   |      |  ✅ |       A2/A3        |     | ✅ |  |  |  | ✅ |  |  |  |  |  |  |  |  | [DeepSeekOCR2](../../tutorials/models/DeepSeekOCR2.md) |
    | MiniMax-M2.5/2.7    |   🔴     |      |  ✅ |   A2/A3/Ascend950 (Ascend950 experimental) | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 🟡 | ✅ | ✅ | ✅ | 🟡 | ✅ | 200k | [MiniMax-M2](../../tutorials/models/MiniMax-M2.md) |
    | Qwen2.5-Math-RM-72B |   🔴    | vllm-rm, tensor_parallel_size=4, max_model_len=4096 | ✅ | A2 | ✅ | 🟡 | 🟡 | ❌ | 🟡 | ✅ | ✅ | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 4096 | [Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md) |
    | Hy3-preview | 🔴 |   |  |A3|  |  |  |  |  |  | |  |  |  |  |  |  |  | [Hy3-preview](../../tutorials/models/Hy3-preview.md)  |
    | Hy4-preview (Experimental)  | 🔵  |   |  |A3|   |   |   |   |   |   |   |   |   |   |   |   |  |   |  [Hy4-preview (Experimental) ](../../tutorials/models/Hy4-preview.md) |

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

### Generative Models

#### Core Supported Models

=== "950PR&950DT Products"

    | Model           | Support  | Note   | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallelism | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | MLP Weight Prefetch | Doc |
    |-----------------|----------|--------|------|------|------|---------|-------|------|------|--------|-------|--------|--------|-------|-------|--------|----------|---------|----------|-----|
    | GLM-5.3-Flash | ✅ | Text, image, video | ✅ | 950DT Products | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 133120 |  | [GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md) |
    |Qwen3.5-397B-A17B|✅        | Text, image, video |✅    | 950DT Products |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|1010000|| [Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md) |
    |Qwen3.6-27B      |✅        |       |✅    | 950DT Products |✅|✅|✅||✅|✅|✅||✅|✅|✅|✅|✅|262144|| [Qwen3.5-27B / Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) |
    |Qwen3.6-35B-A3B  |✅        |       |✅    | 950DT Products |✅|✅|✅||✅ | ✅ | ✅ |  | ✅ | ✅ | ❌ | ✅ | ✅ | 135168 | | [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) |
    |Kimi-K3          |✅        |W4A8; DSpark; FULL_DECODE_ONLY; KV Cache Pool||950DT Products|||✅||✅||✅||✅|✅|✅||✅|131072||[Kimi-K3](../../tutorials/models/Kimi-K3.md)|
    |MiniMax-M3       |✅        |MXFP8; text/image/video||950DT Products||✅|✅||✅|✅|✅|✅|✅|✅|✅|🟡|✅|1M|| [MiniMax-M3](../../tutorials/models/MiniMax-M3.md) |
    |Step-5-Preview       |🔵 | Experimental || 950DT Products |✅|✅|✅||✅|✅|✅|✅|✅|✅|||✅|1M|| [Step-5-Preview](../../tutorials/models/Step-5-Preview.md) |
    | MiniMax-M3.1-Flash | 🔵 | Experimental |  | 950DT products | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 1M |  | [MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md) |

=== "A2/A3"

    |                Model                | Support | Note | BF16 | Supported Hardware | W8A8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallelism | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | max-model-len | Doc |
    | ----------------------------------- | ------- | ---- | ---- | ------------------ | ---- | --------------- | ---------------------- | ---- | -------------------- | ---------------- | --------------- | ----------------- | --------------- | ------------- | ----------------------------- | ------------------ | ------------------ | ------------- | --- |
    | GLM-5.3-Flash | ✅ | Text, image, video | ✅ | A2/A3 | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 200k (A2) / 1M (A3) | [GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md) |
    | Qwen3-VL                            | ✅      |      |      | A2/A3              |  |  |  |  |  |  | ✅ |  |  |  |  | ✅ | ✅ |  | [Qwen-VL-Dense](../../tutorials/models/Qwen-VL-Dense.md) |
    | Qwen3-VL-30B-A3B | ✅      |      | ✅  | A2/A3              | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Qwen3-VL-30B-A3B](../../tutorials/models/Qwen3-VL-30B-A3B-Instruct.md) |
    | Qwen3-VL-235B-A22B | 🔴      |      | ✅  | A2/A3              | ✅ | ✅ | ✅ |  |  | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Qwen3-VL-235B-A22B](../../tutorials/models/Qwen3-VL-235B-A22B-Instruct.md) |
    | DeepSeek-V4.1-Flash                 | ✅      | W8A8; 2-node A3 or 4-node A2 colocated deployment | 🟡 | A2/A3 | ✅ | 🟡 | ✅ | ❌ | ✅ | 🟡 | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ✅ | 1048576 | [DeepSeek-V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md) |
    | Qwen3.5-397B-A17B                   | ✅     | Text, image, video | ✅  | A2/A3              | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 1010000 | [Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md) |
    | Qwen3.5-27B / Qwen3.6-27B           | ✅      |      | ✅  | A2/A3              | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Qwen3.5-27B / Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md) |
    | Qwen3.6-35B-A3B                     | ✅      |      | ✅  | A2/A3              | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ❌ | ✅ | ✅ | 135168 | [Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md) |
    | Qwen3.8-Flash-Next           | 🔵      |      | ✅  | A3              | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Qwen3.8-Flash-Next](../../tutorials/models/Qwen3.8-Flash-Next.md) |
    | Qwen3-Omni-30B-A3B-Thinking         | 🔴      |      |      | A2/A3              |  |  |  |  |  |  | ✅ |  | ✅ |  |  |  |  |  | [Qwen3-Omni-30B-A3B-Thinking](../../tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md) |
    | Kimi-K2.5            | 🔴      |      |      | A2/A3              |  | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Kimi-K2.5](../../tutorials/models/Kimi-K2.5.md) |
    | Kimi-K2.6                 | ✅      |      |      | A2/A3              |  | ✅ | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | 262144 | [Kimi-K2.6](../../tutorials/models/Kimi-K2.6.md) |
    | Kimi-K3 | ✅ | W4A8; DSpark; FULL_DECODE_ONLY; KV Cache Pool |  | A3 |  |  | ✅ |  | ✅ |  | ✅ |  | ✅ | ✅ | ✅ |  | ✅ | 131072 | [Kimi-K3](../../tutorials/models/Kimi-K3.md) |
    | MiniMax-M3                          | ✅      | Text, image, video, and ViT DP | ✅ | A2/A3 | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 1M | [MiniMax-M3](../../tutorials/models/MiniMax-M3.md) |
    | MiniMax-M3.1-Flash | 🔵 | Experimental |  | A2/A3 | ✅ | ✅ | ✅ |  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 | ✅ | 1M | [MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md) |
    | Cohere Transcribe                  | 🔵      | 2B Conformer encoder-decoder ASR, 14 languages (03-2026 & arabic-07-2026) | ✅ | A2/A3 | ❌ |  |  | ❌ | ❌ |  | 🟡 |  |  |  | ❌ | 🟡 | 🟡 |  | [Cohere-Transcribe](../../tutorials/models/Cohere-Transcribe.md) |
    | PaddleOCR-VL | 🔴 |  |  | A2/A3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  | [PaddleOCR-VL](../../tutorials/models/PaddleOCR-VL.md) |

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
