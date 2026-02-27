# Release Notes

## v0.15.0rc1 - 2026.02.26

This is the first release candidate of v0.15.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights

- **NPU Graph EX (npugraph_ex) Enabled by Default**: The npugraph_ex feature is now enabled by default, providing better graph optimization with integrated inductor pass and MatmulAllReduceAddRMSNorm fusion. [#6354](https://github.com/vllm-project/vllm-ascend/pull/6354) [#6664](https://github.com/vllm-project/vllm-ascend/pull/6664) [#6006](https://github.com/vllm-project/vllm-ascend/pull/6006)
- **310P MoE and W8A8 Support**: 310P now supports MoE models, W8A8 quantization, and weightNZ feature, significantly expanding hardware capabilities. [#6530](https://github.com/vllm-project/vllm-ascend/pull/6530) [#6641](https://github.com/vllm-project/vllm-ascend/pull/6641) [#6454](https://github.com/vllm-project/vllm-ascend/pull/6454) [#6705](https://github.com/vllm-project/vllm-ascend/pull/6705)
- **Qwen3-VL-MoE EAGLE Support**: Added EAGLE speculative decoding support for Qwen3-VL-MoE model. [#6327](https://github.com/vllm-project/vllm-ascend/pull/6327)
- **Kimi-K2.5 Model Support**: Added support for Kimi-K2.5 models. [#6755](https://github.com/vllm-project/vllm-ascend/pull/6755)

### Features

- **Auto-detect Quantization Format**: Quantization format can now be auto-detected from model files. [#6645](https://github.com/vllm-project/vllm-ascend/pull/6645)
- **GPT-OSS Attention Support**: Added GPT-OSS attention implementation. [#5901](https://github.com/vllm-project/vllm-ascend/pull/5901)
- **DCP Support for SFA**: Added Decode Context Parallel (DCP) support for SFA architecture. [#6563](https://github.com/vllm-project/vllm-ascend/pull/6563)
- **Mooncake Layerwise PCP Support**: Mooncake layerwise connector now supports PCP function. [#6627](https://github.com/vllm-project/vllm-ascend/pull/6627)
- **Mooncake Connector Remote PTP Size**: Mooncake connector can now get remote PTP size. [#5822](https://github.com/vllm-project/vllm-ascend/pull/5822)
- **KV Pool Sparse Attention**: KV pool now supports sparse attention. [#6339](https://github.com/vllm-project/vllm-ascend/pull/6339)
- **Batch Invariant with AscendC**: Implemented batch invariant feature with AscendC. [#6590](https://github.com/vllm-project/vllm-ascend/pull/6590)
- **Routing Replay**: Added routing replay feature. [#6696](https://github.com/vllm-project/vllm-ascend/pull/6696)
- **Compressed Tensors MoE W4A8 Dynamic Weight**: Added support for compressed tensors moe w4a8 dynamic weight quantization. [#5889](https://github.com/vllm-project/vllm-ascend/pull/5889)
- **GLM4.7-Flash W8A8 Quantization**: Added W8A8 quantization support for GLM4.7-Flash. [#6492](https://github.com/vllm-project/vllm-ascend/pull/6492)
- **DispatchGmmCombineDecode Enhancement**: DispatchGmmCombineDecode now supports bf16/float16 gmm1/gmm2 weight and ND format weight. [#6393](https://github.com/vllm-project/vllm-ascend/pull/6393)
- **RMSNorm Dynamic Quant Fusion**: Added rmsnorm dynamic quant fusion pass. [#6274](https://github.com/vllm-project/vllm-ascend/pull/6274)
- **Worker Health Check Interface**: Added `check_health` interface for worker. [#6681](https://github.com/vllm-project/vllm-ascend/pull/6681)

### Hardware and Operator Support

- **310P Support Expansion**: Multiple improvements for 310P hardware:
    - Fixed attention accuracy issue on 310P. [#6803](https://github.com/vllm-project/vllm-ascend/pull/6803)
    - Added weightNZ feature for 310P with quant or unquant support. [#6705](https://github.com/vllm-project/vllm-ascend/pull/6705)
    - Added addrmsnorm support for 300I DUO. [#6704](https://github.com/vllm-project/vllm-ascend/pull/6704)
    - 310P now supports PrefillCacheHit state. [#6756](https://github.com/vllm-project/vllm-ascend/pull/6756)
- **ARM-only CPU Binding**: Enabled ARM-only CPU binding with NUMA-balanced A3 policy. [#6686](https://github.com/vllm-project/vllm-ascend/pull/6686)
- **Triton Rope Enhancement**: Triton rope now supports index_selecting from cos_sin_cache. [#5450](https://github.com/vllm-project/vllm-ascend/pull/5450)
- **AscendC Fused Op**: Added AscendC fused op transpose_kv_cache_by_block to speed up GQA transfer. [#6366](https://github.com/vllm-project/vllm-ascend/pull/6366)
- **Rotary_dim Parameter**: Added support for rotary_dim parameter when using partial rope in rotary_embedding. [#6581](https://github.com/vllm-project/vllm-ascend/pull/6581)

### Performance

- **Multimodal seq_lens CPU Cache**: Use `seq_lens` CPU cache to avoid frequent D2H copy for better multimodal performance. [#6448](https://github.com/vllm-project/vllm-ascend/pull/6448)
- **DispatchFFNCombine Optimization**: Optimized DispatchFFNCombine kernel performance and resolved vector error caused by unaligned UB access. [#6468](https://github.com/vllm-project/vllm-ascend/pull/6468) [#6707](https://github.com/vllm-project/vllm-ascend/pull/6707)
- **DeepSeek V3.2 KVCache Optimization**: Optimized KV cache usage for DeepSeek V3.2. [#6610](https://github.com/vllm-project/vllm-ascend/pull/6610)
- **MLA/SFA Weight Prefetch**: Refactored MLA/SFA weight prefetch to be consistent with MoE weight prefetch. [#6629](https://github.com/vllm-project/vllm-ascend/pull/6629)
- **MLP Weight Prefetch**: Refactored MLP weight prefetch to be consistent with MoE model's prefetching. [#6442](https://github.com/vllm-project/vllm-ascend/pull/6442)
- **Adaptive Block Size Selection**: Added adaptive block size selection in linear_persistent kernel. [#6537](https://github.com/vllm-project/vllm-ascend/pull/6537)
- **EPLB Memory Optimization**: Reduced memory used for heat aggregation in EPLB. [#6729](https://github.com/vllm-project/vllm-ascend/pull/6729)
- **Memory Migration and Interrupt Core Binding**: Improved binding logic with memory migration and interrupt core binding functions. [#6785](https://github.com/vllm-project/vllm-ascend/pull/6785)
- **Triton Stability**: Improved Triton stability on Ascend for large grids. [#6301](https://github.com/vllm-project/vllm-ascend/pull/6301)

### Dependencies

- **Mooncake**: Upgraded to v0.3.8.post1. [#6428](https://github.com/vllm-project/vllm-ascend/pull/6428)

### Deprecation & Breaking Changes

- **ProfileExecuteDuration**: Cleaned up and deprecated ProfileExecuteDuration feature. [#6461](https://github.com/vllm-project/vllm-ascend/pull/6461)
- **Custom rotary_embedding Operator**: Removed custom rotary_embedding operator. [#6523](https://github.com/vllm-project/vllm-ascend/pull/6523)
- **USE_OPTIMIZED_MODEL**: Cleaned up unused env `USE_OPTIMIZED_MODEL`. [#6618](https://github.com/vllm-project/vllm-ascend/pull/6618)

### Documentation

- Added AI-assisted model-adaptation workflow documentation for vllm-ascend. [#6731](https://github.com/vllm-project/vllm-ascend/pull/6731)
- Added vLLM Ascend development guidelines (AGETNS.md). [#6797](https://github.com/vllm-project/vllm-ascend/pull/6797)
- Added GLM5 tutorial documentation. [#6709](https://github.com/vllm-project/vllm-ascend/pull/6709) [#6717](https://github.com/vllm-project/vllm-ascend/pull/6717)
- Added Memcache Usage Guide. [#6476](https://github.com/vllm-project/vllm-ascend/pull/6476)
- Added request forwarding documentation. [#6780](https://github.com/vllm-project/vllm-ascend/pull/6780)
- Added Benchmark Tutorial for Suffix Speculative Decoding. [#6323](https://github.com/vllm-project/vllm-ascend/pull/6323)
- Restructured tutorial documentation. [#6501](https://github.com/vllm-project/vllm-ascend/pull/6501)
- Added npugraph_ex introduction documentation. [#6306](https://github.com/vllm-project/vllm-ascend/pull/6306)

### Others

- **MTP in PD Fullgraph**: Fixed support for ALL D-Nodes in fullgraph when running MTP in PD deployment. [#5472](https://github.com/vllm-project/vllm-ascend/pull/5472)
- **DeepSeekV3.1 Accuracy**: Fixed DeepSeekV3.1 accuracy issue. [#6805](https://github.com/vllm-project/vllm-ascend/pull/6805)
- **EAGLE Refactor**: Routed MTP to EAGLE except for PCP/DCP+MTP cases. [#6349](https://github.com/vllm-project/vllm-ascend/pull/6349)
- **Speculative Decoding Accuracy**: Fixed spec acceptance rate problem in vLLM 0.15.0. [#6606](https://github.com/vllm-project/vllm-ascend/pull/6606)
- **PCP/DCP Accuracy**: Fixed accuracy issue in PCP/DCP with speculative decoding. [#6491](https://github.com/vllm-project/vllm-ascend/pull/6491)
- **Dynamic EPLB**: Fixed ineffective dynamic EPLB bug and EPLB no longer depends on a specified model. [#6653](https://github.com/vllm-project/vllm-ascend/pull/6653) [#6528](https://github.com/vllm-project/vllm-ascend/pull/6528)
- **KV Pool Mooncake Backend**: Correctly initialized head_or_tp_rank for mooncake backend. [#6498](https://github.com/vllm-project/vllm-ascend/pull/6498)
- **Layerwise Connector Recompute Scheduler**: Layerwise connector now supports recompute scheduler. [#5900](https://github.com/vllm-project/vllm-ascend/pull/5900)
- **Memcache Pool**: Fixed service startup failure when memcache pool is enabled. [#6229](https://github.com/vllm-project/vllm-ascend/pull/6229)
- **AddRMSNormQuant**: Fixed AddRMSNormQuant not taking effect. [#6620](https://github.com/vllm-project/vllm-ascend/pull/6620)
- **Pooling Code**: Fixed pooling code issues and updated usage guide. [#6126](https://github.com/vllm-project/vllm-ascend/pull/6126)
- **Context Parallel**: Fixed and unified the PD request discrimination logic. [#5939](https://github.com/vllm-project/vllm-ascend/pull/5939)
- **npugraph_ex**: Fixed duplicate pattern issue and added extra check for allreduce rmsnorm fusion pass. [#6513](https://github.com/vllm-project/vllm-ascend/pull/6513) [#6430](https://github.com/vllm-project/vllm-ascend/pull/6430)
- **RecomputeScheduler**: Fixed incompatibility of RecomputeScheduler with vLLM v0.14.1. [#6286](https://github.com/vllm-project/vllm-ascend/pull/6286)

## v0.13.0 - 2026.02.06

This is the final release of v0.13.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to get started.

### Highlights

**Model Support**

- **DeepSeek-R1 & DeepSeek-V3.2**: [Experimental]Performance optimizations, and async scheduling enhancements. [#3631](https://github.com/vllm-project/vllm-ascend/pull/3631) [#3900](https://github.com/vllm-project/vllm-ascend/pull/3900) [#3908](https://github.com/vllm-project/vllm-ascend/pull/3908) [#4191](https://github.com/vllm-project/vllm-ascend/pull/4191) [#4805](https://github.com/vllm-project/vllm-ascend/pull/4805)
- **Qwen3-Next**: [Experimental]Full support for Qwen3-Next series including 80B-A3B-Instruct with full graph mode, MTP, quantization (W8A8), NZ optimization, and chunked prefill. Fixed multiple accuracy and stability issues. [#3450](https://github.com/vllm-project/vllm-ascend/pull/3450) [#3572](https://github.com/vllm-project/vllm-ascend/pull/3572) [#3428](https://github.com/vllm-project/vllm-ascend/pull/3428) [#3918](https://github.com/vllm-project/vllm-ascend/pull/3918) [#4058](https://github.com/vllm-project/vllm-ascend/pull/4058) [#4245](https://github.com/vllm-project/vllm-ascend/pull/4245) [#4070](https://github.com/vllm-project/vllm-ascend/pull/4070) [#4477](https://github.com/vllm-project/vllm-ascend/pull/4477) [#4770](https://github.com/vllm-project/vllm-ascend/pull/4770)
- **InternVL**: Added support for InternVL models with comprehensive e2e tests and accuracy evaluation. [#3796](https://github.com/vllm-project/vllm-ascend/pull/3796) [#3964](https://github.com/vllm-project/vllm-ascend/pull/3964)
- **LongCat-Flash**: [Experimental]Added support for LongCat-Flash model. [#3833](https://github.com/vllm-project/vllm-ascend/pull/3833)
- **minimax_m2**: [Experimental]Added support for minimax_m2 model. [#5624](https://github.com/vllm-project/vllm-ascend/pull/5624)
- **Whisper & Cross-Attention**: [Experimental]Added support for cross-attention and Whisper models. [#5592](https://github.com/vllm-project/vllm-ascend/pull/5592)
- **Pooling Models**: [Experimental]Added support for pooling models with PCP adaptation and fixed multiple pooling-related bugs. [#3122](https://github.com/vllm-project/vllm-ascend/pull/3122) [#4143](https://github.com/vllm-project/vllm-ascend/pull/4143) [#6056](https://github.com/vllm-project/vllm-ascend/pull/6056) [#6057](https://github.com/vllm-project/vllm-ascend/pull/6057) [#6146](https://github.com/vllm-project/vllm-ascend/pull/6146)
- **PanguUltraMoE**: [Experimental]Added support for PanguUltraMoE model. [#4615](https://github.com/vllm-project/vllm-ascend/pull/4615)

**Core Features**

- **Context Parallel (PCP/DCP)**: [Experimental] Added comprehensive support for Prefill Context Parallel (PCP) and Decode Context Parallel (DCP) with ACLGraph, MTP, chunked prefill, MLAPO, and Mooncake connector integration. This is an experimental feature - feedback welcome. [#3260](https://github.com/vllm-project/vllm-ascend/pull/3260) [#3731](https://github.com/vllm-project/vllm-ascend/pull/3731) [#3801](https://github.com/vllm-project/vllm-ascend/pull/3801) [#3980](https://github.com/vllm-project/vllm-ascend/pull/3980) [#4066](https://github.com/vllm-project/vllm-ascend/pull/4066) [#4098](https://github.com/vllm-project/vllm-ascend/pull/4098) [#4183](https://github.com/vllm-project/vllm-ascend/pull/4183) [#5672](https://github.com/vllm-project/vllm-ascend/pull/5672)
- **Full Graph Mode (ACLGraph)**: [Experimental]Enhanced full graph mode with GQA support, memory optimizations, unified logic between ACLGraph and Torchair, and improved stability. [#3560](https://github.com/vllm-project/vllm-ascend/pull/3560) [#3970](https://github.com/vllm-project/vllm-ascend/pull/3970) [#3812](https://github.com/vllm-project/vllm-ascend/pull/3812) [#3879](https://github.com/vllm-project/vllm-ascend/pull/3879) [#3888](https://github.com/vllm-project/vllm-ascend/pull/3888) [#3894](https://github.com/vllm-project/vllm-ascend/pull/3894) [#5118](https://github.com/vllm-project/vllm-ascend/pull/5118)
- **Multi-Token Prediction (MTP)**: Significantly improved MTP support with chunked prefill for DeepSeek, quantization support, full graph mode, PCP/DCP integration, and async scheduling. MTP now works in most cases and is recommended for use. [#2711](https://github.com/vllm-project/vllm-ascend/pull/2711) [#2713](https://github.com/vllm-project/vllm-ascend/pull/2713) [#3620](https://github.com/vllm-project/vllm-ascend/pull/3620) [#3845](https://github.com/vllm-project/vllm-ascend/pull/3845) [#3910](https://github.com/vllm-project/vllm-ascend/pull/3910) [#3915](https://github.com/vllm-project/vllm-ascend/pull/3915) [#4102](https://github.com/vllm-project/vllm-ascend/pull/4102) [#4111](https://github.com/vllm-project/vllm-ascend/pull/4111) [#4770](https://github.com/vllm-project/vllm-ascend/pull/4770) [#5477](https://github.com/vllm-project/vllm-ascend/pull/5477)
- **Eagle Speculative Decoding**: Eagle spec decode now works with full graph mode and is more stable. [#5118](https://github.com/vllm-project/vllm-ascend/pull/5118) [#4893](https://github.com/vllm-project/vllm-ascend/pull/4893) [#5804](https://github.com/vllm-project/vllm-ascend/pull/5804)
- **PD Disaggregation**: Set ADXL engine as default backend for disaggregated prefill with improved performance and stability. Added support for KV NZ feature for DeepSeek decode node. [#3761](https://github.com/vllm-project/vllm-ascend/pull/3761) [#3950](https://github.com/vllm-project/vllm-ascend/pull/3950) [#5008](https://github.com/vllm-project/vllm-ascend/pull/5008) [#3072](https://github.com/vllm-project/vllm-ascend/pull/3072)
- **KV Pool & Mooncake**: Enhanced KV pool with Mooncake connector support for PCP/DCP, multiple input suffixes, and improved performance of Layerwise Connector. [#3690](https://github.com/vllm-project/vllm-ascend/pull/3690) [#3752](https://github.com/vllm-project/vllm-ascend/pull/3752) [#3849](https://github.com/vllm-project/vllm-ascend/pull/3849) [#4183](https://github.com/vllm-project/vllm-ascend/pull/4183) [#5303](https://github.com/vllm-project/vllm-ascend/pull/5303)
- **EPLB (Elastic Prefill Load Balancing)**: [Experimental]EPLB is now more stable with many bug fixes. Mix placement now works. [#6086](https://github.com/vllm-project/vllm-ascend/pull/6086)
- **Full Decode Only Mode**: Added support for Qwen3-Next and DeepSeekv32 in full_decode_only mode with bug fixes. [#3949](https://github.com/vllm-project/vllm-ascend/pull/3949) [#3986](https://github.com/vllm-project/vllm-ascend/pull/3986) [#3763](https://github.com/vllm-project/vllm-ascend/pull/3763)
- **Model Runner V2**: [Experimental]Added basic support for Model Runner V2, the next generation of vLLM. It will be used by default in future releases. [#5210](https://github.com/vllm-project/vllm-ascend/pull/5210)

### Features

- **W8A16 Quantization**: [Experimental]Added new W8A16 quantization method support. [#4541](https://github.com/vllm-project/vllm-ascend/pull/4541)
- **UCM Connector**: [Experimental]Added UCMConnector for KV Cache Offloading. [#4411](https://github.com/vllm-project/vllm-ascend/pull/4411)
- **Batch Invariant**: [Experimental]Implemented basic framework for batch invariant feature. [#5517](https://github.com/vllm-project/vllm-ascend/pull/5517)
- **Sampling**: Enhanced sampling with async_scheduler and disable_padded_drafter_batch support in Eagle. [#4893](https://github.com/vllm-project/vllm-ascend/pull/4893)

### Hardware and Operator Support

- **Custom Operators**: Added multiple custom operators including:
    - Fused matmul/reduce-scatter kernel [#3693](https://github.com/vllm-project/vllm-ascend/pull/3693)
    - mrope fusion op [#3708](https://github.com/vllm-project/vllm-ascend/pull/3708)
    - Triton chunk_gated_delta_rule ops for Qwen3-Next [#4070](https://github.com/vllm-project/vllm-ascend/pull/4070)
    - l2norm triton kernel [#4595](https://github.com/vllm-project/vllm-ascend/pull/4595)
    - RejectSampler, MoeInitRoutingCustom, DispatchFFNCombine custom ops
- **Operator Fusion**: Added AddRmsnormQuant fusion pattern with SP support and inductor fusion for quantization. [#5077](https://github.com/vllm-project/vllm-ascend/pull/5077) [#4168](https://github.com/vllm-project/vllm-ascend/pull/4168)
- **MLA/SFA**: Refactored SFA into MLA architecture for better maintainability. [#3769](https://github.com/vllm-project/vllm-ascend/pull/3769)
- **FIA Operator**: Adapted to npu_fused_infer_attention_score with flash decoding function. To optimize performance in small batch size scenarios, this attention operator is now available. Please refer to item 22 in [FAQs](https://docs.vllm.ai/projects/ascend/en/v0.13.0/faqs.html) to enable it. [#4025](https://github.com/vllm-project/vllm-ascend/pull/4025)
- **CANN 8.5 Support**: Removed CP redundant variables after FIA operator enables for CANN 8.5. [#6039](https://github.com/vllm-project/vllm-ascend/pull/6039)

### Performance

Many custom ops and triton kernels were added in this release to speed up model performance:

- **DeepSeek Performance**: [Experimental]Improved performance for DeepSeek V3.2 by eliminating HD synchronization in async scheduling and optimizing memory usage for MTP. [#4805](https://github.com/vllm-project/vllm-ascend/pull/4805) [#2713](https://github.com/vllm-project/vllm-ascend/pull/2713)
- **Qwen3-Next Performance**: [Experimental]Improved performance with Triton ops and optimizations. [#5664](https://github.com/vllm-project/vllm-ascend/pull/5664) [#5984](https://github.com/vllm-project/vllm-ascend/pull/5984) [#5765](https://github.com/vllm-project/vllm-ascend/pull/5765)
- **FlashComm**: Enhanced FlashComm v2 optimization with o_shared linear and communication domain fixes. [#3232](https://github.com/vllm-project/vllm-ascend/pull/3232) [#4188](https://github.com/vllm-project/vllm-ascend/pull/4188) [#4458](https://github.com/vllm-project/vllm-ascend/pull/4458) [#5848](https://github.com/vllm-project/vllm-ascend/pull/5848)
- **MoE Optimization**: Optimized all2allv for MoE models and enhanced all-reduce skipping logic. [#3738](https://github.com/vllm-project/vllm-ascend/pull/3738) [#5329](https://github.com/vllm-project/vllm-ascend/pull/5329)
- **Attention Optimization**: Moved attention update stream out of loop, converted BSND to TND format for long sequence optimization, and removed transpose step after attention switching to transpose_batchmatmul. [#3848](https://github.com/vllm-project/vllm-ascend/pull/3848) [#3778](https://github.com/vllm-project/vllm-ascend/pull/3778) [#5390](https://github.com/vllm-project/vllm-ascend/pull/5390)
- **Quantization Performance**: Moved quantization before allgather in Allgather EP. [#3420](https://github.com/vllm-project/vllm-ascend/pull/3420)
- **Layerwise Connector**: [Experimental]Improved performance of Layerwise Connector. [#5303](https://github.com/vllm-project/vllm-ascend/pull/5303)
- **Prefix Cache**: Improved performance of prefix cache features. [#4022](https://github.com/vllm-project/vllm-ascend/pull/4022)
- **Async Scheduling**: Fixed async copy and eliminated hangs in async scheduling. [#4113](https://github.com/vllm-project/vllm-ascend/pull/4113) [#4233](https://github.com/vllm-project/vllm-ascend/pull/4233)
- **Memory Operations**: Removed redundant D2H operations and deleted redundant operations in model_runner. [#4063](https://github.com/vllm-project/vllm-ascend/pull/4063) [#3677](https://github.com/vllm-project/vllm-ascend/pull/3677)
- **Rope Embedding**: Optimized rope embedding with triton kernel for huge performance gain. [#5918](https://github.com/vllm-project/vllm-ascend/pull/5918)
- **Sampling**: Added support for advanced apply_top_k_top_p without top_k constraint. [#6098](https://github.com/vllm-project/vllm-ascend/pull/6098)
- **Multimodal**: Parallelized Q/K/V padding in AscendMMEncoderAttention for better performance. [#6204](https://github.com/vllm-project/vllm-ascend/pull/6204)

### Dependencies

- **CANN**: Upgraded to 8.5.0 [#6112](https://github.com/vllm-project/vllm-ascend/pull/6112)
- **torch-npu**: Upgraded to 2.8.0.post2. It's installed in the docker container by default.
- **triton-ascend**: Upgraded to 3.2.0 [#6105](https://github.com/vllm-project/vllm-ascend/pull/6105)
- **vLLM**: Upgraded to 0.13.0 and dropped 0.12.0 support. [#5146](https://github.com/vllm-project/vllm-ascend/pull/5146)
- **Transformers**: Upgraded to >= 4.57.4 [#5250](https://github.com/vllm-project/vllm-ascend/pull/5250)

### Deprecation & Breaking Changes

- **CPUOffloadingConnector** is deprecated. We'll remove it in the next release. It'll be replaced by CPUOffload feature from vLLM in the future.
- **ProfileExecuteDuration** [feature](https://docs.vllm.ai/projects/ascend/en/v0.13.0/developer_guide/performance_and_debug/profile_execute_duration.html) is deprecated.
- **Ascend Scheduler** has been dropped. [#4623](https://github.com/vllm-project/vllm-ascend/pull/4623)
- **Torchair** has been dropped. [#4814](https://github.com/vllm-project/vllm-ascend/pull/4814)
- **VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE** is removed and `VLLM_ASCEND_ENABLE_PREFETCH_MLP` is recommended to replace as they were always enabled together. [#5272](https://github.com/vllm-project/vllm-ascend/pull/5272)
- **VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP** is dropped now. [#5270](https://github.com/vllm-project/vllm-ascend/pull/5270)
- **VLLM_ASCEND_ENABLE_NZ** is disabled for float weight case, since we noticed that the performance is not good in some float cases. Feel free to set it to 2 if you make sure it works for your case. [#4878](https://github.com/vllm-project/vllm-ascend/pull/4878)
- **chunked_prefill_for_mla** in `additional_config` is dropped now. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)
- **dump_config** in `additional_config` is renamed to `dump_config_path` and the type is changed from `dict` to `string`. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)
- **--task parameter** for embedding models is deprecated. [#5257](https://github.com/vllm-project/vllm-ascend/pull/5257)
- **The value of VLLM_ASCEND_ENABLE_MLAPO** env will be set to True by default in the next release. It'll be enabled in decode node by default. Please note that this feature will cost more memory. If you are memory sensitive, please set it to False.

### Documentation

- Added comprehensive developer guides for ACLGraph, MTP, KV Pool, EPLB, and PD disaggregation features
- Added tutorials for multiple models including DeepSeek-V3.2-Exp, Qwen3-Next, and various multimodal models
- Updated FAQ and configuration documentation

### Others

- **OOM Fix**: OOM error on VL models is fixed now. We're keeping observing it. If you hit OOM problem again, please submit an issue. [#5136](https://github.com/vllm-project/vllm-ascend/pull/5136)
- **Qwen3-Next-MTP Accuracy**: Fixed an accuracy bug of Qwen3-Next-MTP when batched inferring. [#4932](https://github.com/vllm-project/vllm-ascend/pull/4932)
- **ZMQ Bug Fix**: Fixed zmq send/receive failed bug. [#5503](https://github.com/vllm-project/vllm-ascend/pull/5503)
- **Weight Transpose**: Fixed weight transpose in RL scenarios. [#5567](https://github.com/vllm-project/vllm-ascend/pull/5567)
- **Eagle3 SP**: Adapted SP to eagle3. [#5562](https://github.com/vllm-project/vllm-ascend/pull/5562)
- **GLM4.6 MTP**: GLM4.6 now supports MTP with fullgraph. [#5460](https://github.com/vllm-project/vllm-ascend/pull/5460)
- **Flashcomm2 Oshard**: Flashcomm2 now works with oshard generalized feature. [#4723](https://github.com/vllm-project/vllm-ascend/pull/4723)
- **Fine-grained Shared Expert Overlap**: Support fine-grained shared expert overlap. [#5962](https://github.com/vllm-project/vllm-ascend/pull/5962)

### Known Issue

- Due to the upgrade of `transformers` package, some models quantization weight, such as `qwen2.5vl`, `gemma3`, `minimax`, may not work. We'll fix it in the next post release. [#6302](https://github.com/vllm-project/vllm-ascend/pull/6302)
- The performance of `Qwen3-32B` will not be good with 128K input case, it's suggested to enable pcp&dcp feature for this case. This will be improved in the next CANN release.
- The performance of `Qwen3-235B`, `Qwen3-480B` under prefill-decode scenario and EP=32 scenario is not good as expect. We'll improve it in the next post release.
- When deploy deepseek3.1 under prefill-decode scenario, please make sure the tp size for decode node is great than 1. `TP=1` doesn't work. This will be fixed in the next CANN release.

## v0.14.0rc1 - 2026.01.26

This is the first release candidate of v0.14.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started. This release includes all the changes in v0.13.0rc2. So We just list the differences from v0.13.0rc2. If you are upgrading from v0.13.0rc1, please read both v0.14.0rc1 and v0.13.0rc2 release notes.

### Highlights

- 310P support is back now. In this release, only basic dense and vl models are supported with eager mode. We'll keep improving and maintaining the support for 310P. [#5776](https://github.com/vllm-project/vllm-ascend/pull/5776)
- Support compressed tensors moe w8a8-int8 quantization. [#5718](https://github.com/vllm-project/vllm-ascend/pull/5718)
- Support Medusa speculative decoding. [#5668](https://github.com/vllm-project/vllm-ascend/pull/5668)
- Support Eagle3 speculative decoding for Qwen3vl. [#4848](https://github.com/vllm-project/vllm-ascend/pull/4848)

### Features

- Xlite Backend supports Qwen3 MoE now. [#5951](https://github.com/vllm-project/vllm-ascend/pull/5951)
- Support DSA-CP for PD-mix deployment case. [#5702](https://github.com/vllm-project/vllm-ascend/pull/5702)
- Add support of new W4A4_LAOS_DYNAMIC quantization method. [#5143](https://github.com/vllm-project/vllm-ascend/pull/5143)

### Performance

- The performance of Qwen3-next has been improved. [#5664](https://github.com/vllm-project/vllm-ascend/pull/5664) [#5984](https://github.com/vllm-project/vllm-ascend/pull/5984) [#5765](https://github.com/vllm-project/vllm-ascend/pull/5765)
- The CPU bind logic and performance has been improved. [#5555](https://github.com/vllm-project/vllm-ascend/pull/5555)
- Merge Q/K split to simplify AscendApplyRotaryEmb for better performance. [#5799](https://github.com/vllm-project/vllm-ascend/pull/5799)
- Add Matmul Allreduce Rmsnorm fusion Pass. It's disabled by default. Set `fuse_allreduce_rms=True` in `--additional_config` to enable it. [#5034](https://github.com/vllm-project/vllm-ascend/pull/5034)
- Optimize rope embedding with triton kernel for huge performance gain. [#5918](https://github.com/vllm-project/vllm-ascend/pull/5918)
- support advanced apply_top_k_top_p without top_k constraint. [#6098](https://github.com/vllm-project/vllm-ascend/pull/6098)
- Parallelize Q/K/V padding in AscendMMEncoderAttention for better performance. [#6204](https://github.com/vllm-project/vllm-ascend/pull/6204)

### Others

- model runner v2 support triton of penalty. [#5854](https://github.com/vllm-project/vllm-ascend/pull/5854)
- model runner v2 support eagle spec decoding. [#5840](https://github.com/vllm-project/vllm-ascend/pull/5840)
- Fix multimodal inference OOM issues by setting `expandable_segments:True` by default. [#5855](https://github.com/vllm-project/vllm-ascend/pull/5855)
- `VLLM_ASCEND_ENABLE_MLAPO` is set to `True` by default. It's enabled automatically on decode node in PD deployment case. Please note that this feature will cost more memory. If you are memory sensitive, please set it to False. [#5952](https://github.com/vllm-project/vllm-ascend/pull/5952)
- SSL config can be set to kv_extra_config for PD deployment with mooncake layerwise connector. [#5875](https://github.com/vllm-project/vllm-ascend/pull/5875)
- support `--max_model_len=auto`. [#6193](https://github.com/vllm-project/vllm-ascend/pull/6193)

### Dependencies

- torch-npu is upgraded to 2.9.0 [#6112](https://github.com/vllm-project/vllm-ascend/pull/6112)

### Deprecation & Breaking Changes

- EPLB config options is moved to `eplb_config` in [additional config](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html). The old ones are removed in this release.
- The profiler envs, such as `VLLM_TORCH_PROFILER_DIR` and `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` do not work with vLLM Ascend now. Please use vLLM `--profiler-config` parameters instead. [#5928](https://github.com/vllm-project/vllm-ascend/pull/5928)

### Known Issues

- If you hit the pickle error from `EngineCore` process sometimes, please cherry-pick the [PR](https://github.com/vllm-project/vllm/pull/32022) into your local vLLM code. This known issue will be fixed in vLLM in the next release.

## v0.13.0rc2 - 2026.01.24

This is the second release candidate of v0.13.0 for vLLM Ascend. In this rc release, we fixed lots of bugs and improved the performance of many models. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.13.0/) to get started. Any feedback is welcome to help us to improve the final version of v0.13.0.

### Highlights

We mainly focus on quality and performance improvement in this release. The spec decode, graph mode, context parallel and EPLB have been improved significantly. A lot of bugs have been fixed and the performance has been improved for DeepSeek3.1/3.2, Qwen3 Dense/MOE models.

### Features

- implement basic framework for batch invariant [#5517](https://github.com/vllm-project/vllm-ascend/pull/5517)
- Eagle spec decode feature now works with full graph mode. [#5118](https://github.com/vllm-project/vllm-ascend/pull/5118)
- Context Parallel(PCP&DCP) feature is more stable now. And it works for most case. Please try it out.
- MTP and eagle spec decode feature now works in most cases. And it's suggested to use them in most cases.
- EPLB feature more stable now. Many bugs have been fixed. Mix placement works now [#6086](https://github.com/vllm-project/vllm-ascend/pull/6086)
- Support kv nz feature for DeepSeek decode node in disagg-prefill scenario [#3072](https://github.com/vllm-project/vllm-ascend/pull/3072)

### Model Support

- LongCat-Flash is supported now.[#3833](https://github.com/vllm-project/vllm-ascend/pull/3833)
- minimax_m2 is supported now. [#5624](https://github.com/vllm-project/vllm-ascend/pull/5624)
- Support for cross-attention and whisper models [#5592](https://github.com/vllm-project/vllm-ascend/pull/5592)

### Performance

- Many custom ops and triton kernels are added in this release to speed up the performance of models. Such as `RejectSampler`, `MoeInitRoutingCustom`, `DispatchFFNCombine` and so on.
- Improved the performance of Layerwise Connector [#5303](https://github.com/vllm-project/vllm-ascend/pull/5303)

### Others

- Basic support Model Runner v2. Model Runner V2 is the next generation of vLLM. It will be used by default in the future release. [#5210](https://github.com/vllm-project/vllm-ascend/pull/5210)
- Fixed a bug that the zmq send/receive may failed [#5503](https://github.com/vllm-project/vllm-ascend/pull/5503)
- Supported to use full-graph with Qwen3-Next-MTP [#5477](https://github.com/vllm-project/vllm-ascend/pull/5477)
- Fix weight transpose in RL scenarios [#5567](https://github.com/vllm-project/vllm-ascend/pull/5567)
- Adapted SP to eagle3 [#5562](https://github.com/vllm-project/vllm-ascend/pull/5562)
- Context Parallel(PCP&DCP) support mlapo [#5672](https://github.com/vllm-project/vllm-ascend/pull/5672)
- GLM4.6 support mtp with fullgraph [#5460](https://github.com/vllm-project/vllm-ascend/pull/5460)
- Flashcomm2 now works with oshard generalized feature [#4723](https://github.com/vllm-project/vllm-ascend/pull/4723)
- Support setting tp=1 for the Eagle draft model [#5804](https://github.com/vllm-project/vllm-ascend/pull/5804)
- Flashcomm1 feature now works with qwen3-vl [#5848](https://github.com/vllm-project/vllm-ascend/pull/5848)
- Support fine-grained shared expert overlap [#5962](https://github.com/vllm-project/vllm-ascend/pull/5962)

### Dependencies

- CANN is upgraded to 8.5.0
- torch-npu is upgraded to 2.8.0.post1. Please note that the post version will not be installed by default. Please install it by hand from [pypi mirror](https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/).
- triton-ascend is upgraded to 3.2.0

### Deprecation & Breaking Changes

- `CPUOffloadingConnector` is deprecated. We'll remove it in the next release. It'll be replaced by CPUOffload feature from vLLM in the future.
- eplb config options is moved to `eplb_config` in [additional config](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html). The old ones will be removed in the next release.
- `ProfileExecuteDuration` [feature](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/profile_execute_duration.html) is deprecated. It's replaced by `ObservabilityConfig` from vLLM.
- The value of `VLLM_ASCEND_ENABLE_MLAPO` env will be set to True by default in the next release. It'll be enabled in decode node by default. Please note that this feature will cost more memory. If you are memory sensitive, please set it to False.

## v0.13.0rc1 - 2025.12.27

This is the first release candidate of v0.13.0 for vLLM Ascend. We landed lots of bug fix, performance improvement and feature support in this release. Any feedback is welcome to help us to improve vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights

- Improved the performance of DeepSeek V3.2, please refer to [tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/DeepSeek-V3.2.html)
- Qwen3-Next MTP with chunked prefill is supported now [#4770](https://github.com/vllm-project/vllm-ascend/pull/4770), please refer to [tutorials](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/Qwen3-Next.html)
- [Experimental] Prefill Context Parallel and Decode Context Parallel are supported, but notice that it is an experimental feature now, welcome any feedback. please refer to [context parallel feature guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)

### Features

- Support openPangu Ultra MoE [4615](https://github.com/vllm-project/vllm-ascend/pull/4615)
- A new quantization method W8A16 is supported now. [#4541](https://github.com/vllm-project/vllm-ascend/pull/4541)
- Cross-machine Disaggregated Prefill is supported now. [#5008](https://github.com/vllm-project/vllm-ascend/pull/5008)
- Add UCMConnector for KV Cache Offloading. [#4411](https://github.com/vllm-project/vllm-ascend/pull/4411)
- Support async_scheduler and disable_padded_drafter_batch in eagle. [#4893](https://github.com/vllm-project/vllm-ascend/pull/4893)
- Support pcp + mtp in full graph mode. [#4572](https://github.com/vllm-project/vllm-ascend/pull/4572)
- Enhance all-reduce skipping logic for MoE models in NPUModelRunner [#5329](https://github.com/vllm-project/vllm-ascend/pull/5329)

### Performance

Some general performance improvement:

- Add l2norm triton kernel [#4595](https://github.com/vllm-project/vllm-ascend/pull/4595)
- Add new pattern for AddRmsnormQuant with SP, which could only take effect in graph mode. [#5077](https://github.com/vllm-project/vllm-ascend/pull/5077)
- Add async exponential while model executing. [#4501](https://github.com/vllm-project/vllm-ascend/pull/4501)
- Remove the transpose step after attention and switch to transpose_batchmatmul [#5390](https://github.com/vllm-project/vllm-ascend/pull/5390)
- To optimize the performance in small batch size scenario, an attention operator with flash decoding function is offered, please refer to item 22 in [FAQs](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html) to enable it.

### Other

- OOM error on VL models is fixed now. We're keeping observing it, if you hit OOM problem again, please submit an issue. [#5136](https://github.com/vllm-project/vllm-ascend/pull/5136)
- Fixed an accuracy bug of Qwen3-Next-MTP when batched inferring. [#4932](https://github.com/vllm-project/vllm-ascend/pull/4932)
- Fix npu-cpu offloading interface change bug. [#5290](https://github.com/vllm-project/vllm-ascend/pull/5290)
- Fix MHA model runtime error in aclgraph mode [#5397](https://github.com/vllm-project/vllm-ascend/pull/5397)
- Fix unsuitable moe_comm_type under ep=1 scenario [#5388](https://github.com/vllm-project/vllm-ascend/pull/5388)

### Deprecation & Breaking Changes

- `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE` is removed and `VLLM_ASCEND_ENABLE_PREFETCH_MLP` is recommend to replace as they always be enabled together. [#5272](https://github.com/vllm-project/vllm-ascend/pull/5272)
- `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` is dropped now. [#5270](https://github.com/vllm-project/vllm-ascend/pull/5270)
- `VLLM_ASCEND_ENABLE_NZ` is disabled for float weight case, since we notice that the performance is not good in some float case. Feel free to set it to 2 if you make sure it works for your case. [#4878](https://github.com/vllm-project/vllm-ascend/pull/4878)
- `chunked_prefill_for_mla` in `additional_config` is dropped now. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)
- `dump_config` in `additional_config` is renamed to `dump_config_path` and the type is change from `dict` to `string`. [#5296](https://github.com/vllm-project/vllm-ascend/pull/5296)

### Dependencies

- vLLM version has been upgraded to 0.13.0 and drop 0.12.0 support. [#5146](https://github.com/vllm-project/vllm-ascend/pull/5146)
- Transformer version has been upgraded >= 4.57.3 [#5250](https://github.com/vllm-project/vllm-ascend/pull/5250)

### Known Issues

- Qwen3-Next doesn't support long sequence scenario, and we should limit `gpu-memory-utilization` according to the doc to run Qwen3-Next. We'll improve it in the next release
- The functional break on Qwen3-Next when the input/output is around 3.5k/1.5k is fixed, but it introduces a regression on performance. We'll fix it in next release. [#5357](https://github.com/vllm-project/vllm-ascend/issues/5357)
- There is a precision issue with curl on ultra-short sequences in DeepSeek-V3.2. We'll fix it in next release. [#5370](https://github.com/vllm-project/vllm-ascend/issues/5370)

## v0.11.0 - 2025.12.16

We're excited to announce the release of v0.11.0 for vLLM Ascend. This is the official release for v0.11.0. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.11.0) to get started. We'll consider to release post version in the future if needed. This release note will only contain the important change and note from v0.11.0rc3.

### Highlights

- Improved the performance for deepseek 3/3.1. [#3995](https://github.com/vllm-project/vllm-ascend/pull/3995)
- Fixed the accuracy bug for qwen3-vl. [#4811](https://github.com/vllm-project/vllm-ascend/pull/4811)
- Improved the performance of sample. [#4153](https://github.com/vllm-project/vllm-ascend/pull/4153)
- Eagle3 is back now. [#4721](https://github.com/vllm-project/vllm-ascend/pull/4721)

### Other

- Improved the performance for kimi-k2.  [#4555](https://github.com/vllm-project/vllm-ascend/pull/4555)
- Fixed a quantization bug for deepseek3.2-exp. [#4797](https://github.com/vllm-project/vllm-ascend/pull/4797)
- Fixed qwen3-vl-moe bug under high concurrency. [#4658](https://github.com/vllm-project/vllm-ascend/pull/4658)
- Fixed an accuracy bug for Prefill Decode disaggregation case. [#4437](https://github.com/vllm-project/vllm-ascend/pull/4437)
- Fixed some bugs for EPLB [#4576](https://github.com/vllm-project/vllm-ascend/pull/4576) [#4777](https://github.com/vllm-project/vllm-ascend/pull/4777)
- Fixed the version incompatibility issue for openEuler docker image. [#4745](https://github.com/vllm-project/vllm-ascend/pull/4745)

### Deprecation announcement

- LLMdatadist connector has been deprecated, it'll be removed in v0.12.0rc1
- Torchair graph has been deprecated, it'll be removed in v0.12.0rc1
- Ascend scheduler has been deprecated, it'll be removed in v0.12.0rc1

### Upgrade notice

- torch-npu is upgraded to 2.7.1.post1. Please note that the package is pushed to [pypi mirror](https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/). So it's hard to add it to auto dependence. Please install it by yourself.
- CANN is upgraded to 8.3.rc2.

### Known Issues

- Qwen3-Next doesn't support expert parallel and MTP features in this release. And it'll be oom if the input is too long. We'll improve it in the next release
- Deepseek 3.2 only work with torchair graph mode in this release. We'll make it work with aclgraph mode in the next release.
- Qwen2-audio doesn't work by default. Temporary solution is to set `--gpu-memory-utilization` to a suitable value, such as 0.8.
- CPU bind feature doesn't work if more than one vLLM instance is running on the same node.

## v0.12.0rc1 - 2025.12.13

This is the first release candidate of v0.12.0 for vLLM Ascend. We landed lots of bug fix, performance improvement and feature support in this release. Any feedback is welcome to help us to improve vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights

- DeepSeek 3.2 is stable and performance is improved. In this release, you don't need to install any other packages now. Following the [official tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/DeepSeek-V3.2.html) to start using it.
- Async scheduler is more stable and ready to enable now. Please set `--async-scheduling` to enable it.
- More new models, such as Qwen3-omni, DeepSeek OCR, PaddleOCR, OpenCUA are supported now.

### Core

- [Experimental] Full decode only graph mode is supported now. Although it is not enabled by default, we suggest to enable it by `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` in most case. Let us know if you hit any error. We'll keep improve it and enable it by default in next few release.
- Lots of triton kernel are added. The performance of vLLM Ascend, especially Qwen3-Next and DeepSeek 3.2 is improved. Please note that triton is not installed and enabled by default, but we suggest to enable it in most case. You can download and install it by hand from [package url](https://vllm-ascend.obs.cn-north-4.myhuaweicloud.com/vllm-ascend/triton_ascend-3.2.0.dev2025110717-cp311-cp311-manylinux_2_27_aarch64.whl). If you're running vLLM Ascend with X86, you need to build triton ascend by yourself from [source](https://gitcode.com/Ascend/triton-ascend)
- Lots of Ascend ops are added to improve the performance. It means that from this release vLLM Ascend only works with custom ops built. So we removed the env `COMPILE_CUSTOM_KERNELS`. You can not set it to 0 now.
- speculative decode method `MTP` is more stable now. It can be enabled with most case and decode token number can be 1,2,3.
- speculative decode method `suffix` is supported now. Thanks for the contribution from China Merchants Bank.
- llm-comppressor quantization tool with W8A8 works now. You can now deploy the model with W8A8 quantization from this tool directly.
- W4A4 quantization works now.
- Support features flashcomm1 and flashcomm2 in paper [flashcomm](https://arxiv.org/pdf/2412.04964) [#3004](https://github.com/vllm-project/vllm-ascend/pull/3004) [#3334](https://github.com/vllm-project/vllm-ascend/pull/3334)
- Pooling model, such as bge, reranker, etc. are supported now
- Official doc has been improved. we refactored the tutorial to make it more clear. The user guide and developer guide is more complete now. We'll keep improving it.

### Other

- [Experimental] Mooncake layerwise connector is supported now.
- [Experimental] [KV cache pool](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/KV_Cache_Pool_Guide.html) feature is added
- [Experimental] A new graph mode `xlite` is introduced. It performs good with some models. Following the [official tutorial](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html#using-xlitegraph) to start using it.
- LLMdatadist kv connector is removed. Please use mooncake connector instead.
- Ascend scheduler is removed. `--additional-config {"ascend_scheduler": {"enabled": true}` doesn't work anymore.
- Torchair graph mode is removed. `--additional-config {"torchair_graph_config": {"enabled": true}}` doesn't work anymore. Please use aclgraph instead.
- `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` env is removed. This feature is stable enough. We enable it by default now.
- speculative decode method `Ngram` is back now.
- msprobe tool is added to help user to check the model accuracy. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/msprobe_guide.html) to get started.
- msserviceprofiler tool is added to help user to profile the model performance. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html) to get started.

### Upgrade Note

- vLLM Ascend self maintained modeling file has been removed. The related python entrypoint is removed as well. So please uninstall the old version of vLLM Ascend in your env before upgrade.
- CANN is upgraded to 8.3.RC2, Pytorch and torch-npu are upgraded to 2.8.0. Don't forget to install them.
- Python 3.9 support is dropped to keep the same with vLLM v0.12.0

### Known Issues

- DeepSeek 3/3.1 and Qwen3 doesn't work with FULL_DECODE_ONLY graph mode. We'll fix it in next release. [#4990](https://github.com/vllm-project/vllm-ascend/pull/4990)
- Hunyuan OCR doesn't work. We'll fix it in the next release. [#4989](https://github.com/vllm-project/vllm-ascend/pull/4989) [#4992](https://github.com/vllm-project/vllm-ascend/pull/4992)
- DeepSeek 3.2 doesn't work with chat template. It because that vLLM v0.12.0 doesn't support it. We'll support in the next v0.13.0rc1 version.
- DeepSeek 3.2 doesn't work with high concurrency in some case. We'll fix it in next release. [#4996](https://github.com/vllm-project/vllm-ascend/pull/4996)
- We notice that bf16/fp16 model doesn't perform well, it's mainly because that `VLLM_ASCEND_ENABLE_NZ` is enabled by default. Please set `VLLM_ASCEND_ENABLE_NZ=0` to disable it. We'll add the auto detection mechanism in next release.
- speculative decode method `suffix` doesn't work. We'll fix it in next release. You can pick this commit to fix the issue: [#5010](https://github.com/vllm-project/vllm-ascend/pull/5010)

## v0.11.0rc3 - 2025.12.03

This is the third release candidate of v0.11.0 for vLLM Ascend. For quality reasons, we released a new rc before the official release. Thanks for all your feedback. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.11.0) to get started.

### Highlights

- torch-npu is upgraded to 2.7.1.post1. Please note that the package is pushed to [pypi mirror](https://mirrors.huaweicloud.com/ascend/repos/pypi/torch-npu/). So it's hard to add it to auto dependence. Please install it by yourself.
- Disable NZ weight loader to speed up dense model. Please note that this is a temporary solution. If you find the performance becomes bad, please let us know. We'll keep improving it. [#4495](https://github.com/vllm-project/vllm-ascend/pull/4495)
- mooncake is installed in official docker image now. You can use it directly in container now. [#4506](https://github.com/vllm-project/vllm-ascend/pull/4506)

### Other

- Fix an OOM issue for moe models. [#4367](https://github.com/vllm-project/vllm-ascend/pull/4367)
- Fix hang issue of multimodal model when running with DP>1 [#4393](https://github.com/vllm-project/vllm-ascend/pull/4393)
- Fix some bugs for EPLB [#4416](https://github.com/vllm-project/vllm-ascend/pull/4416)
- Fix bug for mtp>1 + lm_head_tp>1 case [#4360](https://github.com/vllm-project/vllm-ascend/pull/4360)
- Fix a accuracy issue when running vLLM serve for long time. [#4117](https://github.com/vllm-project/vllm-ascend/pull/4117)
- Fix a function bug when running qwen2.5 vl under high concurrency. [#4553](https://github.com/vllm-project/vllm-ascend/pull/4553)

## v0.11.0rc2 - 2025.11.21

This is the second release candidate of v0.11.0 for vLLM Ascend. In this release, we solved many bugs to improve the quality. Thanks for all your feedback. We'll keep working on bug fix and performance improvement. The v0.11.0 official release will come soon. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.11.0) to get started.

### Highlights

- CANN is upgraded to 8.3.RC2. [#4332](https://github.com/vllm-project/vllm-ascend/pull/4332)
- Ngram spec decode method is back now. [#4092](https://github.com/vllm-project/vllm-ascend/pull/4092)
- The performance of aclgraph is improved by updating default capture size. [#4205](https://github.com/vllm-project/vllm-ascend/pull/4205)

### Core

- Speed up vLLM startup time. [#4099](https://github.com/vllm-project/vllm-ascend/pull/4099)
- Kimi k2 with quantization works now. [#4190](https://github.com/vllm-project/vllm-ascend/pull/4190)
- Fix a bug for qwen3-next. It's more stable now. [#4025](https://github.com/vllm-project/vllm-ascend/pull/4025)

### Other

- Fix an issue for full decode only mode. Full graph mode is more stable now. [#4106](https://github.com/vllm-project/vllm-ascend/pull/4106) [#4282](https://github.com/vllm-project/vllm-ascend/pull/4282)
- Fix a allgather ops bug for DeepSeek V3 series models. [#3711](https://github.com/vllm-project/vllm-ascend/pull/3711)
- Fix some bugs for EPLB feature. [#4150](https://github.com/vllm-project/vllm-ascend/pull/4150) [#4334](https://github.com/vllm-project/vllm-ascend/pull/4334)
- Fix a bug that vl model doesn't work on x86 machine. [#4285](https://github.com/vllm-project/vllm-ascend/pull/4285)
- Support ipv6 for prefill disaggregation proxy. Please note that mooncake connector doesn't work with ipv6. We're working on it. [#4242](https://github.com/vllm-project/vllm-ascend/pull/4242)
- Add a check that to ensure EPLB only support w8a8 method for quantization case. [#4315](https://github.com/vllm-project/vllm-ascend/pull/4315)
- Add a check that to ensure FLASHCOMM feature doesn't work with vl model. It'll be supported in 2025 Q4 [#4222](https://github.com/vllm-project/vllm-ascend/pull/4222)
- Audio required library is installed in container. [#4324](https://github.com/vllm-project/vllm-ascend/pull/4324)

### Known Issues

- Ray + EP doesn't work, if you run vLLM Ascend with ray, please disable expert parallelism. [#4123](https://github.com/vllm-project/vllm-ascend/pull/4123)
- `response_format` parameter is not supported yet. We'll support it soon. [#4175](https://github.com/vllm-project/vllm-ascend/pull/4175)
- cpu bind feature doesn't work for multi instance case(Such as multi DP on one node). We'll fix it in the next release.

## v0.11.0rc1 - 2025.11.10

This is the first release candidate of v0.11.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.11.0) to get started.
v0.11.0 will be the next official release version of vLLM Ascend. We'll release it in the next few days. Any feedback is welcome to help us to improve v0.11.0.

### Highlights

- CANN is upgrade to 8.3.RC1. Torch-npu is upgrade to 2.7.1. [#3945](https://github.com/vllm-project/vllm-ascend/pull/3945) [#3896](https://github.com/vllm-project/vllm-ascend/pull/3896)
- PrefixCache and Chunked Prefill are enabled by default. [#3967](https://github.com/vllm-project/vllm-ascend/pull/3967)
- W4A4 quantization is supported now. [#3427](https://github.com/vllm-project/vllm-ascend/pull/3427) Official tutorial is available at [here](<https://docs.vllm.ai/projects/ascend/en/latest/tutorials/single_npu_qwen3_w4a4.html>.

### Core

- Performance of Qwen3 and Deepseek V3 series models are improved.
- Mooncake layerwise connector is supported now [#2602](https://github.com/vllm-project/vllm-ascend/pull/2602). Find tutorial [here](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/pd_disaggregation_mooncake_multi_node.html).
- MTP > 1 is supported now. [#2708](https://github.com/vllm-project/vllm-ascend/pull/2708)
- [Experimental] Graph mode `FULL_DECODE_ONLY` is supported now! And `FULL` will be landing in the next few weeks. [#2128](https://github.com/vllm-project/vllm-ascend/pull/2128)
- Pooling models, such as bge-m3, are supported now. [#3171](https://github.com/vllm-project/vllm-ascend/pull/3171)

### Other

- Refactor the MOE module to make it clearer and easier to understand and the performance has improved in both quantitative and non-quantitative scenarios.
- Refactor model register module to make it easier to maintain. We'll remove this module in Q4 2025. [#3004](https://github.com/vllm-project/vllm-ascend/pull/3004)
- Torchair is deprecated. We'll remove it once the performance of ACL Graph is good enough. The deadline is Q1 2026.
- LLMDatadist KV Connector is deprecated. We'll remove it in Q1 2026.
- Refactor the linear module to support features flashcomm1 and flashcomm2 in paper [flashcomm](https://arxiv.org/pdf/2412.04964) [#3004](https://github.com/vllm-project/vllm-ascend/pull/3004) [#3334](https://github.com/vllm-project/vllm-ascend/pull/3334)

### Known issue

- The memory may be leaked and the service may be stuck after long time serving. This is a bug from torch-npu, we'll upgrade and fix it soon.
- The accuracy of qwen2.5 VL is not very good. This is a bug lead by CANN, we fix it soon.
- For long sequence input case, there is no response sometimes and the kv cache usage is become higher. This is a bug for scheduler. We are working on it.
- Qwen2-audio doesn't work by default, we're fixing it. Temporary solution is to set `--gpu-memory-utilization` to a suitable value, such as 0.8.
- When running Qwen3-Next with expert parallel enabled, please set `HCCL_BUFFSIZE` environment variable to a suitable value, such as 1024.
- The accuracy of DeepSeek3.2 with aclgraph is not correct. Temporary solution is to set `cudagraph_capture_sizes` to a suitable value depending on the batch size for the input.

## v0.11.0rc0 - 2025.09.30

This is the special release candidate of v0.11.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started.

### Highlights

- DeepSeek V3.2 is supported now. [#3270](https://github.com/vllm-project/vllm-ascend/pull/3270)
- Qwen3-vl is supported now. [#3103](https://github.com/vllm-project/vllm-ascend/pull/3103)

### Core

- DeepSeek works with aclgraph now. [#2707](https://github.com/vllm-project/vllm-ascend/pull/2707)
- MTP works with aclgraph now. [#2932](https://github.com/vllm-project/vllm-ascend/pull/2932)
- EPLB is supported now. [#2956](https://github.com/vllm-project/vllm-ascend/pull/2956)
- Mooncacke store kvcache connector is supported now. [#2913](https://github.com/vllm-project/vllm-ascend/pull/2913)
- CPU offload connector is supported now. [#1659](https://github.com/vllm-project/vllm-ascend/pull/1659)

### Others

- Qwen3-next is stable now. [#3007](https://github.com/vllm-project/vllm-ascend/pull/3007)
- Fixed a lot of bugs introduced in v0.10.2 by Qwen3-next. [#2964](https://github.com/vllm-project/vllm-ascend/pull/2964) [#2781](https://github.com/vllm-project/vllm-ascend/pull/2781) [#3070](https://github.com/vllm-project/vllm-ascend/pull/3070) [#3113](https://github.com/vllm-project/vllm-ascend/pull/3113)
- The LoRA feature is back now. [#3044](https://github.com/vllm-project/vllm-ascend/pull/3044)
- Eagle3 spec decode method is back now. [#2949](https://github.com/vllm-project/vllm-ascend/pull/2949)

## v0.10.2rc1 - 2025.09.16

This is the 1st release candidate of v0.10.2 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started.

### Highlights

- Added support for Qwen3-Next. Please note that the expert parallel and MTP features do not work with this release. We will be adding support for them soon. Follow the [official guide](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_npu_qwen3_next.html) to get started. [#2917](https://github.com/vllm-project/vllm-ascend/pull/2917)
- Added quantization support for aclgraph [#2841](https://github.com/vllm-project/vllm-ascend/pull/2841)

### Core

- Aclgraph now works with Ray backend. [#2589](https://github.com/vllm-project/vllm-ascend/pull/2589)
- MTP now works with the token > 1. [#2708](https://github.com/vllm-project/vllm-ascend/pull/2708)
- Qwen2.5 VL now works with quantization. [#2778](https://github.com/vllm-project/vllm-ascend/pull/2778)
- Improved the performance with async scheduler enabled. [#2783](https://github.com/vllm-project/vllm-ascend/pull/2783)
- Fixed the performance regression with non MLA model when using default scheduler. [#2894](https://github.com/vllm-project/vllm-ascend/pull/2894)

### Others

- The performance of W8A8 quantization is improved. [#2275](https://github.com/vllm-project/vllm-ascend/pull/2275)
- The performance is improved for moe models. [#2689](https://github.com/vllm-project/vllm-ascend/pull/2689) [#2842](https://github.com/vllm-project/vllm-ascend/pull/2842)
- Fixed resources limit error when apply speculative decoding and aclgraph. [#2472](https://github.com/vllm-project/vllm-ascend/pull/2472)
- Fixed the git config error in Docker images. [#2746](https://github.com/vllm-project/vllm-ascend/pull/2746)
- Fixed the sliding windows attention bug with prefill. [#2758](https://github.com/vllm-project/vllm-ascend/pull/2758)
- The official doc for Prefill-Decode Disaggregation with Qwen3 is added. [#2751](https://github.com/vllm-project/vllm-ascend/pull/2751)
- `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` env works again. [#2740](https://github.com/vllm-project/vllm-ascend/pull/2740)
- A new improvement for oproj in deepseek is added. Set `oproj_tensor_parallel_size` to enable this feature. [#2167](https://github.com/vllm-project/vllm-ascend/pull/2167)
- Fix a bug that deepseek with torchair doesn't work as expect when `graph_batch_sizes` is set. [#2760](https://github.com/vllm-project/vllm-ascend/pull/2760)
- Avoid duplicate generation of sin_cos_cache in rope when kv_seqlen > 4k. [#2744](https://github.com/vllm-project/vllm-ascend/pull/2744)
- The performance of Qwen3 dense model is improved with flashcomm_v1. Set `VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1` and `VLLM_ASCEND_ENABLE_FLASHCOMM=1` to enable it. [#2779](https://github.com/vllm-project/vllm-ascend/pull/2779)
- The performance of Qwen3 dense model is improved with prefetch feature. Set `VLLM_ASCEND_ENABLE_PREFETCH_MLP=1` to enable it. [#2816](https://github.com/vllm-project/vllm-ascend/pull/2816)
- The performance of Qwen3 MoE model is improved with rope ops update. [#2571](https://github.com/vllm-project/vllm-ascend/pull/2571)
- Fix the weight load error for RLHF case. [#2756](https://github.com/vllm-project/vllm-ascend/pull/2756)
- Add warm_up_atb step to speed up the inference. [#2823](https://github.com/vllm-project/vllm-ascend/pull/2823)
- Fixed the aclgraph steam error for moe model. [#2827](https://github.com/vllm-project/vllm-ascend/pull/2827)

### Known Issues

- The server will hang when running Prefill Decode Disaggregation with different TP size for P and D. It's fixed by [vLLM commit](https://github.com/vllm-project/vllm/pull/23917) which is not included in v0.10.2. You can pick this commit to fix the issue.
- The HBM usage of Qwen3-Next is higher than expected. It is a [known issue](https://github.com/vllm-project/vllm-ascend/issues/2884) and we are working on it. You can set `max_model_len` and `gpu_memory_utilization` to suitable value based on your parallel configuration to avoid oom error.
- We notice that LoRA does not work with this release due to the refactor of KV cache. We will fix it soon. [2941](https://github.com/vllm-project/vllm-ascend/issues/2941)
- Please do not enable chunked prefill with prefix cache when running with Ascend scheduler. The performance and accuracy is not good/correct. [#2943](https://github.com/vllm-project/vllm-ascend/issues/2943)

## v0.10.1rc1 - 2025.09.04

This is the 1st release candidate of v0.10.1 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started.

### Highlights

- LoRA Performance improved much through adding Custom Kernels by China Merchants Bank. [#2325](https://github.com/vllm-project/vllm-ascend/pull/2325)
- Support Mooncake TransferEngine for kv cache register and pull_blocks style disaggregate prefill implementation. [#1568](https://github.com/vllm-project/vllm-ascend/pull/1568)
- Support capture custom ops into aclgraph now. [#2113](https://github.com/vllm-project/vllm-ascend/pull/2113)

### Core

- Added MLP tensor parallel to improve performance, but note that this will increase memory usage. [#2120](https://github.com/vllm-project/vllm-ascend/pull/2120)
- openEuler is upgraded to 24.03. [#2631](https://github.com/vllm-project/vllm-ascend/pull/2631)
- Added custom lmhead tensor parallel to achieve reduced memory consumption and improved TPOT performance. [#2309](https://github.com/vllm-project/vllm-ascend/pull/2309)
- Qwen3 MoE/Qwen2.5 support torchair graph now. [#2403](https://github.com/vllm-project/vllm-ascend/pull/2403)
- Support Sliding Window Attention with AscendSceduler, thus fixing Gemma3 accuracy issue. [#2528](https://github.com/vllm-project/vllm-ascend/pull/2528)

### Others

- Bug fixes:
    - Updated the graph capture size calculation, somehow alleviated the problem that NPU stream not enough in some scenarios. [#2511](https://github.com/vllm-project/vllm-ascend/pull/2511)
    - Fixed bugs and refactor cached mask generation logic. [#2442](https://github.com/vllm-project/vllm-ascend/pull/2442)
    - Fixed the nz format does not work in quantization scenarios. [#2549](https://github.com/vllm-project/vllm-ascend/pull/2549)
    - Fixed the accuracy issue on Qwen series caused by enabling `enable_shared_pert_dp` by default. [#2457](https://github.com/vllm-project/vllm-ascend/pull/2457)
    - Fixed the accuracy issue on models whose rope dim is not equal to head dim, e.g., GLM4.5. [#2601](https://github.com/vllm-project/vllm-ascend/pull/2601)
- Performance improved through a lot of prs:
    - Removed torch.cat and replaced it with List[0]. [#2153](https://github.com/vllm-project/vllm-ascend/pull/2153)
    - Converted the format of gmm to nz. [#2474](https://github.com/vllm-project/vllm-ascend/pull/2474)
    - Optimized parallel strategies to reduce communication overhead. [#2198](https://github.com/vllm-project/vllm-ascend/pull/2198)
    - Optimized reject sampler in greedy situation. [#2137](https://github.com/vllm-project/vllm-ascend/pull/2137)
- A batch of refactoring PRs to enhance the code architecture:
    - Refactor on MLA. [#2465](https://github.com/vllm-project/vllm-ascend/pull/2465)
    - Refactor on torchair fused_moe. [#2438](https://github.com/vllm-project/vllm-ascend/pull/2438)
    - Refactor on allgather/mc2-related fused_experts. [#2369](https://github.com/vllm-project/vllm-ascend/pull/2369)
    - Refactor on torchair model runner. [#2208](https://github.com/vllm-project/vllm-ascend/pull/2208)
    - Refactor on CI. [#2276](https://github.com/vllm-project/vllm-ascend/pull/2276)
- Parameters changes:
    - Added `lmhead_tensor_parallel_size` in `additional_config`, set it to enable lmhead tensor parallel. [#2309](https://github.com/vllm-project/vllm-ascend/pull/2309)
    - Some unused environment variables `HCCN_PATH`, `PROMPT_DEVICE_ID`, `DECODE_DEVICE_ID`, `LLMDATADIST_COMM_PORT` and `LLMDATADIST_SYNC_CACHE_WAIT_TIME`  are removed. [#2448](https://github.com/vllm-project/vllm-ascend/pull/2448)
    - Environment variable `VLLM_LLMDD_RPC_PORT` is renamed to `VLLM_ASCEND_LLMDD_RPC_PORT` now. [#2450](https://github.com/vllm-project/vllm-ascend/pull/2450)
    - Added `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` in environment variables, Whether to enable mlp optimize when tensor parallel is enabled. This feature provides better performance in eager mode. [#2120](https://github.com/vllm-project/vllm-ascend/pull/2120)
    - Removed `MOE_ALL2ALL_BUFFER` and `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ` in environment variables. [#2612](https://github.com/vllm-project/vllm-ascend/pull/2612)
    - Added `enable_prefetch` in `additional_config`, Whether to enable weight prefetch. [#2465](https://github.com/vllm-project/vllm-ascend/pull/2465)
    - Added `mode` in `additional_config.torchair_graph_config`, When using reduce-overhead mode for torchair, mode needs to be set. [#2461](https://github.com/vllm-project/vllm-ascend/pull/2461)
    - `enable_shared_expert_dp` in `additional_config` is disabled by default now, and it is recommended to be enabled when inferencing with deepseek. [#2457](https://github.com/vllm-project/vllm-ascend/pull/2457)

### Known Issues

- Sliding window attention not support chunked prefill currently, thus we could only enable AscendScheduler to run with it. [#2729](https://github.com/vllm-project/vllm-ascend/issues/2729)
- There is a bug with creating mc2_mask when MultiStream is enabled, will fix it in next release. [#2681](https://github.com/vllm-project/vllm-ascend/pull/2681)

## v0.9.1 - 2025.09.03

We are excited to announce the newest official release of vLLM Ascend. This release includes many feature supports, performance improvements and bug fixes. We recommend users to upgrade from 0.7.3 to this version. Please always set `VLLM_USE_V1=1` to use V1 engine.

In this release, we added many enhancements for large scale expert parallel case. It's recommended to follow the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html).

Please note that this release note will list all the important changes from last official release(v0.7.3)

### Highlights

- DeepSeek V3/R1 is supported with high quality and performance. MTP can work with DeepSeek as well. Please refer to [muliti node tutorials](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/multi_node.html) and [Large Scale Expert Parallelism](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html).
- Qwen series models work with graph mode now. It works by default with V1 Engine. Please refer to [Qwen tutorials](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/index.html).
- Disaggregated Prefilling support for V1 Engine. Please refer to [Large Scale Expert Parallelism](https://docs.vllm.ai/projects/ascend/en/v0.9.1/tutorials/large_scale_ep.html) tutorials.
- Automatic prefix caching and chunked prefill feature is supported.
- Speculative decoding feature works with Ngram and MTP method.
- MOE and dense w4a8 quantization support now. Please refer to [quantization guide](https://docs.vllm.ai/projects/ascend/en/v0.9.1/user_guide/feature_guide/quantization.html).
- Sleep Mode feature is supported for V1 engine. Please refer to [Sleep mode tutorials](https://docs.vllm.ai/projects/ascend/en/v0.9.1/user_guide/feature_guide/sleep_mode.html).
- Dynamic and Static EPLB support is added. This feature is still experimental.

### Note

The following notes are especially for reference when upgrading from last final release (v0.7.3):

- V0 Engine is not supported from this release. Please always set `VLLM_USE_V1=1` to use V1 engine with vLLM Ascend.
- Mindie Turbo is not needed with this release. And the old version of Mindie Turbo is not compatible. Please do not install it. Currently all the function and enhancement is included in vLLM Ascend already. We'll consider to add it back in the future in needed.
- Torch-npu is upgraded to 2.5.1.post1. CANN is upgraded to 8.2.RC1. Don't forget to upgrade them.

### Core

- The Ascend scheduler is added for V1 engine. This scheduler is more affine with Ascend hardware.
- Structured output feature works now on V1 Engine.
- A batch of custom ops are added to improve the performance.

### Changes

- EPLB support for Qwen3-moe model. [#2000](https://github.com/vllm-project/vllm-ascend/pull/2000)
- Fix the bug that MTP doesn't work well with Prefill Decode Disaggregation. [#2610](https://github.com/vllm-project/vllm-ascend/pull/2610) [#2554](https://github.com/vllm-project/vllm-ascend/pull/2554) [#2531](https://github.com/vllm-project/vllm-ascend/pull/2531)
- Fix few bugs to make sure Prefill Decode Disaggregation works well. [#2538](https://github.com/vllm-project/vllm-ascend/pull/2538) [#2509](https://github.com/vllm-project/vllm-ascend/pull/2509) [#2502](https://github.com/vllm-project/vllm-ascend/pull/2502)
- Fix file not found error with shutil.rmtree in torchair mode. [#2506](https://github.com/vllm-project/vllm-ascend/pull/2506)

### Known Issues

- When running MoE model, Aclgraph mode only work with tensor parallel. DP/EP doesn't work in this release.
- Pipeline parallelism is not supported in this release for V1 engine.
- If you use w4a8 quantization with eager mode, please set `VLLM_ASCEND_MLA_PARALLEL=1` to avoid oom error.
- Accuracy test with some tools may not be correct. It doesn't affect the real user case. We'll fix it in the next post release. [#2654](https://github.com/vllm-project/vllm-ascend/pull/2654)
- We notice that there are still some problems when running vLLM Ascend with Prefill Decode Disaggregation. For example, the memory may be leaked and the service may be stuck. It's caused by known issue by vLLM and vLLM Ascend. We'll fix it in the next post release. [#2650](https://github.com/vllm-project/vllm-ascend/pull/2650) [#2604](https://github.com/vllm-project/vllm-ascend/pull/2604) [vLLM#22736](https://github.com/vllm-project/vllm/pull/22736) [vLLM#23554](https://github.com/vllm-project/vllm/pull/23554) [vLLM#23981](https://github.com/vllm-project/vllm/pull/23981)

## v0.9.1rc3 - 2025.08.22

This is the 3rd release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.9.1/) to get started.

### Core

- MTP supports V1 scheduler [#2371](https://github.com/vllm-project/vllm-ascend/pull/2371)
- Add LMhead TP communication groups [#1956](https://github.com/vllm-project/vllm-ascend/pull/1956)
- Fix the bug that qwen3 moe doesn't work with aclgraph [#2478](https://github.com/vllm-project/vllm-ascend/pull/2478)
- Fix `grammar_bitmask` IndexError caused by outdated `apply_grammar_bitmask` method [#2314](https://github.com/vllm-project/vllm-ascend/pull/2314)
- Remove `chunked_prefill_for_mla` [#2177](https://github.com/vllm-project/vllm-ascend/pull/2177)
- Fix bugs and refactor cached mask generation logic [#2326](https://github.com/vllm-project/vllm-ascend/pull/2326)
- Fix configuration check logic about ascend scheduler [#2327](https://github.com/vllm-project/vllm-ascend/pull/2327)
- Cancel the verification between deepseek-mtp and non-ascend scheduler in disaggregated-prefill deployment [#2368](https://github.com/vllm-project/vllm-ascend/pull/2368)
- Fix issue that failed with ray distributed backend [#2306](https://github.com/vllm-project/vllm-ascend/pull/2306)
- Fix incorrect req block length in ascend scheduler [#2394](https://github.com/vllm-project/vllm-ascend/pull/2394)
- Fix header include issue in rope [#2398](https://github.com/vllm-project/vllm-ascend/pull/2398)
- Fix mtp config bug [#2412](https://github.com/vllm-project/vllm-ascend/pull/2412)
- Fix error info and adapt `attn_metadata` refactor [#2402](https://github.com/vllm-project/vllm-ascend/pull/2402)
- Fix torchair runtime error caused by configuration mismatches and `.kv_cache_bytes` file missing [#2312](https://github.com/vllm-project/vllm-ascend/pull/2312)
- Move `with_prefill` allreduce from cpu to npu [#2230](https://github.com/vllm-project/vllm-ascend/pull/2230)

### Docs

- Add document for deepseek large EP [#2339](https://github.com/vllm-project/vllm-ascend/pull/2339)

### Known Issues

- `test_aclgraph.py` failed with `"full_cuda_graph": True` on A2 (910B1) [#2182](https://github.com/vllm-project/vllm-ascend/issues/2182)

## v0.10.0rc1 - 2025.08.07

This is the 1st release candidate of v0.10.0 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started. V0 is completely removed from this version.

### Highlights

- Disaggregate prefill works with V1 engine now. You can take a try with DeepSeek model [#950](https://github.com/vllm-project/vllm-ascend/pull/950), following this [tutorial](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/README.md).
- W4A8 quantization method is supported for dense and MoE model now. [#2060](https://github.com/vllm-project/vllm-ascend/pull/2060) [#2172](https://github.com/vllm-project/vllm-ascend/pull/2172)

### Core

- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.7.1.dev20250724`. [#1562](https://github.com/vllm-project/vllm-ascend/pull/1562) And CANN hase been upgraded to `8.2.RC1`. [#1653](https://github.com/vllm-project/vllm-ascend/pull/1653) Don’t forget to update them in your environment or using the latest images.
- vLLM Ascend works on Atlas 800I A3 now, and the image on A3 will be released from this version on. [#1582](https://github.com/vllm-project/vllm-ascend/pull/1582)
- Kimi-K2 with w8a8 quantization, Qwen3-Coder and GLM-4.5 is supported in vLLM Ascend, please following this [tutorial](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_node_kimi.md.html) to have a try. [#2162](https://github.com/vllm-project/vllm-ascend/pull/2162)
- Pipeline Parallelism is supported in V1 now. [#1800](https://github.com/vllm-project/vllm-ascend/pull/1800)
- Prefix cache feature now work with the Ascend Scheduler. [#1446](https://github.com/vllm-project/vllm-ascend/pull/1446)
- Torchair graph mode works with tp > 4 now. [#1508](https://github.com/vllm-project/vllm-ascend/issues/1508)
- MTP support torchair graph mode now [#2145](https://github.com/vllm-project/vllm-ascend/pull/2145)

### Others

- Bug fixes:
    - Fix functional problem of multimodality models like Qwen2-audio with Aclgraph. [#1803](https://github.com/vllm-project/vllm-ascend/pull/1803)
    - Fix the process group creating error with external launch scenario. [#1681](https://github.com/vllm-project/vllm-ascend/pull/1681)
    - Fix the functional problem with guided decoding. [#2022](https://github.com/vllm-project/vllm-ascend/pull/2022)
    - Fix the accuracy issue with common MoE models in DP scenario. [#1856](https://github.com/vllm-project/vllm-ascend/pull/1856)
- Performance improved through a lot of prs:
    - Caching sin/cos instead of calculate it every layer. [#1890](https://github.com/vllm-project/vllm-ascend/pull/1890)
    - Improve shared expert multi-stream parallelism [#1891](https://github.com/vllm-project/vllm-ascend/pull/1891)
    - Implement the fusion of allreduce and matmul in prefill phase when tp is enabled. Enable this feature by setting `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` to `1`. [#1926](https://github.com/vllm-project/vllm-ascend/pull/1926)
    - Optimize Quantized MoE Performance by Reducing All2All Communication. [#2195](https://github.com/vllm-project/vllm-ascend/pull/2195)
    - Use AddRmsNormQuant ops in the custom model to optimize Qwen3's performance [#1806](https://github.com/vllm-project/vllm-ascend/pull/1806)
    - Use multicast to avoid padding decode request to prefill size [#1555](https://github.com/vllm-project/vllm-ascend/pull/1555)
    - The performance of LoRA has been improved. [#1884](https://github.com/vllm-project/vllm-ascend/pull/1884)
- A batch of refactoring prs to enhance the code architecture:
    - Torchair model runner refactor [#2205](https://github.com/vllm-project/vllm-ascend/pull/2205)
    - Refactoring forward_context and model_runner_v1. [#1979](https://github.com/vllm-project/vllm-ascend/pull/1979)
    - Refactor AscendMetaData Comments. [#1967](https://github.com/vllm-project/vllm-ascend/pull/1967)
    - Refactor torchair utils. [#1892](https://github.com/vllm-project/vllm-ascend/pull/1892)
    - Refactor torchair worker. [#1885](https://github.com/vllm-project/vllm-ascend/pull/1885)
    - Register activation customop instead of overwrite forward_oot. [#1841](https://github.com/vllm-project/vllm-ascend/pull/1841)
- Parameters changes:
    - `expert_tensor_parallel_size` in `additional_config` is removed now, and the EP and TP is aligned with vLLM now. [#1681](https://github.com/vllm-project/vllm-ascend/pull/1681)
    - Add `VLLM_ASCEND_MLA_PA` in environ variables, use this to enable mla paged attention operator for deepseek mla decode.
    - Add `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` in environ variables, enable `MatmulAllReduce` fusion kernel when tensor parallel is enabled. This feature is supported in A2, and eager mode will get better performance.
    - Add `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ` in environ variables, Whether to enable moe all2all seq, this provides a basic framework on the basis of alltoall for easy expansion.

- UT coverage reached 76.34% after a batch of prs followed by this rfc: [#1298](https://github.com/vllm-project/vllm-ascend/issues/1298)
- Sequence Parallelism works for Qwen3 MoE. [#2209](https://github.com/vllm-project/vllm-ascend/issues/2209)
- Chinese online document is added now. [#1870](https://github.com/vllm-project/vllm-ascend/issues/1870)

### Known Issues

- Aclgraph could not work with DP + EP currently, the mainly gap is the number of npu stream that Aclgraph needed to capture graph is not enough. [#2229](https://github.com/vllm-project/vllm-ascend/issues/2229)
- There is an accuracy issue on W8A8 dynamic quantized DeepSeek with multistream enabled. This will be fixed in the next release. [#2232](https://github.com/vllm-project/vllm-ascend/issues/2232)
- In Qwen3 MoE, SP cannot be incorporated into the Aclgraph. [#2246](https://github.com/vllm-project/vllm-ascend/issues/2246)
- MTP not support V1 scheduler currently, will fix it in Q3. [#2254](https://github.com/vllm-project/vllm-ascend/issues/2254)
- When running MTP with DP > 1, we need to disable metrics logger due to some issue on vLLM. [#2254](https://github.com/vllm-project/vllm-ascend/issues/2254)

## v0.9.1rc2 - 2025.08.04

This is the 2nd release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.9.1/) to get started.

### Highlights

- MOE and dense w4a8 quantization support now: [#1320](https://github.com/vllm-project/vllm-ascend/pull/1320) [#1910](https://github.com/vllm-project/vllm-ascend/pull/1910) [#1275](https://github.com/vllm-project/vllm-ascend/pull/1275) [#1480](https://github.com/vllm-project/vllm-ascend/pull/1480)
- Dynamic EPLB support in [#1943](https://github.com/vllm-project/vllm-ascend/pull/1943)
- Disaggregated Prefilling support for V1 Engine and improvement, continued development and stabilization of the disaggregated prefill feature, including performance enhancements and bug fixes for single-machine setups:[#1953](https://github.com/vllm-project/vllm-ascend/pull/1953) [#1612](https://github.com/vllm-project/vllm-ascend/pull/1612) [#1361](https://github.com/vllm-project/vllm-ascend/pull/1361) [#1746](https://github.com/vllm-project/vllm-ascend/pull/1746) [#1552](https://github.com/vllm-project/vllm-ascend/pull/1552) [#1801](https://github.com/vllm-project/vllm-ascend/pull/1801) [#2083](https://github.com/vllm-project/vllm-ascend/pull/2083) [#1989](https://github.com/vllm-project/vllm-ascend/pull/1989)

### Model Improvement

- DeepSeek DeepSeek DBO support and improvement: [#1285](https://github.com/vllm-project/vllm-ascend/pull/1285) [#1291](https://github.com/vllm-project/vllm-ascend/pull/1291) [#1328](https://github.com/vllm-project/vllm-ascend/pull/1328) [#1420](https://github.com/vllm-project/vllm-ascend/pull/1420) [#1445](https://github.com/vllm-project/vllm-ascend/pull/1445) [#1589](https://github.com/vllm-project/vllm-ascend/pull/1589) [#1759](https://github.com/vllm-project/vllm-ascend/pull/1759) [#1827](https://github.com/vllm-project/vllm-ascend/pull/1827) [#2093](https://github.com/vllm-project/vllm-ascend/pull/2093)
- DeepSeek MTP improvement and bugfix: [#1214](https://github.com/vllm-project/vllm-ascend/pull/1214) [#943](https://github.com/vllm-project/vllm-ascend/pull/943) [#1584](https://github.com/vllm-project/vllm-ascend/pull/1584) [#1473](https://github.com/vllm-project/vllm-ascend/pull/1473) [#1294](https://github.com/vllm-project/vllm-ascend/pull/1294) [#1632](https://github.com/vllm-project/vllm-ascend/pull/1632) [#1694](https://github.com/vllm-project/vllm-ascend/pull/1694) [#1840](https://github.com/vllm-project/vllm-ascend/pull/1840) [#2076](https://github.com/vllm-project/vllm-ascend/pull/2076) [#1990](https://github.com/vllm-project/vllm-ascend/pull/1990) [#2019](https://github.com/vllm-project/vllm-ascend/pull/2019)
- Qwen3 MoE support improvement and bugfix around graph mode and DP:  [#1940](https://github.com/vllm-project/vllm-ascend/pull/1940) [#2006](https://github.com/vllm-project/vllm-ascend/pull/2006) [#1832](https://github.com/vllm-project/vllm-ascend/pull/1832)
- Qwen3 performance improvement around rmsnorm/repo/mlp ops: [#1545](https://github.com/vllm-project/vllm-ascend/pull/1545) [#1719](https://github.com/vllm-project/vllm-ascend/pull/1719) [#1726](https://github.com/vllm-project/vllm-ascend/pull/1726) [#1782](https://github.com/vllm-project/vllm-ascend/pull/1782) [#1745](https://github.com/vllm-project/vllm-ascend/pull/1745)
- DeepSeek MLA chunked prefill/graph mode/multistream improvement and bugfix: [#1240](https://github.com/vllm-project/vllm-ascend/pull/1240) [#933](https://github.com/vllm-project/vllm-ascend/pull/933) [#1135](https://github.com/vllm-project/vllm-ascend/pull/1135) [#1311](https://github.com/vllm-project/vllm-ascend/pull/1311) [#1750](https://github.com/vllm-project/vllm-ascend/pull/1750) [#1872](https://github.com/vllm-project/vllm-ascend/pull/1872) [#2170](https://github.com/vllm-project/vllm-ascend/pull/2170) [#1551](https://github.com/vllm-project/vllm-ascend/pull/1551)
- Qwen2.5 VL improvement via mrope/padding mechanism improvement: [#1261](https://github.com/vllm-project/vllm-ascend/pull/1261) [#1705](https://github.com/vllm-project/vllm-ascend/pull/1705) [#1929](https://github.com/vllm-project/vllm-ascend/pull/1929) [#2007](https://github.com/vllm-project/vllm-ascend/pull/2007)
- Ray: Fix the device error when using ray and add initialize_cache and improve warning info: [#1234](https://github.com/vllm-project/vllm-ascend/pull/1234) [#1501](https://github.com/vllm-project/vllm-ascend/pull/1501)

### Graph Mode Improvement

- Fix DeepSeek with deepseek with mc2 in [#1269](https://github.com/vllm-project/vllm-ascend/pull/1269)
- Fix accuracy problem for deepseek V3/R1 models with torchair graph in long sequence predictions in [#1332](https://github.com/vllm-project/vllm-ascend/pull/1332)
- Fix torchair_graph_batch_sizes bug in [#1570](https://github.com/vllm-project/vllm-ascend/pull/1570)
- Enable the limit of tp <= 4 for torchair graph mode in [#1404](https://github.com/vllm-project/vllm-ascend/pull/1404)
- Fix rope accuracy bug [#1887](https://github.com/vllm-project/vllm-ascend/pull/1887)
- Support multistream of shared experts in FusedMoE [#997](https://github.com/vllm-project/vllm-ascend/pull/997)
- Enable kvcache_nz for the decode process in torchair graph mode[#1098](https://github.com/vllm-project/vllm-ascend/pull/1098)
- Fix chunked-prefill with torchair case to resolve UnboundLocalError: local variable 'decode_hs_or_q_c' issue in [#1378](https://github.com/vllm-project/vllm-ascend/pull/1378)
- Improve shared experts multi-stream perf for w8a8 dynamic. in [#1561](https://github.com/vllm-project/vllm-ascend/pull/1561)
- Repair moe error when set multistream. in [#1882](https://github.com/vllm-project/vllm-ascend/pull/1882)
- Round up graph batch size to tp size in EP case [#1610](https://github.com/vllm-project/vllm-ascend/pull/1610)
- Fix torchair bug when DP is enabled in [#1727](https://github.com/vllm-project/vllm-ascend/pull/1727)
- Add extra checking to torchair_graph_config. in [#1675](https://github.com/vllm-project/vllm-ascend/pull/1675)
- Fix rope bug in torchair+chunk-prefill scenario in [#1693](https://github.com/vllm-project/vllm-ascend/pull/1693)
- torchair_graph bugfix when chunked_prefill is true in [#1748](https://github.com/vllm-project/vllm-ascend/pull/1748)
- Improve prefill optimization to support torchair graph mode in [#2090](https://github.com/vllm-project/vllm-ascend/pull/2090)
- Fix rank set in DP scenario [#1247](https://github.com/vllm-project/vllm-ascend/pull/1247)
- Reset all unused positions to prevent out-of-bounds to resolve GatherV3 bug in [#1397](https://github.com/vllm-project/vllm-ascend/pull/1397)
- Remove duplicate multimodal codes in ModelRunner in [#1393](https://github.com/vllm-project/vllm-ascend/pull/1393)
- Fix block table shape to resolve accuracy issue in [#1297](https://github.com/vllm-project/vllm-ascend/pull/1297)
- Implement primal full graph with limited scenario in [#1503](https://github.com/vllm-project/vllm-ascend/pull/1503)
- Restore paged attention kernel in Full Graph for performance in [#1677](https://github.com/vllm-project/vllm-ascend/pull/1677)
- Fix DeepSeek OOM issue in extreme `--gpu-memory-utilization` scenario in [#1829](https://github.com/vllm-project/vllm-ascend/pull/1829)
- Turn off aclgraph when enabling TorchAir in [#2154](https://github.com/vllm-project/vllm-ascend/pull/2154)

### Operator Improvement

- Added custom AscendC kernel `vocabparallelembedding` [#796](https://github.com/vllm-project/vllm-ascend/pull/796)
- Fixed rope sin/cos cache bug in [#1267](https://github.com/vllm-project/vllm-ascend/pull/1267)
- Refactored AscendFusedMoE (#1229) in [#1264](https://github.com/vllm-project/vllm-ascend/pull/1264)
- Used fused ops npu_top_k_top_p in sampler [#1920](https://github.com/vllm-project/vllm-ascend/pull/1920)

### Core

- Upgraded CANN to 8.2.rc1 in [#2036](https://github.com/vllm-project/vllm-ascend/pull/2036)
- Upgraded torch-npu to 2.5.1.post1 in [#2135](https://github.com/vllm-project/vllm-ascend/pull/2135)
- Upgraded python to 3.11 in [#2136](https://github.com/vllm-project/vllm-ascend/pull/2136)
- Disabled quantization in mindie_turbo  in [#1749](https://github.com/vllm-project/vllm-ascend/pull/1749)
- Fixed v0 spec decode in [#1323](https://github.com/vllm-project/vllm-ascend/pull/1323)
- Enabled `ACL_OP_INIT_MODE=1` directly only when using V0 spec decode in [#1271](https://github.com/vllm-project/vllm-ascend/pull/1271)
- Refactoring forward_context and model_runner_v1 in [#1422](https://github.com/vllm-project/vllm-ascend/pull/1422)
- Fixed sampling params in [#1423](https://github.com/vllm-project/vllm-ascend/pull/1423)
- Added a switch for enabling NZ layout in weights and enable NZ for GMM. in [#1409](https://github.com/vllm-project/vllm-ascend/pull/1409)
- Resolved bug in ascend_forward_context in [#1449](https://github.com/vllm-project/vllm-ascend/pull/1449) [#1554](https://github.com/vllm-project/vllm-ascend/pull/1554) [#1598](https://github.com/vllm-project/vllm-ascend/pull/1598)
- Address PrefillCacheHit state to fix prefix cache accuracy bug in [#1492](https://github.com/vllm-project/vllm-ascend/pull/1492)
- Fixed load weight error and add new e2e case in [#1651](https://github.com/vllm-project/vllm-ascend/pull/1651)
- Optimized the number of rope-related index selections in deepseek. in [#1614](https://github.com/vllm-project/vllm-ascend/pull/1614)
- Added mc2 mask in [#1642](https://github.com/vllm-project/vllm-ascend/pull/1642)
- Fixed static EPLB log2phy condition and improve unit test in [#1667](https://github.com/vllm-project/vllm-ascend/pull/1667) [#1896](https://github.com/vllm-project/vllm-ascend/pull/1896) [#2003](https://github.com/vllm-project/vllm-ascend/pull/2003)
- Added chunk mc2 for prefill in [#1703](https://github.com/vllm-project/vllm-ascend/pull/1703)
- Fixed mc2 op GroupCoordinator bug in [#1711](https://github.com/vllm-project/vllm-ascend/pull/1711)
- Fixed the failure to recognize the actual type of quantization in [#1721](https://github.com/vllm-project/vllm-ascend/pull/1721)
- Fixed DeepSeek bug when tp_size == 1  in [#1755](https://github.com/vllm-project/vllm-ascend/pull/1755)
- Added support for delay-free blocks in prefill nodes in [#1691](https://github.com/vllm-project/vllm-ascend/pull/1691)
- MoE alltoallv communication optimization for unquantized RL training & alltoallv support dpo in [#1547](https://github.com/vllm-project/vllm-ascend/pull/1547)
- Adapted dispatchV2 interface in [#1822](https://github.com/vllm-project/vllm-ascend/pull/1822)
- Fixed disaggregate prefill hang issue in long output in [#1807](https://github.com/vllm-project/vllm-ascend/pull/1807)
- Fixed flashcomm_v1 when engine v0 in [#1859](https://github.com/vllm-project/vllm-ascend/pull/1859)
- ep_group is not equal to word_size in some cases in [#1862](https://github.com/vllm-project/vllm-ascend/pull/1862).
- Fixed wheel glibc version incompatibility in [#1808](https://github.com/vllm-project/vllm-ascend/pull/1808).
- Fixed mc2 process group to resolve self.cpu_group is None in [#1831](https://github.com/vllm-project/vllm-ascend/pull/1831).
- Pin vllm version to v0.9.1 to make mypy check passed  in [#1904](https://github.com/vllm-project/vllm-ascend/pull/1904).
- Applied npu_moe_gating_top_k_softmax for moe to improve perf in [#1902](https://github.com/vllm-project/vllm-ascend/pull/1902).
- Fixed bug in path_decorator when engine v0 in [#1919](https://github.com/vllm-project/vllm-ascend/pull/1919).
- Avoid performing cpu all_reduce in disaggregated-prefill scenario in [#1644](https://github.com/vllm-project/vllm-ascend/pull/1644).
- Added super kernel in decode MoE in [#1916](https://github.com/vllm-project/vllm-ascend/pull/1916)
- [Prefill Perf] Parallel Strategy Optimizations (VRAM-for-Speed Tradeoff) in [#1802](https://github.com/vllm-project/vllm-ascend/pull/1802).
- Removed unnecessary reduce_results access in shared_experts.down_proj in [#2016](https://github.com/vllm-project/vllm-ascend/pull/2016).
- Optimized greedy reject sampler with vectorization in [#2002](https://github.com/vllm-project/vllm-ascend/pull/2002).
- Made multiple Ps and Ds work on a single machine in [#1936](https://github.com/vllm-project/vllm-ascend/pull/1936).
- Fixed the shape conflicts between shared & routed experts for deepseek model when tp > 1 and multistream_moe enabled in [#2075](https://github.com/vllm-project/vllm-ascend/pull/2075).
- Added CPU binding support [#2031](https://github.com/vllm-project/vllm-ascend/pull/2031).
- Added with_prefill cpu allreduce to handle D-node recomputation in [#2129](https://github.com/vllm-project/vllm-ascend/pull/2129).
- Added D2H & initRoutingQuantV2 to improve prefill perf in [#2038](https://github.com/vllm-project/vllm-ascend/pull/2038).

### Docs

- Provide an e2e guide for execute duration profiling [#1113](https://github.com/vllm-project/vllm-ascend/pull/1113)
- Add Referer header for CANN package download url. [#1192](https://github.com/vllm-project/vllm-ascend/pull/1192)
- Add reinstall instructions doc [#1370](https://github.com/vllm-project/vllm-ascend/pull/1370)
- Update Disaggregate prefill README [#1379](https://github.com/vllm-project/vllm-ascend/pull/1379)
- Disaggregate prefill for kv cache register style  [#1296](https://github.com/vllm-project/vllm-ascend/pull/1296)
- Fix errors and non-standard parts in examples/disaggregate_prefill_v1/README.md in [#1965](https://github.com/vllm-project/vllm-ascend/pull/1965)

### Known Issues

- Full graph mode support are not yet available for specific hardware types with full_cuda_graphenable. [#2182](https://github.com/vllm-project/vllm-ascend/issues/2182)
- Qwen3 MoE aclgraph mode with tp failed when enable ep due to bincount error [#2226](https://github.com/vllm-project/vllm-ascend/issues/2226)
- As mentioned in the v0.9.1rc1 release note, Atlas 300I series support will NOT be included.

## v0.9.2rc1 - 2025.07.11

This is the 1st release candidate of v0.9.2 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started. From this release, V1 engine will be enabled by default, there is no need to set `VLLM_USE_V1=1` any more. And this release is the last version to support V0 engine, V0 code will be clean up in the future.

### Highlights

- Pooling model works with V1 engine now. You can take a try with Qwen3 embedding model [#1359](https://github.com/vllm-project/vllm-ascend/pull/1359).
- The performance on Atlas 300I series has been improved. [#1591](https://github.com/vllm-project/vllm-ascend/pull/1591)
- aclgraph mode works with Moe models now. Currently, only Qwen3 Moe is well tested. [#1381](https://github.com/vllm-project/vllm-ascend/pull/1381)

### Core

- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.5.1.post1.dev20250619`. Don’t forget to update it in your environment. [#1347](https://github.com/vllm-project/vllm-ascend/pull/1347)
- The GatherV3 error has been fixed with aclgraph mode. [#1416](https://github.com/vllm-project/vllm-ascend/pull/1416)
- W8A8 quantization works on Atlas 300I series now. [#1560](https://github.com/vllm-project/vllm-ascend/pull/1560)
- Fix the accuracy problem with deploy models with parallel parameters. [#1678](https://github.com/vllm-project/vllm-ascend/pull/1678)
- The pre-built wheel package now requires lower version of glibc. Users can use it by `pip install vllm-ascend` directly. [#1582](https://github.com/vllm-project/vllm-ascend/pull/1582)

### Others

- Official doc has been updated for better read experience. For example, more deployment tutorials are added, user/developer docs are updated. More guide will coming soon.
- Fix accuracy problem for deepseek V3/R1 models with torchair graph in long sequence predictions. [#1331](https://github.com/vllm-project/vllm-ascend/pull/1331)
- A new env variable `VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP` has been added. It enables the fused allgather-experts kernel for Deepseek V3/R1 models. The default value is `0`. [#1335](https://github.com/vllm-project/vllm-ascend/pull/1335)
- A new env variable `VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION` has been added to improve the performance of topk-topp sampling. The default value is 0, we'll consider to enable it by default in the future[#1732](https://github.com/vllm-project/vllm-ascend/pull/1732)
- A batch of bugs have been fixed for Data Parallelism case [#1273](https://github.com/vllm-project/vllm-ascend/pull/1273) [#1322](https://github.com/vllm-project/vllm-ascend/pull/1322) [#1275](https://github.com/vllm-project/vllm-ascend/pull/1275) [#1478](https://github.com/vllm-project/vllm-ascend/pull/1478)
- The DeepSeek performance has been improved. [#1194](https://github.com/vllm-project/vllm-ascend/pull/1194) [#1395](https://github.com/vllm-project/vllm-ascend/pull/1395) [#1380](https://github.com/vllm-project/vllm-ascend/pull/1380)
- Ascend scheduler works with prefix cache now. [#1446](https://github.com/vllm-project/vllm-ascend/pull/1446)
- DeepSeek now works with prefix cache now. [#1498](https://github.com/vllm-project/vllm-ascend/pull/1498)
- Support prompt logprobs to recover ceval accuracy in V1 [#1483](https://github.com/vllm-project/vllm-ascend/pull/1483)

### Known Issues

- Pipeline parallel does not work with ray and graph mode: <https://github.com/vllm-project/vllm-ascend/issues/1751> <https://github.com/vllm-project/vllm-ascend/issues/1754>

### New Contributors

- @xleoken made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1357>
- @lyj-jjj made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1335>
- @sharonyunyun made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1194>
- @Pr0Wh1teGivee made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1308>
- @leo-pony made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1374>
- @zeshengzong made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1452>
- @GDzhu01 made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1477>
- @Agonixiaoxiao made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1531>
- @zhanghw0354 made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1476>
- @farawayboat made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1591>
- @ZhengWG made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1196>
- @wm901115nwpu made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1654>

**Full Changelog**: <https://github.com/vllm-project/vllm-ascend/compare/v0.9.1rc1...v0.9.2rc1>

## v0.9.1rc1 - 2025.06.22

This is the 1st release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to get started.

### Experimental

- Atlas 300I series is experimental supported in this release (Functional test passed with Qwen2.5-7b-instruct/Qwen2.5-0.5b/Qwen3-0.6B/Qwen3-4B/Qwen3-8B). [#1333](https://github.com/vllm-project/vllm-ascend/pull/1333)
- Support EAGLE-3 for speculative decoding. [#1032](https://github.com/vllm-project/vllm-ascend/pull/1032)

After careful consideration, above features **will NOT be included in v0.9.1-dev branch (v0.9.1 final release)** taking into account the v0.9.1 release quality and the feature rapid iteration. We will improve this from 0.9.2rc1 and later.

### Core

- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.5.1.post1.dev20250528`. Don’t forget to update it in your environment. [#1235](https://github.com/vllm-project/vllm-ascend/pull/1235)
- Support Atlas 300I series container image. You can get it from [quay.io](https://quay.io/repository/vllm/vllm-ascend)
- Fix token-wise padding mechanism to make multi-card graph mode work. [#1300](https://github.com/vllm-project/vllm-ascend/pull/1300)
- Upgrade vLLM to 0.9.1 [#1165](https://github.com/vllm-project/vllm-ascend/pull/1165)

### Other Improvements

- Initial support Chunked Prefill for MLA. [#1172](https://github.com/vllm-project/vllm-ascend/pull/1172)
- An example of best practices to run DeepSeek with ETP has been added. [#1101](https://github.com/vllm-project/vllm-ascend/pull/1101)
- Performance improvements for DeepSeek using the TorchAir graph. [#1098](https://github.com/vllm-project/vllm-ascend/pull/1098), [#1131](https://github.com/vllm-project/vllm-ascend/pull/1131)
- Supports the speculative decoding feature with AscendScheduler. [#943](https://github.com/vllm-project/vllm-ascend/pull/943)
- Improve `VocabParallelEmbedding` custom op performance. It will be enabled in the next release. [#796](https://github.com/vllm-project/vllm-ascend/pull/796)
- Fixed a device discovery and setup bug when running vLLM Ascend on Ray [#884](https://github.com/vllm-project/vllm-ascend/pull/884)
- DeepSeek with [MC2](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0043.html) (Merged Compute and Communication) now works properly. [#1268](https://github.com/vllm-project/vllm-ascend/pull/1268)
- Fixed log2phy NoneType bug with static EPLB feature. [#1186](https://github.com/vllm-project/vllm-ascend/pull/1186)
- Improved performance for DeepSeek with DBO enabled. [#997](https://github.com/vllm-project/vllm-ascend/pull/997), [#1135](https://github.com/vllm-project/vllm-ascend/pull/1135)
- Refactoring AscendFusedMoE [#1229](https://github.com/vllm-project/vllm-ascend/pull/1229)
- Add initial user stories page (include LLaMA-Factory/TRL/verl/MindIE Turbo/GPUStack) [#1224](https://github.com/vllm-project/vllm-ascend/pull/1224)
- Add unit test framework [#1201](https://github.com/vllm-project/vllm-ascend/pull/1201)

### Known Issues

- In some cases, the vLLM process may crash with a **GatherV3** error when **aclgraph** is enabled. We are working on this issue and will fix it in the next release. [#1038](https://github.com/vllm-project/vllm-ascend/issues/1038)
- Prefix cache feature does not work with the Ascend Scheduler but without chunked prefill enabled. This will be fixed in the next release. [#1350](https://github.com/vllm-project/vllm-ascend/issues/1350)

### Full Changelog

<https://github.com/vllm-project/vllm-ascend/compare/v0.9.0rc2...v0.9.1rc1>

### New Contributors

- @farawayboat made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1333>
- @yzim made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1159>
- @chenwaner made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1098>
- @wangyanhui-cmss made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1184>
- @songshanhu07 made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1186>
- @yuancaoyaoHW made their first contribution in <https://github.com/vllm-project/vllm-ascend/pull/1032>

**Full Changelog**: <https://github.com/vllm-project/vllm-ascend/compare/v0.9.0rc2...v0.9.1rc1>

## v0.9.0rc2 - 2025.06.10

This release contains some quick fixes for v0.9.0rc1. Please use this release instead of v0.9.0rc1.

### Highlights

- Fix the import error when vllm-ascend is installed without editable way. [#1152](https://github.com/vllm-project/vllm-ascend/pull/1152)

## v0.9.0rc1 - 2025.06.09

This is the 1st release candidate of v0.9.0 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to start the journey. From this release, V1 Engine is recommended to use. The code of V0 Engine is frozen and will not be maintained any more. Please set environment `VLLM_USE_V1=1` to enable V1 Engine.

### Highlights

- DeepSeek works with graph mode now. Follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html) to take a try. [#789](https://github.com/vllm-project/vllm-ascend/pull/789)
- Qwen series models work with graph mode now. It works by default with V1 Engine. Please note that in this release, only Qwen series models are well tested with graph mode. We'll make it stable and generalize in the next release. If you hit any issues, please feel free to open an issue on GitHub and fallback to eager mode temporarily by set `enforce_eager=True` when initializing the model.

### Core

- The performance of multi-step scheduler has been improved. Thanks for the contribution from China Merchants Bank. [#814](https://github.com/vllm-project/vllm-ascend/pull/814)
- LoRA、Multi-LoRA And Dynamic Serving is supported for V1 Engine now. Thanks for the contribution from China Merchants Bank. [#893](https://github.com/vllm-project/vllm-ascend/pull/893)
- Prefix cache and chunked prefill feature works now [#782](https://github.com/vllm-project/vllm-ascend/pull/782) [#844](https://github.com/vllm-project/vllm-ascend/pull/844)
- Spec decode and MTP features work with V1 Engine now. [#874](https://github.com/vllm-project/vllm-ascend/pull/874) [#890](https://github.com/vllm-project/vllm-ascend/pull/890)
- DP feature works with DeepSeek now. [#1012](https://github.com/vllm-project/vllm-ascend/pull/1012)
- Input embedding feature works with V0 Engine now. [#916](https://github.com/vllm-project/vllm-ascend/pull/916)
- Sleep mode feature works with V1 Engine now. [#1084](https://github.com/vllm-project/vllm-ascend/pull/1084)

### Models

- Qwen2.5 VL works with V1 Engine now. [#736](https://github.com/vllm-project/vllm-ascend/pull/736)
- Llama4 works now. [#740](https://github.com/vllm-project/vllm-ascend/pull/740)
- A new kind of DeepSeek model called dual-batch overlap(DBO) is added. Please set `VLLM_ASCEND_ENABLE_DBO=1` to use it. [#941](https://github.com/vllm-project/vllm-ascend/pull/941)

### Others

- online serve with ascend quantization works now. [#877](https://github.com/vllm-project/vllm-ascend/pull/877)
- A batch of bugs for graph mode and moe model have been fixed. [#773](https://github.com/vllm-project/vllm-ascend/pull/773) [#771](https://github.com/vllm-project/vllm-ascend/pull/771) [#774](https://github.com/vllm-project/vllm-ascend/pull/774) [#816](https://github.com/vllm-project/vllm-ascend/pull/816) [#817](https://github.com/vllm-project/vllm-ascend/pull/817) [#819](https://github.com/vllm-project/vllm-ascend/pull/819) [#912](https://github.com/vllm-project/vllm-ascend/pull/912) [#897](https://github.com/vllm-project/vllm-ascend/pull/897) [#961](https://github.com/vllm-project/vllm-ascend/pull/961) [#958](https://github.com/vllm-project/vllm-ascend/pull/958) [#913](https://github.com/vllm-project/vllm-ascend/pull/913) [#905](https://github.com/vllm-project/vllm-ascend/pull/905)
- A batch of performance improvement PRs have been merged. [#784](https://github.com/vllm-project/vllm-ascend/pull/784) [#803](https://github.com/vllm-project/vllm-ascend/pull/803) [#966](https://github.com/vllm-project/vllm-ascend/pull/966) [#839](https://github.com/vllm-project/vllm-ascend/pull/839) [#970](https://github.com/vllm-project/vllm-ascend/pull/970) [#947](https://github.com/vllm-project/vllm-ascend/pull/947) [#987](https://github.com/vllm-project/vllm-ascend/pull/987) [#1085](https://github.com/vllm-project/vllm-ascend/pull/1085)
- From this release, binary wheel package will be released as well. [#775](https://github.com/vllm-project/vllm-ascend/pull/775)
- The contributor doc site is [added](https://docs.vllm.ai/projects/ascend/en/latest/community/contributors.html)

### Known Issue

- In some case, vLLM process may be crashed with aclgraph enabled. We're working this issue and it'll be fixed in the next release.
- Multi node data-parallel doesn't work with this release. This is a known issue in vllm and has been fixed on main branch. [#18981](https://github.com/vllm-project/vllm/pull/18981)

## v0.7.3.post1 - 2025.05.29

This is the first post release of 0.7.3. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.3) to start the journey. It includes the following changes:

### Highlights

- Qwen3 and Qwen3MOE is supported now. The performance and accuracy of Qwen3 is well tested. You can try it now. Mindie Turbo is recommended to improve the performance of Qwen3. [#903](https://github.com/vllm-project/vllm-ascend/pull/903) [#915](https://github.com/vllm-project/vllm-ascend/pull/915)
- Added a new performance guide. The guide aims to help users to improve vllm-ascend performance on system level. It includes OS configuration, library optimization, deploy guide and so on. [#878](https://github.com/vllm-project/vllm-ascend/pull/878) [Doc Link](https://docs.vllm.ai/projects/ascend/en/v0.7.3/developer_guide/performance/optimization_and_tuning.html)

### Bug Fixes

- Qwen2.5-VL  works for RLHF scenarios now. [#928](https://github.com/vllm-project/vllm-ascend/pull/928)
- Users can launch the model from online weights now. e.g. from huggingface or modelscope directly [#858](https://github.com/vllm-project/vllm-ascend/pull/858) [#918](https://github.com/vllm-project/vllm-ascend/pull/918)
- The meaningless log info `UserWorkspaceSize0` has been cleaned. [#911](https://github.com/vllm-project/vllm-ascend/pull/911)
- The log level for `Failed to import vllm_ascend_C` has been changed to `warning` instead of `error`. [#956](https://github.com/vllm-project/vllm-ascend/pull/956)
- DeepSeek MLA now works with chunked prefill in V1 Engine. Please note that V1 engine in 0.7.3 is just expermential and only for test usage. [#849](https://github.com/vllm-project/vllm-ascend/pull/849) [#936](https://github.com/vllm-project/vllm-ascend/pull/936)

### Docs

- The benchmark doc is updated for Qwen2.5 and Qwen2.5-VL [#792](https://github.com/vllm-project/vllm-ascend/pull/792)
- Add the note to clear that only "modelscope<1.23.0" works with 0.7.3. [#954](https://github.com/vllm-project/vllm-ascend/pull/954)

## v0.7.3 - 2025.05.08

🎉 Hello, World!

We are excited to announce the release of 0.7.3 for vllm-ascend. This is the first official release. The functionality, performance, and stability of this release are fully tested and verified. We encourage you to try it out and provide feedback. We'll post bug fix versions in the future if needed. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.3) to start the journey.

### Highlights

- This release includes all features landed in the previous release candidates ([v0.7.1rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.1rc1), [v0.7.3rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc1), [v0.7.3rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc2)). And all the features are fully tested and verified. Visit the official doc the get the detail [feature](https://docs.vllm.ai/projects/ascend/en/v0.7.3/user_guide/suppoted_features.html) and [model](https://docs.vllm.ai/projects/ascend/en/v0.7.3/user_guide/supported_models.html) support matrix.
- Upgrade CANN to 8.1.RC1 to enable chunked prefill and automatic prefix caching features. You can now enable them now.
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#662](https://github.com/vllm-project/vllm-ascend/pull/662)
- Integrate MindIE Turbo into vLLM Ascend to improve DeepSeek V3/R1, Qwen 2 series performance. [#708](https://github.com/vllm-project/vllm-ascend/pull/708)

### Core

- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the official doc for more usage information. Thanks for the contribution from China Merchants Bank. [#700](https://github.com/vllm-project/vllm-ascend/pull/700)

### Models

- The performance of Qwen2 vl and Qwen2.5 vl is improved. [#702](https://github.com/vllm-project/vllm-ascend/pull/702)
- The performance of `apply_penalties` and `topKtopP` ops are improved. [#525](https://github.com/vllm-project/vllm-ascend/pull/525)

### Others

- Fixed a issue that may lead CPU memory leak. [#691](https://github.com/vllm-project/vllm-ascend/pull/691) [#712](https://github.com/vllm-project/vllm-ascend/pull/712)
- A new environment `SOC_VERSION` is added. If you hit any soc detection error when building with custom ops enabled, please set `SOC_VERSION` to a suitable value. [#606](https://github.com/vllm-project/vllm-ascend/pull/606)
- openEuler container image supported with v0.7.3-openeuler tag. [#665](https://github.com/vllm-project/vllm-ascend/pull/665)
- Prefix cache feature works on V1 engine now. [#559](https://github.com/vllm-project/vllm-ascend/pull/559)

## v0.8.5rc1 - 2025.05.06

This is the 1st release candidate of v0.8.5 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to start the journey. Now you can enable V1 egnine by setting the environment variable `VLLM_USE_V1=1`, see the feature support status of vLLM Ascend in [here](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html).

### Highlights

- Upgrade CANN version to 8.1.RC1 to support chunked prefill and automatic prefix caching (`--enable_prefix_caching`) when V1 is enabled [#747](https://github.com/vllm-project/vllm-ascend/pull/747)
- Optimize Qwen2 VL and Qwen 2.5 VL [#701](https://github.com/vllm-project/vllm-ascend/pull/701)
- Improve Deepseek V3 eager mode and graph mode performance, now you can use --additional_config={'enable_graph_mode': True} to enable graph mode. [#598](https://github.com/vllm-project/vllm-ascend/pull/598) [#719](https://github.com/vllm-project/vllm-ascend/pull/719)

### Core

- Upgrade vLLM to 0.8.5.post1 [#715](https://github.com/vllm-project/vllm-ascend/pull/715)
- Fix early return in CustomDeepseekV2MoE.forward during profile_run [#682](https://github.com/vllm-project/vllm-ascend/pull/682)
- Adapts for new quant model generated by modelslim [#719](https://github.com/vllm-project/vllm-ascend/pull/719)
- Initial support on P2P Disaggregated Prefill based on llm_datadist [#694](https://github.com/vllm-project/vllm-ascend/pull/694)
- Use `/vllm-workspace` as code path and include `.git` in container image to fix issue when start vllm under `/workspace` [#726](https://github.com/vllm-project/vllm-ascend/pull/726)
- Optimize NPU memory usage to make DeepSeek R1 W8A8 32K model len work. [#728](https://github.com/vllm-project/vllm-ascend/pull/728)
- Fix `PYTHON_INCLUDE_PATH` typo in setup.py [#762](https://github.com/vllm-project/vllm-ascend/pull/762)

### Others

- Add Qwen3-0.6B test [#717](https://github.com/vllm-project/vllm-ascend/pull/717)
- Add nightly CI [#668](https://github.com/vllm-project/vllm-ascend/pull/668)
- Add accuracy test report [#542](https://github.com/vllm-project/vllm-ascend/pull/542)

## v0.8.4rc2 - 2025.04.29

This is the second release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to start the journey. Some experimental features are included in this version, such as W8A8 quantization and EP/DP support. We'll make them stable enough in the next release.

### Highlights

- Qwen3 and Qwen3MOE is supported now. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/single_npu.html) to run the quick demo. [#709](https://github.com/vllm-project/vllm-ascend/pull/709)
- Ascend W8A8 quantization method is supported now. Please take the [official doc](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/multi_npu_quantization.html) for example. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/619) is welcome. [#580](https://github.com/vllm-project/vllm-ascend/pull/580)
- DeepSeek V3/R1 works with DP, TP and MTP now. Please note that it's still in experimental status. Let us know if you hit any problem. [#429](https://github.com/vllm-project/vllm-ascend/pull/429) [#585](https://github.com/vllm-project/vllm-ascend/pull/585)  [#626](https://github.com/vllm-project/vllm-ascend/pull/626) [#636](https://github.com/vllm-project/vllm-ascend/pull/636) [#671](https://github.com/vllm-project/vllm-ascend/pull/671)

### Core

- ACLGraph feature is supported with V1 engine now. It's disabled by default because this feature rely on CANN 8.1 release. We'll make it available by default in the next release [#426](https://github.com/vllm-project/vllm-ascend/pull/426)
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#661](https://github.com/vllm-project/vllm-ascend/pull/661)

### Others

- MiniCPM model works now. [#645](https://github.com/vllm-project/vllm-ascend/pull/645)
- openEuler container image supported with `v0.8.4-openeuler` tag and customs Ops build is enabled by default for openEuler OS. [#689](https://github.com/vllm-project/vllm-ascend/pull/689)
- Fix ModuleNotFoundError bug to make Lora work [#600](https://github.com/vllm-project/vllm-ascend/pull/600)
- Add "Using EvalScope evaluation" doc [#611](https://github.com/vllm-project/vllm-ascend/pull/611)
- Add a `VLLM_VERSION` environment to make vLLM version configurable to help developer set correct vLLM version if the code of vLLM is changed by hand locally. [#651](https://github.com/vllm-project/vllm-ascend/pull/651)

## v0.8.4rc1 - 2025.04.18

This is the first release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/) to start the journey. From this version, vllm-ascend will follow the newest version of vllm and release every two weeks. For example, if vllm releases v0.8.5 in the next two weeks, vllm-ascend will release v0.8.5rc1 instead of v0.8.4rc2. Please find the detail from the [official documentation](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#release-window).

### Highlights

- vLLM V1 engine experimental support is included in this version. You can visit [official guide](https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html) to get more detail. By default, vLLM will fallback to V0 if V1 doesn't work, please set `VLLM_USE_V1=1` environment if you want to use V1 forcibly.
- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the [official doc](https://docs.vllm.ai/en/latest/features/lora.html) for more usage information. Thanks for the contribution from China Merchants Bank. [#521](https://github.com/vllm-project/vllm-ascend/pull/521).
- Sleep Mode feature is supported. Currently it only works on V0 engine. V1 engine support will come soon. [#513](https://github.com/vllm-project/vllm-ascend/pull/513)

### Core

- The Ascend scheduler is added for V1 engine. This scheduler is more affinity with Ascend hardware. More scheduler policy will be added in the future. [#543](https://github.com/vllm-project/vllm-ascend/pull/543)
- Disaggregated Prefill feature is supported. Currently only 1P1D works. NPND is under design by vllm team. vllm-ascend will support it once it's ready from vLLM. Follow the [official guide](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) to use. [#432](https://github.com/vllm-project/vllm-ascend/pull/432)
- Spec decode feature works now. Currently it only works on V0 engine. V1 engine support will come soon. [#500](https://github.com/vllm-project/vllm-ascend/pull/500)
- Structured output feature works now on V1 Engine. Currently it only supports xgrammar backend while using guidance backend may get some errors. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

### Others

- A new communicator `pyhccl` is added. It's used for call CANN HCCL library directly instead of using `torch.distribute`. More usage of it will be added in the next release [#503](https://github.com/vllm-project/vllm-ascend/pull/503)
- The custom ops build is enabled by default. You should install the packages like `gcc`, `cmake` first to build `vllm-ascend` from source. Set `COMPILE_CUSTOM_KERNELS=0` environment to disable the compilation if you don't need it. [#466](https://github.com/vllm-project/vllm-ascend/pull/466)
- The custom op `rotary embedding` is enabled by default now to improve the performance. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

## v0.7.3rc2 - 2025.03.29

This is 2nd release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.3) to start the journey.

- Quickstart with container: <https://docs.vllm.ai/projects/ascend/en/v0.7.3/quick_start.html>
- Installation: <https://docs.vllm.ai/projects/ascend/en/v0.7.3/installation.html>

### Highlights

- Add Ascend Custom Ops framework. Developers now can write customs ops using AscendC. An example ops `rotary_embedding` is added. More tutorials will come soon. The Custom Ops compilation is disabled by default when installing vllm-ascend. Set `COMPILE_CUSTOM_KERNELS=1` to enable it.  [#371](https://github.com/vllm-project/vllm-ascend/pull/371)
- V1 engine is basic supported in this release. The full support will be done in 0.8.X release. If you hit any issue or have any requirement of V1 engine. Please tell us [here](https://github.com/vllm-project/vllm-ascend/issues/414). [#376](https://github.com/vllm-project/vllm-ascend/pull/376)
- Prefix cache feature works now. You can set `enable_prefix_caching=True` to enable it. [#282](https://github.com/vllm-project/vllm-ascend/pull/282)

### Core

- Bump torch_npu version to dev20250320.3 to improve accuracy to fix `!!!` output problem. [#406](https://github.com/vllm-project/vllm-ascend/pull/406)

### Models

- The performance of Qwen2-vl is improved by optimizing patch embedding (Conv3D). [#398](https://github.com/vllm-project/vllm-ascend/pull/398)

### Others

- Fixed a bug to make sure multi step scheduler feature work. [#349](https://github.com/vllm-project/vllm-ascend/pull/349)
- Fixed a bug to make prefix cache feature works with correct accuracy. [#424](https://github.com/vllm-project/vllm-ascend/pull/424)

## v0.7.3rc1 - 2025.03.14

🎉 Hello, World! This is the first release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.3) to start the journey.

- Quickstart with container: <https://docs.vllm.ai/projects/ascend/en/v0.7.3/quick_start.html>
- Installation: <https://docs.vllm.ai/projects/ascend/en/v0.7.3/installation.html>

### Highlights

- DeepSeek V3/R1 works well now. Read the [official guide](https://docs.vllm.ai/projects/ascend/en/v0.7.3/tutorials/multi_node.html) to start! [#242](https://github.com/vllm-project/vllm-ascend/pull/242)
- Speculative decoding feature is supported. [#252](https://github.com/vllm-project/vllm-ascend/pull/252)
- Multi step scheduler feature is supported. [#300](https://github.com/vllm-project/vllm-ascend/pull/300)

### Core

- Bump torch_npu version to dev20250308.3 to improve `_exponential` accuracy
- Added initial support for pooling models. Bert based model, such as `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-v2-m3` works now. [#229](https://github.com/vllm-project/vllm-ascend/pull/229)

### Models

- The performance of Qwen2-VL is improved. [#241](https://github.com/vllm-project/vllm-ascend/pull/241)
- MiniCPM is now supported [#164](https://github.com/vllm-project/vllm-ascend/pull/164)

### Others

- Support MTP(Multi-Token Prediction) for DeepSeek V3/R1 [#236](https://github.com/vllm-project/vllm-ascend/pull/236)
- [Docs] Added more model tutorials, include DeepSeek, QwQ, Qwen and Qwen 2.5VL. See the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.3/tutorials/index.html) for detail
- Pin modelscope<1.23.0 on vLLM v0.7.3 to resolve: <https://github.com/vllm-project/vllm/pull/13807>

### Known Issues

- In [some cases](https://github.com/vllm-project/vllm-ascend/issues/324), especially when the input/output is very long, the accuracy of output may be incorrect. We are working on it. It'll be fixed in the next release.
- Improved and reduced the garbled code in model output. But if you still hit the issue, try to change the generation config value, such as `temperature`, and try again. There is also a known issue shown below. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/267) is welcome. [#277](https://github.com/vllm-project/vllm-ascend/pull/277)

## v0.7.1rc1 - 2025.02.19

🎉 Hello, World!

We are excited to announce the first release candidate of v0.7.1 for vllm-ascend.

vLLM Ascend Plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM on the Ascend NPU. With this release, users can now enjoy the latest features and improvements of vLLM on the Ascend NPU.

Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/v0.7.1) to start the journey. Note that this is a release candidate, and there may be some bugs or issues. We appreciate your feedback and suggestions [here](https://github.com/vllm-project/vllm-ascend/issues/19)

### Highlights

- Initial supports for Ascend NPU on vLLM. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- DeepSeek is now supported. [#88](https://github.com/vllm-project/vllm-ascend/pull/88) [#68](https://github.com/vllm-project/vllm-ascend/pull/68)
- Qwen, Llama series and other popular models are also supported, you can see more details in [here](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/supported_models.html).

### Core

- Added the Ascend quantization config option, the implementation will coming soon. [#7](https://github.com/vllm-project/vllm-ascend/pull/7) [#73](https://github.com/vllm-project/vllm-ascend/pull/73)
- Add silu_and_mul and rope ops and add mix ops into attention layer. [#18](https://github.com/vllm-project/vllm-ascend/pull/18)

### Others

- [CI] Enable Ascend CI to actively monitor and improve quality for vLLM on Ascend. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- [Docker] Add vllm-ascend container image [#64](https://github.com/vllm-project/vllm-ascend/pull/64)
- [Docs] Add a [live doc](https://vllm-ascend.readthedocs.org) [#55](https://github.com/vllm-project/vllm-ascend/pull/55)

### Known Issues

- This release relies on an unreleased torch_npu version. It has been installed within official container image already. Please [install](https://docs.vllm.ai/projects/ascend/en/v0.7.1rc1/installation.html) it manually if you are using non-container environment.
- There are logs like `No platform detected, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")` shown when running vllm-ascend. It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/12432) which will be included in v0.7.3 soon.
- There are logs like `# CPU blocks: 35064, # CPU blocks: 2730` shown when running vllm-ascend which should be `# NPU blocks:` . It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/13378) which will be included in v0.7.3 soon.
