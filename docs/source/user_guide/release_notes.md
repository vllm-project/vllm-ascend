# Release note

## v0.10.1rc1 - 2025.09.04

This is the 1st release candidate of v0.10.1 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started.

### Highlights
- LoRA Performance improved much through adding Custom Kernels by China Merchants Bank. [#2325](https://github.com/vllm-project/vllm-ascend/pull/2325)
- Support Mooncake TransferEngine for kv cache register and pull_blocks style disaggregate prefill implementation. [#1568](https://github.com/vllm-project/vllm-ascend/pull/1568)
- Support capture custom ops into aclgraph now. [#2113](https://github.com/vllm-project/vllm-ascend/pull/2113)

### Core
- Add MLP tensor parallel to improve performance, but note that this will increase memory usage. [#2120](https://github.com/vllm-project/vllm-ascend/pull/2120)
- openEuler is upgraded to 24.03. [#2631](https://github.com/vllm-project/vllm-ascend/pull/2631)
- Add custom lmhead tensor parallel to achieve reduced memory consumption and improved TPOT performance. [#2309](https://github.com/vllm-project/vllm-ascend/pull/2309)
- Qwen3 MoE/Qwen2.5 support torchair graph now. [#2403](https://github.com/vllm-project/vllm-ascend/pull/2403)
- Support Sliding Window Attention with AscendSceduler, thus fixing Gemma3 accuracy issue. [#2528](https://github.com/vllm-project/vllm-ascend/pull/2528)

### Other

- Bug fixes:
    * Update the graph capture size calculation, somehow alleviated the problem that npu stream not enough in some scenarios [#2511](https://github.com/vllm-project/vllm-ascend/pull/2511)
    * Fix bugs and refactor cached mask generation logic. [#2442](https://github.com/vllm-project/vllm-ascend/pull/2442)
    * Fix the nz format does not work in quantization scenarios. [#2549](https://github.com/vllm-project/vllm-ascend/pull/2549)
    * Fix accuracy issue on Qwen series caused by enabling `enable_shared_pert_dp` by default. [#2457](https://github.com/vllm-project/vllm-ascend/pull/2457)
    * Fix accuracy issue on models whose rope dim is not equal to head dim, e.g., GLM4.5. [#2601](https://github.com/vllm-project/vllm-ascend/pull/2601)
- Performance improved through a lot of prs:
    * Remove torch.cat and replace it by List[0]. [#2153](https://github.com/vllm-project/vllm-ascend/pull/2153)
    * Convert the format of gmm to nz. [#2474](https://github.com/vllm-project/vllm-ascend/pull/2474)
    * Optimize parallel strategies to reduce communication overhead [#2198](https://github.com/vllm-project/vllm-ascend/pull/2198)
    * Optimize reject sampler in greedy situation [#2137](https://github.com/vllm-project/vllm-ascend/pull/2137)
- A batch of refactoring prs to enhance the code architecture:
    * Refactor on MLA. [#2465](https://github.com/vllm-project/vllm-ascend/pull/2465)
    * Refactor on torchair fused_moe. [#2438](https://github.com/vllm-project/vllm-ascend/pull/2438)
    * Refactor on allgather/mc2-related fused_experts. [#2369](https://github.com/vllm-project/vllm-ascend/pull/2369)
    * Refactor on torchair model runner. [#2208](https://github.com/vllm-project/vllm-ascend/pull/2208)
    * Refactor on CI. [#2276](https://github.com/vllm-project/vllm-ascend/pull/2276)
- Parameters changes:
    * Add `lmhead_tensor_parallel_size` in `additional_config`, set it to enable lmhead tensor parallel. [#2309](https://github.com/vllm-project/vllm-ascend/pull/2309)
    * Some unused environ variables `HCCN_PATH`, `PROMPT_DEVICE_ID`, `DECODE_DEVICE_ID`, `LLMDATADIST_COMM_PORT` and `LLMDATADIST_SYNC_CACHE_WAIT_TIME`  are removed. [#2448](https://github.com/vllm-project/vllm-ascend/pull/2448)
    * Environ variable `VLLM_LLMDD_RPC_PORT` is renamed to `VLLM_ASCEND_LLMDD_RPC_PORT` now. [#2450](https://github.com/vllm-project/vllm-ascend/pull/2450)
    * Add `VLLM_ASCEND_ENABLE_MLP_OPTIMIZE` in environ variables, Whether to enable mlp optimize when tensor parallel is enabled, this feature in eager mode will get better performance. [#2120](https://github.com/vllm-project/vllm-ascend/pull/2120)
    * Remove `MOE_ALL2ALL_BUFFER` and `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ` in environ variables.[#2612](https://github.com/vllm-project/vllm-ascend/pull/2612)
    * Add `enable_prefetch` in `additional_config`, whether to enable weight prefetch. [#2465](https://github.com/vllm-project/vllm-ascend/pull/2465)
    * Add `mode` in `additional_config.torchair_graph_config`, When using reduce-overhead mode for torchair, mode needs to be set. [#2461](https://github.com/vllm-project/vllm-ascend/pull/2461)
    * `enable_shared_expert_dp` in `additional_config` is disabled by default now, and it is recommended to enable when inferencing with deepseek. [#2457](https://github.com/vllm-project/vllm-ascend/pull/2457)

### Known Issues

- Sliding window attention not support chunked prefill currently, thus we could only enable AscendScheduler to run with it. [#2729](https://github.com/vllm-project/vllm-ascend/issues/2729)
- There is a bug with creating mc2_mask when MultiStream is enabled, will fix it in next release. [#2681](https://github.com/vllm-project/vllm-ascend/pull/2681)

## v0.9.1 - 2025.09.03

We are excited to announce the newest official release of vLLM Ascend. This release includes many feature supports, performance improvements and bug fixes. We recommend users to upgrade from 0.7.3 to this version. Please always set `VLLM_USE_V1=1` to use V1 engine.

In this release, we added many enhancements for large scale expert parallel case. It's recommended to follow the [official guide](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/large_scale_ep.html).

Please note that this release note will list all the important changes from last official release(v0.7.3)

### Highlights

- DeepSeek V3/R1 is supported with high quality and performance. MTP can work with DeepSeek as well. Please refer to [muliti node tutorials](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/multi_node.html) and [Large Scale Expert Parallelism](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/large_scale_ep.html).
- Qwen series models work with graph mode now. It works by default with V1 Engine. Please refer to [Qwen tutorials](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/index.html).
- Disaggregated Prefilling support for V1 Engine. Please refer to [Large Scale Expert Parallelism](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/tutorials/large_scale_ep.html) tutorials.
- Automatic prefix caching and chunked prefill feature is supported.
- Speculative decoding feature works with Ngram and MTP method.
- MOE and dense w4a8 quantization support now. Please refer to [quantization guide](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/user_guide/feature_guide/quantization.html).
- Sleep Mode feature is supported for V1 engine. Please refer to [Sleep mode tutorials](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/user_guide/feature_guide/sleep_mode.html).
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

This is the 3rd release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/) to get started.

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
- Fix error info and adapt `attn_metedata` refactor [#2402](https://github.com/vllm-project/vllm-ascend/pull/2402)
- Fix torchair runtime error caused by configuration mismtaches and `.kv_cache_bytes` file missing [#2312](https://github.com/vllm-project/vllm-ascend/pull/2312)
- Move `with_prefill` allreduce from cpu to npu [#2230](https://github.com/vllm-project/vllm-ascend/pull/2230)

### Docs

- Add document for deepseek large EP [#2339](https://github.com/vllm-project/vllm-ascend/pull/2339)

### Known Issues

- `test_aclgraph.py` failed with `"full_cuda_graph": True` on A2 (910B1) [#2182](https://github.com/vllm-project/vllm-ascend/issues/2182)

## v0.10.0rc1 - 2025.08.07

This is the 1st release candidate of v0.10.0 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started. V0 is completely removed from this version.

### Highlights
- Disaggregate prefill works with V1 engine now. You can take a try with DeepSeek model [#950](https://github.com/vllm-project/vllm-ascend/pull/950), following this [tutorial](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/README.md).
- W4A8 quantization method is supported for dense and MoE model now. [#2060](https://github.com/vllm-project/vllm-ascend/pull/2060) [#2172](https://github.com/vllm-project/vllm-ascend/pull/2172)

### Core
- Ascend PyTorch adapter (torch_npu) has been upgraded to `2.7.1.dev20250724`. [#1562](https://github.com/vllm-project/vllm-ascend/pull/1562) And CANN hase been upgraded to `8.2.RC1`. [#1653](https://github.com/vllm-project/vllm-ascend/pull/1653) Don’t forget to update them in your environment or using the latest images.
- vLLM Ascend works on Atlas 800I A3 now, and the image on A3 will be released from this version on. [#1582](https://github.com/vllm-project/vllm-ascend/pull/1582)
- Kimi-K2 with w8a8 quantization, Qwen3-Coder and GLM-4.5 is supported in vLLM Ascend, please following this [tutorial](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_node_kimi.md.html) to have a try. [#2162](https://github.com/vllm-project/vllm-ascend/pull/2162)
- Pipeline Parallelism is supported in V1 now. [#1800](https://github.com/vllm-project/vllm-ascend/pull/1800)
- Prefix cache feature now work with the Ascend Scheduler. [#1446](https://github.com/vllm-project/vllm-ascend/pull/1446)
- Torchair graph mode works with tp > 4 now. [#1508](https://github.com/vllm-project/vllm-ascend/issues/1508)
- MTP support torchair graph mode now [#2145](https://github.com/vllm-project/vllm-ascend/pull/2145)

### Other

- Bug fixes:
    * Fix functional problem of multi-modality models like Qwen2-audio with Aclgraph. [#1803](https://github.com/vllm-project/vllm-ascend/pull/1803)
    * Fix the process group creating error with external launch scenario. [#1681](https://github.com/vllm-project/vllm-ascend/pull/1681)
    * Fix the functional problem with guided decoding. [#2022](https://github.com/vllm-project/vllm-ascend/pull/2022)
    * Fix the accuracy issue with common MoE models in DP scenario. [#1856](https://github.com/vllm-project/vllm-ascend/pull/1856)
- Performance improved through a lot of prs:
    * Caching sin/cos instead of calculate it every layer. [#1890](https://github.com/vllm-project/vllm-ascend/pull/1890)
    * Improve shared expert multi-stream parallelism [#1891](https://github.com/vllm-project/vllm-ascend/pull/1891)
    * Implement the fusion of allreduce and matmul in prefill phase when tp is enabled. Enable this feature by setting `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` to `1`. [#1926](https://github.com/vllm-project/vllm-ascend/pull/1926)
    * Optimize Quantized MoE Performance by Reducing All2All Communication. [#2195](https://github.com/vllm-project/vllm-ascend/pull/2195)
    * Use AddRmsNormQuant ops in the custom model to optimize Qwen3's performance [#1806](https://github.com/vllm-project/vllm-ascend/pull/1806)
    * Use multicast to avoid padding decode request to prefill size [#1555](https://github.com/vllm-project/vllm-ascend/pull/1555)
    * The performance of LoRA has been improved. [#1884](https://github.com/vllm-project/vllm-ascend/pull/1884)
- A batch of refactoring prs to enhance the code architecture:
    * Torchair model runner refactor [#2205](https://github.com/vllm-project/vllm-ascend/pull/2205)
    * Refactoring forward_context and model_runner_v1. [#1979](https://github.com/vllm-project/vllm-ascend/pull/1979)
    * Refactor AscendMetaData Comments. [#1967](https://github.com/vllm-project/vllm-ascend/pull/1967)
    * Refactor torchair utils. [#1892](https://github.com/vllm-project/vllm-ascend/pull/1892)
    * Refactor torchair worker. [#1885](https://github.com/vllm-project/vllm-ascend/pull/1885)
    * Register activation customop instead of overwrite forward_oot. [#1841](https://github.com/vllm-project/vllm-ascend/pull/1841)
- Parameters changes:
    * `expert_tensor_parallel_size` in `additional_config` is removed now, and the EP and TP is aligned with vLLM now. [#1681](https://github.com/vllm-project/vllm-ascend/pull/1681)
    * Add `VLLM_ASCEND_MLA_PA` in environ variables, use this to enable mla paged attention operator for deepseek mla decode.
    * Add `VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE` in environ variables, enable `MatmulAllReduce` fusion kernel when tensor parallel is enabled. This feature is supported in A2, and eager mode will get better performance.
    * Add `VLLM_ASCEND_ENABLE_MOE_ALL2ALL_SEQ` in environ variables, Whether to enable moe all2all seq, this provides a basic framework on the basis of alltoall for easy expansion.

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
This is the 2nd release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.9.1-dev/) to get started.

### Highlights
- MOE and dense w4a8 quantization support now: [#1320](https://github.com/vllm-project/vllm-ascend/pull/1320) [#1910](https://github.com/vllm-project/vllm-ascend/pull/1910) [#1275](https://github.com/vllm-project/vllm-ascend/pull/1275) [#1480](https://github.com/vllm-project/vllm-ascend/pull/1480)
- Dynamic EPLB support in [#1943](https://github.com/vllm-project/vllm-ascend/pull/1943)
- Disaggregated Prefilling support for V1 Engine and improvement, continued development and stabilization of the disaggregated prefill feature, including performance enhancements and bug fixes for single-machine setups:[#1953](https://github.com/vllm-project/vllm-ascend/pull/1953) [#1612](https://github.com/vllm-project/vllm-ascend/pull/1612) [#1361](https://github.com/vllm-project/vllm-ascend/pull/1361) [#1746](https://github.com/vllm-project/vllm-ascend/pull/1746) [#1552](https://github.com/vllm-project/vllm-ascend/pull/1552) [#1801](https://github.com/vllm-project/vllm-ascend/pull/1801) [#2083](https://github.com/vllm-project/vllm-ascend/pull/2083) [#1989](https://github.com/vllm-project/vllm-ascend/pull/1989)

### Models improvement:
- DeepSeek DeepSeek DBO support and improvement: [#1285](https://github.com/vllm-project/vllm-ascend/pull/1285) [#1291](https://github.com/vllm-project/vllm-ascend/pull/1291) [#1328](https://github.com/vllm-project/vllm-ascend/pull/1328) [#1420](https://github.com/vllm-project/vllm-ascend/pull/1420) [#1445](https://github.com/vllm-project/vllm-ascend/pull/1445) [#1589](https://github.com/vllm-project/vllm-ascend/pull/1589) [#1759](https://github.com/vllm-project/vllm-ascend/pull/1759) [#1827](https://github.com/vllm-project/vllm-ascend/pull/1827) [#2093](https://github.com/vllm-project/vllm-ascend/pull/2093)
- DeepSeek MTP improvement and bugfix: [#1214](https://github.com/vllm-project/vllm-ascend/pull/1214) [#943](https://github.com/vllm-project/vllm-ascend/pull/943) [#1584](https://github.com/vllm-project/vllm-ascend/pull/1584) [#1473](https://github.com/vllm-project/vllm-ascend/pull/1473) [#1294](https://github.com/vllm-project/vllm-ascend/pull/1294) [#1632](https://github.com/vllm-project/vllm-ascend/pull/1632) [#1694](https://github.com/vllm-project/vllm-ascend/pull/1694) [#1840](https://github.com/vllm-project/vllm-ascend/pull/1840) [#2076](https://github.com/vllm-project/vllm-ascend/pull/2076) [#1990](https://github.com/vllm-project/vllm-ascend/pull/1990) [#2019](https://github.com/vllm-project/vllm-ascend/pull/2019)
- Qwen3 MoE support improvement and bugfix around graph mode and DP:  [#1940](https://github.com/vllm-project/vllm-ascend/pull/1940) [#2006](https://github.com/vllm-project/vllm-ascend/pull/2006) [#1832](https://github.com/vllm-project/vllm-ascend/pull/1832)
- Qwen3 performance improvement around rmsnorm/repo/mlp ops: [#1545](https://github.com/vllm-project/vllm-ascend/pull/1545) [#1719](https://github.com/vllm-project/vllm-ascend/pull/1719) [#1726](https://github.com/vllm-project/vllm-ascend/pull/1726) [#1782](https://github.com/vllm-project/vllm-ascend/pull/1782) [#1745](https://github.com/vllm-project/vllm-ascend/pull/1745)
- DeepSeek MLA chunked prefill/graph mode/multistream improvement and bugfix: [#1240](https://github.com/vllm-project/vllm-ascend/pull/1240) [#933](https://github.com/vllm-project/vllm-ascend/pull/933) [#1135](https://github.com/vllm-project/vllm-ascend/pull/1135) [#1311](https://github.com/vllm-project/vllm-ascend/pull/1311) [#1750](https://github.com/vllm-project/vllm-ascend/pull/1750) [#1872](https://github.com/vllm-project/vllm-ascend/pull/1872) [#2170](https://github.com/vllm-project/vllm-ascend/pull/2170) [#1551](https://github.com/vllm-project/vllm-ascend/pull/1551)
- Qwen2.5 VL improvement via mrope/padding mechanism improvement: [#1261](https://github.com/vllm-project/vllm-ascend/pull/1261) [#1705](https://github.com/vllm-project/vllm-ascend/pull/1705) [#1929](https://github.com/vllm-project/vllm-ascend/pull/1929) [#2007](https://github.com/vllm-project/vllm-ascend/pull/2007)
- Ray: Fix the device error when using ray and add initialize_cache and improve warning info: [#1234](https://github.com/vllm-project/vllm-ascend/pull/1234) [#1501](https://github.com/vllm-project/vllm-ascend/pull/1501)

### Graph mode improvement:
- Fix DeepSeek with deepseek with mc2 in [#1269](https://github.com/vllm-project/vllm-ascend/pull/1269)
- Fix accuracy problem for deepseek V3/R1 models with torchair graph in long sequence predictions in [#1332](https://github.com/vllm-project/vllm-ascend/pull/1332)
- Fix torchair_graph_batch_sizes bug in [#1570](https://github.com/vllm-project/vllm-ascend/pull/1570)
- Enable the limit of tp <= 4 for torchair graph mode in [#1404](https://github.com/vllm-project/vllm-ascend/pull/1404)
- Fix rope accruracy bug [#1887](https://github.com/vllm-project/vllm-ascend/pull/1887)
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

### Ops improvement:
- add custom ascendc kernel vocabparallelembedding [#796](https://github.com/vllm-project/vllm-ascend/pull/796)
- fix rope sin/cos cache bug in [#1267](https://github.com/vllm-project/vllm-ascend/pull/1267)
- Refactoring AscendFusedMoE (#1229) in [#1264](https://github.com/vllm-project/vllm-ascend/pull/1264)
- Use fused ops npu_top_k_top_p in sampler [#1920](https://github.com/vllm-project/vllm-ascend/pull/1920)

### Core:
- Upgrade CANN to 8.2.rc1 in [#2036](https://github.com/vllm-project/vllm-ascend/pull/2036)
- Upgrade torch-npu to 2.5.1.post1 in [#2135](https://github.com/vllm-project/vllm-ascend/pull/2135)
- Upgrade python to 3.11 in [#2136](https://github.com/vllm-project/vllm-ascend/pull/2136)
- Disable quantization in mindie_turbo  in [#1749](https://github.com/vllm-project/vllm-ascend/pull/1749)
- fix v0 spec decode in [#1323](https://github.com/vllm-project/vllm-ascend/pull/1323)
- Enable `ACL_OP_INIT_MODE=1` directly only when using V0 spec decode in [#1271](https://github.com/vllm-project/vllm-ascend/pull/1271)
- Refactoring forward_context and model_runner_v1 in [#1422](https://github.com/vllm-project/vllm-ascend/pull/1422)
- Fix sampling params in [#1423](https://github.com/vllm-project/vllm-ascend/pull/1423)
- add a switch for enabling NZ layout in weights and enable NZ for GMM. in [#1409](https://github.com/vllm-project/vllm-ascend/pull/1409)
- Resolved bug in ascend_forward_context in [#1449](https://github.com/vllm-project/vllm-ascend/pull/1449) [#1554](https://github.com/vllm-project/vllm-ascend/pull/1554) [#1598](https://github.com/vllm-project/vllm-ascend/pull/1598)
- Address PrefillCacheHit state to fix prefix cache accuracy bug in [#1492](https://github.com/vllm-project/vllm-ascend/pull/1492)
- Fix load weight error and add new e2e case in [#1651](https://github.com/vllm-project/vllm-ascend/pull/1651)
- Optimize the number of rope-related index selections in deepseek. in [#1614](https://github.com/vllm-project/vllm-ascend/pull/1614)
- add mc2 mask in [#1642](https://github.com/vllm-project/vllm-ascend/pull/1642)
- Fix static EPLB log2phy condition and improve unit test in [#1667](https://github.com/vllm-project/vllm-ascend/pull/1667) [#1896](https://github.com/vllm-project/vllm-ascend/pull/1896) [#2003](https://github.com/vllm-project/vllm-ascend/pull/2003)
- add chunk mc2 for prefill in [#1703](https://github.com/vllm-project/vllm-ascend/pull/1703)
- Fix mc2 op GroupCoordinator bug in [#1711](https://github.com/vllm-project/vllm-ascend/pull/1711)
- Fix the failure to recognize the actual type of quantization in [#1721](https://github.com/vllm-project/vllm-ascend/pull/1721)
- Fix deepseek bug when tp_size == 1  in [#1755](https://github.com/vllm-project/vllm-ascend/pull/1755)
- Added support for delay-free blocks in prefill nodes in [#1691](https://github.com/vllm-project/vllm-ascend/pull/1691)
- Moe alltoallv communication optimization for unquantized RL training & alltoallv support dpo in [#1547](https://github.com/vllm-project/vllm-ascend/pull/1547)
- Adapt dispatchV2 interface in [#1822](https://github.com/vllm-project/vllm-ascend/pull/1822)
- Fix disaggregate prefill hang issue in long output in [#1807](https://github.com/vllm-project/vllm-ascend/pull/1807)
- Fix flashcomm_v1 when engine v0 in [#1859](https://github.com/vllm-project/vllm-ascend/pull/1859)
- ep_group is not equal to word_size in some cases. in [#1862](https://github.com/vllm-project/vllm-ascend/pull/1862)
- Fix wheel glibc version incompatibility in [#1808](https://github.com/vllm-project/vllm-ascend/pull/1808)
- Fix mc2 process group to resolve self.cpu_group is None  in [#1831](https://github.com/vllm-project/vllm-ascend/pull/1831)
- Pin vllm version to v0.9.1 to make mypy check passed  in [#1904](https://github.com/vllm-project/vllm-ascend/pull/1904)
- Apply npu_moe_gating_top_k_softmax for moe to improve perf in [#1902](https://github.com/vllm-project/vllm-ascend/pull/1902)
- Fix bug in path_decorator when engine v0 in [#1919](https://github.com/vllm-project/vllm-ascend/pull/1919)
- Avoid performing cpu all_reduce in disaggregated-prefill scenario.  in [#1644](https://github.com/vllm-project/vllm-ascend/pull/1644)
- add super kernel in decode moe in [#1916](https://github.com/vllm-project/vllm-ascend/pull/1916)
- [Prefill Perf] Parallel Strategy Optimizations (VRAM-for-Speed Tradeoff) in [#1802](https://github.com/vllm-project/vllm-ascend/pull/1802)
- Remove unnecessary reduce_results access in shared_experts.down_proj in [#2016](https://github.com/vllm-project/vllm-ascend/pull/2016)
- Optimize greedy reject sampler with vectorization. in [#2002](https://github.com/vllm-project/vllm-ascend/pull/2002)
- Make multiple Ps and Ds work on a single machine in [#1936](https://github.com/vllm-project/vllm-ascend/pull/1936)
- Fixes the shape conflicts between shared & routed experts for deepseek model when tp > 1 and multistream_moe enabled in [#2075](https://github.com/vllm-project/vllm-ascend/pull/2075)
- Add cpu binding support [#2031](https://github.com/vllm-project/vllm-ascend/pull/2031)
- Add with_prefill cpu allreduce to handle D-node recomputatio in [#2129](https://github.com/vllm-project/vllm-ascend/pull/2129)
- Add D2H & initRoutingQuantV2 to improve prefill perf in [#2038](https://github.com/vllm-project/vllm-ascend/pull/2038)

### Docs:
- Provide an e2e guide for execute duration profiling [#1113](https://github.com/vllm-project/vllm-ascend/pull/1113)
- Add Referer header for CANN package download url. [#1192](https://github.com/vllm-project/vllm-ascend/pull/1192)
- Add reinstall instructions doc [#1370](https://github.com/vllm-project/vllm-ascend/pull/1370)
- Update Disaggregate prefill README [#1379](https://github.com/vllm-project/vllm-ascend/pull/1379)
- Disaggregate prefill for kv cache register style  [#1296](https://github.com/vllm-project/vllm-ascend/pull/1296)
- Fix errors and non-standard parts in examples/disaggregate_prefill_v1/README.md in [#1965](https://github.com/vllm-project/vllm-ascend/pull/1965)

### Known Issues
- Full graph mode support are not yet available for specific hardware types with full_cuda_graphenable. [#2182](https://github.com/vllm-project/vllm-ascend/issues/2182)
- Qwen3 MoE aclgraph mode with tp failed when enable ep due to bincount error [#2226](https://github.com/vllm-project/vllm-ascend/issues/2226)
- As mentioend in v0.9.1rc1 release note, Altlas 300I series support will NOT be included.

## v0.9.2rc1 - 2025.07.11

This is the 1st release candidate of v0.9.2 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started. From this release, V1 engine will be enabled by default, there is no need to set `VLLM_USE_V1=1` any more. And this release is the last version to support V0 engine, V0 code will be clean up in the future.

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

### Other
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

- Pipeline parallel does not work with ray and graph mode: https://github.com/vllm-project/vllm-ascend/issues/1751 https://github.com/vllm-project/vllm-ascend/issues/1754

### New Contributors
- @xleoken made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1357
- @lyj-jjj made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1335
- @sharonyunyun made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1194
- @Pr0Wh1teGivee made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1308
- @leo-pony made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1374
- @zeshengzong made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1452
- @GDzhu01 made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1477
- @Agonixiaoxiao made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1531
- @zhanghw0354 made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1476
- @farawayboat made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1591
- @ZhengWG made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1196
- @wm901115nwpu made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1654

**Full Changelog**: https://github.com/vllm-project/vllm-ascend/compare/v0.9.1rc1...v0.9.2rc1

## v0.9.1rc1 - 2025.06.22

This is the 1st release candidate of v0.9.1 for vLLM Ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to get started.

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
https://github.com/vllm-project/vllm-ascend/compare/v0.9.0rc2...v0.9.1rc1

### New Contributors
- @farawayboat made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1333
- @yzim made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1159
- @chenwaner made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1098
- @wangyanhui-cmss made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1184
- @songshanhu07 made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1186
- @yuancaoyaoHW made their first contribution in https://github.com/vllm-project/vllm-ascend/pull/1032

**Full Changelog**: https://github.com/vllm-project/vllm-ascend/compare/v0.9.0rc2...v0.9.1rc1

## v0.9.0rc2 - 2025.06.10

This release contains some quick fixes for v0.9.0rc1. Please use this release instead of v0.9.0rc1.

### Highlights

- Fix the import error when vllm-ascend is installed without editable way. [#1152](https://github.com/vllm-project/vllm-ascend/pull/1152)

## v0.9.0rc1 - 2025.06.09

This is the 1st release candidate of v0.9.0 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. From this release, V1 Engine is recommended to use. The code of V0 Engine is frozen and will not be maintained any more. Please set environment `VLLM_USE_V1=1` to enable V1 Engine.

### Highlights

- DeepSeek works with graph mode now. Follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/user_guide/feature_guide/graph_mode.html) to take a try. [#789](https://github.com/vllm-project/vllm-ascend/pull/789)
- Qwen series models works with graph mode now. It works by default with V1 Engine. Please note that in this release, only Qwen series models are well tested with graph mode. We'll make it stable and generalize in the next release. If you hit any issues, please feel free to open an issue on GitHub and fallback to eager mode temporarily by set `enforce_eager=True` when initializing the model.

### Core

- The performance of multi-step scheduler has been improved. Thanks for the contribution from China Merchants Bank. [#814](https://github.com/vllm-project/vllm-ascend/pull/814)
- LoRA、Multi-LoRA And Dynamic Serving is supported for V1 Engine now. Thanks for the contribution from China Merchants Bank. [#893](https://github.com/vllm-project/vllm-ascend/pull/893)
- Prefix cache and chunked prefill feature works now [#782](https://github.com/vllm-project/vllm-ascend/pull/782) [#844](https://github.com/vllm-project/vllm-ascend/pull/844)
- Spec decode and MTP features work with V1 Engine now. [#874](https://github.com/vllm-project/vllm-ascend/pull/874) [#890](https://github.com/vllm-project/vllm-ascend/pull/890)
- DP feature works with DeepSeek now. [#1012](https://github.com/vllm-project/vllm-ascend/pull/1012)
- Input embedding feature works with V0 Engine now. [#916](https://github.com/vllm-project/vllm-ascend/pull/916)
- Sleep mode feature works with V1 Engine now. [#1084](https://github.com/vllm-project/vllm-ascend/pull/1084)

### Model

- Qwen2.5 VL works with V1 Engine now. [#736](https://github.com/vllm-project/vllm-ascend/pull/736)
- LLama4 works now. [#740](https://github.com/vllm-project/vllm-ascend/pull/740)
- A new kind of DeepSeek model called dual-batch overlap(DBO) is added. Please set `VLLM_ASCEND_ENABLE_DBO=1` to use it. [#941](https://github.com/vllm-project/vllm-ascend/pull/941)

### Other

- online serve with ascend quantization works now. [#877](https://github.com/vllm-project/vllm-ascend/pull/877)
- A batch of bugs for graph mode and moe model have been fixed. [#773](https://github.com/vllm-project/vllm-ascend/pull/773) [#771](https://github.com/vllm-project/vllm-ascend/pull/771) [#774](https://github.com/vllm-project/vllm-ascend/pull/774) [#816](https://github.com/vllm-project/vllm-ascend/pull/816) [#817](https://github.com/vllm-project/vllm-ascend/pull/817) [#819](https://github.com/vllm-project/vllm-ascend/pull/819) [#912](https://github.com/vllm-project/vllm-ascend/pull/912) [#897](https://github.com/vllm-project/vllm-ascend/pull/897) [#961](https://github.com/vllm-project/vllm-ascend/pull/961) [#958](https://github.com/vllm-project/vllm-ascend/pull/958) [#913](https://github.com/vllm-project/vllm-ascend/pull/913) [#905](https://github.com/vllm-project/vllm-ascend/pull/905)
- A batch of performance improvement PRs have been merged. [#784](https://github.com/vllm-project/vllm-ascend/pull/784) [#803](https://github.com/vllm-project/vllm-ascend/pull/803) [#966](https://github.com/vllm-project/vllm-ascend/pull/966) [#839](https://github.com/vllm-project/vllm-ascend/pull/839) [#970](https://github.com/vllm-project/vllm-ascend/pull/970) [#947](https://github.com/vllm-project/vllm-ascend/pull/947) [#987](https://github.com/vllm-project/vllm-ascend/pull/987) [#1085](https://github.com/vllm-project/vllm-ascend/pull/1085)
- From this release, binary wheel package will be released as well. [#775](https://github.com/vllm-project/vllm-ascend/pull/775)
- The contributor doc site is [added](https://vllm-ascend.readthedocs.io/en/latest/community/contributors.html)

### Known Issue

- In some case, vLLM process may be crashed with aclgraph enabled. We're working this issue and it'll be fixed in the next release.
- Multi node data-parallel doesn't work with this release. This is a known issue in vllm and has been fixed on main branch. [#18981](https://github.com/vllm-project/vllm/pull/18981)

## v0.7.3.post1 - 2025.05.29

This is the first post release of 0.7.3. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey. It includes the following changes:

### Highlights

- Qwen3 and Qwen3MOE is supported now. The performance and accuracy of Qwen3 is well tested. You can try it now. Mindie Turbo is recomanded to improve the performance of Qwen3. [#903](https://github.com/vllm-project/vllm-ascend/pull/903) [#915](https://github.com/vllm-project/vllm-ascend/pull/915)
- Added a new performance guide. The guide aims to help users to improve vllm-ascend performance on system level. It includes OS configuration, library optimization, deploy guide and so on. [#878](https://github.com/vllm-project/vllm-ascend/pull/878) [Doc Link](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/developer_guide/performance/optimization_and_tuning.html)

### Bug Fix

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

We are excited to announce the release of 0.7.3 for vllm-ascend. This is the first official release. The functionality, performance, and stability of this release are fully tested and verified. We encourage you to try it out and provide feedback. We'll post bug fix versions in the future if needed. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.

### Highlights
- This release includes all features landed in the previous release candidates ([v0.7.1rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.1rc1), [v0.7.3rc1](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc1), [v0.7.3rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.7.3rc2)). And all the features are fully tested and verified. Visit the official doc the get the detail [feature](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/user_guide/suppoted_features.html) and [model](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/user_guide/supported_models.html) support matrix.
- Upgrade CANN to 8.1.RC1 to enable chunked prefill and automatic prefix caching features. You can now enable them now.
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#662](https://github.com/vllm-project/vllm-ascend/pull/662)
- Integrate MindIE Turbo into vLLM Ascend to improve DeepSeek V3/R1, Qwen 2 series performance. [#708](https://github.com/vllm-project/vllm-ascend/pull/708)

### Core
- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the official doc for more usage information. Thanks for the contribution from China Merchants Bank. [#700](https://github.com/vllm-project/vllm-ascend/pull/700)

### Model
- The performance of Qwen2 vl and Qwen2.5 vl is improved. [#702](https://github.com/vllm-project/vllm-ascend/pull/702)
- The performance of `apply_penalties` and `topKtopP` ops are improved. [#525](https://github.com/vllm-project/vllm-ascend/pull/525)

### Other
- Fixed a issue that may lead CPU memory leak. [#691](https://github.com/vllm-project/vllm-ascend/pull/691) [#712](https://github.com/vllm-project/vllm-ascend/pull/712)
- A new environment `SOC_VERSION` is added. If you hit any soc detection error when building with custom ops enabled, please set `SOC_VERSION` to a suitable value. [#606](https://github.com/vllm-project/vllm-ascend/pull/606)
- openEuler container image supported with v0.7.3-openeuler tag. [#665](https://github.com/vllm-project/vllm-ascend/pull/665)
- Prefix cache feature works on V1 engine now. [#559](https://github.com/vllm-project/vllm-ascend/pull/559)

## v0.8.5rc1 - 2025.05.06

This is the 1st release candidate of v0.8.5 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. Now you can enable V1 egnine by setting the environment variable `VLLM_USE_V1=1`, see the feature support status of vLLM Ascend in [here](https://vllm-ascend.readthedocs.io/en/latest/user_guide/support_matrix/supported_features.html).

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

### Other
- Add Qwen3-0.6B test [#717](https://github.com/vllm-project/vllm-ascend/pull/717)
- Add nightly CI [#668](https://github.com/vllm-project/vllm-ascend/pull/668)
- Add accuracy test report [#542](https://github.com/vllm-project/vllm-ascend/pull/542)

## v0.8.4rc2 - 2025.04.29

This is the second release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. Some experimental features are included in this version, such as W8A8 quantization and EP/DP support. We'll make them stable enough in the next release.

### Highlights
- Qwen3 and Qwen3MOE is supported now. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/single_npu.html) to run the quick demo. [#709](https://github.com/vllm-project/vllm-ascend/pull/709)
- Ascend W8A8 quantization method is supported now. Please take the [official doc](https://vllm-ascend.readthedocs.io/en/latest/tutorials/multi_npu_quantization.html) for example. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/619) is welcome. [#580](https://github.com/vllm-project/vllm-ascend/pull/580)
- DeepSeek V3/R1 works with DP, TP and MTP now. Please note that it's still in experimental status. Let us know if you hit any problem. [#429](https://github.com/vllm-project/vllm-ascend/pull/429) [#585](https://github.com/vllm-project/vllm-ascend/pull/585)  [#626](https://github.com/vllm-project/vllm-ascend/pull/626) [#636](https://github.com/vllm-project/vllm-ascend/pull/636) [#671](https://github.com/vllm-project/vllm-ascend/pull/671)

### Core
- ACLGraph feature is supported with V1 engine now. It's disabled by default because this feature rely on CANN 8.1 release. We'll make it available by default in the next release [#426](https://github.com/vllm-project/vllm-ascend/pull/426)
- Upgrade PyTorch to 2.5.1. vLLM Ascend no longer relies on the dev version of torch-npu now. Now users don't need to install the torch-npu by hand. The 2.5.1 version of torch-npu will be installed automatically. [#661](https://github.com/vllm-project/vllm-ascend/pull/661)

### Other
- MiniCPM model works now. [#645](https://github.com/vllm-project/vllm-ascend/pull/645)
- openEuler container image supported with `v0.8.4-openeuler` tag and customs Ops build is enabled by default for openEuler OS. [#689](https://github.com/vllm-project/vllm-ascend/pull/689)
- Fix ModuleNotFoundError bug to make Lora work [#600](https://github.com/vllm-project/vllm-ascend/pull/600)
- Add "Using EvalScope evaluation" doc [#611](https://github.com/vllm-project/vllm-ascend/pull/611)
- Add a `VLLM_VERSION` environment to make vLLM version configurable to help developer set correct vLLM version if the code of vLLM is changed by hand locally. [#651](https://github.com/vllm-project/vllm-ascend/pull/651)

## v0.8.4rc1 - 2025.04.18

This is the first release candidate of v0.8.4 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/) to start the journey. From this version, vllm-ascend will follow the newest version of vllm and release every two weeks. For example, if vllm releases v0.8.5 in the next two weeks, vllm-ascend will release v0.8.5rc1 instead of v0.8.4rc2. Please find the detail from the [official documentation](https://vllm-ascend.readthedocs.io/en/latest/community/versioning_policy.html#release-window).

### Highlights

- vLLM V1 engine experimental support is included in this version. You can visit [official guide](https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html) to get more detail. By default, vLLM will fallback to V0 if V1 doesn't work, please set `VLLM_USE_V1=1` environment if you want to use V1 forcely.
- LoRA、Multi-LoRA And Dynamic Serving is supported now. The performance will be improved in the next release. Please follow the [official doc](https://docs.vllm.ai/en/latest/features/lora.html) for more usage information. Thanks for the contribution from China Merchants Bank. [#521](https://github.com/vllm-project/vllm-ascend/pull/521).
- Sleep Mode feature is supported. Currently it's only work on V0 engine. V1 engine support will come soon. [#513](https://github.com/vllm-project/vllm-ascend/pull/513)

### Core

- The Ascend scheduler is added for V1 engine. This scheduler is more affinity with Ascend hardware. More scheduler policy will be added in the future. [#543](https://github.com/vllm-project/vllm-ascend/pull/543)
- Disaggregated Prefill feature is supported. Currently only 1P1D works. NPND is under design by vllm team. vllm-ascend will support it once it's ready from vLLM. Follow the [official guide](https://docs.vllm.ai/en/latest/features/disagg_prefill.html) to use. [#432](https://github.com/vllm-project/vllm-ascend/pull/432)
- Spec decode feature works now. Currently it's only work on V0 engine. V1 engine support will come soon. [#500](https://github.com/vllm-project/vllm-ascend/pull/500)
- Structured output feature works now on V1 Engine. Currently it only supports xgrammar backend while using guidance backend may get some errors. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

### Other

- A new communicator `pyhccl` is added. It's used for call CANN HCCL library directly instead of using `torch.distribute`. More usage of it will be added in the next release [#503](https://github.com/vllm-project/vllm-ascend/pull/503)
- The custom ops build is enabled by default. You should install the packages like `gcc`, `cmake` first to build `vllm-ascend` from source. Set `COMPILE_CUSTOM_KERNELS=0` environment to disable the compilation if you don't need it. [#466](https://github.com/vllm-project/vllm-ascend/pull/466)
- The custom op `rotay embedding` is enabled by default now to improve the performance. [#555](https://github.com/vllm-project/vllm-ascend/pull/555)

## v0.7.3rc2 - 2025.03.29

This is 2nd release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.
- Quickstart with container: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html

### Highlights
- Add Ascend Custom Ops framewrok. Developers now can write customs ops using AscendC. An example ops `rotary_embedding` is added. More tutorials will come soon. The Custom Ops compilation is disabled by default when installing vllm-ascend. Set `COMPILE_CUSTOM_KERNELS=1` to enable it.  [#371](https://github.com/vllm-project/vllm-ascend/pull/371)
- V1 engine is basic supported in this release. The full support will be done in 0.8.X release. If you hit any issue or have any requirement of V1 engine. Please tell us [here](https://github.com/vllm-project/vllm-ascend/issues/414). [#376](https://github.com/vllm-project/vllm-ascend/pull/376)
- Prefix cache feature works now. You can set `enable_prefix_caching=True` to enable it. [#282](https://github.com/vllm-project/vllm-ascend/pull/282)

### Core
- Bump torch_npu version to dev20250320.3 to improve accuracy to fix `!!!` output problem. [#406](https://github.com/vllm-project/vllm-ascend/pull/406)

### Model
- The performance of Qwen2-vl is improved by optimizing patch embedding (Conv3D). [#398](https://github.com/vllm-project/vllm-ascend/pull/398)

### Other

- Fixed a bug to make sure multi step scheduler feature work. [#349](https://github.com/vllm-project/vllm-ascend/pull/349)
- Fixed a bug to make prefix cache feature works with correct accuracy. [#424](https://github.com/vllm-project/vllm-ascend/pull/424)

## v0.7.3rc1 - 2025.03.14

🎉 Hello, World! This is the first release candidate of v0.7.3 for vllm-ascend. Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev) to start the journey.
- Quickstart with container: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/quick_start.html
- Installation: https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/installation.html

### Highlights
- DeepSeek V3/R1 works well now. Read the [official guide](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/tutorials/multi_node.html) to start! [#242](https://github.com/vllm-project/vllm-ascend/pull/242)
- Speculative decoding feature is supported. [#252](https://github.com/vllm-project/vllm-ascend/pull/252)
- Multi step scheduler feature is supported. [#300](https://github.com/vllm-project/vllm-ascend/pull/300)

### Core
- Bump torch_npu version to dev20250308.3 to improve `_exponential` accuracy
- Added initial support for pooling models. Bert based model, such as `BAAI/bge-base-en-v1.5` and `BAAI/bge-reranker-v2-m3` works now. [#229](https://github.com/vllm-project/vllm-ascend/pull/229)

### Model
- The performance of Qwen2-VL is improved. [#241](https://github.com/vllm-project/vllm-ascend/pull/241)
- MiniCPM is now supported [#164](https://github.com/vllm-project/vllm-ascend/pull/164)

### Other
- Support MTP(Multi-Token Prediction) for DeepSeek V3/R1 [#236](https://github.com/vllm-project/vllm-ascend/pull/236)
- [Docs] Added more model tutorials, include DeepSeek, QwQ, Qwen and Qwen 2.5VL. See the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.3-dev/tutorials/index.html) for detail
- Pin modelscope<1.23.0 on vLLM v0.7.3 to resolve: https://github.com/vllm-project/vllm/pull/13807

### Known issues
- In [some cases](https://github.com/vllm-project/vllm-ascend/issues/324), especially when the input/output is very long, the accuracy of output may be incorrect. We are working on it. It'll be fixed in the next release.
- Improved and reduced the garbled code in model output. But if you still hit the issue, try to change the generation config value, such as `temperature`, and try again. There is also a knonwn issue shown below. Any [feedback](https://github.com/vllm-project/vllm-ascend/issues/267) is welcome. [#277](https://github.com/vllm-project/vllm-ascend/pull/277)

## v0.7.1rc1 - 2025.02.19

🎉 Hello, World!

We are excited to announce the first release candidate of v0.7.1 for vllm-ascend.

vLLM Ascend Plugin (vllm-ascend) is a community maintained hardware plugin for running vLLM on the Ascend NPU. With this release, users can now enjoy the latest features and improvements of vLLM on the Ascend NPU.

Please follow the [official doc](https://vllm-ascend.readthedocs.io/en/v0.7.1-dev) to start the journey. Note that this is a release candidate, and there may be some bugs or issues. We appreciate your feedback and suggestions [here](https://github.com/vllm-project/vllm-ascend/issues/19)

### Highlights

- Initial supports for Ascend NPU on vLLM. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- DeepSeek is now supported. [#88](https://github.com/vllm-project/vllm-ascend/pull/88) [#68](https://github.com/vllm-project/vllm-ascend/pull/68)
- Qwen, Llama series and other popular models are also supported, you can see more details in [here](https://vllm-ascend.readthedocs.io/en/latest/user_guide/supported_models.html).

### Core

- Added the Ascend quantization config option, the implementation will coming soon. [#7](https://github.com/vllm-project/vllm-ascend/pull/7) [#73](https://github.com/vllm-project/vllm-ascend/pull/73)
- Add silu_and_mul and rope ops and add mix ops into attention layer. [#18](https://github.com/vllm-project/vllm-ascend/pull/18)

### Other

- [CI] Enable Ascend CI to actively monitor and improve quality for vLLM on Ascend. [#3](https://github.com/vllm-project/vllm-ascend/pull/3)
- [Docker] Add vllm-ascend container image [#64](https://github.com/vllm-project/vllm-ascend/pull/64)
- [Docs] Add a [live doc](https://vllm-ascend.readthedocs.org) [#55](https://github.com/vllm-project/vllm-ascend/pull/55)

### Known issues

- This release relies on an unreleased torch_npu version. It has been installed within official container image already. Please [install](https://vllm-ascend.readthedocs.io/en/v0.7.1rc1/installation.html) it manually if you are using non-container environment.
- There are logs like `No platform detected, vLLM is running on UnspecifiedPlatform` or `Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")` shown when running vllm-ascend. It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/12432) which will be included in v0.7.3 soon.
- There are logs like `# CPU blocks: 35064, # CPU blocks: 2730` shown when running vllm-ascend which should be `# NPU blocks:` . It actually doesn't affect any functionality and performance. You can just ignore it. And it has been fixed in this [PR](https://github.com/vllm-project/vllm/pull/13378) which will be included in v0.7.3 soon.
