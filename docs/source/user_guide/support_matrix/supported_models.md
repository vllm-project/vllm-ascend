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

    <table markdown="1">
    <thead>
        <tr style="text-align: center; vertical-align: middle;">
        <th rowspan="2">Model</th><th rowspan="2">Support</th><th rowspan="2">Hardware</th>
        <th colspan="4">[Graph Mode](../../user_guide/feature_guide/graph_mode.md)</th>
        <th colspan="2">[Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex)</th>
        <th colspan="2">[CP](../../user_guide/feature_guide/context_parallel.md)</th>
        <th rowspan="2">[CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)</th>
        <th rowspan="2">[KVPP(only D node)](../../user_guide/feature_guide/kvpp.md)</th>
        <th rowspan="2">TP</th>
        <th rowspan="2">EP</th>
        <th rowspan="2">DP</th>
        <th rowspan="2">MLAPO</th>
        <th rowspan="2">[EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md)</th>
        <th rowspan="2">Multistream MoE(only D node)</th>
        <th rowspan="2">Shared Expert DP</th>
        <th rowspan="2">Fused MC2</th>
        <th colspan="4">[Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md)</th>
        <th colspan="3">[Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="3">[Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="2">[KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)</th>
        <th rowspan="2">Prefix Cache</th><th rowspan="2">Weight nz</th><th rowspan="2">Async Scheduling</th><th rowspan="2">[CPU Binding](../../user_guide/feature_guide/cpu_binding.md)</th>
        <th rowspan="2">flashcomml(only P node)</th>
        <th colspan="4">[Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md)</th>
        </tr>
        <tr style="text-align: center; vertical-align: middle;">
        <th>PIECEWISE</th><th>FULL_DECODE_ONLY</th><th>FULL</th><th>FULL_AND_PIECEWISE</th>
        <th>Static kernel</th><th>Super kernel</th>
        <th>PCP</th><th>DCP</th>
        <th>MTP</th><th>eagle3</th><th>DSpark</th><th>DFlash</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Sparse</th>
        <th>o_proj</th><th>embedding</th><th>MLP</th><th>LM_head</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>[DeepSeek V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>🟡</td><td>5</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>✅</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[DeepSeek V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>🟡</td><td>5</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>✅</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td></tr>
        <tr><td>[DeepSeek-V3.1](../../tutorials/models/DeepSeek-V3.1.md)</td><td>🔴</td><td>950DT Products</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[GLM-5.1](../../tutorials/models/GLM5.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[GLM-5.2](../../tutorials/models/GLM5.2.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[GLM-5.3](../../tutorials/models/GLM5.2.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>✅</td><td>🟡</td><td>5</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td></tr>
    </tbody>
    </table>

=== "A2/A3"

    <table markdown="1">
    <thead>
        <tr style="text-align: center; vertical-align: middle;">
        <th rowspan="2">Model</th><th rowspan="2">Support</th><th rowspan="2">Hardware</th>
        <th colspan="4">[Graph Mode](../../user_guide/feature_guide/graph_mode.md)</th>
        <th colspan="2">[Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex)</th>
        <th colspan="2">[CP](../../user_guide/feature_guide/context_parallel.md)</th>
        <th rowspan="2">[CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)</th>
        <th rowspan="2">[KVPP(only D node)](../../user_guide/feature_guide/kvpp.md)</th>
        <th rowspan="2">TP</th>
        <th rowspan="2">EP</th>
        <th rowspan="2">DP</th>
        <th rowspan="2">MLAPO</th>
        <th rowspan="2">[EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md)</th>
        <th rowspan="2">Multistream MoE(only D node)</th>
        <th rowspan="2">Shared Expert DP</th>
        <th rowspan="2">Fused MC2</th>
        <th colspan="4">[Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md)</th>
        <th colspan="3">[Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="3">[Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="2">[KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)</th>
        <th rowspan="2">Prefix Cache</th><th rowspan="2">Weight nz</th><th rowspan="2">Async Scheduling</th><th rowspan="2">[CPU Binding](../../user_guide/feature_guide/cpu_binding.md)</th>
        <th rowspan="2">flashcomml(only P node)</th>
        <th colspan="4">[Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md)</th>
        </tr>
        <tr style="text-align: center; vertical-align: middle;">
        <th>PIECEWISE</th><th>FULL_DECODE_ONLY</th><th>FULL</th><th>FULL_AND_PIECEWISE</th>
        <th>Static kernel</th><th>Super kernel</th>
        <th>PCP</th><th>DCP</th>
        <th>MTP</th><th>eagle3</th><th>DSpark</th><th>DFlash</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Sparse</th>
        <th>o_proj</th><th>embedding</th><th>MLP</th><th>LM_head</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>[DeepSeek V4-Flash](../../tutorials/models/DeepSeek-V4-Flash.md)</td><td>✅</td><td>A2/A3</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>🟡</td><td>✅</td><td>🟡</td><td>❌</td><td>5</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[DeepSeek V4-Pro](../../tutorials/models/DeepSeek-V4-Pro.md)</td><td>✅</td><td>A2/A3</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>🟡</td><td>✅</td><td>🟡</td><td>❌</td><td>5</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[DeepSeek V3/3.1](../../tutorials/models/DeepSeek-V3.1.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[DeepSeek V3.2](../../tutorials/models/DeepSeek-V3.2.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[DeepSeek R1](../../tutorials/models/DeepSeek-R1.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[DeepSeekOCR2](../../tutorials/models/DeepSeekOCR2.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[GLM-4.x](../../tutorials/models/GLM4.x.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[GLM-5](../../tutorials/models/GLM5.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[GLM-5.1](../../tutorials/models/GLM5.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[GLM-5.2](../../tutorials/models/GLM5.2.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[GLM-5.3](../../tutorials/models/GLM5.2.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>5</td><td>🟡</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[Gemma4](../../tutorials/models/Gemma4.md)</td><td>🔵</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Hy3-preview](../../tutorials/models/Hy3-preview.md)</td><td>🔴</td><td>A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Hy4-preview (Experimental) ](../../tutorials/models/Hy4-preview.md)</td><td>🔵</td><td>A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[InternLM2-20B-Chat](../../tutorials/models/InternLM2-20B-Chat.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K2-Thinking](../../tutorials/models/Kimi-K2-Thinking.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[MiniMax-M2.5/2.7](../../tutorials/models/MiniMax-M2.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3-Dense](../../tutorials/models/Qwen3-Dense.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3-30B-A3B](../../tutorials/models/Qwen3-30B-A3B.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3-Coder-30B-A3B](../../tutorials/models/Qwen3-Coder-30B-A3B.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3-235B-A22B](../../tutorials/models/Qwen3-235B-A22B.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3-Next](../../tutorials/models/Qwen3-Next.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen2.5-Math-RM-72B](../../tutorials/models/Qwen2.5-Math-RM-72B.md)</td><td>🔴</td><td>A2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>

    </tbody>
    </table>    

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

    <table markdown="1">
    <thead>
        <tr style="text-align: center; vertical-align: middle;">
        <th rowspan="2">Model</th><th rowspan="2">Support</th><th rowspan="2">Hardware</th>
        <th colspan="4">[Graph Mode](../../user_guide/feature_guide/graph_mode.md)</th>
        <th colspan="2">[Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex)</th>
        <th colspan="2">[CP](../../user_guide/feature_guide/context_parallel.md)</th>
        <th rowspan="2">[CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)</th>
        <th rowspan="2">[KVPP(only D node)](../../user_guide/feature_guide/kvpp.md)</th>
        <th rowspan="2">TP</th>
        <th rowspan="2">EP</th>
        <th rowspan="2">DP</th>
        <th rowspan="2">MLAPO</th>
        <th rowspan="2">[EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md)</th>
        <th rowspan="2">Multistream MoE(only D node)</th>
        <th rowspan="2">Shared Expert DP</th>
        <th rowspan="2">Fused MC2</th>
        <th colspan="4">[Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md)</th>
        <th colspan="3">[Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="3">[Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="2">[KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)</th>
        <th rowspan="2">Prefix Cache</th><th rowspan="2">Weight nz</th><th rowspan="2">Async Scheduling</th><th rowspan="2">[CPU Binding](../../user_guide/feature_guide/cpu_binding.md)</th>
        <th rowspan="2">flashcomml(only P node)</th>
        <th colspan="4">[Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md)</th>
        </tr>
        <tr style="text-align: center; vertical-align: middle;">
        <th>PIECEWISE</th><th>FULL_DECODE_ONLY</th><th>FULL</th><th>FULL_AND_PIECEWISE</th>
        <th>Static kernel</th><th>Super kernel</th>
        <th>PCP</th><th>DCP</th>
        <th>MTP</th><th>eagle3</th><th>DSpark</th><th>DFlash</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Sparse</th>
        <th>o_proj</th><th>embedding</th><th>MLP</th><th>LM_head</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>[DeepSeek V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>❌</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>🟡</td><td>5</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>✅</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md)</td><td>✅</td><td>950DT Products</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K3](../../tutorials/models/Kimi-K3.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>3</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K3.1-Pro](../../tutorials/models/Kimi-K3.1-Pro.md)</td><td>🔵</td><td>950DT Products</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr> 
        <tr><td>[MiniMax-M3](../../tutorials/models/MiniMax-M3.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>3</td><td>❌</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>   
        <tr><td>[MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md)</td><td>🔵</td><td>950DT Products</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>    
        <tr><td>[Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
        <tr><td>[Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md)</td><td>✅</td><td>950DT Products</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>🟡</td><td>🟡</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
        <tr><td>[Qwen3.8-Flash-Next](../../tutorials/models/Qwen3.8-Flash-Next.md)</td><td>✅</td><td>950DT Products</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Step-5-Preview](../../tutorials/models/Step-5-Preview.md)</td><td>🔵</td><td>950DT Products</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
    </tbody>
    </table>
    
=== "A2/A3"

    <table markdown="1">
    <thead>
        <tr style="text-align: center; vertical-align: middle;">
        <th rowspan="2">Model</th><th rowspan="2">Support</th><th rowspan="2">Hardware</th>
        <th colspan="4">[Graph Mode](../../user_guide/feature_guide/graph_mode.md)</th>
        <th colspan="2">[Npugraph_ex](../../user_guide/feature_guide/graph_mode.md#using-npugraph_ex)</th>
        <th colspan="2">[CP](../../user_guide/feature_guide/context_parallel.md)</th>
        <th rowspan="2">[CPP(only P node)](../../user_guide/feature_guide/dynamic_chunk_pipeline_parallel.md)</th>
        <th rowspan="2">[KVPP(only D node)](../../user_guide/feature_guide/kvpp.md)</th>
        <th rowspan="2">TP</th>
        <th rowspan="2">EP</th>
        <th rowspan="2">DP</th>
        <th rowspan="2">MLAPO</th>
        <th rowspan="2">[EPLB](../../user_guide/feature_guide/expert_parallelism_load_balancer.md)</th>
        <th rowspan="2">Multistream MoE(only D node)</th>
        <th rowspan="2">Shared Expert DP</th>
        <th rowspan="2">Fused MC2</th>
        <th colspan="4">[Speculative Decoding](../../user_guide/feature_guide/speculative_decoding.md)</th>
        <th colspan="3">[Mooncake Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="3">[Memcache Backend Pool](../../user_guide/feature_guide/kv_cache_pool_ascend_store.md)</th>
        <th colspan="2">[KV Cache Offloading](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)</th>
        <th rowspan="2">Prefix Cache</th><th rowspan="2">Weight nz</th><th rowspan="2">Async Scheduling</th><th rowspan="2">[CPU Binding](../../user_guide/feature_guide/cpu_binding.md)</th>
        <th rowspan="2">flashcomml(only P node)</th>
        <th colspan="4">[Fine-grained TP](../../user_guide/feature_guide/Fine_grained_TP.md)</th>
        </tr>
        <tr style="text-align: center; vertical-align: middle;">
        <th>PIECEWISE</th><th>FULL_DECODE_ONLY</th><th>FULL</th><th>FULL_AND_PIECEWISE</th>
        <th>Static kernel</th><th>Super kernel</th>
        <th>PCP</th><th>DCP</th>
        <th>MTP</th><th>eagle3</th><th>DSpark</th><th>DFlash</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Non-layerwise</th><th>SSD</th>
        <th>Layerwise</th><th>Sparse</th>
        <th>o_proj</th><th>embedding</th><th>MLP</th><th>LM_head</th>
        </tr>
    </thead>
    <tbody>
        <tr><td>[DeepSeek-V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md)</td><td>✅</td><td>A2</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>5</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[DeepSeek-V4.1-Flash](../../tutorials/models/DeepSeek-V4.1-Flash.md)</td><td>✅</td><td>A3</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>5</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td></tr>
        <tr><td>[GLM-5.3-Flash](../../tutorials/models/GLM5.3-Flash.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>❌</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>✅</td><td>3</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>✅</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K2.5](../../tutorials/models/Kimi-K2.5.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
        <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Kimi-K2.6](../../tutorials/models/Kimi-K2.6.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K3](../../tutorials/models/Kimi-K3.md)</td><td>✅</td><td>A3</td><td>🟡</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>7</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Kimi-K3.1-Pro](../../tutorials/models/Kimi-K3.1-Pro.md)</td><td>🔵</td><td>A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[MiniMax-M3](../../tutorials/models/MiniMax-M3.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>❌</td><td>3</td><td>❌</td><td>❌</td><td>🟡</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>        
        <tr><td>[MiniMax-M3.1-Flash](../../tutorials/models/MiniMax-M3.1-Flash.md)</td><td>✅</td><td>A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3.8-Flash-Next](../../tutorials/models/Qwen3.8-Flash-Next.md)</td><td>✅</td><td>A2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3-VL](../../tutorials/models/Qwen-VL-Dense.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3-VL-30B-A3B](../../tutorials/models/Qwen3-VL-30B-A3B-Instruct.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3-VL-235B-A22B](../../tutorials/models/Qwen3-VL-235B-A22B-Instruct.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Qwen3.5-397B-A17B](../../tutorials/models/Qwen3.5-397B-A17B.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td></tr>
        <tr><td>[Qwen3.5-27B/Qwen3.6-27B](../../tutorials/models/Qwen3.5-27B-Qwen3.6-27B.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>❌</td><td>✅</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3.6-35B-A3B](../../tutorials/models/Qwen3.6-35B-A3B.md)</td><td>✅</td><td>A2/A3</td><td>🟡</td><td>✅</td><td>🟡</td><td>🟡</td><td>✅</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>❌</td><td>✅</td><td>✅</td><td>✅</td><td>✅</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[Qwen3-Omni-30B-A3B-Thinking](../../tutorials/models/Qwen3-Omni-30B-A3B-Thinking.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
        <tr><td>[Cohere-Transcribe](../../tutorials/models/Cohere-Transcribe.md)</td><td>🔵</td><td>A2/A3</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td><td>🟡</td></tr>
        <tr><td>[PaddleOCR-VL](../../tutorials/models/PaddleOCR-VL.md)</td><td>🔴</td><td>A2/A3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
    </tbody>
    </table>

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
