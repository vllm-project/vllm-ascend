# Customization of Models

This doc will elaborate our customization on models in vllm-ascend, including what is added or patched to the model and related reasons as well.

## DeepSeek Series

### DeepSeek-V2

In our DeepSeek-V2 model, `CustomDeepseekV2MLAAttention`, `CustomDeepseekV2MoE` and `CustomDeepseekV2MLP` are used in our `CustomDeepseekV2DecoderLayer`.

In `CustomDeepseekV2MLAAttention` layer, we can enbale multi-stream mla for `npu_prefetch()`, and we have cutomized the forward code when using torchair graph.

In `CustomDeepseekV2MoE` layer, we use contomized `AscendFusedMoE` ops to replace the experts layers.

In `CustomDeepseekV2MLP` layer, we use customized layers including `CustomDeepseekV2MergedReplicatedLinear` and `CustomDeepseekV2SiluAndMul` (will call `torch_npu.npu_dequant_swiglu_quant()`) for better performance on NPU device.

### DeepSeek-MTP

In our DeepSeek-MTP model, `CustomDeepSeekMultiTokenPredictorLayer` is added and we just replaced the original mtp layers to `CustomDeepseekV2DecoderLayer`.

### DeepSeek-DBO

In our DeepSeek-DBO model, if `VLLM_ASCEND_ENABLE_DBO` is enabled, the Multi-Stream feature can be used to optimize the foward of sparse layers. Specificly, we add a `ms_pre_layer` before MoE layers and a `ms_post_layer` after MoE layers, respectively.

Plus, `CustomDeepseekDBOMLAAttention`, `CustomDeepseekDBOMoE` and `CustomDeepseekDBOMLP` are also added to the decode layers. Find more detalis at the [code](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/deepseek_dbo.py).

## Qwen Series

### Qwen2/2.5-VL

In our Qwen2-VL model, we use `AscendQwen2VisionTransformer` to pad the weights of attention to get a performance improvement on NPU devices. Plus, we also replaced the ops in ViT modules with `torch_npu._npu_flash_attention_unpad()`.

As for our Qwen2.5-VL model, the costimization is very similar to that of Qwen2-VL. Find more details at the [code](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/models/qwen2_5_vl.py). Plus, an unpadded version of Qwen2.5-VL is also added to vllm-ascend to avoid errors in verl scenario, you can set `USE_OPTIMIZED_MODEL=False` to use this model.

### Qwen3-MoE

In our Qwen3-MoE model, we just added `packed_modules_mapping` to the modeling class and others keep the same with upstream.
