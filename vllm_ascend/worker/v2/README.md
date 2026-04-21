# [Experimental] Model Runner V2

This directory contains the new model runner which is under active development.

please see [Model Runner V2](https://github.com/vllm-project/vllm-ascend/issues/5208)
to get specific plans.

## Gaps with vLLM (To Be Addressed)

- [ ] `set_cos_and_sin` & `update_cos_sin`

    Why: DeepSeek-like models (mla) still need cos/sin setting and updating in model_runner. These should be removed when mla can solve cos/sin internally.

    Location: `NPUModelRunner.__init__`, `NPUModelRunner.prepare_inputs`, `AscendInputBatch.make_dummy`.

- [ ] `_allocate_kv_cache` & `_reshape_kv_cache`

    Why: KV cache requires continuous space (thus divided as K cache and V cache separately) and PD disaggregation requires 2M-aligned tensors for KV cache, so custom KV cache initialization is needed. These should be removed when the above 2 requirements are no longer needed.

    Location: `attn_utils._get_layer_kv_cache_specs`, `attn_utils._get_attention_kv_cache_dims`, `attn_utils._align_memory`, `attn_utils._allocate_kv_cache`, `attn_utils._reshape_kv_cache`.

- [ ] `torch_npu_graph_wrapper`

    Why: FIA ops in FULL mode need explicit workspace allocating, and each workspace corresponding to each graph (a specific batch_size) should be released via `weak_ref_workspaces` when each capturing is exactly completed to avoid OOM, thus that leads us to regard `weak_ref_workspaces` as post-processing in `torch.npu.graph` and patch it. This should be removed when we don't need such special operations.

    Location: `utils.torch_cuda_wrapper`, `utils.torch_npu_graph_wrapper`.
