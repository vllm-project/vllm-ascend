# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from contextlib import AbstractContextManager

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu import model_runner as upstream

from vllm_ascend.worker.utils import AscendKVBlockZeroer, copy_kv_cache_blocks_inplace


# Adapted from vLLM ced6857afa GPUModelRunner.initialize_kv_cache.
def initialize_kv_cache(
    self,
    kv_cache_config: KVCacheConfig,
    is_profiling: bool = False,
    kv_cache_allocation_context: AbstractContextManager | None = None,
) -> None:
    # GPUWorker finalizes the PD interleave before KV cache initialization.
    self.cp_interleave = self.parallel_config.cp_kv_cache_interleave_size
    kv_cache_config = upstream.deepcopy(kv_cache_config)
    self.kv_cache_config = kv_cache_config

    block_table_max_model_len = self.max_model_len
    if self.is_encoder_decoder:
        # Cross-attention block tables need to index encoder tokens, which
        # can exceed the decoder's max_model_len.
        block_table_max_model_len = max(
            block_table_max_model_len,
            self.scheduler_config.max_num_encoder_input_tokens,
            getattr(self.model_config.hf_config, "max_source_positions", 0),
        )

    block_sizes = []
    max_num_blocks_per_group = []
    slot_mapping_enabled = []
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        spec = kv_cache_group.kv_cache_spec
        block_sizes.append(spec.block_size)
        layer_spec = spec.first_spec if isinstance(spec, upstream.UniformTypeKVCacheSpecs) else spec
        slot_mapping_enabled.append(layer_spec.uses_slot_mapping)
        # Let each cache type account for CP. Attention KV is DCP-sharded,
        # while Mamba/GDN recurrent state is replicated across DCP ranks.
        max_num_blocks = spec.max_num_blocks_per_req(self.vllm_config, block_table_max_model_len)
        # Preserve each cache type's alignment requirements after applying
        # its topology-aware block-table width.
        max_num_blocks = upstream.get_block_table_width(
            max_num_blocks,
            spec.block_size,
            token_alignment=spec.block_table_token_alignment,
        )
        max_num_blocks_per_group.append(max_num_blocks)

    target_attn_layer_names = None
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        # Adaptive verification validates target attention separately.
        target_attn_layer_names = {
            layer_name for group in self.kv_cache_config.kv_cache_groups for layer_name in group.layer_names
        } - self.speculator.draft_attn_layer_names
    draft_attn_layer_names = None
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        draft_attn_layer_names = self.speculator.draft_attn_layer_names
    # Metadata builders select attention kernels that need JIT warmup.
    with self.jit_warmup_registry.activate():
        (
            self.attn_groups,
            attn_cg_support,
            self.kernel_block_sizes,
        ) = upstream.init_attn_backend(
            self.kv_cache_config,
            self.vllm_config,
            self.device,
            draft_layer_names=draft_attn_layer_names,
        )
    additional_attn_cg_support = self.model_state.get_additional_cg_support()
    attn_cg_support = attn_cg_support.narrow(*additional_attn_cg_support)
    # The speculator clears the flag at load time when the checkpoint has
    # no confidence head, so it holds the effective value.
    self.adaptive_verification = upstream.maybe_create_adaptive_verification_manager(
        enable_adaptive_verification=getattr(self.speculator, "enable_adaptive_verification", False),
        attn_groups=self.attn_groups,
        attn_cg_support=attn_cg_support,
        req_states=self.req_states,
        query_start_loc=self.input_buffers.query_start_loc,
        num_bonus_tokens=self.model_state.num_new_sampled_tokens_per_step,
        max_total_logits=upstream.get_max_chunk_logits(self.vocab_size),
        vllm_config=self.vllm_config,
        target_layer_names=target_attn_layer_names,
        additional_attn_cg_support=additional_attn_cg_support,
    )

    self.block_tables = upstream.BlockTables(
        block_sizes=block_sizes,
        max_num_reqs=self.max_num_reqs,
        max_num_batched_tokens=self.max_num_tokens,
        max_num_blocks_per_group=max_num_blocks_per_group,
        device=self.device,
        kernel_block_sizes=self.kernel_block_sizes,
        slot_mapping_enabled=slot_mapping_enabled,
        cp_size=self.dcp_size,
        cp_rank=self.dcp_rank,
        cp_interleave=self.cp_interleave,
    )
    self.pcp_manager = upstream.pcp.maybe_build_pcp_manager(
        self.vllm_config,
        self.device,
        self.supports_mm_inputs,
        self.block_tables,
        cls=self.pcp_manager_cls,
    )
    self.ubatch_runner = upstream.maybe_build_ubatch_runner(
        self.vllm_config,
        self.device,
        self.model_state,
        self.attn_groups,
        self.kv_cache_config,
        self.max_num_reqs,
    )
    if self.speculator is not None:
        self.speculator.pcp_manager = self.pcp_manager
    upstream.initialize_mamba_ssu_backend(
        self.vllm_config.mamba_config,
        self.kv_cache_config,
        use_replayssm=self.cache_config.use_replayssm,
    )
    piecewise_capture_available = bool(
        upstream.envs.VLLM_USE_BREAKABLE_CUDAGRAPH or upstream.has_compiled_submodule(self.model)
    )
    if self.adaptive_verification is not None:
        self.compilation_config.cudagraph_mode = upstream.resolve_adaptive_cudagraph_mode(
            self.compilation_config.cudagraph_mode,
            piecewise_capture_available=piecewise_capture_available,
        )
    cudagraph_mode = self.compilation_config.resolve_cudagraph_mode_and_sizes(
        attn_cg_support.min_cg_support,
        attn_cg_support.min_cg_attn_backend,
        self.decode_query_len,
        use_v2_model_runner=True,
        tensor_parallel_size=self.parallel_config.tensor_parallel_size,
        kv_cache_config=self.kv_cache_config,
        max_num_reqs=self.max_num_reqs,
        is_profiling=is_profiling,
        piecewise_capture_available=piecewise_capture_available,
    )
    self.cudagraph_manager = upstream.ModelCudaGraphManager(
        self.vllm_config,
        self.device,
        cudagraph_mode,
        decode_query_len=self.decode_query_len,
        lora_capture_cases=self.lora_capture_cases,
        varlen_decode=self.adaptive_verification is not None,
        ubatch_runner=self.ubatch_runner,
    )
    if self.cache_config.kv_sharing_fast_prefill and self.pcp_manager is None:
        self.fast_prefill = upstream.FastPrefillHelper(self.cudagraph_manager, self.max_num_tokens)
    upstream.check_attention_cp_compatibility(self.vllm_config, target_attn_layer_names)
    if isinstance(self.speculator, upstream.DraftModelSpeculator):
        # HACK(woosuk)
        self.speculator.set_attn(
            self.model_state,
            self.kv_cache_config,
            self.block_tables,
            self.input_buffers,
            self.attn_groups,
        )
    if self.speculator is not None:
        # After set_attn, so the speculator can size its cudagraph mode
        # to its own attention support.
        self.speculator.init_cudagraph_manager(cudagraph_mode)

    # Capture warmup providers that depend on allocated KV-cache strides.
    with self.jit_warmup_registry.activate():
        kv_caches_dict = upstream.init_kv_cache(
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.device,
            self.kernel_block_sizes,
            self.vllm_config,
            kv_cache_allocation_context=kv_cache_allocation_context,
            block_tables=self.block_tables,
        )
    self.kv_caches = [
        tensor
        for cache in kv_caches_dict.values()
        for tensor in (cache if isinstance(cache, (tuple, list)) else (cache,))
        if tensor.device == self.device
    ]
    if is_profiling:
        self.kv_connector = upstream.NO_OP_KV_CONNECTOR
    else:
        self.kv_connector = upstream.get_kv_connector(self.vllm_config, kv_caches_dict)


def init_kv_zero_meta(self) -> None:
    """Adapt MRv2 arguments to the existing Ascend zeroer contract."""
    self.kv_block_zeroer = AscendKVBlockZeroer(self.device, is_pin_memory_available())
    self.kv_block_zeroer.init_meta(
        attn_groups_iter=(group for groups in self.attn_groups for group in groups),
        kernel_block_sizes=[[block_size] for block_size in self.kernel_block_sizes],
        cache_dtype=self.cache_config.cache_dtype,
        runner_only_attn_layers=set(),
        static_forward_context=self.compilation_config.static_forward_context,
    )


upstream.GPUModelRunner._init_kv_zero_meta = init_kv_zero_meta
upstream.copy_kv_cache_blocks_inplace = copy_kv_cache_blocks_inplace

upstream.GPUModelRunner.initialize_kv_cache = initialize_kv_cache
