# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration contract for GLM pooled-key DCP execution."""


def validate_kpool_dcp(vllm_config, *, check_cache_layout: bool = True) -> None:
    parallel = vllm_config.parallel_config
    world = parallel.decode_context_parallel_size
    if world <= 1:
        return
    if parallel.prefill_context_parallel_size != 1:
        # Candidate exchange requires the same query rows on every rank.
        # PCP DualChunkSwap has different query rows and needs another exchange.
        raise NotImplementedError("GLM KPool DCP currently requires PCP=1 on the DCP worker.")
    if parallel.tensor_parallel_size % world:
        raise ValueError("GLM KPool DCP size must divide tensor_parallel_size.")
    if not check_cache_layout:
        # ReplaySSM validation precedes the platform's cache-size/interleave
        # normalization. The platform and backend validate the final layout.
        return
    pool = vllm_config.model_config.hf_text_config.index_kpool
    topk = vllm_config.model_config.hf_text_config.index_topk
    if pool <= 1 or topk <= 0 or topk % pool:
        raise ValueError("GLM KPool DCP requires index_topk divisible by index_kpool > 1.")
    interleave = parallel.cp_kv_cache_interleave_size
    block = vllm_config.cache_config.block_size
    if interleave <= 0 or interleave % pool or block % interleave:
        raise ValueError(
            "GLM KPool DCP requires a positive cp_kv_cache_interleave_size divisible by index_kpool "
            "and dividing the cache block size; "
            f"got interleave={interleave}, pool={pool}, block={block}."
        )
    # Shared candidate sorting represents IDs exactly in FP32 comparison keys.
    if vllm_config.model_config.max_model_len > 1 << 24:
        raise ValueError("GLM KPool DCP supports at most 2**24 tokens.")
