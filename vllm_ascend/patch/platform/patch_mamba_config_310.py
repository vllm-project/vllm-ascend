# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from math import lcm

import vllm.model_executor.models.config
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


@classmethod
def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
    """
    Ensure that page size of attention layers is greater than or
    equal to the mamba layers. If not, automatically set the attention
    block size to ensure that it is. If the attention page size is
    strictly greater than the mamba page size, we pad the mamba page size
    to make them equal.

    Args:
        vllm_config: vLLM Config
    """
    logger = init_logger(__name__)
    # Save the user input before it gets modified by MambaModelConfig
    mamba_block_size = vllm_config.cache_config.mamba_block_size
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    attention_config = vllm_config.attention_config
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    # get attention page size (for 1 token)
    # Attention backend constraints:
    # - FlashAttention (FA) requires block size to be 64 or 128
    if model_config.use_mla:
        raise RuntimeError("MLA is not supported on 310P currently.")
    kernel_block_alignment_size = 128
    if (model_config.get_head_size() == 256):
        # https://github.com/flashinfer-ai/flashinfer/issues/1993 reports that`
        # head size 256 and block size 16 is not supported on blackwell.
        kernel_block_alignment_size = 64
    attn_page_size_1_token = FullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
    ).page_size_bytes

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba page size
    mamba_page_size = MambaSpec(
        shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=-1,  # block_size doesn't matter for mamba page size
    ).page_size_bytes

    # Model may be marked as is_hybrid
    #  but mamba is skipped via config,
    #  return directly
    if mamba_page_size == 0:
        return

    if cache_config.mamba_cache_mode == "all":
        # With prefix caching, select attention block size to
        # optimize for mamba kernel performance

        # Mamba2 SSD kernel uses a chunk_size, e.g. 256
        # Align the block to the kernel: use lowest multiple of chunk_size
        # of attention tokens that would fit mamba_page_size:
        # e.g. for mamba page size = 788kB
        #          attn_1_token = 2kB -> fits ~394 tokens
        #      then round up to a multiple of 256 -> 512 tokens
        # End result:
        #  attn_block_size = 512
        #  mamba_block_size = 512 (aligned to a multiple of chunk_size)
        # TODO(tdoublep): this constraint can be relaxed fairly
        # easily by changing the way we layout chunks in the
        # mamba2 kernels.

        base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
        attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
        chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
        attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
        cache_config.mamba_block_size = attn_block_size
    else:
        # Without prefix caching, select minimum valid attention block size
        # to minimize mamba state padding

        # Calculate minimum attention block size that satisfies both:
        # 1. Backend alignment requirements (kernel_block_alignment_size)
        # 2. Mamba page size compatibility (attn_page_size >= mamba_page_size)
        attn_block_size = kernel_block_alignment_size * cdiv(
            mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token
        )

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens "
            "to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # By default, mamba block size will be set to max_model_len.
    # When enabling prefix caching and using align mamba cache
    # mode, we align mamba block size to the block size as the
    # basic granularity for prefix caching.
    if cache_config.mamba_cache_mode == "align":
        cache_config.mamba_block_size = cache_config.block_size

    # compute new attention page size
    attn_page_size = cache_config.block_size * attn_page_size_1_token

    assert attn_page_size >= mamba_page_size

    if attn_page_size == mamba_page_size:
        # don't need to pad mamba page size
        return

    # pad mamba page size to exactly match attention
    if (
        cache_config.mamba_page_size_padded is None
        or cache_config.mamba_page_size_padded != attn_page_size
    ):
        cache_config.mamba_page_size_padded = attn_page_size
        mamba_padding_pct = (
            100 * (attn_page_size - mamba_page_size) / mamba_page_size
        )
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.",
            mamba_padding_pct,
        )

vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
