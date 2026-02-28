# mypy: ignore-errors
import vllm.model_executor.models.config
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

import numpy as np

@classmethod
def verify_and_update_config(cls, vllm_config) -> None:
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
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    block_alignment_bytes = 128
    # get attention block size
    attn_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    attn_head_size = model_config.get_head_size()
    attn_single_block_page_size = attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba block size
    dummy_mamba_spec = MambaSpec(
        shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=model_config.max_model_len,
    )
    ssm_shape, ssm_dtype = None, None
    conv_shape, conv_dtype = None, None
    for shape, dtype in zip(dummy_mamba_spec.shapes, dummy_mamba_spec.dtypes):
        if len(shape) == 3:
            ssm_shape, ssm_dtype = shape, dtype
        else:
            conv_shape, conv_dtype = shape, dtype

    # NOTE(zxr): because of the limit of Ascend Hardware, we need to keep
    # all cache tensors contiguous, so we align the page size of ssm_block
    # and single attn_block
    ssm_block_page_size = int(np.prod(ssm_shape) * get_dtype_size(ssm_dtype))
    conv_block_page_size = int(np.prod(conv_shape) * get_dtype_size(conv_dtype))
    attn_block_size = block_alignment_bytes * cdiv(ssm_block_page_size, block_alignment_bytes * attn_single_block_page_size)
    assert attn_single_block_page_size * block_alignment_bytes != ssm_block_page_size, "Cannot align ssm_page_size and attn_page_size."

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens to ensure that attention page size is >= mamba page size.",
            attn_block_size,
        )

    # compute new attention page size
    attn_page_size = cache_config.block_size * 2 * attn_head_size * attn_num_kv_heads * get_dtype_size(kv_cache_dtype)

    # pad mamba page size for conv_blocks
    if cache_config.mamba_page_size_padded is None or cache_config.mamba_page_size_padded != attn_page_size + conv_block_page_size:
        cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size
        mamba_padding_pct = 100 * conv_block_page_size / cache_config.mamba_page_size_padded
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.",
            mamba_padding_pct,
        )


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
