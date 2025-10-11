# mypy: ignore-errors
import vllm.model_executor.models.config
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.ascend_config import get_ascend_config


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
    try:
        ascend_config = get_ascend_config()
    except RuntimeError:
        ascend_config = None

    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config

    if cache_config.cache_dtype == "auto":
        kv_cache_dtype = model_config.dtype
    else:
        kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

    # get attention page size (for 1 token)
    attn_page_size_1_token = FullAttentionSpec(
        block_size=1,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
        use_mla=model_config.use_mla,
        use_sfa=ascend_config.use_sfa
        if ascend_config is not None else False).page_size_bytes

    model_cls, _ = ModelRegistry.resolve_model_cls(
        model_config.architecture,
        model_config=model_config,
    )

    # get mamba page size
    mamba_page_size = MambaSpec(
        shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
        dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
        block_size=model_config.max_model_len,
    ).page_size_bytes

    block_alignment_bytes = 64

    # some attention backends (e.g. FA) only support setting
    # block size to multiple of 16, so let's suggest a value
    # that would work (note: FA is currently not compatible
    # with mamba layers, use FlashInfer instead).
    attn_block_size = block_alignment_bytes * cdiv(
        mamba_page_size, block_alignment_bytes * attn_page_size_1_token)

    # override attention block size if either (a) the
    # user has not set it or (b) the user has set it
    # too small.
    if (cache_config.block_size is None
            or cache_config.block_size < attn_block_size):
        cache_config.block_size = attn_block_size
        logger.info(
            "Setting attention block size to %d tokens "
            "to ensure that attention page size is >= mamba page size.",
            attn_block_size)

    # compute new attention page size
    attn_page_size = \
        cache_config.block_size * attn_page_size_1_token

    assert attn_page_size >= mamba_page_size

    if attn_page_size == mamba_page_size:
        # don't need to pad mamba page size
        return

    # pad mamba page size to exactly match attention
    if (cache_config.mamba_page_size_padded is None
            or cache_config.mamba_page_size_padded != attn_page_size):
        cache_config.mamba_page_size_padded = (attn_page_size)
        mamba_padding_pct = 100 * (attn_page_size -
                                   mamba_page_size) / mamba_page_size
        logger.info(
            "Padding mamba page size by %.2f%% to ensure "
            "that mamba page size and attention page size are "
            "exactly equal.", mamba_padding_pct)


vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
