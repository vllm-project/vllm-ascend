# mypy: ignore-errors
import math

import vllm.model_executor.models.config
from vllm.logger import logger
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.config import MambaModelConfig
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size


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
    # Enable FULL_AND_PIECEWISE by default
    MambaModelConfig.verify_and_update_config(vllm_config)

    cache_config = vllm_config.cache_config
    using_kv_transfer_with_hybrid = (
        not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and vllm_config.kv_transfer_config
    )
    if using_kv_transfer_with_hybrid:
        if cache_config.mamba_cache_mode == "none":
            cache_config.mamba_cache_mode = "align"
        else:
            assert cache_config.mamba_cache_mode == "align", (
                "mamba_cache_mode only support 'align' when kv_transfer enabled now!"
            )

vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = verify_and_update_config
