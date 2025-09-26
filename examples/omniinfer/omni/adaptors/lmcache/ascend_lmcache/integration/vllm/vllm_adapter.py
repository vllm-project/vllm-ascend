# Standard
import os
from typing import Optional, Union

# Third Party
import torch

from vllm.config import VllmConfig
from vllm.utils import get_kv_cache_torch_dtype

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME, lmcache_get_config
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from ascend_lmcache.v1.npu_connector import VLLMPagedMemNPUConnectorV2

logger = init_logger(name=__name__)

def init_lmcache_engine(
    vllm_config: "VllmConfig"
) -> Optional[LMCacheEngine]:
    if LMCacheEngineBuilder.get(ENGINE_NAME) is not None:
        return None

    config = lmcache_get_config()
    if not isinstance(config, LMCacheEngineConfig):
        logger.error("config is not LMCacheEngineConfig")
        return None
    if vllm_config is None:
        logger.error("vllm_config is null")
        return None
    cache_config = vllm_config.cache_config
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    speculative_config = vllm_config.speculative_config

    kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

    use_mla = False
    if (
        hasattr(model_config, "use_mla")
        and isinstance(model_config.use_mla, bool)
        and model_config.use_mla
    ):
        use_mla = True
    if use_mla and (config.remote_serde != "naive" and config.remote_serde is not None):
        raise ValueError("MLA only works with naive serde mode..")

    # construct kv shape (for mem pool)
    num_layer = model_config.get_num_layers(parallel_config)
    if speculative_config is not None:
        num_layer += speculative_config.num_speculative_tokens
        logger.info(f"use speculative, num_speculative_tokens is : {speculative_config.num_speculative_tokens}")
    chunk_size = config.chunk_size
    num_kv_head = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
    logger.info(f"use mla: {use_mla}, kv shape: {kv_shape}")

    # Change current device.
    torch.npu.device(parallel_config.rank)
    device = torch.device(f"npu:{parallel_config.rank}")
    metadata = LMCacheEngineMetadata(
        model_config.model,
        parallel_config.world_size,
        parallel_config.rank,
        "vllm",
        kv_dtype,
        kv_shape,
        use_mla=use_mla,
    )

    if use_mla:
        metadata.world_size = 1
        metadata.worker_id = 0

    hidden_dim_size = num_kv_head * head_size
    if config.use_layerwise:
        raise ValueError("layerwise is not supported")
    else:
        vllm_gpu_connector = VLLMPagedMemNPUConnectorV2(
            hidden_dim_size,
            num_layer,
            block_size=cache_config.block_size,
            chunk_size=chunk_size,
            dtype=kv_dtype,
            device=device,
            use_mla=use_mla,
        )

    engine = LMCacheEngineBuilder.get_or_create(
        ENGINE_NAME, config, metadata, vllm_gpu_connector
    )

    return engine