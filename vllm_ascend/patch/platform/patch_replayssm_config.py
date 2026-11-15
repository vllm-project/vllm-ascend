"""Extend the upstream ReplaySSM selector to Ascend's GLM KDA and Qwen GDN implementations."""

import logging
from functools import wraps

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, validate_glm_recoverssm
from vllm_ascend.models.qwen_gdn_recoverssm import (
    _uses_gdn_recoverssm,
    is_gdn_model,
    validate_qwen_gdn_recoverssm,
)

logger = logging.getLogger(__name__)

_original_validate = VllmConfig.validate_mamba_cached_kernel


@wraps(_original_validate)
def validate_mamba_cached_kernel(config):
    if config.cache_config.use_replayssm and is_gdn_model(config):
        if _uses_gdn_recoverssm(config):
            validate_qwen_gdn_recoverssm(config)
            # Upstream's historical name selects the generic RecoverSSMState
            # lifecycle as well as single-slot metadata, not a particular kernel.
            config.cache_config.use_kda_recoverssm = True
            return config
        # Unsupported method on GDN: fall through to the original handler.
        logger.info_once(
            "RecoverSSM not enabled for speculative method %r on GDN model.",
            getattr(config.speculative_config, "method", None),
        )
        return _original_validate(config)
    if config.cache_config.use_replayssm and is_glm_recoverssm_model(config):
        validate_glm_recoverssm(config)
        config.cache_config.use_kda_recoverssm = True
        return config
    return _original_validate(config)


VllmConfig.validate_mamba_cached_kernel = validate_mamba_cached_kernel
VllmConfig.__pydantic_decorators__.model_validators["validate_mamba_cached_kernel"].func = validate_mamba_cached_kernel
rebuild_dataclass(VllmConfig, force=True)
