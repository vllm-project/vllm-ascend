# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select Ascend's GLM and Kimi KDA RecoverSSM implementations."""

from copy import copy
from functools import wraps

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, validate_glm_recoverssm
from vllm_ascend.models.kimi_k3.recoverssm_config import (
    use_request_indexed_kda_recoverssm,
    validate_request_indexed_kda_recoverssm,
)
from vllm_ascend.models.qwen_gdn_recoverssm import is_gdn_model, validate_qwen_gdn_recoverssm

_original_validate = VllmConfig.validate_mamba_cached_kernel


@wraps(_original_validate)
def validate_mamba_cached_kernel(config):
    speculative = getattr(config, "speculative_config", None)
    if (
        config.cache_config.use_replayssm
        and getattr(config.model_config, "architecture", None) == "Qwen4ExpMTP"
        and speculative is not None
        and config.model_config is getattr(speculative, "draft_model_config", None)
    ):
        # Qwen4Exp constructs an internal draft VllmConfig by replacing only
        # ModelConfig. CacheConfig is still shared with the target at this
        # point. Its QSA-only MTP blocks must not inherit target GDN recovery;
        # copy before clearing flags, or target recovery is disabled as well.
        config.cache_config = copy(config.cache_config)
        config.cache_config.use_replayssm = False
        config.cache_config.use_kda_recoverssm = False
    if config.cache_config.use_replayssm:
        if is_gdn_model(config):
            validate_qwen_gdn_recoverssm(config)
            # Upstream's historical name selects the generic RecoverSSMState
            # lifecycle as well as single-slot metadata, not a particular kernel.
            config.cache_config.use_kda_recoverssm = True
            return config
        if is_glm_recoverssm_model(config):
            validate_glm_recoverssm(config)
            config.cache_config.use_kda_recoverssm = True
            return config
    if use_request_indexed_kda_recoverssm(config):
        validate_request_indexed_kda_recoverssm(config)
        config.cache_config.use_kda_recoverssm = True
        return config
    return _original_validate(config)


VllmConfig.validate_mamba_cached_kernel = validate_mamba_cached_kernel
VllmConfig.__pydantic_decorators__.model_validators["validate_mamba_cached_kernel"].func = validate_mamba_cached_kernel
rebuild_dataclass(VllmConfig, force=True)
