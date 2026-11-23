# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select Ascend's GLM and Kimi KDA RecoverSSM implementations."""

from functools import wraps

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, validate_glm_recoverssm
from vllm_ascend.models.kimi_k3.recoverssm_config import (
    use_request_indexed_kda_recoverssm,
    validate_request_indexed_kda_recoverssm,
)

_original_validate = VllmConfig.validate_mamba_cached_kernel


@wraps(_original_validate)
def validate_mamba_cached_kernel(config):
    if config.cache_config.use_replayssm and is_glm_recoverssm_model(config):
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
