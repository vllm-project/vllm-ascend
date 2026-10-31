# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend the upstream ReplaySSM selector to Ascend's GLM KDA implementation."""

from functools import wraps

from pydantic.dataclasses import rebuild_dataclass
from vllm.config import VllmConfig

from vllm_ascend.models.glm5next.recoverssm_config import is_glm_recoverssm_model, validate_glm_recoverssm

_original_validate = VllmConfig.validate_mamba_cached_kernel


@wraps(_original_validate)
def validate_mamba_cached_kernel(config):
    if config.cache_config.use_replayssm and is_glm_recoverssm_model(config):
        validate_glm_recoverssm(config)
        config.cache_config.use_kda_recoverssm = True
        return config
    return _original_validate(config)


VllmConfig.validate_mamba_cached_kernel = validate_mamba_cached_kernel
VllmConfig.__pydantic_decorators__.model_validators["validate_mamba_cached_kernel"].func = validate_mamba_cached_kernel
rebuild_dataclass(VllmConfig, force=True)
