#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Backport vLLM PP + MTP runtime support.

The local Eagle/MTP drafter returns the draft tokens that belong to the model
output being processed. With PP batch_queue, EngineCore schedules a newer batch
before consuming the older output, so updating ``request.spec_token_ids`` from
``post_step`` observes live Request state from the newer schedule step.
"""

from __future__ import annotations

import copy
from functools import wraps

from vllm.logger import logger

_PATCHED = False


def _patch_model_config_validation() -> None:
    from typing import get_args

    from vllm.config.model import ModelConfig
    from vllm.config.speculative import MTPModelTypes

    original_verify = ModelConfig.verify_with_parallel_config
    if getattr(original_verify, "_vllm_ascend_pp_mtp_patched", False):
        return

    mtp_model_types = set(get_args(MTPModelTypes))

    @wraps(original_verify)
    def _patched_verify_with_parallel_config(self, parallel_config):
        hf_config = getattr(self, "hf_config", None)
        model_type = getattr(hf_config, "model_type", None)
        is_eagle_drafter = (model_type == "eagle" or model_type == "speculators") and any(
            arch.startswith("Eagle") or arch.endswith("Eagle3") for arch in getattr(self, "architectures", ())
        )
        is_mtp_drafter = model_type in mtp_model_types
        if (
            getattr(self, "runner", None) == "draft"
            and (is_eagle_drafter or is_mtp_drafter)
            and getattr(parallel_config, "pipeline_parallel_size", 1) > 1
        ):
            # Local Eagle/MTP drafters are loaded on the last PP stage rather
            # than partitioned across all PP stages. Keep normal target-model
            # validation intact, but validate these draft models as PP=1.
            logger.warning(
                "Validating local Eagle/MTP drafter with pipeline_parallel_size=1 "
                "because it is loaded locally on the last pipeline stage."
            )
            patched_config = copy.copy(parallel_config)
            patched_config.pipeline_parallel_size = 1
            return original_verify(self, patched_config)
        return original_verify(self, parallel_config)

    _patched_verify_with_parallel_config._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
    ModelConfig.verify_with_parallel_config = _patched_verify_with_parallel_config


def _patch_v2_model_runner_supported() -> None:
    """Patch VllmConfig._is_default_v2_model_runner_model to exclude DeepSeek-V2.

    Upstream vLLM added DeepseekV2ForCausalLM to DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES,
    but vllm-ascend's v2 model runner does not fully support DeepSeek-V2 in PP scenarios.
    This ensures DeepSeek-V2 uses the v1 model runner which has proper PP support.

    We patch the _is_default_v2_model_runner_model method which is called by
    the use_v2_model_runner property.
    """
    try:
        import vllm.config.vllm as vllm_config_module

        original_is_v2_model = vllm_config_module.VllmConfig._is_default_v2_model_runner_model
        if getattr(original_is_v2_model, "_vllm_ascend_pp_mtp_patched", False):
            return

        @wraps(original_is_v2_model)
        def _patched_is_default_v2_model_runner_model(self):
            # First check if this is DeepSeek-V2
            model_config = getattr(self, "model_config", None)
            if model_config is not None:
                architectures = getattr(model_config, "architectures", [])
                if any("DeepseekV2" in arch for arch in architectures):
                    # DeepSeek-V2 should use v1 model runner for PP support
                    logger.info("[Patch] DeepSeek-V2 detected, using v1 model runner for proper PP support.")
                    return False
            # For other models, use the original logic
            return original_is_v2_model(self)

        _patched_is_default_v2_model_runner_model._vllm_ascend_pp_mtp_patched = True  # type: ignore[attr-defined]
        vllm_config_module.VllmConfig._is_default_v2_model_runner_model = _patched_is_default_v2_model_runner_model
        logger.info("[Patch] Patched VllmConfig._is_default_v2_model_runner_model for DeepSeek-V2.")
    except ImportError:
        pass
    except Exception as e:
        logger.error("[Patch] Failed to patch VllmConfig._is_default_v2_model_runner_model: %s", e)


def _apply_patch() -> None:
    global _PATCHED
    if _PATCHED:
        logger.info("[Patch] patch_pp_mtp already applied, skipping")
        return
    _PATCHED = True
    logger.info("[Patch] Applying patch_pp_mtp patches...")
    _patch_model_config_validation()
    _patch_v2_model_runner_supported()
    logger.info("[Patch] patch_pp_mtp patches applied successfully")


_apply_patch()
