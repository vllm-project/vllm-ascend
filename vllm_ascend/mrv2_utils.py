#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.envs as envs_vllm
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

from vllm_ascend.utils import is_310p

# Architectures for which Model Runner V2 is enabled by default on Ascend.
DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
    }
)


def _validate_v2_model_runner(vllm_config: VllmConfig) -> None:
    """No-op replacement for the upstream V2 model runner validation.

    Ascend fully owns the V2 model runner enablement decision through the model
    / feature whitelists in :func:`use_v2_model_runner`, so the upstream checks
    -- Triton availability plus the list of features the *upstream* GPU V2
    runner does not yet support -- are intentionally decoupled. Otherwise a V2
    enablement decision made here (e.g. via an explicit
    ``VLLM_USE_V2_MODEL_RUNNER=1``) could fail at config construction with
    upstream checks that do not apply to the Ascend runner.
    """


def apply_v2_model_runner_config_patch() -> None:
    """Apply the Ascend V2 model runner overrides to VllmConfig.

    Installs two overrides on the ``VllmConfig`` class:

    * ``use_v2_model_runner`` is driven by the Ascend whitelist default instead
      of the upstream default decision (see :func:`use_v2_model_runner`).
    * ``_validate_v2_model_runner`` is neutralized because the upstream checks
      describe the upstream GPU runner and do not apply to the Ascend runner.

    Must run wherever ``VllmConfig`` is (re)created or its properties are read
    in a separate process -- the frontend during config construction, each
    worker process, and the engine-core process (the scheduler reads
    ``use_v2_model_runner`` there from a pickled config, so the class-level
    patch does not carry over from the frontend). Repeated application is
    harmless: it just re-assigns the same overrides.
    """
    from vllm.config.vllm import VllmConfig

    VllmConfig.use_v2_model_runner = property(use_v2_model_runner)
    VllmConfig._validate_v2_model_runner = _validate_v2_model_runner


def is_default_v2_model_runner_model(vllm_config: VllmConfig) -> bool:
    """Model whitelist: enable V2 for default-V2 architectures."""
    model_config = vllm_config.model_config
    if model_config is None:
        return False

    if getattr(model_config, "runner_type", "generate") != "generate":
        return False

    if getattr(model_config, "is_hybrid", False):
        return False

    if getattr(model_config, "is_attention_free", False):
        return False

    architectures = getattr(model_config, "architectures", [])
    return any(arch in DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES for arch in architectures)


def is_supported_v2_model_runner_feature(vllm_config: VllmConfig) -> bool:
    """Feature whitelist: only whitelisted features may be enabled with a whitelisted model."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return True
    if speculative_config.method in ("eagle3", "mtp", "dflash"):
        logger.info_once(
            "Model Runner V2 is enabled by default for speculative method '%s'.",
            speculative_config.method,
        )
        return True
    return False


def _v2_model_runner_environment_ready(vllm_config: VllmConfig) -> bool:
    """Check the remaining V2 gates (feature whitelist + platform + Triton)."""
    if not is_supported_v2_model_runner_feature(vllm_config):
        return False

    if is_310p():
        logger.warning_once("Model Runner V2 is not supported on 310P; using the V1 model runner instead.")
        return False

    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON:
        logger.warning_once("Model Runner V2 requires Triton; using the V1 model runner instead.")
        return False

    return True


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Return whether the V2 model runner should be used on Ascend.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER`` override wins. Otherwise the V2
    runner is enabled by default only when all of the following hold:

    * the model is on the default-V2 model whitelist,
    * the enabled features are on the V2 feature whitelist,
    * the platform is not 310P and the runtime provides Triton.
    """
    use_v2_model_runner = envs_vllm.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        logger.info_once(
            "VLLM_USE_V2_MODEL_RUNNER=%s is set; using Model Runner %s.",
            use_v2_model_runner,
            "V2" if use_v2_model_runner else "V1",
        )
        return use_v2_model_runner

    if is_default_v2_model_runner_model(vllm_config):
        if _v2_model_runner_environment_ready(vllm_config):
            architectures = getattr(vllm_config.model_config, "architectures", [])
            logger.info_once(
                "Model Runner V2 is enabled for %s.",
                ", ".join(architectures),
            )
            return True
        return False

    logger.warning_once(
        "Model Runner V2 model whitelist does not include this model; using the V1 model runner instead."
    )
    return False
