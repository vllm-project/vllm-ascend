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

from vllm_ascend.device.device_config import is_310p

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None


def _validate_v2_model_runner(vllm_config: VllmConfig) -> None:
    """No-op replacement for the upstream V2 model runner validation.

    Ascend defaults to V2 and uses ``VLLM_USE_V2_MODEL_RUNNER`` as the explicit
    runner-selection override. Features on the Ascend V2 blacklist fall back to
    V1 only when that env var is unset. Upstream GPU-specific model, feature,
    and Triton checks do not apply to the Ascend runner.
    """


def apply_v2_model_runner_config_patch() -> None:
    """Apply the Ascend V2 model runner overrides to VllmConfig.

    Installs two overrides on the ``VllmConfig`` class:

    * ``use_v2_model_runner`` defaults to V2, honors an explicit
      ``VLLM_USE_V2_MODEL_RUNNER`` override, and falls back to V1 for
      blacklisted features (see :func:`use_v2_model_runner`).
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


def _is_configured(value: object) -> bool:
    """Return whether an optional config object is actually set.

    ``unittest.mock`` test doubles are not None and are truthy, so they must
    not be treated as enabled LoRA, pooling, or EC-transfer configs.
    """
    if value is None:
        return False
    return not type(value).__module__.startswith("unittest.mock")


_KV_POOL_CONNECTORS = frozenset({"AscendStoreConnector"})
_DFLASH2_ARCHITECTURES = frozenset({"DFlash2DraftModel"})
_NGRAM_SPEC_METHODS = frozenset({"ngram", "ngram_gpu"})
_HY3_ARCHITECTURES = frozenset({"HYV3ForCausalLM"})
_GEMMA4_ARCHITECTURES = frozenset(
    {
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    }
)


def _additional_config_value(vllm_config: VllmConfig, key: str) -> object:
    additional_config = getattr(vllm_config, "additional_config", None)
    if not _is_configured(additional_config):
        return None
    if isinstance(additional_config, dict):
        return additional_config.get(key)
    return getattr(additional_config, key, None)


def _draft_window_size(vllm_config: VllmConfig) -> object:
    return _additional_config_value(vllm_config, "draft_window_size")


def _is_kvpp_enabled(vllm_config: VllmConfig) -> bool:
    return _additional_config_value(vllm_config, "enable_kvpp") is True


def _collect_architectures(model_config: object) -> list[str]:
    architectures: list[str] = []
    architecture = getattr(model_config, "architecture", None)
    if isinstance(architecture, str):
        architectures.append(architecture)
    architectures.extend(getattr(model_config, "architectures", None) or ())
    for config_name in ("hf_config", "hf_text_config"):
        config = getattr(model_config, config_name, None)
        architectures.extend(getattr(config, "architectures", None) or ())
    return [name for name in architectures if isinstance(name, str)]


def _is_dflash2_spec(speculative_config: object) -> bool:
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return any(architecture in _DFLASH2_ARCHITECTURES for architecture in _collect_architectures(draft_model_config))


def _blacklisted_architecture(model_config: object) -> str | None:
    architectures = _collect_architectures(model_config)
    if any(architecture in _HY3_ARCHITECTURES or architecture.startswith("HYV3") for architecture in architectures):
        return "Hy3-preview"
    if any(
        architecture in _GEMMA4_ARCHITECTURES or architecture.startswith("Gemma4") for architecture in architectures
    ):
        return "Gemma4"
    return None


def _is_kv_pool(kv_transfer_config: object) -> bool:
    connector = getattr(kv_transfer_config, "kv_connector", None)
    if isinstance(connector, str) and connector in _KV_POOL_CONNECTORS:
        return True
    extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None)
    return isinstance(extra_config, dict) and extra_config.get("backend") == "memcache"


def _get_v2_model_runner_blacklist(vllm_config: VllmConfig) -> list[str]:
    """Collect Ascend features that are not V2-ready and default to V1."""
    unsupported: list[str] = []

    if is_310p():
        unsupported.append("310P")

    if _is_configured(getattr(vllm_config, "lora_config", None)):
        unsupported.append("LoRA")

    model_config = getattr(vllm_config, "model_config", None)
    if _is_configured(model_config):
        architecture = _blacklisted_architecture(model_config)
        if architecture is not None:
            unsupported.append(architecture)
        if (
            getattr(model_config, "runner_type", None) == "pooling"
            or getattr(model_config, "is_pooling_model", False) is True
        ):
            unsupported.append("pooling KV")
        if getattr(model_config, "is_encoder_decoder", False) is True:
            unsupported.append("encoder-decoder")

    if _is_kvpp_enabled(vllm_config):
        unsupported.append("KVPP")

    if _is_configured(getattr(vllm_config, "ec_transfer_config", None)):
        unsupported.append("VL encoder disaggregation")
    elif _is_configured(model_config):
        mm_config = getattr(model_config, "multimodal_config", None)
        if getattr(mm_config, "mm_encoder_only", False) is True:
            unsupported.append("VL encoder-only")

    if _draft_window_size(vllm_config) is not None:
        unsupported.append("draft_window_size")

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if _is_configured(speculative_config):
        spec_method = getattr(speculative_config, "method", None)
        if spec_method == "suffix":
            unsupported.append("suffix speculative decoding")
        if spec_method in _NGRAM_SPEC_METHODS:
            unsupported.append("ngram speculative decoding")
        if getattr(speculative_config, "parallel_drafting", False) is True:
            unsupported.append("parallel_drafting")
        if _is_dflash2_spec(speculative_config) and getattr(speculative_config, "enforce_eager", False) is not True:
            unsupported.append("dflash2 graph")

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if _is_configured(kv_transfer_config) and _is_kv_pool(kv_transfer_config):
        unsupported.append("KV pool")

    return unsupported


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Return whether the V2 model runner should be used on Ascend.

    An explicit ``VLLM_USE_V2_MODEL_RUNNER`` override wins. Otherwise V2 is
    the default, except for blacklisted models and features that still default
    to V1:

    * 310P
    * Hy3-preview (``HYV3*``)
    * Gemma4 (``Gemma4*``)
    * LoRA
    * pooling KV (``runner_type="pooling"``)
    * encoder-decoder (Whisper)
    * KVPP (``additional_config.enable_kvpp``)
    * VL encoder disaggregation (``ec_transfer_config`` / encoder-only)
    * draft_window_size
    * suffix speculative decoding
    * ngram speculative decoding (``ngram`` / ``ngram_gpu``)
    * parallel_drafting
    * dflash2 graph (DFlash2 drafts without ``enforce_eager``)
    * KV pool (``AscendStoreConnector`` / memcache)

    Set ``VLLM_USE_V2_MODEL_RUNNER=0`` to select V1 explicitly.
    """
    use_v2_model_runner = envs_vllm.VLLM_USE_V2_MODEL_RUNNER
    if use_v2_model_runner is not None:
        logger.info_once(
            "VLLM_USE_V2_MODEL_RUNNER=%s is set; using Model Runner %s.",
            use_v2_model_runner,
            "V2" if use_v2_model_runner else "V1",
        )
        return use_v2_model_runner

    unsupported = _get_v2_model_runner_blacklist(vllm_config)
    if unsupported:
        logger.warning_once(
            "Model Runner V2 does not yet support %s; using the V1 model runner instead.",
            ", ".join(unsupported),
        )
        return False

    logger.info_once("VLLM_USE_V2_MODEL_RUNNER is unset; using Model Runner V2 by default.")
    return True
