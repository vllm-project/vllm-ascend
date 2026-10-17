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

_NGRAM_SPEC_METHODS = frozenset({"ngram", "ngram_gpu"})
_P_EAGLE_SPEC_METHODS = frozenset({"eagle", "eagle3"})
_DFLASH2_ARCHITECTURES = frozenset({"DFlash2DraftModel"})
_TRUTHY_STRINGS = frozenset({"1", "true", "yes"})


def _is_configured(value: object) -> bool:
    """Treat unset and ``unittest.mock`` doubles as missing config."""
    return value is not None and not type(value).__module__.startswith("unittest.mock")


def _additional(vllm_config: VllmConfig, key: str) -> object:
    extra = getattr(vllm_config, "additional_config", None)
    if not _is_configured(extra):
        return None
    return extra.get(key) if isinstance(extra, dict) else getattr(extra, key, None)


def _is_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    if isinstance(value, int):
        return value == 1
    return False


def _architectures(model_config: object) -> list[str]:
    names: list[str] = []
    architecture = getattr(model_config, "architecture", None)
    if isinstance(architecture, str):
        names.append(architecture)
    for owner in (
        model_config,
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
    ):
        names.extend(getattr(owner, "architectures", None) or ())
    return [name for name in names if isinstance(name, str)]


def _is_dflash2_graph(speculative_config: object) -> bool:
    """DFlash2 graph capture still fails on NPU; eager drafts stay on V2."""
    if getattr(speculative_config, "enforce_eager", False) is True:
        return False
    draft = getattr(speculative_config, "draft_model_config", None)
    return any(name in _DFLASH2_ARCHITECTURES for name in _architectures(draft))


def _v2_blacklist(vllm_config: VllmConfig) -> list[str]:
    """Reasons this config is not V2-ready and should default to V1."""
    reasons: list[str] = []
    model_config = getattr(vllm_config, "model_config", None)
    spec_config = getattr(vllm_config, "speculative_config", None)

    if is_310p():
        reasons.append("310P")

    if _is_configured(model_config):
        if (
            getattr(model_config, "runner_type", None) == "pooling"
            or getattr(model_config, "is_pooling_model", False) is True
        ):
            reasons.append("pooling KV")
        if getattr(model_config, "is_encoder_decoder", False) is True:
            reasons.append("encoder-decoder")

    if _is_configured(getattr(vllm_config, "ec_transfer_config", None)):
        reasons.append("VL encoder disaggregation")
    elif _is_configured(model_config):
        mm_config = getattr(model_config, "multimodal_config", None)
        if getattr(mm_config, "mm_encoder_only", False) is True:
            reasons.append("VL encoder-only")
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if _is_configured(compilation_config) and getattr(compilation_config, "cudagraph_mm_encoder", False) is True:
        reasons.append("VL encoder graph")
    if _additional(vllm_config, "draft_window_size") is not None:
        reasons.append("draft_window_size")
    # V2 MTP draft pads the forward to the DP-wide request count, while eager
    # attention metadata keeps this rank's request count. DSA-CP rejects that
    # split once the hidden states outgrow the TP-aligned metadata.
    if _is_enabled(_additional(vllm_config, "enable_dsa_cp")):
        reasons.append("enable_dsa_cp")

    if _is_configured(spec_config):
        method = getattr(spec_config, "method", None)
        if method == "suffix":
            reasons.append("suffix speculative decoding")
        if method in _NGRAM_SPEC_METHODS:
            reasons.append("ngram speculative decoding")
        # V2 init_speculator has no draft_model branch. Parallel drafting with
        # a separate draft model still runs on the V1 proposer.
        if method == "draft_model":
            reasons.append("draft_model speculative decoding")
        if _is_dflash2_graph(spec_config):
            reasons.append("dflash2 graph")
        # V2 FULL capture builds attention metadata inside ACL graph GLOBAL
        # capture. p-eagle reads NPU seq_lens there, and that copy is rejected.
        if method in _P_EAGLE_SPEC_METHODS and getattr(spec_config, "parallel_drafting", False) is True:
            reasons.append("p-eagle")

    return reasons


def use_v2_model_runner(vllm_config: VllmConfig) -> bool:
    """Select the Ascend model runner.

    ``VLLM_USE_V2_MODEL_RUNNER`` wins when set. Otherwise V2 is the default,
    except for these models and features, which fall back to V1:

    * 310P
    * pooling KV (``runner_type="pooling"``)
    * encoder-decoder (Whisper)
    * VL encoder disaggregation (``ec_transfer_config`` / encoder-only)
    * VL encoder graph (``compilation_config.cudagraph_mm_encoder``)
    * draft_window_size
    * enable_dsa_cp
    * suffix speculative decoding
    * ngram speculative decoding (``ngram`` / ``ngram_gpu``)
    * draft_model speculative decoding
    * dflash2 graph (DFlash2 drafts without ``enforce_eager``)
    * p-eagle (``eagle`` / ``eagle3`` with ``parallel_drafting``)

    Set ``VLLM_USE_V2_MODEL_RUNNER=0`` to force V1.
    """
    env_override = envs_vllm.VLLM_USE_V2_MODEL_RUNNER
    if env_override is not None:
        logger.info_once(
            "VLLM_USE_V2_MODEL_RUNNER=%s is set; using Model Runner %s.",
            env_override,
            "V2" if env_override else "V1",
        )
        return env_override

    unsupported = _v2_blacklist(vllm_config)
    if unsupported:
        logger.warning_once(
            "Model Runner V2 does not yet support %s; using the V1 model runner instead.",
            ", ".join(unsupported),
        )
        return False

    logger.info_once("VLLM_USE_V2_MODEL_RUNNER is unset; using Model Runner V2 by default.")
    return True


def _validate_v2_model_runner(vllm_config: VllmConfig) -> None:
    """Skip upstream GPU/Triton V2 checks; Ascend uses :func:`use_v2_model_runner`."""


def apply_v2_model_runner_config_patch() -> None:
    """Install Ascend runner selection on ``VllmConfig``.

    Re-apply in every process that reads ``use_v2_model_runner`` (frontend,
    workers, engine-core). Pickled configs do not carry this class patch.
    """
    from vllm.config.vllm import VllmConfig

    VllmConfig.use_v2_model_runner = property(use_v2_model_runner)
    VllmConfig._validate_v2_model_runner = _validate_v2_model_runner
