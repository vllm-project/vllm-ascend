# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Quantize the drafter's copy of a shared target lm_head online.

additional_config ``draft_lm_head_quantization="w8a8_dynamic"`` selects
online 8-bit dynamic quantization for the copy of the target lm_head that
a drafter shares (EAGLE/EAGLE3/MTP). The format is adjudicated by
hardware: MXFP8 on A5, INT8 on A2/A3.

The copy itself is derived entirely by the upstream mechanism driven from
``load_eagle_model``: the copy's weight shares the source storage, MTP
per-layer shared heads are fixed up, and RL weight updates re-derive the
copy in place through the upstream ``_SOURCES`` registry. This patch only
bridges the configuration gap: the upstream
``SpeculativeConfig.draft_lm_head_quantization`` field is validated at
construction and only accepts the platform formats ("fp8"/"nvfp4"), so the
Ascend format is resolved here and assigned to the field around the call
only.

The Ascend lm_head schemes (shared with the target ``lm_head_quantization``
feature) implement the scheme interface, not the vLLM quant-method
interface, so they are registered under internal format keys in the
upstream ``_LM_HEAD_METHODS`` table wrapped in ``AscendEmbeddingMethod``,
the adapter that ``_quantize_copy`` instantiates through.

Requires Model Runner V2 and a drafter that shares the target lm_head
(upstream rejects the rest). Mutually exclusive with the target
``lm_head_quantization``: the draft copy is derived from the unquantized
target head.
"""

import functools

import torch
import vllm.v1.worker.gpu.spec_decode.eagle.speculator as eagle_speculator_module
import vllm.v1.worker.gpu.spec_decode.eagle.utils as eagle_utils_module
import vllm.v1.worker.gpu.spec_decode.mtp.speculator as mtp_speculator_module
import vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator as multi_module_mtp_speculator_module
from vllm.config import VllmConfig
from vllm.logger import logger

import vllm_ascend.worker.v2.spec_decode.multi_module_mtp.speculator as ascend_mtp_speculator_module
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, HardwareCapability, get_current_hardware_profile

DRAFT_LM_HEAD_QUANTIZATION = "w8a8_dynamic"

_original_load_eagle_model = eagle_utils_module.load_eagle_model


def _make_int8_lm_head_method():
    # Registered lazily: the scheme reads the current vllm config at
    # construction, which is only valid inside _quantize_copy.
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod

    return AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())


def _make_mxfp8_lm_head_method():
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLmHeadMethod

    return AscendEmbeddingMethod(AscendW8A8MXFP8DynamicLmHeadMethod())


def _register_lm_head_methods() -> None:
    # _LM_HEAD_METHODS maps a format name to a zero-argument factory for the
    # copy's quant method; upstream instantiates it inside _quantize_copy.
    from vllm.model_executor.layers.quantization.online import lm_head as online_lm_head_module

    methods = online_lm_head_module._LM_HEAD_METHODS
    methods.setdefault("int8", _make_int8_lm_head_method)
    methods.setdefault("mxfp8", _make_mxfp8_lm_head_method)


def _resolve_format() -> str:
    device_family = get_current_hardware_profile().device_adaptor_family
    if device_family == DeviceAdaptorFamily.COMPATIBILITY:
        raise ValueError("draft_lm_head_quantization is not yet adapted for 310P.")
    return "mxfp8" if device_family == DeviceAdaptorFamily.FP8_OPTIMIZED else "int8"


@functools.wraps(_original_load_eagle_model)
def _load_eagle_model_with_draft_quant(
    target_model: torch.nn.Module, vllm_config: VllmConfig
) -> torch.nn.Module:
    speculative_config = vllm_config.speculative_config
    additional_config = vllm_config.additional_config or {}
    quantization = additional_config.get("draft_lm_head_quantization")
    if quantization is None:
        return _original_load_eagle_model(target_model, vllm_config)
    if speculative_config is None or speculative_config.draft_lm_head_quantization is not None:
        raise ValueError(
            "draft_lm_head_quantization in additional_config conflicts with the "
            "speculative_config field of the same name; select the Ascend "
            "feature through additional_config only."
        )
    if quantization != DRAFT_LM_HEAD_QUANTIZATION:
        raise ValueError(f"Unsupported draft_lm_head_quantization: {quantization!r}.")
    if additional_config.get("lm_head_quantization") is not None:
        raise ValueError(
            "draft_lm_head_quantization is mutually exclusive with "
            "lm_head_quantization: the draft copy is derived from the "
            "unquantized target lm_head."
        )
    if not vllm_config.use_v2_model_runner:
        raise ValueError("draft_lm_head_quantization requires Model Runner V2.")
    quant_format = _resolve_format()
    _register_lm_head_methods()
    logger.info_once("Quantizing the draft lm_head copy online with %s.", quant_format)
    # The field only accepts the platform formats at construction, so assign
    # the resolved format around the call only. RL weight updates re-derive
    # the copy through the upstream _SOURCES registry, not this field.
    previous = speculative_config.draft_lm_head_quantization
    try:
        object.__setattr__(speculative_config, "draft_lm_head_quantization", quant_format)
        return _original_load_eagle_model(target_model, vllm_config)
    finally:
        object.__setattr__(speculative_config, "draft_lm_head_quantization", previous)


# The speculators bind load_eagle_model by name at import time, so every
# namespace holding a direct reference must point at the wrapper.
eagle_utils_module.load_eagle_model = _load_eagle_model_with_draft_quant
eagle_speculator_module.load_eagle_model = _load_eagle_model_with_draft_quant
mtp_speculator_module.load_eagle_model = _load_eagle_model_with_draft_quant
multi_module_mtp_speculator_module.load_eagle_model = _load_eagle_model_with_draft_quant
ascend_mtp_speculator_module.load_eagle_model = _load_eagle_model_with_draft_quant

# 310P keeps its own MTP speculator binding; wire it too so setting the
# config there fails with the explicit "not yet adapted" error instead of
# being silently ignored.
if not get_current_hardware_profile().supports(HardwareCapability.STANDARD_WORKER_PATCHES):
    import vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator as mtp_speculator_310p_module

    mtp_speculator_310p_module.load_eagle_model = _load_eagle_model_with_draft_quant
