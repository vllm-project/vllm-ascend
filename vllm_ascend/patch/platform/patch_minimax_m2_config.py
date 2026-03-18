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
#
# Patch target: vllm/config/model.py
# - MiniMax-M2 fp8 checkpoint on NPU: disable fp8 quantization (load bf16
#   dequantized weights in worker patch) instead of failing validation.
# - For ACL graph capture, set HCCL_OP_EXPANSION_MODE=AIV if user didn't set it.
#

import os

from vllm.config.model import ModelConfig
from vllm.logger import logger
from vllm.platforms import current_platform

_original_verify_quantization = getattr(ModelConfig, "_verify_quantization", None)
_original_verify_cuda_graph = getattr(ModelConfig, "_verify_cuda_graph", None)

_DISABLE_FP8_LOG = (
    "Detected fp8 MiniMax-M2 checkpoint on NPU. "
    "Disabling fp8 quantization and loading dequantized bf16 "
    "weights instead."
)


def _get_model_type(cfg: ModelConfig) -> str | None:
    # vLLM config fields have changed across versions; try multiple sources.
    model_arch_cfg = getattr(cfg, "model_arch_config", None)
    if model_arch_cfg is not None:
        mt = getattr(model_arch_cfg, "model_type", None)
        if mt:
            return mt

    hf_text_cfg = getattr(cfg, "hf_text_config", None)
    if hf_text_cfg is not None:
        mt = getattr(hf_text_cfg, "model_type", None)
        if mt:
            return mt

    hf_cfg = getattr(cfg, "hf_config", None)
    if hf_cfg is not None:
        mt = getattr(hf_cfg, "model_type", None)
        if mt:
            return mt

    return getattr(cfg, "model_type", None)


def _should_disable_fp8(cfg: ModelConfig, quant_method: str | None) -> bool:
    return current_platform.device_name == "npu" and _get_model_type(cfg) == "minimax_m2" and quant_method == "fp8"


def _disable_fp8(cfg: ModelConfig, *, log: bool) -> bool:
    if not _should_disable_fp8(cfg, getattr(cfg, "quantization", None)):
        return False
    if log:
        logger.warning(_DISABLE_FP8_LOG)
    cfg.quantization = None
    return True


def _patched_verify_quantization(self: ModelConfig) -> None:
    """Inject mid-function behavior for ModelConfig._verify_quantization.

    Upstream validates quantization inside this method via:
        current_platform.verify_quantization(self.quantization)

    We emulate a mid-function patch without copying upstream code by temporarily
    overriding current_platform.verify_quantization while the original verifier
    executes.
    """
    assert _original_verify_quantization is not None

    orig_platform_verify = getattr(current_platform, "verify_quantization", None)

    def _platform_verify_hook(quant_method: str | None) -> None:
        if _should_disable_fp8(self, quant_method):
            # This is the effective "middle of _verify_quantization" interception.
            _disable_fp8(self, log=True)
            return
        assert orig_platform_verify is not None
        return orig_platform_verify(quant_method)

    # Some versions may read self.quantization before calling platform verifier.
    _disable_fp8(self, log=True)

    try:
        if orig_platform_verify is not None:
            current_platform.verify_quantization = _platform_verify_hook
        return _original_verify_quantization(self)
    finally:
        if orig_platform_verify is not None:
            current_platform.verify_quantization = orig_platform_verify
        # Ensure fp8 isn't restored by upstream logic.
        _disable_fp8(self, log=False)


def _patched_verify_cuda_graph(self: ModelConfig) -> None:
    assert _original_verify_cuda_graph is not None

    if (
        current_platform.device_name == "npu"
        and _get_model_type(self) == "minimax_m2"
        and not getattr(self, "enforce_eager", True)
    ):
        expansion_mode = os.environ.get("HCCL_OP_EXPANSION_MODE")
        if expansion_mode is None:
            os.environ["HCCL_OP_EXPANSION_MODE"] = "AIV"
            logger.info("Set HCCL_OP_EXPANSION_MODE=AIV for MiniMax-M2 ACL graph capture on NPU.")
        elif expansion_mode != "AIV":
            logger.warning(
                "HCCL_OP_EXPANSION_MODE=%s may reduce ACL graph shape "
                "coverage for MiniMax-M2 on NPU. Recommended value: AIV.",
                expansion_mode,
            )

    return _original_verify_cuda_graph(self)


if _original_verify_quantization is not None:
    ModelConfig._verify_quantization = _patched_verify_quantization

if _original_verify_cuda_graph is not None:
    ModelConfig._verify_cuda_graph = _patched_verify_cuda_graph
