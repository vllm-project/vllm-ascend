# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .amd.model import (
        Qwen4ExpForCausalLM,
        Qwen4ExpForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen4ExpMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen4ExpForCausalLM",
        "Qwen4ExpForConditionalGeneration",
        "Qwen4ExpMTP",
    }:
        from vllm.platforms import current_platform

        if current_platform.device_type in ("npu", "rocm"):
            from .amd.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .amd.mtp import Qwen4ExpMTP
        elif current_platform.device_type == "cuda":
            from .nvidia.model import (
                Qwen4ExpForCausalLM,
                Qwen4ExpForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen4ExpMTP
        else:
            raise NotImplementedError(
                "Qwen4Exp supports CUDA, ROCm, and NPU platforms"
            )

        return {
            "Qwen4ExpForCausalLM": Qwen4ExpForCausalLM,
            "Qwen4ExpForConditionalGeneration": (Qwen4ExpForConditionalGeneration),
            "Qwen4ExpMTP": Qwen4ExpMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]
