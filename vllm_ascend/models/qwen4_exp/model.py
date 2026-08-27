# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend wrappers for the upstream Qwen4Exp model implementation."""

import sys
import types

import torch
import vllm.envs as vllm_envs
from torch import nn
from vllm.config import VllmConfig
from vllm.platforms import current_platform

from vllm_ascend.ops.triton.qwen4_exp import hc as ascend_hc_ops
from vllm_ascend.ops.triton.qwen4_exp import qsa as ascend_qsa_ops

from .ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .ple_offload_layer import PleOffloadLayer

# The vendored NVIDIA modules use relative imports for their platform kernels.
# Bind those names to Ascend implementations before importing any model module.
_PACKAGE = __package__
sys.modules[f"{_PACKAGE}.nvidia.ops.hc"] = ascend_hc_ops
sys.modules[f"{_PACKAGE}.nvidia.ops.qsa"] = ascend_qsa_ops

from .nvidia import hyperconnection as upstream_hc  # noqa: E402

# Keep the upstream model and weight loader authoritative. HyperConnection's
# CUDA glue functions are module globals, so replace them before importing the
# model classes that instantiate GatedResidual.
upstream_hc.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
upstream_hc.hc_combine = hc_combine
upstream_hc.hc_combine_norm = hc_combine_norm
upstream_hc.hc_gate_mix = hc_gate_mix
upstream_hc.hc_silu = hc_silu


def _raise_for_ple_cpu_offload() -> None:
    if getattr(vllm_envs, "VLLM_PLE_CPU_OFFLOAD", False):
        raise NotImplementedError(
            "VLLM_PLE_CPU_OFFLOAD uses CUDA IPC stream semaphores and is not "
            "supported on Ascend. Disable it to run Qwen3.8-Flash-Next."
        )

def _get_ascend_ple_device(cls: type[PleOffloadLayer]) -> torch.device:
    del cls
    _raise_for_ple_cpu_offload()
    return torch.device(
        current_platform.device_type,
        torch.accelerator.current_device_index(),
    )


PleOffloadLayer.get_target_device = classmethod(_get_ascend_ple_device)


# The NVIDIA model imports a Blackwell-only CuTe DSL GEMM hook eagerly. It is
# an optional decode optimization and the standard vLLM linear path is the
# correct Ascend fallback, so prevent that CUDA module from becoming a hard
# import dependency while retaining the current upstream model definition.
low_latency_compat = types.ModuleType(
    "vllm_ascend.models.qwen4_exp.nvidia.low_latency_gemm"
)


def _keep_standard_linear_methods(module: nn.Module, dtype: torch.dtype) -> None:
    del module, dtype


low_latency_compat.enable_qwen4_exp_low_latency_gemm = _keep_standard_linear_methods
sys.modules[low_latency_compat.__name__] = low_latency_compat


from .nvidia import model as upstream_model  # noqa: E402
from .qsa import AscendQwen4ExpQSAAttention  # noqa: E402

upstream_model.Qwen4ExpQSAAttention = AscendQwen4ExpQSAAttention


class AscendQwen4ExpModel(upstream_model.Qwen4ExpModel):
    """Qwen4Exp backbone with Ascend-safe graph shape annotations."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # With max_num_seqs=1 the PLE query-start buffer has a static length of
        # two. Marking that length dynamic makes TorchDynamo reject the static
        # specialization used by PLE during FULL_DECODE_ONLY graph capture.
        dynamic_arg_dims = getattr(self, "_dynamic_arg_dims", None)
        if dynamic_arg_dims is not None:
            self._dynamic_arg_dims = {
                name: dims for name, dims in dynamic_arg_dims.items() if name != "query_start_loc"
            }


# Qwen4ExpForCausalLM resolves this module global when it creates the backbone.
upstream_model.Qwen4ExpModel = AscendQwen4ExpModel


class AscendQwen4ExpForCausalLM(upstream_model.Qwen4ExpForCausalLM):
    """Qwen3.8-Flash-Next text model using Ascend platform operators."""

    @staticmethod
    def get_model_state_cls() -> type:
        from vllm_ascend.models.qwen4_exp_model_state import (
            AscendQwen4ExpModelState,
        )

        return AscendQwen4ExpModelState


class AscendQwen4ExpForConditionalGeneration(upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""

    get_model_state_cls = staticmethod(
        AscendQwen4ExpForCausalLM.get_model_state_cls
    )


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
