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

from vllm.config import VllmConfig
from vllm.models.qwen4_exp.amd import (
    hyperconnection as upstream_hc,
)

from .ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .ple import patch_upstream_ple_short_conv

# Keep the upstream model and weight loader authoritative. HyperConnection's
# CUDA glue functions are module globals, so replace them before importing the
# model classes that instantiate GatedResidual.
upstream_hc.grouped_gemma_rmsnorm = grouped_gemma_rmsnorm
upstream_hc.hc_combine = hc_combine
upstream_hc.hc_combine_norm = hc_combine_norm
upstream_hc.hc_gate_mix = hc_gate_mix
upstream_hc.hc_silu = hc_silu

# PLE's depthwise F.conv1d otherwise lowers to ACLop Conv2D when torch-npu
# internal formats are enabled. Scope the graph-safe option override to the
# upstream PLE custom op instead of changing it globally for the worker.
patch_upstream_ple_short_conv()

from vllm.models.qwen4_exp.amd import model as upstream_model  # noqa: E402

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


class AscendQwen4ExpForConditionalGeneration(upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
