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
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.models.qwen4_exp.amd import (
    hyperconnection as upstream_hc,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5VLProcessor,
    _AscendProcessorMixin,
    _AscendVLPreprocessMixin,
)
from vllm_ascend.ops.triton.qwen4_exp.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)

from .ple import AscendPCPPLELayer, patch_upstream_ple_short_conv

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
        parallel = vllm_config.parallel_config
        use_pcp = parallel.prefill_context_parallel_size > 1
        if use_pcp:
            if parallel.decode_context_parallel_size != 1:
                raise NotImplementedError("Qwen4Exp PCP requires DCP=1")

            from vllm_ascend.ops.head_parallel import get_head_parallel_size

            text_config = vllm_config.model_config.hf_text_config
            head_shards = get_head_parallel_size(parallel)
            if text_config.linear_num_key_heads % head_shards or text_config.linear_num_value_heads % head_shards:
                raise ValueError("Qwen4Exp GDN key and value heads must be divisible by TP * PCP")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if use_pcp:
            from vllm_ascend.ops.gdn import shard_gdn_heads_over_pcp

            static_context = vllm_config.compilation_config.static_forward_context
            for layer in self.layers[self.start_layer : self.end_layer]:
                if hasattr(layer, "linear_attn"):
                    shard_gdn_heads_over_pcp(layer.linear_attn, parallel)
                if layer.ple is not None:
                    ple_prefix = layer.ple.prefix
                    ple_id = layer.ple.ple_dense_layer_id
                    del static_context[ple_prefix]
                    del layer.ple
                    layer.ple = AscendPCPPLELayer(
                        self.config,
                        vllm_config,
                        layer_idx=layer.layer_idx,
                        ple_dense_layer_id=ple_id,
                        prefix=ple_prefix,
                        use_sequence_parallel=layer.use_hc_sequence_parallel,
                    )
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

    @classmethod
    def get_gdn_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        upstream_shape = super().get_gdn_mamba_state_shape_from_config
        if vllm_config.parallel_config.prefill_context_parallel_size == 1:
            return upstream_shape(vllm_config)
        from vllm_ascend.ops.head_parallel import get_head_sharded_state_shape

        return get_head_sharded_state_shape(vllm_config, upstream_shape)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        # vLLM uses these specs to align attention pages. PLE has a private
        # dense bank; only GDN's complete conv+SSM state shares QSA pages.
        return tuple(spec for spec in super().get_mamba_specs_from_config(vllm_config) if len(spec.shapes) == 2)

    @staticmethod
    def get_model_state_cls():
        from .model_state import AscendQwen4ExpModelState

        return AscendQwen4ExpModelState


class AscendQwen4ExpProcessingInfo(_AscendProcessorMixin, upstream_model.Qwen4ExpProcessingInfo):
    """Preserve Qwen4Exp config handling while emitting raw image payloads."""

    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen4ExpForConditionalGeneration(_AscendVLPreprocessMixin, upstream_model.Qwen4ExpForConditionalGeneration):
    """Qwen3.8-Flash-Next multimodal model using Ascend operators."""

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        return AscendQwen4ExpForCausalLM.get_mamba_specs_from_config(vllm_config)

    @staticmethod
    def get_model_state_cls():
        from .model_state import AscendQwen4ExpModelState

        return AscendQwen4ExpModelState


__all__ = [
    "AscendQwen4ExpForCausalLM",
    "AscendQwen4ExpForConditionalGeneration",
    "AscendQwen4ExpModel",
]
