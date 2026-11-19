# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from types import MethodType

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.model_executor.models.lfm2_vl import (
    Lfm2VLDummyInputsBuilder,
    Lfm2VLForConditionalGeneration,
    Lfm2VLMultiModalProcessor,
    Lfm2VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata


@lru_cache(maxsize=1)
def _decode_kernel():
    # Load the upstream implementation in a private namespace: worker patches
    # replace its public entry point with a graph-unsafe fallback.
    from vllm.model_executor.layers.mamba.ops import causal_conv1d

    spec = spec_from_file_location(__name__ + "._causal_conv1d", causal_conv1d.__file__)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.causal_conv1d_update


def _short_conv_forward(
    self: ShortConv,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
):
    triton_causal_conv1d_update = _decode_kernel()
    from vllm_ascend.patch.worker.patch_ascend_ops import (
        _npu_causal_conv1d_fn,
    )

    metadata = get_forward_context().attn_metadata
    if metadata is not None:
        assert isinstance(metadata, dict)
        metadata = metadata.get(self.prefix)
    if metadata is not None:
        assert isinstance(metadata, ShortConvAttentionMetadata)

    BCx, _ = self.in_proj(hidden_states)

    B, C, x = BCx.chunk(3, dim=-1)

    conv_weights = self.conv.weight.view(self.conv.weight.size(0), self.conv.weight.size(2))

    if metadata is None:
        # V1 profile run
        Bx = (B * x).contiguous()
        hidden_states = C * Bx
        contextualized_states, _ = self.out_proj(hidden_states)
        return contextualized_states

    conv_state = self.kv_cache[0]
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)
    num_decodes = metadata.num_decode_tokens
    num_prefill_tokens = metadata.num_prefill_tokens
    num_actual_tokens = num_decodes + num_prefill_tokens
    # Upstream batching places decode tokens before prefill tokens.
    B_d, B_p = B[:num_decodes], B[num_decodes:num_actual_tokens]
    C_d, C_p = C[:num_decodes], C[num_decodes:num_actual_tokens]
    x_d, x_p = x[:num_decodes], x[num_decodes:num_actual_tokens]
    conv_output_list = []

    if metadata.num_prefills > 0:
        Bx_p = (B_p * x_p).transpose(0, 1)
        prefill_state = conv_state.to(conv_weights.dtype)
        Bx = _npu_causal_conv1d_fn(
            Bx_p,
            conv_weights,
            self.conv.bias,
            activation=None,
            conv_states=prefill_state,
            has_initial_state=metadata.has_initial_states_p,
            cache_indices=metadata.state_indices_tensor_p,
            metadata=metadata,
            query_start_loc=metadata.query_start_loc_p,
        ).transpose(0, 1)[:num_prefill_tokens]

        if prefill_state is not conv_state:
            conv_state.copy_(prefill_state)
        y = C_p * Bx
        conv_output_list.append(y)

    if num_decodes > 0:
        Bx_d = (B_d * x_d).contiguous()
        spec_kwargs = {}
        if metadata.num_accepted_tokens is not None:
            assert metadata.state_indices_tensor_d is not None
            spec_kwargs = dict(
                num_accepted_tokens=metadata.num_accepted_tokens,
                query_start_loc=metadata.query_start_loc_d,
                max_query_len=metadata.state_indices_tensor_d.size(-1),
            )
        Bx = triton_causal_conv1d_update(
            Bx_d,
            conv_state,
            conv_weights,
            self.conv.bias,
            activation=None,
            conv_state_indices=metadata.state_indices_tensor_d,
            **spec_kwargs,
        )
        y = C_d * Bx
        conv_output_list.insert(0, y)

    # Merge prefill and decode outputs before passing to gated MLP
    hidden_states = torch.vstack(conv_output_list)

    # Final linear projection
    output[:num_actual_tokens], _ = self.out_proj(hidden_states)


@MULTIMODAL_REGISTRY.register_processor(
    Lfm2VLMultiModalProcessor,
    info=Lfm2VLProcessingInfo,
    dummy_inputs=Lfm2VLDummyInputsBuilder,
)
class AscendLfm2VLForConditionalGeneration(Lfm2VLForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.language_model.modules():
            if isinstance(layer, ShortConv):
                layer.forward_cuda = MethodType(_short_conv_forward, layer)
