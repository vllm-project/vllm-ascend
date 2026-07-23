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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    qwen_gdn_attention_core,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNPrefillMetadata,
)
from vllm_ascend.utils import vllm_version_is


class _Linear(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states[:, : self.output_size], None


class _Norm(nn.Module):
    def forward(self, hidden_states: torch.Tensor, z: torch.Tensor):
        del z
        return hidden_states


class _OutputProjection(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _GDNForwardWrapper(nn.Module):
    forward = AscendGatedDeltaNetAttention.forward
    _forward_core = AscendGatedDeltaNetAttention._forward_core
    _split_ba_for_tp = AscendGatedDeltaNetAttention._split_ba_for_tp

    def __init__(self):
        super().__init__()
        self.in_proj_qkv = _Linear(2)
        self.in_proj_ba = _Linear(2)
        self.in_proj_z = _Linear(2)
        self.norm = _Norm()
        self.out_proj = _OutputProjection()
        self.conv1d = nn.Conv1d(1, 2, kernel_size=2)
        self.num_v_heads = 1
        self.tp_size = 1
        self.head_v_dim = 2
        self.activation = None
        self.register_buffer("A_log", torch.zeros(1))
        self.register_buffer("dt_bias", torch.zeros(1))
        self.register_buffer("conv_state", torch.zeros(1, 1, 2))
        self.register_buffer("ssm_state", torch.zeros(1, 1, 2, 2))
        self.prefix = "layers.0.linear_attn"

    @property
    def kv_cache(self):
        return self.conv_state, self.ssm_state

    def split_ba(self, ba: torch.Tensor):
        # Avoid torch.chunk: torch-npu's aten.split fallback conflicts with
        # PyTorch's decomposition check when CI is set.
        midpoint = ba.shape[-1] // 2
        return ba[..., :midpoint], ba[..., midpoint:]

    def rearrange_mixed_qkv(self, mixed_qkv: torch.Tensor | None):
        if mixed_qkv is None:
            return None, None, None
        projected = mixed_qkv.reshape(1, mixed_qkv.shape[0], 1, 2)
        return projected, projected, projected


def _run_gdn_forward(
    model: nn.Module,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if vllm_version_is("0.25.1"):
        result = model(hidden_states, output)
        return output if result is None else result
    return model(hidden_states)


def _make_prefill_metadata(device: torch.device | str = "cpu") -> GDNAttentionMetadata:
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_state_indices_tensor=torch.tensor([0], dtype=torch.int32, device=device),
    )
    # These fields are constructor arguments only in newer vLLM releases.
    # Assigning them after construction also supports the older metadata class
    # while exercising the same production GDN prefill path.
    metadata.prefill_query_start_loc = torch.tensor([0, 2], dtype=torch.int32, device=device)
    metadata.prefill_state_indices = torch.tensor([0], dtype=torch.int64, device=device)
    metadata.prefill_has_initial_state = torch.tensor([True], device=device)
    metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
        causal_conv1d=GDNCausalConv1dMetadata(
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device=device),
            cache_indices=torch.tensor([0], dtype=torch.int32, device=device),
            initial_state_mode=torch.tensor([1], dtype=torch.int32, device=device),
        ),
        chunk=Mock(),
    )
    return metadata


def test_connector_observes_updated_gdn_state_for_each_compiled_call():
    # vLLM registers this production dispatcher for NPU only. Keep a CPU
    # registration alive for this test so Inductor sees the same custom op.
    cpu_impl = torch.library.Library("vllm", "IMPL", "CPU")
    cpu_impl.impl("qwen_gdn_attention_core", qwen_gdn_attention_core)
    model = _GDNForwardWrapper()
    hidden_states = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    output = torch.empty_like(hidden_states)
    metadata = _make_prefill_metadata()
    forward_context = ForwardContext(
        no_compile_layers={model.prefix: model},
        attn_metadata={model.prefix: metadata},
        slot_mapping={},
    )
    connector = Mock()
    observed_states = []

    def record_ready_state(layer_name, kv_cache_layer, attn_metadata):
        assert layer_name == ""
        assert kv_cache_layer == []
        assert attn_metadata is forward_context.attn_metadata
        observed_states.append(tuple(state.clone() for state in model.kv_cache))

    connector.save_kv_layer.side_effect = record_ready_state

    def causal_conv1d(output_tensor, mixed_qkv, conv_weights, **kwargs):
        del conv_weights, kwargs
        output_tensor.copy_(mixed_qkv)
        model.kv_cache[0].add_(1)

    def chunk_attention(**kwargs):
        initial_state = kwargs["initial_state"]
        value = kwargs["v"]
        return value + 1, initial_state + 1

    gating = (
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, 1),
    )

    with (
        override_forward_context(forward_context),
        patch.object(torch.accelerator, "is_available", return_value=False),
        patch("vllm_ascend.ops.gdn.get_pcp_group", return_value=SimpleNamespace(world_size=1)),
        patch("vllm_ascend.ops.gdn.DeviceOperator.fused_gdn_gating", return_value=gating),
        patch("vllm_ascend.ops.gdn.clear_ssm_states"),
        patch("vllm_ascend.ops.gdn.chunk_gated_delta_rule", side_effect=chunk_attention),
        patch.object(
            torch.ops._C_ascend,
            "npu_causal_conv1d_custom",
            side_effect=causal_conv1d,
            create=True,
        ),
        patch("vllm_ascend.attention.utils.has_kv_transfer_group", return_value=True),
        patch("vllm_ascend.attention.utils.is_v1_kv_transfer_group", return_value=True),
        patch("vllm_ascend.attention.utils.get_kv_transfer_group", return_value=connector),
    ):
        eager_output = _run_gdn_forward(model, hidden_states, output)
        torch.testing.assert_close(eager_output, hidden_states + 1)
        compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
        for _ in range(2):
            compiled_output = _run_gdn_forward(compiled_model, hidden_states, output)
            torch.testing.assert_close(compiled_output, hidden_states + 1)

    assert connector.save_kv_layer.call_count == 3
    for execution, (conv_state, ssm_state) in enumerate(observed_states, start=1):
        torch.testing.assert_close(conv_state, torch.full_like(conv_state, execution))
        torch.testing.assert_close(ssm_state, torch.full_like(ssm_state, execution))
