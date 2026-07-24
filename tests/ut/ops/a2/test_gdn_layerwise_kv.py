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
from torch._inductor import config as inductor_config
from vllm.forward_context import ForwardContext, override_forward_context

from tests.ut.ops.test_gdn_layerwise_kv import (
    _GDNForwardWrapper,
    _make_prefill_metadata,
    _run_gdn_forward,
)


def test_npu_connector_observes_updated_gdn_state_after_compile():
    device = torch.device("npu")
    model = _GDNForwardWrapper().to(device)
    hidden_states = torch.arange(4, dtype=torch.float32, device=device).reshape(2, 2)
    output = torch.empty_like(hidden_states)
    metadata = _make_prefill_metadata(device)
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
        model.conv_state.add_(1)

    def chunk_attention(**kwargs):
        initial_state = kwargs["initial_state"]
        value = kwargs["v"]
        return value + 1, initial_state + 1

    gating = (
        torch.zeros(1, 2, 1, device=device),
        torch.zeros(1, 2, 1, device=device),
    )

    with (
        inductor_config.patch(compile_threads=1),
        override_forward_context(forward_context),
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
        torch.npu.synchronize()

    assert connector.save_kv_layer.call_count == 3
    for execution, (conv_state, ssm_state) in enumerate(observed_states, start=1):
        torch.testing.assert_close(conv_state, torch.full_like(conv_state, execution))
        torch.testing.assert_close(ssm_state, torch.full_like(ssm_state, execution))
