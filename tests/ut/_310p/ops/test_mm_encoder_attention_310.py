#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest import mock

import torch

from vllm_ascend._310p.ops.mm_encoder_attention import AscendMMEncoderAttention310
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def test_register_customop_overrides_mm_encoder_attention_for_310p():
    import vllm.model_executor.custom_op as custom_op_module

    from vllm_ascend.ops import registry as ops_registry

    with (
        mock.patch("vllm.model_executor.custom_op.CustomOp.register_oot") as mock_register_oot,
        mock.patch(
            "vllm_ascend.ops.registry.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType._310P),
        ),
        mock.patch("vllm_ascend.ops.registry._registered_all_custom_ops", False),
        mock.patch.dict(custom_op_module.op_registry_oot, clear=True),
    ):
        mock_register_oot.side_effect = lambda _decorated_op_cls=None, name=None: (
            custom_op_module.op_registry_oot.__setitem__(name, _decorated_op_cls)
        )
        ops_registry.register_all_custom_ops()

        mock_register_oot.assert_any_call(_decorated_op_cls=AscendMMEncoderAttention310, name="MMEncoderAttention")
        assert custom_op_module.op_registry_oot["MMEncoderAttention"] is AscendMMEncoderAttention310


def test_mm_encoder_attention_310_forward_oot_with_padding():
    layer = AscendMMEncoderAttention310.__new__(AscendMMEncoderAttention310)
    layer.num_heads = 4
    layer.num_kv_heads = 2
    layer.head_size = 80
    layer.enable_pad = True
    layer.scale_value = layer.head_size**-0.5
    layer.support_approximate_calculation = False

    bsz, q_len, kv_len = 2, 3, 3
    query = torch.randn(bsz, q_len, layer.num_heads, layer.head_size)
    key = torch.randn(bsz, kv_len, layer.num_kv_heads, layer.head_size)
    value = torch.randn(bsz, kv_len, layer.num_kv_heads, layer.head_size)

    capture = {}

    def fake_flash_attention_unpad(*, query, key, value, seq_len, scale_value, num_heads, num_kv_heads, out):
        capture["query_shape"] = query.shape
        capture["key_shape"] = key.shape
        capture["value_shape"] = value.shape
        capture["seq_len"] = seq_len
        capture["scale_value"] = scale_value
        capture["num_heads"] = num_heads
        capture["num_kv_heads"] = num_kv_heads
        out.copy_(query + 1.0)

    with mock.patch(
        "vllm_ascend._310p.ops.mm_encoder_attention.torch_npu._npu_flash_attention_unpad",
        side_effect=fake_flash_attention_unpad,
        create=True,
    ):
        out = layer.forward_oot(query, key, value)

    assert capture["query_shape"] == (bsz * q_len, layer.num_heads, 128)
    assert capture["key_shape"] == (bsz * kv_len, layer.num_heads, 128)
    assert capture["value_shape"] == (bsz * kv_len, layer.num_heads, 128)
    assert capture["seq_len"].device.type == "cpu"
    torch.testing.assert_close(capture["seq_len"], torch.tensor([q_len, q_len], dtype=torch.int32))
    assert capture["num_heads"] == layer.num_heads
    assert capture["num_kv_heads"] == layer.num_kv_heads

    assert out.shape == query.shape
    torch.testing.assert_close(out, query + 1.0)
