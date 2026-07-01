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

from vllm_ascend import utils
from vllm_ascend._310p.ops.mm_encoder_attention import AscendMMEncoderAttention310


def test_register_customop_overrides_mm_encoder_attention_for_310p():
    original_registered = utils._ASCEND_CUSTOMOP_IS_REIGISTERED
    try:
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = False
        with (
            mock.patch("vllm.model_executor.custom_op.CustomOp.register_oot"),
            mock.patch("vllm_ascend.utils.is_310p", return_value=True),
        ):
            utils.register_ascend_customop()

        assert utils.REGISTERED_ASCEND_OPS["MMEncoderAttention"] is AscendMMEncoderAttention310
    finally:
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = original_registered


def test_mm_encoder_attention_310_forward_oot_with_padding():
    layer = AscendMMEncoderAttention310.__new__(AscendMMEncoderAttention310)
    layer.num_heads = 4
    layer.num_kv_heads = 2
    layer.head_size = 80
    layer.enable_pad = True
    layer.scale_value = layer.head_size**-0.5

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

