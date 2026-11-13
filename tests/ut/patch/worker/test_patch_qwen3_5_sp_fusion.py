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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.patch.worker import patch_qwen3_5


def _decoder_layer(*, fused_output: bool, gdn_op=None):
    input_norm = MagicMock(
        side_effect=lambda hidden, residual=None: hidden if residual is None else (hidden, residual)
    )
    post_norm = MagicMock(side_effect=lambda hidden, residual: (hidden, residual))
    mlp = MagicMock(side_effect=lambda hidden, **kwargs: hidden)
    return SimpleNamespace(
        use_attn_reduce_scatter_for_moe=True,
        layer_type="linear_attention",
        input_layernorm=input_norm,
        linear_attn=MagicMock(side_effect=lambda hidden_states: hidden_states),
        self_attn=MagicMock(),
        layer_scale=False,
        fuse_o_proj_mm_reduce_scatter=fused_output,
        gdn_in_proj_ag_mm_op=gdn_op,
        post_attention_layernorm=post_norm,
        mlp=mlp,
    )


def test_decoder_uses_gdn_ag_mm_without_standalone_collectives():
    hidden = torch.randn((3, 8))
    fused_output = torch.randn((3, 8))
    gdn_op = MagicMock(return_value=fused_output)
    layer = _decoder_layer(fused_output=True, gdn_op=gdn_op)

    with patch.object(patch_qwen3_5, "sp_all_gather") as all_gather, patch.object(
        patch_qwen3_5,
        "sp_reduce_scatter",
    ) as reduce_scatter:
        output, residual = patch_qwen3_5.AscendQwen3_5DecoderLayer.forward(
            layer,
            hidden,
            None,
            positions=torch.arange(6),
        )

    gdn_op.assert_called_once_with(hidden)
    all_gather.assert_not_called()
    reduce_scatter.assert_not_called()
    layer.linear_attn.assert_not_called()
    layer.mlp.assert_called_once_with(fused_output, already_sequence_parallel=True)
    assert output is fused_output
    assert residual is hidden


def test_decoder_keeps_baseline_collectives_when_fusion_is_disabled():
    hidden = torch.randn((3, 8))
    gathered = torch.randn((6, 8))
    sharded = torch.randn((3, 8))
    layer = _decoder_layer(fused_output=False)

    with patch.object(patch_qwen3_5, "sp_all_gather", return_value=gathered) as all_gather, patch.object(
        patch_qwen3_5,
        "sp_reduce_scatter",
        return_value=sharded,
    ) as reduce_scatter:
        output, _ = patch_qwen3_5.AscendQwen3_5DecoderLayer.forward(
            layer,
            hidden,
            None,
            positions=torch.arange(6),
        )

    all_gather.assert_called_once_with(hidden)
    layer.linear_attn.assert_called_once_with(hidden_states=gathered)
    reduce_scatter.assert_called_once_with(gathered)
    assert output is sharded


def test_layer_init_attaches_both_linear_attention_fusions():
    projection = SimpleNamespace(custom_op=None)
    linear_attn = SimpleNamespace(out_proj=projection)

    def original_init(layer, vllm_config, layer_type, prefix=""):
        del vllm_config, layer_type, prefix
        layer.use_attn_reduce_scatter_for_moe = True
        layer.linear_attn = linear_attn
        layer.layer_idx = 4

    mm_rs_op = object()
    ag_mm_op = object()
    config = SimpleNamespace(lora_config=None)
    profile = SimpleNamespace(supports=lambda capability: True)

    with patch.object(patch_qwen3_5, "_QWEN3_5_LAYER_ORIGINAL_INIT", side_effect=original_init), patch.object(
        patch_qwen3_5,
        "get_current_hardware_profile",
        return_value=profile,
    ), patch.object(
        patch_qwen3_5,
        "get_ascend_config",
        return_value=SimpleNamespace(weight_nz_mode=0),
    ), patch.object(
        patch_qwen3_5.Qwen35OProjMMReduceScatterOp,
        "unsupported_reason",
        return_value=None,
    ), patch.object(
        patch_qwen3_5,
        "Qwen35OProjMMReduceScatterOp",
        return_value=mm_rs_op,
    ), patch.object(
        patch_qwen3_5.Qwen35GdnInProjAllGatherMatmulOp,
        "unsupported_reason",
        return_value=None,
    ), patch.object(
        patch_qwen3_5,
        "Qwen35GdnInProjAllGatherMatmulOp",
        return_value=ag_mm_op,
    ):
        layer = SimpleNamespace()
        patch_qwen3_5._qwen3_5_decoder_layer_init_with_sp_fusion(
            layer,
            config,
            "linear_attention",
            prefix="model.layers.4",
        )

    assert projection.custom_op is mm_rs_op
    assert layer.fuse_o_proj_mm_reduce_scatter
    assert layer.gdn_in_proj_ag_mm_op is ag_mm_op
