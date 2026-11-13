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

from vllm_ascend.ops import qwen3_5_sp


def test_mm_rs_fake_pads_rows_to_tp_shard():
    output = qwen3_5_sp._qwen35_fused_bf16_mm_rs_fake(
        torch.empty((17, 32), dtype=torch.bfloat16),
        torch.empty((64, 32), dtype=torch.bfloat16),
        "comm",
        8,
    )

    assert output.shape == (3, 64)
    assert output.dtype == torch.bfloat16


def test_dispatch_fake_keeps_local_token_rows():
    output = qwen3_5_sp._qwen35_gdn_in_proj_ag_mm_fake(
        torch.empty((3, 32), dtype=torch.bfloat16),
        torch.empty((64, 32), dtype=torch.bfloat16),
        128,
        8,
    )

    assert output.shape == (3, 128)


def test_gdn_input_projection_fuses_all_gather_with_qkvz_matmul():
    op = object.__new__(qwen3_5_sp.Qwen35GdnInProjAllGatherMatmulOp)
    weight = torch.randn((12, 8), dtype=torch.bfloat16)
    gathered = torch.randn((6, 8), dtype=torch.bfloat16)
    qkvz_output = torch.randn((6, 12), dtype=torch.bfloat16)
    ba_output = torch.randn((6, 4), dtype=torch.bfloat16)
    ba_method = MagicMock()
    ba_method.apply.return_value = (ba_output, None)
    op.qkvz = SimpleNamespace(weight=weight)
    op.ba = SimpleNamespace(quant_method=ba_method)
    op.hcom = "comm"
    op.world_size = 2
    op._can_fuse = MagicMock(return_value=True)
    x_shard = torch.randn((3, 8), dtype=torch.bfloat16)

    with patch.object(
        qwen3_5_sp.torch_npu,
        "npu_all_gather_base_mm",
        return_value=(qkvz_output, gathered),
        create=True,
    ) as fused, patch.object(qwen3_5_sp, "sp_all_gather") as all_gather:
        actual_qkvz, actual_ba = op.run_input_projections(x_shard)

    fused.assert_called_once()
    assert fused.call_args.args[1].shape == (8, 12)
    all_gather.assert_not_called()
    ba_method.apply.assert_called_once_with(op.ba, gathered, None)
    assert actual_qkvz is qkvz_output
    assert actual_ba is ba_output


def test_gdn_input_projection_falls_back_for_decode():
    op = object.__new__(qwen3_5_sp.Qwen35GdnInProjAllGatherMatmulOp)
    gathered = torch.randn((6, 8), dtype=torch.bfloat16)
    qkvz_output = torch.randn((6, 12), dtype=torch.bfloat16)
    ba_output = torch.randn((6, 4), dtype=torch.bfloat16)
    qkvz_method = MagicMock()
    qkvz_method.apply.return_value = qkvz_output
    ba_method = MagicMock()
    ba_method.apply.return_value = ba_output
    op.qkvz = SimpleNamespace(quant_method=qkvz_method)
    op.ba = SimpleNamespace(quant_method=ba_method)
    op._can_fuse = MagicMock(return_value=False)
    x_shard = torch.randn((3, 8), dtype=torch.bfloat16)

    with patch.object(qwen3_5_sp, "sp_all_gather", return_value=gathered) as all_gather, patch.object(
        qwen3_5_sp.torch_npu,
        "npu_all_gather_base_mm",
        create=True,
    ) as fused:
        actual_qkvz, actual_ba = op.run_input_projections(x_shard)

    all_gather.assert_called_once_with(x_shard)
    fused.assert_not_called()
    assert actual_qkvz is qkvz_output
    assert actual_ba is ba_output


def test_prefill_only_rejects_decode_and_mixed_batches():
    with patch.object(qwen3_5_sp, "is_forward_context_available", return_value=True), patch.object(
        qwen3_5_sp,
        "get_forward_context",
        return_value=SimpleNamespace(attn_metadata=SimpleNamespace(num_prefills=1, num_decodes=1)),
    ):
        assert not qwen3_5_sp._prefill_only()

    with patch.object(qwen3_5_sp, "is_forward_context_available", return_value=True), patch.object(
        qwen3_5_sp,
        "get_forward_context",
        return_value=SimpleNamespace(attn_metadata=SimpleNamespace(num_prefills=1, num_decodes=0)),
    ):
        assert qwen3_5_sp._prefill_only()
