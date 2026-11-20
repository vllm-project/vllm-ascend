# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify arch35 raw padding IDs against token-zero lookup on real A5 hardware."""

import pytest
import torch

from vllm_ascend.device.device_config import check_ascend_device_type, is_950
from vllm_ascend.utils import enable_custom_op


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_hash_padding_uses_token_zero_row(id_dtype, dtype, execution_mode):
    check_ascend_device_type()
    if not is_950():
        pytest.skip("arch35 padding regression requires Ascend 950")
    enable_custom_op()
    rows, experts, top_k = 8, 384, 6
    generator = torch.Generator().manual_seed(21452)
    logits = torch.randn(rows, experts, generator=generator, dtype=dtype).npu()
    bias = torch.zeros(experts, dtype=dtype, device="npu")
    ids_cpu = torch.tensor([-1, 1, 2, -1, 4, 5, 6, -1], dtype=id_dtype)
    normalized_cpu = torch.where(ids_cpu == -1, 0, ids_cpu).long()
    input_ids = ids_cpu.npu()
    normalized_ids = normalized_cpu.npu()
    table_cpu = (torch.arange(8)[:, None] * 7 + torch.arange(top_k)[None, :] * 17).int()
    table = table_cpu.npu()

    def run(ids):
        return torch.ops._C_ascend.moe_gating_top_k_hash(
            x=logits,
            k=top_k,
            bias=bias,
            input_ids=ids,
            tid2eid=table,
            k_group=1,
            group_count=1,
            routed_scaling_factor=1.0,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
        )

    expected_weights, expected_ids, _ = run(normalized_ids)
    if execution_mode == "graph":
        run(input_ids)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            weights, expert_ids, _ = run(input_ids)
        graph.replay()
        torch.npu.synchronize()
    else:
        weights, expert_ids, _ = run(input_ids)

    torch.testing.assert_close(expert_ids.cpu(), table_cpu[normalized_cpu], rtol=0, atol=0)
    torch.testing.assert_close(expert_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
