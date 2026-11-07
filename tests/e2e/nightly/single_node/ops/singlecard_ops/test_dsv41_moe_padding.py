# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise one AllGather EP rank after routing, without HCCL communication."""

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


def _w4a8_weight(experts, k, n):
    weight = torch.randint(-8, 8, (experts, k, n), dtype=torch.int8, device="cpu")
    scale = torch.rand(experts, n, device="cpu") * 0.005 + 0.005
    bias = weight.sum(dim=1, dtype=torch.int32).float() * scale * 8
    packed = ((weight[..., 0::2] & 15) | (weight[..., 1::2] << 4)).contiguous().npu()
    packed = torch_npu.npu_format_cast(packed, 29).view(torch.int32).contiguous()
    scale_bits = (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).npu()
    return packed, scale_bits, bias.npu()


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="This W4A8 NZ GMM path is configured for Ascend A2/A3",
)
@pytest.mark.parametrize("rank", [0, 7])
@torch.inference_mode()
def test_moe_padding_zero_counts_graph_replay(monkeypatch, rank):
    torch.npu.set_device(0)
    monkeypatch.setattr(torch_npu.npu.config, "allow_internal_format", True)
    enable_custom_op()
    torch.manual_seed(19679 + rank)
    experts, local_experts, topk = 384, 48, 6
    capacity, hidden, intermediate = 16, 128, 128
    start, end = rank * local_experts, (rank + 1) * local_experts
    foreign = local_experts if rank == 0 else 0
    expert_map = torch.full((experts,), -1, dtype=torch.int32, device="cpu")
    expert_map[start:end] = torch.arange(local_experts, dtype=torch.int32, device="cpu")
    expert_map = expert_map.npu()
    w1, s1, b1 = _w4a8_weight(local_experts, hidden, 2 * intermediate)
    w2, s2, b2 = _w4a8_weight(local_experts, intermediate, hidden)
    s2 = s2.unsqueeze(1)

    # Every row has both local and foreign routes. Padding activations are also
    # nonzero, so routing exclusion cannot accidentally pass by multiplying zeros.
    rows = torch.arange(capacity, device="cpu")[:, None]
    choices = torch.arange(topk // 2, device="cpu")[None, :]
    cpu_ids = torch.cat((start + (rows + choices) % local_experts, foreign + (rows + choices) % local_experts), 1)
    cpu_ids = cpu_ids.to(torch.int32)
    x = torch.randn(capacity, hidden, dtype=torch.bfloat16, device="cpu").npu()
    ids = cpu_ids.npu()
    probs = torch.rand(capacity, topk, device="cpu")
    probs = (probs / probs.sum(dim=1, keepdim=True)).npu()
    is_padding = torch.ones(capacity, dtype=torch.bool, device="npu")

    def run_moe(hidden_states, topk_ids, topk_weights, padding_mask):
        routed_ids = topk_ids.masked_fill(padding_mask[:, None], foreign)
        weights = topk_weights * (expert_map[routed_ids.long()] != -1)
        expanded, row_idx, counts, scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            routed_ids,
            active_num=hidden_states.shape[0] * topk,
            expert_num=experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[start, end],
            quant_mode=1,
        )
        counts = counts.to(torch.int64)
        activation, activation_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
            x=expanded,
            weight=[w1],
            weight_scale=[s1],
            x_scale=scale,
            group_list=counts,
            weight_assist_matrix=[b1],
            dequant_mode=0,
            group_list_type=1,
            swiglu_limit=0.0,
        )
        output = torch_npu.npu_grouped_matmul(
            x=[activation],
            weight=[w2],
            scale=[s2],
            bias=[b2],
            per_token_scale=[activation_scale],
            group_list=counts,
            group_list_type=1,
            split_item=2,
            group_type=0,
            output_dtype=torch.bfloat16,
        )[0]
        # Match the A2/A3 DeviceOperator, including abs(sorted_indices).
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=output,
            sorted_indices=row_idx.abs(),
            probs=weights.to(output.dtype),
        )
        return output, counts

    # Capture with no local work, as in an all-padding graph warmup. The graph
    # must still execute GEMMs after real requests update the device mask.
    for _ in range(3):
        run_moe(x, ids, probs, is_padding)
    torch.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        output, counts = run_moe(x, ids, probs, is_padding)

    for step, num_valid in enumerate((0, 1, 9, capacity, 0, 5)):
        valid = (torch.arange(capacity, device="cpu") < num_valid).roll(step)
        updated_ids = cpu_ids.roll(step, dims=0)
        x.copy_(torch.randn(capacity, hidden, dtype=torch.bfloat16, device="cpu"))
        ids.copy_(updated_ids)
        is_padding.copy_(~valid)
        graph.replay()
        torch.npu.synchronize()

        expected_counts = torch.bincount(updated_ids[valid].flatten().long(), minlength=experts)[start:end]
        torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
        actual = output.cpu()
        torch.testing.assert_close(actual[~valid], torch.zeros_like(actual[~valid]), rtol=0, atol=0)
        if num_valid:
            # Removing padding before InitRouting is the reference. It must
            # preserve each valid token's output and the per-expert row counts.
            indices = valid.nonzero().flatten().npu()
            reference, reference_counts = run_moe(
                x.index_select(0, indices),
                ids.index_select(0, indices),
                probs.index_select(0, indices),
                torch.zeros(num_valid, dtype=torch.bool, device="npu"),
            )
            torch.testing.assert_close(actual[valid], reference.cpu(), rtol=0.02, atol=0.02)
            torch.testing.assert_close(counts, reference_counts, rtol=0, atol=0)
