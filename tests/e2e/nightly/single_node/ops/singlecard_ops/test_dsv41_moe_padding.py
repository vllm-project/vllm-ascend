# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise one AllGather EP rank after routing, without HCCL communication."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.utils import enable_custom_op
from vllm_ascend.worker.v2 import input_batch as input_batch_module
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _w4a8_weight(experts, k, n):
    weight = torch.randint(-8, 8, (experts, k, n), dtype=torch.int8, device="cpu")
    scale = torch.rand(experts, n, device="cpu") * 0.005 + 0.005
    bias = weight.sum(dim=1, dtype=torch.int32).float() * scale * 8
    packed = ((weight[..., 0::2] & 15) | (weight[..., 1::2] << 4)).contiguous().npu()
    packed = torch_npu.npu_format_cast(packed, 29).view(torch.int32).contiguous()
    scale_bits = (scale.view(torch.int32).to(torch.int64) & 0xFFFFFFFF).npu()
    return packed, scale_bits, bias.npu()


@pytest.fixture
def allow_internal_format():
    torch.npu.set_device(0)
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT").decode()
    torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": "enable"})
    try:
        yield
    finally:
        torch_npu.npu.set_option({"ALLOW_INTERNAL_FORMAT": previous})


@pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3),
    reason="This W4A8 NZ GMM path is configured for Ascend A2/A3",
)
@pytest.mark.parametrize("rank", [0, 7])
@pytest.mark.parametrize("graph_tokens", [13, 16])
@torch.inference_mode()
def test_moe_padding_zero_counts_graph_replay(allow_internal_format, monkeypatch, rank, graph_tokens):
    enable_custom_op()
    torch.manual_seed(21092 + rank)
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
    runner = NPUModelRunner.__new__(NPUModelRunner)
    parallel = SimpleNamespace(use_sequence_parallel_moe=True, tensor_parallel_size=8)
    config = SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr(input_batch_module, "get_current_vllm_config_or_none", lambda: config)
    runner.input_buffers = AscendInputBuffers(1, graph_tokens, torch.device("npu"))

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

    class Backbone:
        use_sequence_parallel = True
        skip_padding_tokens_in_moe = True

        def __call__(self, *args, **kwargs):
            return run_moe(x, ids, probs, context.is_padding)

    context = SimpleNamespace(is_padding=runner.prepare_padding_mask(0, graph_tokens))
    model = SimpleNamespace(model=Backbone())
    monkeypatch.setattr(model_module, "get_forward_context", lambda: context)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 8)

    # Capture with no local work, as in an all-padding graph warmup. The graph
    # must still execute GEMMs after real requests update the device mask.
    for _ in range(3):
        context.is_padding = runner.input_buffers.is_padding[:graph_tokens]
        model_module.AscendDeepseekV41LLMForCausalLM.forward(model, None, None)
    torch.npu.synchronize()
    graph = torch_npu.npu.NPUGraph()
    context.is_padding = runner.input_buffers.is_padding[:graph_tokens]
    with torch_npu.npu.graph(graph):
        output, counts = model_module.AscendDeepseekV41LLMForCausalLM.forward(model, None, None)
    assert context.is_padding.data_ptr() == runner.input_buffers.is_padding.data_ptr()

    for step, num_valid in enumerate((0, 1, 9, graph_tokens, 0, 5)):
        valid = torch.arange(capacity, device="cpu") < num_valid
        updated_ids = cpu_ids.roll(step, dims=0)
        x.copy_(torch.randn(capacity, hidden, dtype=torch.bfloat16, device="cpu"))
        ids.copy_(updated_ids)
        runner.prepare_padding_mask(num_valid, graph_tokens)
        if graph_tokens < capacity:
            # A replay must re-mark the SP tail even when it contains stale values.
            runner.input_buffers.is_padding[graph_tokens:].fill_(False)
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
