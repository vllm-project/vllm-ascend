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

import gc

import numpy as np
import pytest
import torch
import torch_npu

from vllm_ascend.device.device_op import BaseDeviceAdaptor
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True

NUM_TOKENS = 16
HIDDEN_SIZE = 512
NUM_EXPERTS = 4
INTERMEDIATE_SIZE = 1024
SPLIT_SYNC_NUM_TOKENS = 1024
SPLIT_SYNC_CHUNK_SIZE = 128
# Packed INT8 last dim stores two INT4 values, so logical N is 2x this.
W4_PACKED_N = INTERMEDIATE_SIZE
W4_LOGICAL_N = W4_PACKED_N * 2


@pytest.fixture(autouse=True)
def _release_npu_memory():
    yield
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()


def _group_list(counts: list[int], group_list_type: int) -> torch.Tensor:
    groups = torch.tensor(counts, dtype=torch.int64, device="npu")
    return groups.cumsum(dim=0) if group_list_type == 0 else groups


def _format_nz_weights(weights: torch.Tensor, single_tensor: bool) -> list[torch.Tensor]:
    if single_tensor:
        return [torch_npu.npu_format_cast(weights.npu(), 29)]
    return [torch_npu.npu_format_cast(weight.npu(), 29) for weight in weights]


def _call_v2(
    *,
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    weight_scales: list[torch.Tensor],
    token_scales: torch.Tensor,
    groups: torch.Tensor,
    group_list_type: int,
    weight_assist_matrix: list[torch.Tensor] | None = None,
    dequant_mode: int = 0,
    swiglu_limit: float = 0.0,
):
    return torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=groups,
        weight_assist_matrix=weight_assist_matrix,
        dequant_mode=dequant_mode,
        group_list_type=group_list_type,
        swiglu_limit=swiglu_limit,
    )


def _call_adaptor_v2(
    *,
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    weight_scales: list[torch.Tensor],
    token_scales: torch.Tensor,
    groups: torch.Tensor,
    group_list_type: int,
):
    return BaseDeviceAdaptor.npu_grouped_matmul_swiglu_quant(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=groups,
        group_list_type=group_list_type,
        act_quant_type=torch.int8,
    )


def _call_cann_v2(
    *,
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    weight_scales: list[torch.Tensor],
    token_scales: torch.Tensor,
    groups: torch.Tensor,
    group_list_type: int,
):
    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=groups,
        group_list_type=group_list_type,
        dequant_mode=0,
    )


def _make_a8w8_case(single_tensor: bool):
    weights_cpu = torch.randint(
        -16,
        16,
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.int8,
    )
    weights = _format_nz_weights(weights_cpu, single_tensor)
    scales_cpu = torch.rand(NUM_EXPERTS, INTERMEDIATE_SIZE, dtype=torch.float32) * 0.02 + 0.001
    weight_scales = [scales_cpu.npu()] if single_tensor else [scale.npu() for scale in scales_cpu]
    return weights, weight_scales


def _a8w8_golden(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    weight_scales: torch.Tensor,
    token_scales: torch.Tensor,
    expert_counts: list[int],
):
    output = torch.empty(
        hidden_states.shape[0],
        weights.shape[-1] // 2,
        dtype=torch.int8,
    )
    output_scale = torch.empty(hidden_states.shape[0], dtype=torch.float32)
    start = 0
    for expert, count in enumerate(expert_counts):
        end = start + count
        if count:
            projected = torch.matmul(
                hidden_states[start:end].to(torch.int32),
                weights[expert].to(torch.int32),
            ).to(torch.float32)
            projected.mul_(token_scales[start:end, None])
            projected.mul_(weight_scales[expert])
            swish_input, gate = projected.chunk(2, dim=-1)
            activated = swish_input * torch.sigmoid(swish_input) * gate
            scale = activated.abs().amax(dim=-1) / 127
            output[start:end] = torch.round(activated / scale[:, None]).to(torch.int8)
            output_scale[start:end] = scale
        start = end
    return output, output_scale


def _pack_va_w4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Production W4A8 packing: two INT4 values in one INT8, NZ as INT8, view INT32."""
    weight_nz = torch_npu.npu_format_cast(weight.npu().contiguous(), 29)
    return weight_nz.view(torch.int32).contiguous()


def _pack_a8w4_scale(scale: torch.Tensor) -> torch.Tensor:
    scale_bits = scale.cpu().numpy().view(np.uint32).astype(np.int64)
    return torch.from_numpy(scale_bits).npu()


def _make_a8w4_case(single_tensor: bool, per_group: bool):
    weights_cpu = torch.randint(
        -8,
        8,
        (NUM_EXPERTS, HIDDEN_SIZE, W4_PACKED_N),
        dtype=torch.int8,
    )
    if single_tensor:
        weights = [_pack_va_w4_weight(weights_cpu)]
    else:
        weights = [_pack_va_w4_weight(weights_cpu[i]) for i in range(NUM_EXPERTS)]
    scale_shape = (NUM_EXPERTS, 2, W4_LOGICAL_N) if per_group else (NUM_EXPERTS, W4_LOGICAL_N)
    scales_cpu = torch.rand(scale_shape, dtype=torch.float32) * 0.02 + 0.001
    weight_scales = (
        [_pack_a8w4_scale(scales_cpu)] if single_tensor else [_pack_a8w4_scale(scale) for scale in scales_cpu]
    )
    assist_cpu = torch.rand(NUM_EXPERTS, W4_LOGICAL_N, dtype=torch.float32) * 0.02
    assists = [assist_cpu.npu()] if single_tensor else [assist.npu() for assist in assist_cpu]
    return weights, weight_scales, assists


@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_count_matches_cumsum():
    """Count and cumulative group lists must describe the same grouping."""
    num_tokens, hidden_size, num_experts, intermediate_size = 8, 7168, 4, 4096

    torch.manual_seed(0)
    hidden_states = torch.randint(-128, 127, (num_tokens, hidden_size), dtype=torch.int8).npu()
    weights = [
        torch_npu.npu_format_cast(
            torch.randint(-128, 127, (hidden_size, intermediate_size), dtype=torch.int8).npu(),
            29,
        )
        for _ in range(num_experts)
    ]
    weight_scales = [(torch.rand(intermediate_size, dtype=torch.float32) * 0.9 + 0.1).npu() for _ in range(num_experts)]
    token_scales = (torch.rand(num_tokens, dtype=torch.float32) * 0.9 + 0.1).npu()

    # Include an empty expert to exercise the boundary where count and
    # cumulative encodings differ most visibly.
    count_group_list = torch.tensor([2, 0, 3, 3], dtype=torch.int64, device="npu")
    cumsum_group_list = count_group_list.cumsum(dim=0)

    cumsum_output, cumsum_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=cumsum_group_list,
        group_list_type=0,
    )
    count_output, count_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
        x=hidden_states,
        weight=weights,
        weight_scale=weight_scales,
        x_scale=token_scales,
        group_list=count_group_list,
        group_list_type=1,
    )

    torch.testing.assert_close(count_output.cpu(), cumsum_output.cpu(), atol=1, rtol=2**-13)
    torch.testing.assert_close(count_scale.cpu(), cumsum_scale.cpu(), atol=1e-9, rtol=1e-6)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("single_tensor", [True, False], ids=["single", "multi"])
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w8_split_sync_matches_chunks(single_tensor):
    """Splitting one expert across cube/vector syncs must preserve its scale."""
    torch.manual_seed(3)
    weights, weight_scales = _make_a8w8_case(single_tensor)
    hidden_states = torch.randint(
        -16,
        16,
        (SPLIT_SYNC_NUM_TOKENS, HIDDEN_SIZE),
        dtype=torch.int8,
        device="npu",
    )
    token_scales = torch.rand(SPLIT_SYNC_NUM_TOKENS, dtype=torch.float32, device="npu") * 0.02 + 0.001

    full_output, full_scale = _call_v2(
        hidden_states=hidden_states,
        weights=weights,
        weight_scales=weight_scales,
        token_scales=token_scales,
        groups=_group_list([SPLIT_SYNC_NUM_TOKENS, 0, 0, 0], 1),
        group_list_type=1,
    )

    chunk_outputs = []
    chunk_scales = []
    for start in range(0, SPLIT_SYNC_NUM_TOKENS, SPLIT_SYNC_CHUNK_SIZE):
        end = start + SPLIT_SYNC_CHUNK_SIZE
        output, scale = _call_v2(
            hidden_states=hidden_states[start:end],
            weights=weights,
            weight_scales=weight_scales,
            token_scales=token_scales[start:end],
            groups=_group_list([SPLIT_SYNC_CHUNK_SIZE, 0, 0, 0], 1),
            group_list_type=1,
        )
        chunk_outputs.append(output)
        chunk_scales.append(scale)

    torch.testing.assert_close(full_output.cpu(), torch.cat(chunk_outputs).cpu(), atol=1, rtol=2**-13)
    torch.testing.assert_close(full_scale.cpu(), torch.cat(chunk_scales).cpu(), atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("single_tensor", [True, False], ids=["single", "multi"])
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w8_matches_cpu(single_tensor):
    """Packed and TensorList layouts must both preserve the A8W8 result."""
    torch.manual_seed(5)
    weights_cpu = torch.randint(
        -16,
        16,
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.int8,
    )
    scales_cpu = torch.rand(NUM_EXPERTS, INTERMEDIATE_SIZE, dtype=torch.float32) * 0.02 + 0.001
    weights = _format_nz_weights(weights_cpu, single_tensor)
    weight_scales = [scales_cpu.npu()] if single_tensor else [scale.npu() for scale in scales_cpu]
    hidden_states_cpu = torch.randint(-16, 16, (NUM_TOKENS, HIDDEN_SIZE), dtype=torch.int8)
    token_scales_cpu = torch.rand(NUM_TOKENS, dtype=torch.float32) * 0.02 + 0.001
    expert_counts = [4, 0, 5, 7]

    expected_output, expected_scale = _a8w8_golden(
        hidden_states_cpu,
        weights_cpu,
        scales_cpu,
        token_scales_cpu,
        expert_counts,
    )
    output, output_scale = _call_v2(
        hidden_states=hidden_states_cpu.npu(),
        weights=weights,
        weight_scales=weight_scales,
        token_scales=token_scales_cpu.npu(),
        groups=_group_list(expert_counts, 1),
        group_list_type=1,
    )

    torch.testing.assert_close(output.cpu(), expected_output, atol=1, rtol=2**-13)
    torch.testing.assert_close(output_scale.cpu(), expected_scale, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize(
    "expert_counts",
    [[1, 0, 0, 0], [4, 4, 4, 4], [16, 16, 16, 16]],
    ids=["decode", "small-prefill", "prefill"],
)
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w8_matches_cann_v2(expert_counts):
    """Custom V2 should stay within int8 rounding of CANN GmmSwigluQuantV2."""
    torch.manual_seed(17)
    num_tokens = sum(expert_counts)
    hidden_states = torch.randint(
        -16,
        16,
        (num_tokens, HIDDEN_SIZE),
        dtype=torch.int8,
        device="npu",
    )
    weights, weight_scales = _make_a8w8_case(True)
    token_scales = torch.rand(num_tokens, dtype=torch.float32, device="npu") * 0.02 + 0.001
    groups = _group_list(expert_counts, 0)

    cann_output, cann_scale = _call_cann_v2(
        hidden_states=hidden_states,
        weights=weights,
        weight_scales=weight_scales,
        token_scales=token_scales,
        groups=groups,
        group_list_type=0,
    )
    output, output_scale = _call_v2(
        hidden_states=hidden_states,
        weights=weights,
        weight_scales=weight_scales,
        token_scales=token_scales,
        groups=groups,
        group_list_type=0,
    )

    torch.testing.assert_close(output.cpu(), cann_output.cpu(), atol=1, rtol=2**-13)
    torch.testing.assert_close(output_scale.cpu(), cann_scale.cpu(), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("single_tensor", [True, False], ids=["single", "multi"])
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w8_count_graph_replay(single_tensor):
    """The MoE count contract must consume fresh values on every replay."""
    torch.manual_seed(1)
    group_list_type = 1
    weights, weight_scales = _make_a8w8_case(single_tensor)
    static_x = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.int8, device="npu")
    static_token_scales = torch.ones(NUM_TOKENS, dtype=torch.float32, device="npu")
    static_groups = _group_list([4, 0, 5, 7], group_list_type)

    graph = torch.npu.NPUGraph()
    graph_pool = torch.npu.graph_pool_handle()
    with torch.npu.graph(graph, pool=graph_pool, capture_error_mode="thread_local", auto_dispatch_capture=True):
        graph_output, graph_output_scale = _call_adaptor_v2(
            hidden_states=static_x,
            weights=weights,
            weight_scales=weight_scales,
            token_scales=static_token_scales,
            groups=static_groups,
            group_list_type=group_list_type,
        )

    try:
        for seed, counts in ((11, [4, 0, 5, 7]), (12, [0, 7, 1, 8]), (13, [8, 3, 5, 0])):
            torch.manual_seed(seed)
            current_x = torch.randint(-16, 16, static_x.shape, dtype=torch.int8, device="npu")
            current_token_scales = torch.rand(NUM_TOKENS, dtype=torch.float32, device="npu") * 0.02 + 0.001
            current_groups = _group_list(counts, group_list_type)
            expected_output, expected_scale = _call_adaptor_v2(
                hidden_states=current_x,
                weights=weights,
                weight_scales=weight_scales,
                token_scales=current_token_scales,
                groups=current_groups,
                group_list_type=group_list_type,
            )

            static_x.copy_(current_x)
            static_token_scales.copy_(current_token_scales)
            static_groups.copy_(current_groups)
            graph.replay()
            torch.npu.synchronize()

            torch.testing.assert_close(graph_output.cpu(), expected_output.cpu(), atol=1, rtol=2**-13)
            torch.testing.assert_close(graph_output_scale.cpu(), expected_scale.cpu(), atol=1e-9, rtol=1e-6)
    finally:
        graph.reset()


@pytest.mark.parametrize("per_group", [False, True], ids=["per_channel", "per_group"])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0], ids=["unclamped", "clamped"])
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w4_single_matches_multi(per_group, swiglu_limit):
    """Packed and TensorList VA W4 layouts must implement the same grouping."""
    torch.manual_seed(2)
    single_weights, single_scales, single_assists = _make_a8w4_case(True, per_group)
    torch.manual_seed(2)
    multi_weights, multi_scales, multi_assists = _make_a8w4_case(False, per_group)
    hidden_states = torch.randint(
        -5,
        5,
        (NUM_TOKENS, HIDDEN_SIZE),
        dtype=torch.int8,
        device="npu",
    )
    token_scales = torch.rand(NUM_TOKENS, dtype=torch.float32, device="npu") * 0.02 + 0.001
    counts = [4, 0, 5, 7]

    single_output, single_output_scale = _call_v2(
        hidden_states=hidden_states,
        weights=single_weights,
        weight_scales=single_scales,
        token_scales=token_scales,
        groups=_group_list(counts, 0),
        group_list_type=0,
        weight_assist_matrix=single_assists,
        dequant_mode=int(per_group),
        swiglu_limit=swiglu_limit,
    )
    multi_output, multi_output_scale = _call_v2(
        hidden_states=hidden_states,
        weights=multi_weights,
        weight_scales=multi_scales,
        token_scales=token_scales,
        groups=_group_list(counts, 1),
        group_list_type=1,
        weight_assist_matrix=multi_assists,
        dequant_mode=int(per_group),
        swiglu_limit=swiglu_limit,
    )

    torch.testing.assert_close(single_output.cpu(), multi_output.cpu(), atol=1, rtol=0.005)
    torch.testing.assert_close(single_output_scale.cpu(), multi_output_scale.cpu(), atol=1e-6, rtol=0.005)
    assert tuple(single_output.shape) == (NUM_TOKENS, W4_LOGICAL_N // 2)
