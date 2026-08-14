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
GRAPH_POOL = torch.npu.graph_pool_handle()


@pytest.fixture(autouse=True)
def _release_npu_memory():
    yield
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


def _pack_a8w4_weight(weight: torch.Tensor) -> torch.Tensor:
    weight_nz = torch_npu.npu_format_cast(weight.npu().to(torch.float32), 29)
    return torch_npu.npu_quantize(
        weight_nz,
        torch.tensor([1.0], device="npu"),
        None,
        torch.quint4x2,
        -1,
        False,
    )


def _pack_a8w4_scale(scale: torch.Tensor) -> torch.Tensor:
    scale_bits = scale.cpu().numpy().view(np.uint32).astype(np.int64)
    return torch.from_numpy(scale_bits).npu()


def _make_a8w4_case(single_tensor: bool, per_group: bool):
    weights_cpu = torch.randint(
        -5,
        5,
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=torch.int8,
    )
    packed_weights = _pack_a8w4_weight(weights_cpu)
    # npu_quantize does not support packing a rank-2 FRACTAL_NZ tensor on A3.
    # Slice the correctly packed rank-3 tensor to build the equivalent
    # TensorList representation without changing the underlying NZ layout.
    weights = [packed_weights] if single_tensor else [packed_weights[i].clone() for i in range(NUM_EXPERTS)]
    scale_shape = (NUM_EXPERTS, 2, INTERMEDIATE_SIZE) if per_group else (NUM_EXPERTS, INTERMEDIATE_SIZE)
    scales_cpu = torch.rand(scale_shape, dtype=torch.float32) * 0.02 + 0.001
    weight_scales = (
        [_pack_a8w4_scale(scales_cpu)] if single_tensor else [_pack_a8w4_scale(scale) for scale in scales_cpu]
    )
    assist_cpu = torch.rand(NUM_EXPERTS, INTERMEDIATE_SIZE, dtype=torch.float32) * 0.02
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
@pytest.mark.parametrize("group_list_type", [0, 1], ids=["cumulative", "counts"])
@torch.inference_mode()
def test_grouped_matmul_swiglu_quant_v2_a8w8_graph_replay(single_tensor, group_list_type):
    """Replay must consume fresh x, x-scale, and group-list values."""
    torch.manual_seed(1)
    weights, weight_scales = _make_a8w8_case(single_tensor)
    static_x = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.int8, device="npu")
    static_token_scales = torch.ones(NUM_TOKENS, dtype=torch.float32, device="npu")
    static_groups = _group_list([4, 0, 5, 7], group_list_type)

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=GRAPH_POOL, capture_error_mode="thread_local", auto_dispatch_capture=True):
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
    """Packed and TensorList W4 layouts must implement the same grouping."""
    torch.manual_seed(2)
    single_weights, single_scales, single_assists = _make_a8w4_case(True, per_group)

    # Reuse the unpacked values encoded by the single tensor so both layouts
    # compare the same logical weights and scales.
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
