# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import AscendDeviceType, bootstrap_custom_op_env, get_ascend_device_type

K3_INPUT_WIDTH = 6144
K3_SITU_BETA = 4.0
K3_SITU_LINEAR_BETA = 25.0
SITU_MX_DST_TYPE_E4M3FN = 36
GROUP_LIST_64 = [5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4]
GROUP_LIST_128 = [10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]


def _register_custom_op() -> None:
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("requires Ascend 950")
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401


def _run(x: torch.Tensor, group_list: torch.Tensor | None):
    situ_group_list = None if group_list is None else group_list.to(torch.int64)
    return torch.ops._C_ascend.situ_mx_quant(
        x=x,
        group_list=situ_group_list,
        beta=K3_SITU_BETA,
        linear_beta=K3_SITU_LINEAR_BETA,
        activate_left=True,
        dst_type=SITU_MX_DST_TYPE_E4M3FN,
    )


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_situ_mx_quant_group_list_eager_and_graph():
    _register_custom_op()
    capacity_rows = 128
    group_64 = torch.tensor(GROUP_LIST_64, dtype=torch.int32, device="npu")
    group_full = torch.tensor(GROUP_LIST_128, dtype=torch.int32, device="npu")
    x = torch.randn(capacity_rows, K3_INPUT_WIDTH, dtype=torch.bfloat16, device="npu")

    baseline_y, baseline_scale = _run(x, None)
    full_group_y, full_group_scale = _run(x, group_full)
    optimized_y, optimized_scale = _run(x, group_64)
    direct_y, direct_scale = _run(x[:64], group_64)
    torch.npu.synchronize()

    assert tuple(optimized_y.shape) == tuple(baseline_y.shape)
    assert tuple(optimized_scale.shape) == tuple(baseline_scale.shape)
    torch.testing.assert_close(full_group_y.cpu(), baseline_y.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(full_group_scale.cpu(), baseline_scale.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(optimized_y[:64].cpu(), baseline_y[:64].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(optimized_scale[:64].cpu(), baseline_scale[:64].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(optimized_y[:64].cpu(), direct_y.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(optimized_scale[:64].cpu(), direct_scale.cpu(), rtol=0, atol=0)

    graph_group = group_full.clone()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        graph_y, graph_scale = _run(x, graph_group)
    torch.npu.synchronize()

    graph_group.copy_(group_64)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_y[:64].cpu(), optimized_y[:64].cpu(), rtol=0, atol=0)
    torch.testing.assert_close(graph_scale[:64].cpu(), optimized_scale[:64].cpu(), rtol=0, atol=0)
