# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON, triton

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.activation.routed_swiglu import _routed_swiglu_kernel, can_use_routed_swiglu, routed_swiglu
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

pytestmark = pytest.mark.skipif(
    not HAS_TRITON or get_ascend_device_type() != AscendDeviceType.A3, reason="requires A3 Triton"
)


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize("active", [0, 864, 5760])
def test_routed_swiglu(group_list_type, active):
    capacity, width = 5760, 96
    limit = 3.3
    init_device_properties_triton()
    x = torch.randn(capacity, 2 * width, device="npu", dtype=torch.bfloat16) * 5
    groups = torch.tensor([active // 2, 0, active - active // 2], dtype=torch.int64, device="npu")
    if group_list_type == 0:
        groups = groups.cumsum(0)
    assert can_use_routed_swiglu(x, groups, group_list_type)
    assert not can_use_routed_swiglu(x[:-1], groups, group_list_type)
    gate, up = x.chunk(2, dim=-1)
    expected = torch_npu.npu_swiglu(torch.cat([gate.clamp(max=limit), up.clamp(-limit, limit)], dim=-1))
    actual = routed_swiglu(x, groups, group_list_type, limit)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual[:active], expected[:active], rtol=0.008, atol=1e-6)
    # Pre-fill the output to detect any writes to inactive rows, including on empty ranks.
    out = torch.full_like(actual, float("nan"))
    _routed_swiglu_kernel[(get_vectorcore_num(),)](
        x,
        groups,
        out,
        limit,
        capacity,
        width,
        groups.numel(),
        triton.next_power_of_2(groups.numel()),
        group_list_type == 0,
        triton.next_power_of_2(width),
    )
    assert out[active:].isnan().all()
