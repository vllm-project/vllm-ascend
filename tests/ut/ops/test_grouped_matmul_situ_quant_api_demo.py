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
# This file is a part of the vllm-ascend project.
"""Small, readable hardware demo for the GroupedMatmulSituQuant API.

Run this file on an installed Ascend950 environment independently of the UT
conftest (which supplies CPU/A2 mocks)::

    python -m pytest --noconftest \
        tests/ut/ops/test_grouped_matmul_situ_quant_api_demo.py -v -s -rA

This is an interface-consistency example.  It checks that the public wrapper
agrees across stacked/TensorList NZ inputs and type-1 counts/type-0 cumulative
group lists.  It is not an independent split-chain accuracy golden and it is
not a performance test.  Only routed rows are defined by the operator, so the
capacity tail is deliberately excluded from byte comparisons.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend.ops.grouped_matmul_situ_quant import (  # noqa: E402
    grouped_matmul_situ_quant,
    is_available,
    to_weight_nz,
    to_weight_nz_list,
)
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type  # noqa: E402


DEVICE = "npu:0"
EXPERTS = 3
CAPACITY = 8
K = 128
N = 128
K_GROUPS = K // 64
ACTIVE_ROWS = 5
BETA = 4.0
LINEAR_BETA = 25.0


def _is_a5_hardware() -> bool:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        return False
    return get_ascend_device_type() == AscendDeviceType.A5


pytestmark = pytest.mark.skipif(
    not _is_a5_hardware(),
    reason="this hardware demo requires an Ascend950 (A5) environment",
)


def _cpu_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Move raw bytes to CPU before comparisons to avoid NPU MX dtype gaps."""
    return tensor.detach().view(torch.uint8).cpu()


def _build_inputs():
    """Build deterministic, finite inputs whose byte patterns are easy to audit."""
    # Step 1: small non-zero FP8 activations.  Values stay far from the E4M3
    # finite range limits, and 0 is replaced before conversion.
    x_fp32 = (torch.arange(CAPACITY * K, dtype=torch.float32).reshape(CAPACITY, K) % 15 - 7) / 4
    x_fp32 = torch.where(x_fp32 == 0, torch.full_like(x_fp32, 0.5), x_fp32)
    x = x_fp32.to(torch.float8_e4m3fn).to(DEVICE)

    x_row = torch.arange(CAPACITY).view(CAPACITY, 1, 1)
    x_k_group = torch.arange(K_GROUPS).view(1, K_GROUPS, 1)
    x_lane = torch.arange(2).view(1, 1, 2)
    x_scale_exp = (x_row + 2 * x_k_group + x_lane) % 7 - 3
    x_scale = (x_scale_exp + 127).to(torch.uint8).to(DEVICE).view(torch.float8_e8m0fnu)

    # Step 2: packed FP4 weights.  Each low and high nibble has magnitude 1..7,
    # so neither half of any packed byte is an FP4 zero.
    packed_index = torch.arange(EXPERTS * N * (K // 2)).reshape(EXPERTS, N, K // 2)
    low_nibble = (packed_index % 7 + 1) | (((packed_index // 7) % 2) << 3)
    high_nibble = ((packed_index * 3) % 7 + 1) | (((packed_index // 11) % 2) << 3)
    packed_weight = (low_nibble | (high_nibble << 4)).to(torch.uint8)
    assert torch.all((packed_weight & 0x0F) != 0)
    assert torch.all((packed_weight >> 4) != 0)
    weight_nd = packed_weight.view(torch.float4_e2m1fn_x2).to(DEVICE)

    # Step 3: create finite E8M0 scale bytes in the loader's physical order.
    # Every expert, output channel, and K group contributes to the exponent.
    expert = torch.arange(EXPERTS).view(EXPERTS, 1, 1, 1)
    channel = torch.arange(N).view(1, N, 1, 1)
    k_group = torch.arange(K_GROUPS).view(1, 1, K_GROUPS, 1)
    lane = torch.arange(2).view(1, 1, 1, 2)
    scale_exp = (2 * expert + channel + 3 * k_group + lane) % 7 - 3
    weight_scale_base = (scale_exp + 127).to(torch.uint8).to(DEVICE).view(torch.float8_e8m0fnu)

    # A real loader exposes (E, K_GROUPS, N, 2) by transposing the contiguous
    # N-major (E, N, K_GROUPS, 2) base.  The wrapper restores the base view
    # without materializing an unsupported E8M0 transpose.
    weight_scale_view = weight_scale_base.transpose(-3, -2)
    assert weight_scale_base.is_contiguous()
    assert not weight_scale_view.is_contiguous()
    assert weight_scale_view.data_ptr() == weight_scale_base.data_ptr()

    return x, x_scale, weight_nd, weight_scale_base, weight_scale_view


def _assert_output_contract(output: torch.Tensor, output_scale: torch.Tensor) -> None:
    assert output.shape == (CAPACITY, N // 2)
    assert output.dtype == torch.float8_e4m3fn
    assert output_scale.shape == (CAPACITY, (N // 2 + 63) // 64, 2)
    assert output_scale.dtype == torch.float8_e8m0fnu

    # Only routed rows have defined contents; check their numerical encoding,
    # while leaving the three capacity-tail rows entirely unconstrained.
    output_bytes = _cpu_bytes(output[:ACTIVE_ROWS])
    scale_bytes = _cpu_bytes(output_scale[:ACTIVE_ROWS])
    assert torch.isfinite(output_bytes.view(torch.float8_e4m3fn).float()).all()
    # E8M0 has no infinity encoding; byte 255 is its NaN encoding.
    assert torch.all(scale_bytes != 255)


def test_nz_stacked_and_list_match_for_counts_and_cumsum():
    """Exercise the two public NZ forms and both group-list encodings."""
    torch.npu.set_device(DEVICE)

    # On real A5 hardware, a missing registration is a broken installation,
    # not a reason to skip or silently pass this demo.
    assert is_available(), (
        "Ascend950 detected, but grouped_matmul_situ_quant is not registered; "
        "reinstall vllm-ascend with SOC_VERSION=ascend950"
    )

    x, x_scale, weight_nd, weight_scale_base, weight_scale_view = _build_inputs()

    # Step 4: model-load-time work.  Each representation is converted exactly
    # once here and then reused by both forward calls below.
    weight_nz_stacked = to_weight_nz(weight_nd)
    weight_nz_list = to_weight_nz_list(weight_nd)
    weight_scale_list = [weight_scale_view[expert] for expert in range(EXPERTS)]

    # Expert 1 is empty.  Five rows are routed and rows [5:8] are capacity tail.
    counts = torch.tensor([3, 0, 2], dtype=torch.int64, device=DEVICE)
    cumsum = torch.tensor([3, 3, 5], dtype=torch.int64, device=DEVICE)

    # Step 5: call only the public wrapper.  Stacked weights select the ordinary
    # NZ overload; independently allocated per-expert weights select ``.list``.
    # The calls also cover both accepted scale inputs: the contiguous N-major
    # base and the non-contiguous logical view emitted by a real loader.
    stacked_counts = grouped_matmul_situ_quant(
        x,
        x_scale,
        weight_nz_stacked,
        weight_scale_base,
        counts,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        group_list_type=1,
        weight_format="nz",
    )
    list_counts = grouped_matmul_situ_quant(
        x,
        x_scale,
        weight_nz_list,
        weight_scale_list,
        counts,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        group_list_type=1,
        weight_format="nz",
    )
    stacked_cumsum = grouped_matmul_situ_quant(
        x,
        x_scale,
        weight_nz_stacked,
        weight_scale_view,
        cumsum,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        group_list_type=0,
        weight_format="nz",
    )
    list_cumsum = grouped_matmul_situ_quant(
        x,
        x_scale,
        weight_nz_list,
        weight_scale_list,
        cumsum,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        group_list_type=0,
        weight_format="nz",
    )

    torch.npu.synchronize()
    results = (stacked_counts, list_counts, stacked_cumsum, list_cumsum)
    for output, output_scale in results:
        _assert_output_contract(output, output_scale)

    reference_output, reference_scale = stacked_counts
    for output, output_scale in results[1:]:
        assert torch.equal(_cpu_bytes(output[:ACTIVE_ROWS]), _cpu_bytes(reference_output[:ACTIVE_ROWS]))
        assert torch.equal(_cpu_bytes(output_scale[:ACTIVE_ROWS]), _cpu_bytes(reference_scale[:ACTIVE_ROWS]))
