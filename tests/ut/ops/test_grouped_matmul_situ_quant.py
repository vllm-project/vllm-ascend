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
"""Unit tests for the GroupedMatmulSituQuant A5 fused custom op.

Scope (full bit-exact acceptance against the production split chain lives in
the evolution delivery tree, not here):
  * four weight forms (ND/NZ x stacked/list) byte-identical
  * graph capture-replay twin + dynamic group_list readback
  * negative probes (bad quant combo / bad weight_format)

Byte-level comparison throughout: FP8/E8M0/FP4X2 comparisons go through
.view(torch.uint8) (framework dtype coverage is incomplete on NPU).
"""

import pytest
import torch
import torch_npu  # noqa: F401  -- registers torch.npu APIs used by this test

from vllm_ascend.ops.grouped_matmul_situ_quant import (
    grouped_matmul_situ_quant,
    is_available,
    to_weight_nz,
    to_weight_nz_list,
)

pytestmark = pytest.mark.skipif(
    not is_available(), reason="grouped_matmul_situ_quant requires an Ascend950 vllm_ascend_C build"
)

DEV = "npu:0"
E, M_CAP, K, N = 4, 16, 256, 512


def _build_case(seed: int = 1234):
    gen = torch.Generator().manual_seed(seed)
    kb = (K + 63) // 64

    x = torch.randn(M_CAP, K, generator=gen).to(torch.float8_e4m3fn).to(DEV)
    xs = (127 + torch.randint(-4, 5, (M_CAP, kb, 2), generator=gen)).to(torch.uint8).to(DEV).view(torch.float8_e8m0fnu)

    mag = torch.randint(0, 7, (E, N, K // 2), generator=gen, dtype=torch.uint8)
    sign = torch.randint(0, 2, (E, N, K // 2), generator=gen, dtype=torch.uint8) * 8
    w_nd = (mag | sign).view(torch.float4_e2m1fn_x2).to(DEV)

    e = torch.randint(-2, 3, (E, N, kb, 2), generator=gen).float()
    ws = torch.pow(2.0, e).permute(0, 2, 1, 3).contiguous().to(torch.float8_e8m0fnu).to(DEV)

    gl = torch.tensor([4, 4, 4, 4], dtype=torch.int64, device=DEV)
    return x, xs, w_nd, ws, gl


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


@pytest.mark.parametrize("use_list", [False, True])
def test_meta_shape_uses_weight_geometry(use_list: bool):
    """The canonical N-major scale layout is (E, N, kb, 2)."""
    kb = (K + 63) // 64
    x = torch.empty((M_CAP, K), device="meta", dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M_CAP, kb, 2), device="meta", dtype=torch.float8_e8m0fnu)
    group_list = torch.empty((E,), device="meta", dtype=torch.int64)

    if use_list:
        weight = [
            torch.empty((N, K // 2), device="meta", dtype=torch.float4_e2m1fn_x2) for _ in range(E)
        ]
        weight_scale = [
            torch.empty((N, kb, 2), device="meta", dtype=torch.float8_e8m0fnu) for _ in range(E)
        ]
        op = torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz.list
    else:
        weight = torch.empty((E, N, K // 2), device="meta", dtype=torch.float4_e2m1fn_x2)
        weight_scale = torch.empty((E, N, kb, 2), device="meta", dtype=torch.float8_e8m0fnu)
        op = torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz

    y, y_scale = op(
        x,
        weight,
        weight_scale,
        None,
        None,
        x_scale,
        None,
        group_list,
        1,
        0,
        1,
        1,
        None,
        4.0,
        25.0,
    )

    assert y.shape == (M_CAP, N // 2)
    assert y_scale.shape == (M_CAP, (N // 2 + 63) // 64, 2)


def _build_empty_case(seed: int = 1234):
    x, xs, w_nd, ws, _ = _build_case(seed)
    gl = torch.zeros(E, dtype=torch.int64, device=DEV)
    return x[:0], xs[:0], w_nd, ws, gl


def test_four_weight_forms_byte_identical():
    x, xs, w_nd, ws, gl = _build_case()
    w_nz = to_weight_nz(w_nd)
    # NZ TensorList elements must be independently allocated (slicing a
    # stacked NZ tensor yields non-contiguous internal-format views that the
    # entry-layer cast rejects — hence the dedicated helper).
    w_nz_list = to_weight_nz_list(w_nd)
    ws_list = [ws[ei] for ei in range(E)]

    y_nd, s_nd = grouped_matmul_situ_quant(x, xs, w_nd, ws, gl, beta=4.0, linear_beta=25.0, weight_format="nd")
    y_nz, s_nz = grouped_matmul_situ_quant(x, xs, w_nz, ws, gl, beta=4.0, linear_beta=25.0, weight_format="nz")
    y_l, s_l = grouped_matmul_situ_quant(x, xs, w_nz_list, ws_list, gl, beta=4.0, linear_beta=25.0, weight_format="nz")

    assert torch.equal(_bytes(y_nd), _bytes(y_nz))
    assert torch.equal(_bytes(s_nd), _bytes(s_nz))
    assert torch.equal(_bytes(y_l), _bytes(y_nz))
    assert torch.equal(_bytes(s_l), _bytes(s_nz))
    assert y_nd.shape == (M_CAP, N // 2)


@pytest.mark.parametrize("group_list_type", [0, 1])
@pytest.mark.parametrize(
    "weight_format,use_list",
    [
        ("nd", False),
        ("nd", True),
        ("nz", False),
        ("nz", True),
    ],
)
def test_empty_rank_returns_empty_outputs(group_list_type: int, weight_format: str, use_list: bool):
    x, xs, w_nd, ws, gl = _build_empty_case()
    if weight_format == "nz":
        weight = to_weight_nz_list(w_nd) if use_list else to_weight_nz(w_nd)
    else:
        weight = [w_nd[ei] for ei in range(E)] if use_list else w_nd
    weight_scale = [ws[ei] for ei in range(E)] if use_list else ws

    y, y_scale = grouped_matmul_situ_quant(
        x,
        xs,
        weight,
        weight_scale,
        gl,
        beta=4.0,
        linear_beta=25.0,
        group_list_type=group_list_type,
        weight_format=weight_format,
    )

    assert y.shape == (0, N // 2)
    assert y.dtype == torch.float8_e4m3fn
    assert y_scale.shape == (0, (N // 2 + 63) // 64, 2)
    assert y_scale.dtype == torch.float8_e8m0fnu


def test_empty_rank_graph_capture_replay():
    x, xs, w_nd, ws, gl = _build_empty_case()
    w_nz = to_weight_nz(w_nd)

    def call():
        return grouped_matmul_situ_quant(x, xs, w_nz, ws, gl, beta=4.0, linear_beta=25.0)

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y, y_scale = call()
    g.replay()
    torch.npu.synchronize()

    assert y.shape == (0, N // 2)
    assert y_scale.shape == (0, (N // 2 + 63) // 64, 2)


def test_empty_rank_still_validates_metadata():
    x, xs, w_nd, ws, gl = _build_empty_case()
    w_nz = to_weight_nz(w_nd)

    with pytest.raises(RuntimeError, match="groupList length must equal E"):
        grouped_matmul_situ_quant(x, xs, w_nz, ws, gl[:-1], beta=4.0, linear_beta=25.0)

    with pytest.raises(RuntimeError, match="groupListType must be 0 or 1"):
        grouped_matmul_situ_quant(x, xs, w_nz, ws, gl, beta=4.0, linear_beta=25.0, group_list_type=2)

    with pytest.raises(RuntimeError, match="quantMode=1"):
        torch.ops._C_ascend.grouped_matmul_situ_quant(
            x, w_nd, ws, None, None, xs, None, gl, 1, 0, 0, 1, None, 4.0, 25.0
        )


def test_graph_capture_replay_and_dynamic_group_list():
    x, xs, w_nd, ws, gl = _build_case()
    w_nz = to_weight_nz(w_nd)

    def call():
        return grouped_matmul_situ_quant(x, xs, w_nz, ws, gl, beta=4.0, linear_beta=25.0)

    y_eager, s_eager = call()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        y_g, s_g = call()
    g.replay()
    torch.npu.synchronize()

    assert torch.equal(_bytes(y_g), _bytes(y_eager))
    assert torch.equal(_bytes(s_g), _bytes(s_eager))

    # dynamicity: rewrite group_list values in place, replay must read them
    gl.copy_(torch.tensor([16, 0, 0, 0], dtype=torch.int64, device=DEV))
    y_rot, s_rot = call()  # eager reference with new routing
    g.replay()
    torch.npu.synchronize()
    assert torch.equal(_bytes(y_g), _bytes(y_rot))
    assert torch.equal(_bytes(s_g), _bytes(s_rot))


def test_negative_probes():
    x, xs, w_nd, ws, gl = _build_case()

    with pytest.raises(ValueError, match="weight_format"):
        grouped_matmul_situ_quant(x, xs, w_nd, ws, gl, beta=4.0, linear_beta=25.0, weight_format="nzc")

    # only the MX A8W4 combo is implemented; other quantMode values raise
    with pytest.raises(RuntimeError):
        torch.ops._C_ascend.grouped_matmul_situ_quant(
            x, w_nd, ws, None, None, xs, None, gl, 1, 0, 0, 1, None, 4.0, 25.0
        )
