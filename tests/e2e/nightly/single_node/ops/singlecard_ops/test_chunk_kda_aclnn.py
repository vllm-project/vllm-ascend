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

import gc
import math
from dataclasses import dataclass

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

DETERMINISM_REPEATS = 20
CHUNK_KDA_OUTPUT_NAMES = (
    "o",
    "final_state",
    "g",
    "aqk",
    "akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state_out",
)


@dataclass
class ChunkKdaReferenceResult:
    o: torch.Tensor
    final_state: torch.Tensor | None


def _lower_inverse(mat: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=torch.float32)
    lhs = eye + torch.tril(mat.to(torch.float32), diagonal=-1)
    return torch.linalg.solve_triangular(lhs, eye, upper=False)


def chunk_kda_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> ChunkKdaReferenceResult:
    bsz, total_t, hq, kdim = q.shape
    _, _, hv, vdim = v.shape
    group = hv // hq
    out_dtype = v.dtype
    device = q.device
    o = torch.zeros((bsz, total_t, hv, vdim), device=device, dtype=out_dtype)
    state = (
        torch.zeros((bsz, hv, kdim, vdim), device=device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32).clone()
    )

    for b in range(bsz):
        for start in range(0, total_t, chunk_size):
            end = min(start + chunk_size, total_t)
            cur_t = end - start
            for ihv in range(hv):
                ih = ihv // group
                q_blk = q[b, start:end, ih].to(torch.float32)
                k_blk = k[b, start:end, ih].to(torch.float32)
                v_blk = v[b, start:end, ihv].to(torch.float32)
                g_blk = gk[b, start:end, ihv].to(torch.float32)
                beta_blk = beta[b, start:end, ihv].to(torch.float32)

                causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril()
                strict_causal = torch.ones((cur_t, cur_t), device=device, dtype=torch.bool).tril(diagonal=-1)
                rel = g_blk[:, None, :] - g_blk[None, :, :]
                rel = rel.masked_fill(~causal[:, :, None], 0.0)
                gate = torch.exp2(rel)
                qk = torch.einsum("ik,jk,ijk->ij", q_blk, k_blk, gate) * float(scale)
                kk = torch.einsum("ik,jk,ijk->ij", k_blk, k_blk, gate)
                tril_qk = torch.where(causal, qk, torch.zeros_like(qk))
                tril_kk = torch.where(strict_causal, kk * beta_blk[:, None], torch.zeros_like(kk))
                inv_akk = _lower_inverse(tril_kk)

                k_beta_g = k_blk * beta_blk[:, None] * torch.exp2(g_blk)
                v_beta = v_blk * beta_blk[:, None]
                w_blk = inv_akk @ k_beta_g
                u_blk = inv_akk @ v_beta

                last_g = g_blk[cur_t - 1]
                qg_blk = q_blk * torch.exp2(g_blk)
                kg_blk = k_blk * torch.exp2(last_g[None, :] - g_blk)
                h_prev = state[b, ihv].clone()
                v_new_blk = u_blk - w_blk @ h_prev
                state[b, ihv] = torch.exp2(last_g)[:, None] * h_prev + kg_blk.T @ v_new_blk

                o_inter = qg_blk @ h_prev * float(scale)
                o_local = tril_qk @ v_new_blk
                o[b, start:end, ihv] = (o_inter + o_local).to(out_dtype)

    return ChunkKdaReferenceResult(o=o, final_state=state if output_final_state else None)


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _is_ascend_950() -> bool:
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


def _assert_close(name, actual, expected, rtol=5e-2, atol=5e-2):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=rtol, atol=atol, msg=name)


def _snapshot_outputs(outputs):
    torch.npu.synchronize()
    return tuple(output.detach().cpu().contiguous() for output in outputs)


def _assert_outputs_bitwise_equal(reference, actual, repeat):
    assert len(reference) == len(actual) == len(CHUNK_KDA_OUTPUT_NAMES)
    for name, expected, current in zip(CHUNK_KDA_OUTPUT_NAMES, reference, actual):
        same_metadata = expected.shape == current.shape and expected.dtype == current.dtype
        same_bits = same_metadata and torch.equal(expected.view(torch.uint8), current.view(torch.uint8))
        if same_bits:
            continue

        expected_float = expected.float()
        current_float = current.float()
        finite = torch.isfinite(expected_float) & torch.isfinite(current_float)
        max_abs_diff = (
            (expected_float[finite] - current_float[finite]).abs().max().item() if finite.any() else float("nan")
        )
        expected_nonfinite = (~torch.isfinite(expected_float)).sum().item()
        current_nonfinite = (~torch.isfinite(current_float)).sum().item()
        raise AssertionError(
            f"repeat={repeat} output={name} is not bitwise deterministic: "
            f"expected_nonfinite={expected_nonfinite}, current_nonfinite={current_nonfinite}, "
            f"max_abs_diff={max_abs_diff}"
        )


def _gate_cumsum_reference(g, chunk_size, cu_seqlens=None):
    g_cpu = g.detach().cpu().to(torch.float32)
    ref = torch.empty_like(g_cpu)
    rcp_ln2 = 1.4426950408889634
    if cu_seqlens is None:
        cu_seqlens = [0, g_cpu.shape[1]]
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end)
            ref[:, chunk_start:chunk_end] = torch.cumsum(g_cpu[:, chunk_start:chunk_end] * rcp_ln2, dim=1)
    return ref


def _layout_swap12_reference(x):
    return x.transpose(1, 2).contiguous()


def test_kda_torch_bindings_have_shape_correct_meta_kernels():
    q = torch.empty((1, 64, 1, 128), device="meta", dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty((1, 64, 2, 256), device="meta", dtype=torch.bfloat16)
    raw_gate = torch.empty((1, 64, 2, 128), device="meta", dtype=torch.bfloat16)
    beta = torch.empty((1, 64, 2), device="meta", dtype=torch.float32)
    a_log = torch.empty((2,), device="meta", dtype=torch.float32)
    dt_bias = torch.empty((2 * 128,), device="meta", dtype=torch.float32)

    gk = torch.ops._C_ascend.kda_gate_cumsum(raw_gate, 64, layout="BSND")
    outputs = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        128**-0.5,
        64,
        layout="BSND",
        output_final_state=True,
        safe_gate=True,
        use_gate_in_kernel=True,
        A_log=a_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    swapped = torch.ops._C_ascend.kda_layout_swap12(raw_gate)

    assert gk.shape == raw_gate.shape
    assert gk.dtype == torch.float32
    assert [tuple(output.shape) for output in outputs[:-1]] == [
        (1, 64, 2, 256),
        (1, 2, 128, 256),
        (1, 2, 64, 128),
        (1, 2, 64, 64),
        (1, 2, 64, 64),
        (1, 2, 64, 128),
        (1, 2, 64, 256),
        (1, 2, 64, 128),
        (1, 2, 64, 128),
        (1, 2, 64, 256),
        (1, 1, 2, 128, 256),
    ]
    assert outputs[11] is None
    assert outputs[0].dtype == torch.bfloat16
    assert outputs[1].dtype == torch.float32
    assert swapped.shape == (1, 2, 64, 128)
    assert swapped.dtype == torch.bfloat16


@torch.inference_mode()
def test_chunk_kda_fwd_matches_reference_bsnd():
    torch.manual_seed(20260720)

    bsz, total_t, hq, hv, kdim, vdim = 1, 64, 1, 1, 128, 128
    dtype = torch.float16
    q = (torch.randn(bsz, total_t, hq, kdim, dtype=dtype) * 0.05).npu()
    k = (torch.randn(bsz, total_t, hq, kdim, dtype=dtype) * 0.05).npu()
    v = (torch.randn(bsz, total_t, hv, vdim, dtype=dtype) * 0.05).npu()
    g = (-torch.rand(bsz, total_t, hv, kdim, dtype=torch.float32) * 0.05).npu()
    beta = torch.sigmoid(torch.randn(bsz, total_t, hv, dtype=torch.float32)).npu()
    gk = torch.ops._C_ascend.kda_gate_cumsum(g, 64, layout="BSND")
    initial_state = (torch.randn(bsz, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    got = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        gk.cpu(),
        beta.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    assert torch.isfinite(got[0]).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _cleanup_npu()


@pytest.mark.parametrize(
    ("shape", "dtype", "with_dependency"),
    [
        ((1, 64, 2, 128), torch.float32, False),
        ((1, 64, 2, 128), torch.float16, True),
        ((1, 64, 2, 128), torch.bfloat16, False),
    ],
)
@torch.inference_mode()
def test_kda_layout_swap12_matches_reference(shape, dtype, with_dependency):
    torch.manual_seed(20260720 + len(shape) + shape[-1])

    x = (torch.randn(*shape, dtype=dtype) * 0.04).npu()
    expected = _layout_swap12_reference(x.cpu())
    dependency = torch.empty_like(expected).npu() if with_dependency else None

    got = torch.ops._C_ascend.kda_layout_swap12(x, dependency=dependency)

    assert torch.isfinite(got).all().item()
    _assert_close("layout_swap12", got, expected, rtol=0, atol=0)
    _cleanup_npu()


@pytest.mark.parametrize(
    ("total_t", "hq", "hv", "kdim", "vdim", "dtype"),
    [
        (64, 1, 1, 128, 128, torch.float16),
        (128, 1, 2, 128, 256, torch.float16),
        (128, 2, 2, 128, 256, torch.bfloat16),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fwd_c128_v256_path(total_t, hq, hv, kdim, vdim, dtype):
    torch.manual_seed(20260720 + total_t + hq + hv + vdim)

    q = (torch.randn(1, total_t, hq, kdim, dtype=dtype) * 0.04).npu()
    k = (torch.randn(1, total_t, hq, kdim, dtype=dtype) * 0.04).npu()
    v = (torch.randn(1, total_t, hv, vdim, dtype=dtype) * 0.04).npu()
    g = (-torch.rand(1, total_t, hv, kdim, dtype=torch.float32) * 0.04).npu()
    beta = torch.sigmoid(torch.randn(1, total_t, hv, dtype=torch.float32)).npu()
    initial_state = (torch.randn(1, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    gk = torch.ops._C_ascend.kda_gate_cumsum(g, 64, layout="BSND")

    def run_chunk_kda_fwd():
        return torch.ops._C_ascend.chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True,
            return_intermediate_states=True,
        )

    is_a5_determinism_case = (
        total_t == 128 and hq == 2 and hv == 2 and kdim == 128 and vdim == 256 and dtype == torch.bfloat16
    )
    if is_a5_determinism_case:
        run_chunk_kda_fwd()
        torch.npu.synchronize()
        got = run_chunk_kda_fwd()
        reference_outputs = _snapshot_outputs(got)
        for repeat in range(1, DETERMINISM_REPEATS):
            current_outputs = _snapshot_outputs(run_chunk_kda_fwd())
            _assert_outputs_bitwise_equal(reference_outputs, current_outputs, repeat)
    else:
        got = run_chunk_kda_fwd()
    ref = chunk_kda_forward_reference(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        gk.cpu(),
        beta.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    assert torch.isfinite(got[0]).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", got[0], ref.o)
    _assert_close("final_state", got[1], ref.final_state)
    _cleanup_npu()


@torch.inference_mode()
def test_kda_gate_cumsum_matches_reference():
    torch.manual_seed(20260720)

    g = (-torch.rand(1, 96, 2, 128, dtype=torch.float32) * 0.05).npu()
    cu_seqlens = [0, 31, 96]
    out = torch.ops._C_ascend.kda_gate_cumsum(g, 64, cu_seqlens=cu_seqlens, layout="BSND")
    ref = _gate_cumsum_reference(g, 64, cu_seqlens)

    assert torch.isfinite(out).all().item()
    _assert_close("gk", out, ref, rtol=2e-3, atol=2e-3)
    _cleanup_npu()


@torch.inference_mode()
def test_chunk_kda_fwd_bnsd_layout_matches_reference():
    torch.manual_seed(20260720)

    total_t, hq, hv, kdim, vdim = 64, 1, 1, 128, 128
    q_bsnd = (torch.randn(1, total_t, hq, kdim, dtype=torch.float16) * 0.04).npu()
    k_bsnd = (torch.randn(1, total_t, hq, kdim, dtype=torch.float16) * 0.04).npu()
    v_bsnd = (torch.randn(1, total_t, hv, vdim, dtype=torch.float16) * 0.04).npu()
    g_bsnd = (-torch.rand(1, total_t, hv, kdim, dtype=torch.float32) * 0.04).npu()
    beta_bsn = torch.sigmoid(torch.randn(1, total_t, hv, dtype=torch.float32)).npu()
    initial_state = (torch.randn(1, hv, kdim, vdim, dtype=torch.float32) * 0.01).npu()
    scale = kdim**-0.5

    q_bnsd = q_bsnd.transpose(1, 2).contiguous()
    k_bnsd = k_bsnd.transpose(1, 2).contiguous()
    v_bnsd = v_bsnd.transpose(1, 2).contiguous()
    g_bnsd = g_bsnd.transpose(1, 2).contiguous()
    beta_bns = beta_bsn.transpose(1, 2).contiguous()

    gk_bnsd = torch.ops._C_ascend.kda_gate_cumsum(g_bnsd, 64, layout="BNSD")
    got = torch.ops._C_ascend.chunk_kda_fwd(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        g_bnsd,
        beta_bns,
        scale,
        64,
        layout="BNSD",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    gk_bsnd = gk_bnsd.transpose(1, 2).contiguous()
    ref = chunk_kda_forward_reference(
        q_bsnd.cpu(),
        k_bsnd.cpu(),
        v_bsnd.cpu(),
        gk_bsnd.cpu(),
        beta_bsn.cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.cpu(),
        output_final_state=True,
    )

    out_bsnd = got[0]
    assert torch.isfinite(out_bsnd).all().item()
    assert torch.isfinite(got[1]).all().item()
    _assert_close("o", out_bsnd, ref.o)
    _assert_close("final_state", got[1], ref.final_state)
    _cleanup_npu()


def _canonical_chunk_indices(cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return None
    return [
        value
        for seq_id, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        for chunk_id in range((end - start + chunk_size - 1) // chunk_size)
        for value in (seq_id, chunk_id)
    ]


def _run_chunk_kda_fwd_a5_case(
    layout,
    tokens,
    batch_size,
    query_heads,
    value_heads,
    key_dim,
    value_dim,
    chunk_size,
    dtype,
    cu_seqlens,
):
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    if cu_seqlens is not None:
        assert cu_seqlens[0] == 0
        assert cu_seqlens[-1] == tokens
    is_tnd = layout == "TND"
    q_shape = (tokens, query_heads, key_dim) if is_tnd else (batch_size, tokens, query_heads, key_dim)
    v_shape = (tokens, value_heads, value_dim) if is_tnd else (batch_size, tokens, value_heads, value_dim)
    g_shape = (tokens, value_heads, key_dim) if is_tnd else (batch_size, tokens, value_heads, key_dim)
    beta_shape = (tokens, value_heads) if is_tnd else (batch_size, tokens, value_heads)
    q = torch.full(
        q_shape,
        1.0 / math.sqrt(key_dim),
        dtype=dtype,
        device=device,
    )
    k = torch.full_like(q, 1.0 / math.sqrt(key_dim))
    v = torch.zeros(v_shape, dtype=dtype, device=device)
    raw_gate = torch.full(
        g_shape,
        -0.005 * math.log(2.0),
        dtype=torch.float32,
        device=device,
    )
    beta_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    beta = torch.full(beta_shape, 0.5, dtype=beta_dtype, device=device)
    chunk_indices = _canonical_chunk_indices(cu_seqlens, chunk_size)

    torch.npu.synchronize()
    outputs = torch.ops._C_ascend.chunk_kda_fwd(
        q,
        k,
        v,
        raw_gate,
        beta,
        key_dim**-0.5,
        chunk_size,
        layout=layout,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        disable_recompute=False,
        return_intermediate_states=False,
    )
    torch.npu.synchronize()

    sequence_count = len(cu_seqlens) - 1 if cu_seqlens is not None else batch_size
    expected_output_shape = (tokens, value_heads, value_dim) if is_tnd else (batch_size, tokens, value_heads, value_dim)
    expected_gk_shape = (value_heads, tokens, key_dim) if is_tnd else (batch_size, value_heads, tokens, key_dim)
    assert len(outputs) == len(CHUNK_KDA_OUTPUT_NAMES)
    assert outputs[0].shape == expected_output_shape
    assert outputs[1].shape == (sequence_count, value_heads, key_dim, value_dim)
    assert outputs[2].shape == expected_gk_shape
    assert torch.count_nonzero(outputs[0]).item() == 0
    assert torch.count_nonzero(outputs[1]).item() == 0
    assert torch.isfinite(outputs[2]).all().item()

    _cleanup_npu()


@pytest.mark.parametrize(
    ("layout", "cu_seqlens"),
    [
        pytest.param("BSND", None, id="BSND-dense"),
        pytest.param("TND", [0, 2047, 4096, 8191], id="TND-varlen"),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda_fwd_a5_profile_t8191(layout, cu_seqlens):
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    _run_chunk_kda_fwd_a5_case(
        layout=layout,
        tokens=8191,
        batch_size=1,
        query_heads=16,
        value_heads=32,
        key_dim=128,
        value_dim=128,
        chunk_size=64,
        dtype=torch.bfloat16,
        cu_seqlens=cu_seqlens,
    )


@pytest.mark.parametrize(
    (
        "layout",
        "tokens",
        "batch_size",
        "query_heads",
        "value_heads",
        "key_dim",
        "value_dim",
        "chunk_size",
        "dtype",
        "cu_seqlens",
    ),
    [
        pytest.param(
            "BSND",
            600,
            1,
            6,
            12,
            128,
            256,
            128,
            torch.bfloat16,
            [0, 127, 383, 600],
            id="BSND-varlen-bf16",
        ),
        pytest.param("TND", 257, 1, 6, 12, 128, 256, 128, torch.bfloat16, None, id="TND-dense-bf16"),
        pytest.param(
            "TND",
            300,
            1,
            6,
            12,
            128,
            128,
            64,
            torch.bfloat16,
            [0, 63, 191, 300],
            id="TND-varlen-bf16",
        ),
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda_fwd_a5_generalized_layouts(
    layout,
    tokens,
    batch_size,
    query_heads,
    value_heads,
    key_dim,
    value_dim,
    chunk_size,
    dtype,
    cu_seqlens,
):
    if not _is_ascend_950():
        pytest.skip("requires an Ascend 950 device")

    _run_chunk_kda_fwd_a5_case(
        layout=layout,
        tokens=tokens,
        batch_size=batch_size,
        query_heads=query_heads,
        value_heads=value_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk_size,
        dtype=dtype,
        cu_seqlens=cu_seqlens,
    )
