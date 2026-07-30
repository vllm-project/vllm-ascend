# SPDX-License-Identifier: Apache-2.0
"""E2E correctness test for AscendC fused_gdn_decode kernel.

Validates torch.ops._C_ascend.npu_fused_gdn_decode against a pure PyTorch
golden reference. The custom op updates state in-place, so both output and
state are checked.
"""

import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.set_compile_mode(jit_compile=False)
enable_custom_op()

SEED = 42


def _softplus_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(x <= threshold, torch.log1p(torch.exp(x)), x)


def _split_mixed_qkv(
    mixed_qkv: torch.Tensor,
    h: int,
    hv: int,
    k: int,
    v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, key, value = torch.split(mixed_qkv, [h * k, h * k, hv * v], dim=-1)
    return q.view(-1, h, k), key.view(-1, h, k), value.view(-1, hv, v)


def _golden_fused_gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float,
    softplus_threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure torch golden for one-token-per-request GDN decode."""
    batch = mixed_qkv.shape[0]
    hv = state.shape[1]
    v = state.shape[2]
    k = state.shape[3]
    h = (mixed_qkv.shape[1] - hv * v) // (2 * k)
    heads_per_h = hv // h

    q, key, value = _split_mixed_qkv(mixed_qkv, h, hv, k, v)
    q = torch.nn.functional.normalize(q.to(torch.float32), p=2, dim=-1) * scale
    key = torch.nn.functional.normalize(key.to(torch.float32), p=2, dim=-1)
    value = value.to(torch.float32)

    g = -torch.exp(A_log.to(torch.float32).unsqueeze(0)) * _softplus_threshold(
        a.to(torch.float32) + dt_bias.to(torch.float32).unsqueeze(0),
        softplus_threshold,
    )
    exp_g = torch.exp(g)
    beta = torch.sigmoid(b.to(torch.float32))

    state_ref = state.clone().to(torch.float32)
    out = torch.zeros(batch, 1, hv, v, dtype=torch.float32)

    for batch_idx in range(batch):
        state_idx = int(ssm_state_indices[batch_idx].item())
        if state_idx <= 0:
            continue
        for hv_idx in range(hv):
            h_idx = hv_idx // heads_per_h
            q_i = q[batch_idx, h_idx]
            k_i = key[batch_idx, h_idx]
            v_i = value[batch_idx, hv_idx]
            state_i = state_ref[state_idx, hv_idx]

            state_i.mul_(exp_g[batch_idx, hv_idx])
            delta = (v_i - torch.mv(state_i, k_i)) * beta[batch_idx, hv_idx]
            state_i.add_(delta[:, None] * k_i[None, :])
            out[batch_idx, 0, hv_idx] = torch.mv(state_i, q_i)

    return out.to(mixed_qkv.dtype), state_ref.to(state.dtype)


def _make_inputs(
    batch: int,
    h: int,
    hv: int,
    k: int,
    v: int,
    slots: int,
    mixed_dtype: torch.dtype,
    state_dtype: torch.dtype,
    include_zero_index: bool,
):
    torch.manual_seed(SEED)
    packed_dim = 2 * h * k + hv * v
    mixed_qkv = torch.randn(batch, packed_dim, dtype=mixed_dtype) * 0.01
    a = torch.randn(batch, hv, dtype=mixed_dtype) * 0.01
    b = torch.randn(batch, hv, dtype=mixed_dtype) * 0.01
    A_log = torch.randn(hv, dtype=torch.float32) * 0.01
    dt_bias = torch.randn(hv, dtype=torch.float32) * 0.01
    state = torch.randn(slots, hv, v, k, dtype=state_dtype) * 0.01

    indices = torch.randperm(slots - 1, dtype=torch.int64)[:batch] + 1
    if include_zero_index:
        indices[0] = 0
    return mixed_qkv, a, b, A_log, dt_bias, state, indices


def _run_npu_op(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scale: float,
):
    state_npu = state.npu()
    out = torch.ops._C_ascend.npu_fused_gdn_decode(
        mixed_qkv.npu(),
        a.npu(),
        b.npu(),
        A_log.npu(),
        dt_bias.npu(),
        state_npu,
        ssm_state_indices.npu(),
        scale,
        20.0,
    )
    torch.npu.synchronize()
    return out.cpu(), state_npu.cpu()


@pytest.mark.parametrize(
    (
        "batch",
        "h",
        "hv",
        "k",
        "v",
        "mixed_dtype",
        "state_dtype",
        "include_zero_index",
    ),
    [
        (1, 8, 16, 128, 128, torch.bfloat16, torch.float32, False),
        (4, 8, 16, 128, 128, torch.bfloat16, torch.bfloat16, True),
        (8, 16, 32, 128, 128, torch.float16, torch.float32, False),
    ],
)
def test_fused_gdn_decode_vs_reference(
    batch,
    h,
    hv,
    k,
    v,
    mixed_dtype,
    state_dtype,
    include_zero_index,
):
    slots = batch + 2
    scale = k**-0.5
    inputs = _make_inputs(
        batch,
        h,
        hv,
        k,
        v,
        slots,
        mixed_dtype,
        state_dtype,
        include_zero_index,
    )
    mixed_qkv, a, b, A_log, dt_bias, state, ssm_state_indices = inputs

    ref_out, ref_state = _golden_fused_gdn_decode(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        ssm_state_indices,
        scale,
    )
    npu_out, npu_state = _run_npu_op(
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        state,
        ssm_state_indices,
        scale,
    )

    torch.testing.assert_close(
        npu_out.to(torch.float32),
        ref_out.to(torch.float32),
        rtol=1e-2,
        atol=5e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        npu_state.to(torch.float32),
        ref_state.to(torch.float32),
        rtol=1e-2,
        atol=5e-2,
        equal_nan=True,
    )

    touched = torch.zeros(slots, dtype=torch.bool)
    touched[ssm_state_indices[ssm_state_indices > 0]] = True
    torch.testing.assert_close(
        npu_state[~touched].to(torch.float32),
        state[~touched].to(torch.float32),
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    if include_zero_index:
        assert torch.count_nonzero(npu_out[0]) == 0

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
