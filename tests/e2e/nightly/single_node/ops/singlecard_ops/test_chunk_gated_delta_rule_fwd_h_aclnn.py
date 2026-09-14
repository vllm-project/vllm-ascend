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

import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()

CHUNK_SIZE = 64
DETERMINISM_REPEATS = 20
FWD_H_OUTPUT_NAMES = ("h", "v_new", "final_state")


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _prepare_chunk_offsets(cu_seqlens, chunk_size):
    if cu_seqlens is None:
        return None

    num_chunks = 0
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        num_chunks += math.ceil((seq_end - seq_start) / chunk_size)
    return [0, 0] * num_chunks


def _make_cumulative_gate(shape_batch, v_num_head, seqlen, chunk_size, cu_seqlens):
    torch.manual_seed(20260720 + seqlen + v_num_head)

    g = -torch.rand(shape_batch, v_num_head, seqlen, dtype=torch.float32) * 0.05
    if cu_seqlens is None:
        cu_seqlens = [0, seqlen]

    for batch_idx in range(shape_batch):
        for head_idx in range(v_num_head):
            for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                for chunk_start in range(seq_start, seq_end, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, seq_end)
                    g[batch_idx, head_idx, chunk_start:chunk_end] = torch.cumsum(
                        g[batch_idx, head_idx, chunk_start:chunk_end],
                        dim=0,
                    )
    return g


def _make_inputs(batch, seqlen, k_num_head, v_num_head, k_dim, v_dim, dtype, is_varlen):
    torch.manual_seed(20260720 + batch + seqlen + k_num_head + v_num_head + v_dim)

    shape_batch = 1 if is_varlen else batch
    cu_seqlens = [0, seqlen // 3, seqlen] if is_varlen else None
    k = torch.randn(shape_batch, k_num_head, seqlen, k_dim, dtype=dtype) * 0.04
    w = torch.randn(shape_batch, v_num_head, seqlen, k_dim, dtype=dtype) * 0.04
    u = torch.randn(shape_batch, v_num_head, seqlen, v_dim, dtype=dtype) * 0.04
    g = _make_cumulative_gate(shape_batch, v_num_head, seqlen, CHUNK_SIZE, cu_seqlens)
    chunk_indices = _prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
    return k, w, u, g, cu_seqlens, chunk_indices


def _chunk_gated_delta_rule_fwd_h_reference(k, w, u, g, chunk_size, cu_seqlens):
    dtype = k.dtype
    k = k.float()
    w = w.float()
    u = u.float()
    g = g.float()

    shape_batch, k_num_head, seqlen, k_dim = k.shape
    v_num_head, v_dim = u.shape[1], u.shape[3]
    head_ratio = v_num_head // k_num_head
    if cu_seqlens is None:
        cu_seqlens = [0, seqlen]
        num_sequences = shape_batch
        num_chunks = (seqlen + chunk_size - 1) // chunk_size
    else:
        num_sequences = len(cu_seqlens) - 1
        num_chunks = sum(
            math.ceil((seq_end - seq_start) / chunk_size) for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:])
        )

    h = torch.zeros(shape_batch, v_num_head, num_chunks, k_dim, v_dim, dtype=torch.float32)
    v_new = torch.zeros(shape_batch, v_num_head, seqlen, v_dim, dtype=torch.float32)

    for seq_idx in range(num_sequences):
        shape_batch_idx = 0 if len(cu_seqlens) > 2 else seq_idx
        seq_start = 0 if len(cu_seqlens) == 2 and shape_batch > 1 else cu_seqlens[seq_idx]
        seq_end = seqlen if len(cu_seqlens) == 2 and shape_batch > 1 else cu_seqlens[seq_idx + 1]
        chunk_base = 0
        if len(cu_seqlens) > 2:
            chunk_base = sum(
                math.ceil((end - start) / chunk_size)
                for start, end in zip(cu_seqlens[:seq_idx], cu_seqlens[1 : seq_idx + 1])
            )
        seq_chunks = math.ceil((seq_end - seq_start) / chunk_size)

        for v_head_idx in range(v_num_head):
            k_head_idx = v_head_idx // head_ratio
            for chunk_idx in range(seq_chunks):
                token_start = seq_start + chunk_idx * chunk_size
                actual_len = min(chunk_size, seq_end - token_start)
                h_idx = chunk_base + chunk_idx

                k_sel = torch.zeros(chunk_size, k_dim, dtype=torch.float32)
                w_sel = torch.zeros(chunk_size, k_dim, dtype=torch.float32)
                u_sel = torch.zeros(chunk_size, v_dim, dtype=torch.float32)
                g_sel = torch.zeros(chunk_size, dtype=torch.float32)
                token_slice = slice(token_start, token_start + actual_len)
                k_sel[:actual_len] = k[shape_batch_idx, k_head_idx, token_slice]
                w_sel[:actual_len] = w[shape_batch_idx, v_head_idx, token_slice]
                u_sel[:actual_len] = u[shape_batch_idx, v_head_idx, token_slice]
                g_sel[:actual_len] = g[shape_batch_idx, v_head_idx, token_slice]

                current_h = h[shape_batch_idx, v_head_idx, h_idx]
                v_work = u_sel - w_sel @ current_h
                if chunk_idx != seq_chunks - 1:
                    gate = (g_sel[actual_len - 1] - g_sel).exp().unsqueeze(-1)
                    h[shape_batch_idx, v_head_idx, h_idx + 1] = current_h * g_sel[
                        actual_len - 1
                    ].exp() + k_sel.transpose(-1, -2) @ (v_work * gate)
                v_new[shape_batch_idx, v_head_idx, token_slice] = v_work[:actual_len]

    return h.to(dtype), v_new.to(dtype)


def _chunk_gated_delta_rule_fwd_h_kda_reference(k, w, u, gk, initial_state, chunk_size):
    dtype = k.dtype
    k = k.float()
    w = w.float()
    u = u.float()
    gk = gk.float()
    state = initial_state.float().clone()

    batch, k_num_head, seqlen, k_dim = k.shape
    v_num_head, v_dim = u.shape[1], u.shape[3]
    head_ratio = v_num_head // k_num_head
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    h = torch.zeros(batch, v_num_head, num_chunks, k_dim, v_dim, dtype=torch.float32)
    v_new = torch.zeros(batch, v_num_head, seqlen, v_dim, dtype=torch.float32)

    for batch_idx in range(batch):
        for v_head_idx in range(v_num_head):
            k_head_idx = v_head_idx // head_ratio
            for chunk_idx, token_start in enumerate(range(0, seqlen, chunk_size)):
                token_end = min(token_start + chunk_size, seqlen)
                token_slice = slice(token_start, token_end)
                current_h = state[batch_idx, v_head_idx]
                h[batch_idx, v_head_idx, chunk_idx] = current_h
                v_work = u[batch_idx, v_head_idx, token_slice] - w[batch_idx, v_head_idx, token_slice] @ current_h
                v_new[batch_idx, v_head_idx, token_slice] = v_work
                last_gk = gk[batch_idx, v_head_idx, token_end - 1]
                state[batch_idx, v_head_idx] = (
                    torch.exp2(last_gk)[:, None] * current_h
                    + k[batch_idx, k_head_idx, token_slice].transpose(-1, -2) @ v_work
                )

    return h.to(dtype), v_new.to(dtype), state.to(initial_state.dtype)


def _assert_cosine_close(name, actual, expected, threshold=0.99):
    actual = actual.detach().cpu().float().flatten()
    expected = expected.detach().cpu().float().flatten()
    if actual.norm() == 0 and expected.norm() == 0:
        return

    cosine = torch.nn.functional.cosine_similarity(actual.unsqueeze(0), expected.unsqueeze(0)).item()
    assert cosine >= threshold, f"{name} cosine={cosine:.6f}, expected >= {threshold}"


def _snapshot_outputs(outputs):
    torch.npu.synchronize()
    return tuple(output.detach().cpu().contiguous() for output in outputs)


def _assert_outputs_bitwise_equal(reference, actual, repeat):
    assert len(reference) == len(actual) == len(FWD_H_OUTPUT_NAMES)
    for name, expected, current in zip(FWD_H_OUTPUT_NAMES, reference, actual):
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


@pytest.mark.parametrize(
    ("batch", "seqlen", "k_num_head", "v_num_head", "k_dim", "v_dim", "dtype", "is_varlen"),
    [
        (1, 128, 1, 1, 128, 128, torch.float16, False),
        (1, 128, 1, 2, 128, 256, torch.float16, False),
        (1, 128, 2, 2, 128, 256, torch.bfloat16, False),
        (2, 96, 1, 2, 128, 256, torch.float16, True),
    ],
)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_h_matches_reference(
    batch,
    seqlen,
    k_num_head,
    v_num_head,
    k_dim,
    v_dim,
    dtype,
    is_varlen,
):
    k, w, u, g, cu_seqlens, chunk_indices = _make_inputs(
        batch,
        seqlen,
        k_num_head,
        v_num_head,
        k_dim,
        v_dim,
        dtype,
        is_varlen,
    )
    expected_h, expected_v = _chunk_gated_delta_rule_fwd_h_reference(
        k,
        w,
        u,
        g,
        CHUNK_SIZE,
        cu_seqlens,
    )

    h_out, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k.npu(),
        w.npu(),
        u.npu(),
        g=g.npu(),
        output_final_state=False,
        chunk_size=CHUNK_SIZE,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    assert final_state is None
    assert torch.isfinite(h_out).all().item()
    assert torch.isfinite(v_new).all().item()
    _assert_cosine_close("h_out", h_out, expected_h)
    _assert_cosine_close("v_new", v_new, expected_v)
    _cleanup_npu()


@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_h_kda_is_bitwise_deterministic():
    batch, seqlen, k_num_head, v_num_head, k_dim, v_dim = 1, 128, 2, 2, 128, 256
    dtype = torch.bfloat16
    torch.manual_seed(20260720 + seqlen + k_num_head + v_num_head + v_dim)

    k = torch.randn(batch, k_num_head, seqlen, k_dim, dtype=dtype) * 0.04
    w = torch.randn(batch, v_num_head, seqlen, k_dim, dtype=dtype) * 0.04
    u = torch.randn(batch, v_num_head, seqlen, v_dim, dtype=dtype) * 0.04
    raw_gate = -torch.rand(batch, v_num_head, seqlen, k_dim, dtype=torch.float32) * 0.04
    gk = torch.empty_like(raw_gate)
    rcp_ln2 = 1.4426950408889634
    for token_start in range(0, seqlen, CHUNK_SIZE):
        token_end = min(token_start + CHUNK_SIZE, seqlen)
        gk[:, :, token_start:token_end] = torch.cumsum(
            raw_gate[:, :, token_start:token_end] * rcp_ln2,
            dim=2,
        )
    initial_state = torch.randn(batch, v_num_head, k_dim, v_dim, dtype=torch.float32) * 0.01
    expected = _chunk_gated_delta_rule_fwd_h_kda_reference(
        k,
        w,
        u,
        gk,
        initial_state,
        CHUNK_SIZE,
    )

    k_npu = k.npu()
    w_npu = w.npu()
    u_npu = u.npu()
    gk_npu = gk.npu()
    initial_state_npu = initial_state.npu()

    def run_fwd_h():
        return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
            k_npu,
            w_npu,
            u_npu,
            gk=gk_npu,
            initial_state=initial_state_npu,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            save_new_value=True,
        )

    run_fwd_h()
    torch.npu.synchronize()
    got = run_fwd_h()
    reference_outputs = _snapshot_outputs(got)
    for repeat in range(1, DETERMINISM_REPEATS):
        current_outputs = _snapshot_outputs(run_fwd_h())
        _assert_outputs_bitwise_equal(reference_outputs, current_outputs, repeat)

    for name, actual, expected_output in zip(FWD_H_OUTPUT_NAMES, got, expected):
        assert torch.isfinite(actual).all().item()
        _assert_cosine_close(name, actual, expected_output)
    _cleanup_npu()
