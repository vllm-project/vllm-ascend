#
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
#

"""Tests for lightning attention triton kernels.

Covers the following 4 triton kernels:
  - ``_fwd_diag_kernel``: diagonal block causal attention
  - ``_fwd_kv_parallel``: key-value outer product per block
  - ``_fwd_kv_reduce``: prefix-sum reduction of KV across blocks
  - ``_fwd_none_diag_kernel``: non-diagonal block attention

All kernels are exercised through the public APIs:
  - ``lightning_attention_npu_`` (``_attention.apply``, single d-chunk)
  - ``lightning_attention_npu`` (full function with d-dimension chunking)
  - ``AscendLightningAttentionKernel.jit_linear_forward_prefix``

The naive reference implementations replicate the exact triton tiling
algorithm (diagonal blocks -> KV parallel -> KV reduce -> non-diagonal)
to ensure an apples-to-apples numerical comparison. Low-precision inputs use
bounded random values and mirror the kernel's output stores to avoid comparing
against an unrealistically precise PyTorch recurrence.

The production BailingMoE path promotes QKV to float32 before calling these
kernels and commonly uses a float32 Mamba cache. Therefore larger accuracy
cases use float32, while bf16/fp16 cases are kept as bounded compatibility
coverage for the raw operator entry points.
"""

import gc

import pytest
import torch
from einops import rearrange

from vllm_ascend.ops.triton.mamba.lightning_attn import (
    AscendLightningAttentionKernel,
    lightning_attention_npu,
    lightning_attention_npu_,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

# ---------------------------------------------------------------------------
# Naive reference implementations (pure PyTorch, no Triton)
# ---------------------------------------------------------------------------

# The triton lightning attention algorithm processes the sequence in BLOCK-sized
# tiles and follows these steps:
#   1. _fwd_diag_kernel   : causal attention within each diagonal block
#   2. _fwd_kv_parallel   : per-block KV outer product, decayed to block end
#   3. _fwd_kv_reduce     : prefix-sum across blocks, updates kv_history
#   4. _fwd_none_diag_kernel : non-diagonal attention using prefix KV
#
# IMPORTANT: The non-diagonal kernel applies decay as exp(-s * t_in_block).
# The references below mirror that tiled kernel convention directly rather
# than using an equivalent-looking recurrence with a different history state.

BLOCK_SIZE = 256
DIAG_BLOCK_SIZE = 32
KV_BLOCK_SIZE = 64
TEST_INPUT_SCALE = 0.02
TEST_DECAY_SCALE = 0.02
SEMANTIC_INPUT_SCALE = 0.05
FLOAT32_TOLERANCE = (1e-2, 1e-2)
LOW_PRECISION_TOLERANCE = (5e-2, 5e-2)


def _round_like_kernel_store(x, dtype):
    """Mirror Triton stores to output dtype, then compare in float32."""
    if dtype == torch.float32:
        return x
    return x.to(dtype).float()


def _randn(shape, dtype, device, scale=TEST_INPUT_SCALE):
    return (torch.randn(*shape, dtype=dtype, device=device) * scale).to(dtype)


def _rand_decay(h, device, scale=TEST_DECAY_SCALE):
    return torch.rand(h, dtype=torch.float32, device=device) * scale


def _naive_triton_lightning_attention(q, k, v, s, kv_history, block_size=BLOCK_SIZE):
    """Step-by-step replication of the triton tiling algorithm for a single
    d-chunk.

    This function mirrors the four-kernel pipeline used in
    ``_attention.apply`` (a.k.a. ``lightning_attention_npu_``):
      1. Diagonal blocks (within-block causal attention)
      2. Per-block KV outer product (decayed to block end)
      3. Prefix-sum reduce across blocks (updates kv_history in place)
      4. Non-diagonal blocks (cross-block attention with prefix KV)

    Args:
        q: [b, h, n, d]  queries
        k: [b, h, n, d]  keys
        v: [b, h, n, e]  values
        s: [1, h, 1, 1]  per-head decay rates
        kv_history: [b, h, d, e]  accumulated KV state from previous steps

    Returns:
        (o, kv_return) where
          o:         [b, h, n, e]  attention output
          kv_return: [b, h, num_blocks + 1, d, e]  block KV plus final state
    """
    output_dtype = q.dtype
    b, h, n, d = q.shape
    e_dim = v.shape[-1]

    q = q.float()
    k = k.float()
    v = v.float()
    decay_rate = s.float().reshape(1, h, 1, 1)

    num_blocks = (n + block_size - 1) // block_size

    # ---- Step 1: Diagonal blocks ----
    # Mirror _fwd_diag_kernel's tiled matmul path. The recurrence is
    # mathematically equivalent, but fp16/bf16 accumulation order differs.
    o = torch.zeros(b, h, n, e_dim, dtype=torch.float32, device=q.device)
    for block_idx in range(num_blocks):
        blk_start = block_idx * block_size
        blk_end = min(blk_start + block_size, n)
        block_len = blk_end - blk_start
        for q_start in range(0, block_len, DIAG_BLOCK_SIZE):
            q_end = min(q_start + DIAG_BLOCK_SIZE, block_len)
            q_block = q[:, :, blk_start + q_start : blk_start + q_end, :]
            q_pos = torch.arange(q_start, q_end, dtype=torch.float32, device=q.device)
            q_len = q_end - q_start
            qkv = torch.zeros(b, h, q_len, e_dim, dtype=torch.float32, device=q.device)

            for kv_start in range(0, q_start + DIAG_BLOCK_SIZE, DIAG_BLOCK_SIZE):
                kv_end = min(kv_start + DIAG_BLOCK_SIZE, block_len)
                if kv_start >= kv_end:
                    continue

                k_block = k[:, :, blk_start + kv_start : blk_start + kv_end, :]
                v_block = v[:, :, blk_start + kv_start : blk_start + kv_end, :]
                kv_pos = torch.arange(kv_start, kv_end, dtype=torch.float32, device=q.device)
                kv_len = kv_end - kv_start
                diff = q_pos[:, None] - kv_pos[None, :]
                causal_mask = (diff >= 0).reshape(1, 1, q_len, kv_len)
                decay = torch.exp(-decay_rate * diff.clamp_min(0).reshape(1, 1, q_len, kv_len))
                decay = decay * causal_mask.to(torch.float32)

                qk = torch.matmul(q_block, k_block.transpose(-1, -2)) * decay
                qkv = qkv + torch.matmul(qk, v_block)

            o[:, :, blk_start + q_start : blk_start + q_end, :] = _round_like_kernel_store(qkv, output_dtype)

    # ---- Step 2: Per-block KV outer product ----
    # For each block, accumulate K^T @ V with each row decayed to the end
    # of that block. The last partial block uses the same left-shifted
    # CBLOCK layout as _fwd_kv_parallel.
    kv_block = torch.zeros(b, h, num_blocks, d, e_dim, dtype=torch.float32, device=q.device)
    for block_idx in range(num_blocks):
        blk_start = block_idx * block_size
        blk_end = min(blk_start + block_size, n)
        block_len = blk_end - blk_start
        num_kv_blocks = min((block_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE, block_size // KV_BLOCK_SIZE)
        left_shift = num_kv_blocks * KV_BLOCK_SIZE - block_len
        decay_start = (block_size // KV_BLOCK_SIZE - num_kv_blocks) * KV_BLOCK_SIZE

        for kv_block_idx in range(num_kv_blocks):
            row_offsets = torch.arange(KV_BLOCK_SIZE, device=q.device)
            source_pos = kv_block_idx * KV_BLOCK_SIZE - left_shift + row_offsets
            left_bound = (1 - kv_block_idx) * left_shift
            valid = (row_offsets >= left_bound) & (source_pos >= 0) & (source_pos < block_len)
            safe_pos = source_pos.clamp(0, max(block_len - 1, 0))
            k_block = k[:, :, blk_start + safe_pos, :] * valid.reshape(1, 1, KV_BLOCK_SIZE, 1)
            v_block = v[:, :, blk_start + safe_pos, :] * valid.reshape(1, 1, KV_BLOCK_SIZE, 1)
            decay_pos = decay_start + kv_block_idx * KV_BLOCK_SIZE + row_offsets
            k_decay = torch.exp(-decay_rate * (block_size - 1 - decay_pos).float().reshape(1, 1, KV_BLOCK_SIZE, 1))
            weighted_k = k_block * k_decay
            kv_block[:, :, block_idx, :, :] = kv_block[:, :, block_idx, :, :] + torch.matmul(
                weighted_k.transpose(-1, -2),
                v_block,
            )

    # ---- Step 3: Prefix-sum reduce across blocks ----
    # Replicates _fwd_kv_reduce exactly:
    #   kv_pre starts as the existing kv_history
    #   For each block i:
    #     block_decay = exp(-s * block_size)
    #     store kv_pre into KV[i]
    #     kv_pre = block_decay * kv_pre + kv_cur
    kv = kv_block.clone()
    kv_pre = kv_history.clone().float()
    for i in range(num_blocks):
        blk_size = min(n - i * block_size, block_size)
        block_decay = torch.exp(-decay_rate * blk_size)  # [1, h, 1, 1]
        kv_cur = kv[:, :, i, :, :].clone()
        kv[:, :, i, :, :] = kv_pre
        kv_pre = block_decay * kv_pre + kv_cur
    kv_history_updated = kv_pre

    # ---- Step 4: Non-diagonal blocks ----
    # O[t] += Q[t] @ kv[block_idx] * exp(-s * t_local)
    # Note: the triton kernel uses t_local (NOT t_local + 1),
    # matching _fwd_none_diag_kernel's q_decay = exp(-s * (off_c*CBLOCK + c)).
    for block_idx in range(num_blocks):
        blk_start = block_idx * block_size
        blk_end = min(blk_start + block_size, n)
        block_len = blk_end - blk_start
        for q_start in range(0, block_len, KV_BLOCK_SIZE):
            q_end = min(q_start + KV_BLOCK_SIZE, block_len)
            q_len = q_end - q_start
            q_block = q[:, :, blk_start + q_start : blk_start + q_end, :]
            q_pos = torch.arange(q_start, q_end, dtype=torch.float32, device=q.device)
            q_decay = torch.exp(-decay_rate * q_pos.reshape(1, 1, q_len, 1))
            nondiag = torch.matmul(q_block, kv[:, :, block_idx, :, :]) * q_decay
            out_slice = o[:, :, blk_start + q_start : blk_start + q_end, :] + nondiag
            o[:, :, blk_start + q_start : blk_start + q_end, :] = _round_like_kernel_store(out_slice, output_dtype)

    return _round_like_kernel_store(o, output_dtype), torch.cat([kv, kv_history_updated.unsqueeze(2)], dim=2)


def _naive_lightning_attention_npu(q, k, v, ed, block_size, kv_history):
    """Naive reference that replicates ``lightning_attention_npu``.

    Handles the d-dimension chunking in the same way as the real
    implementation so that the comparison is apples-to-apples.
    """
    d = q.shape[-1]
    e_dim = v.shape[-1]

    if ed.dim() == 1:
        ed = ed.view(1, -1, 1, 1)

    m = 128 if d >= 128 else 64
    arr = [m * i for i in range(d // m + 1)]
    if arr[-1] != d:
        arr.append(d)

    if kv_history is None:
        kv_history = torch.zeros(
            (q.shape[0], q.shape[1], d, e_dim),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        kv_history = kv_history.clone().contiguous().float()

    output = torch.zeros(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        e_dim,
        dtype=torch.float32,
        device=q.device,
    )

    kv_state = None
    for i in range(len(arr) - 1):
        s_idx = arr[i]
        e_idx = arr[i + 1]
        q1 = q[..., s_idx:e_idx]
        k1 = k[..., s_idx:e_idx]
        kv_history_chunk = kv_history[:, :, s_idx:e_idx, :]
        o, kv_state = _naive_triton_lightning_attention(
            q1,
            k1,
            v,
            ed,
            kv_history_chunk,
            block_size=block_size,
        )
        output = _round_like_kernel_store(output + o, q.dtype)
        kv_history[:, :, s_idx:e_idx, :] = kv_state[:, :, -1, :, :]

    return output, kv_state


def _naive_jit_linear_forward_prefix(q, k, v, kv_caches, slope_rate, block_size, layer_idx=None):
    """Naive reference for ``AscendLightningAttentionKernel.jit_linear_forward_prefix``."""
    slope_rate = slope_rate.to(torch.float32)
    should_squeeze = q.dim() == 3
    if should_squeeze:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    b, h, n, d = q.shape
    e_dim = v.shape[-1]

    if slope_rate.dim() == 1:
        ed = slope_rate.view(1, -1, 1, 1)
    else:
        ed = slope_rate

    kv_history = kv_caches.reshape(1, h, d, e_dim).contiguous().float()
    output, kv_state = _naive_lightning_attention_npu(
        q,
        k,
        v,
        ed,
        block_size,
        kv_history,
    )

    # The triton kernel updates kv_caches in-place with the final KV state
    kv_caches_out = kv_state[:, :, -1, :, :].reshape_as(kv_caches)

    assert output.shape[0] == 1, "batch size must be 1"
    result = rearrange(output.squeeze(0), "h n d -> n (h d)")
    return result.float(), kv_caches_out


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------


def _get_tolerances(dtype):
    """Return (rtol, atol) appropriate for the given dtype."""
    if dtype == torch.float32:
        return FLOAT32_TOLERANCE
    elif dtype in (torch.float16, torch.bfloat16):
        return LOW_PRECISION_TOLERANCE
    return FLOAT32_TOLERANCE


# ---------------------------------------------------------------------------
# Tests for lightning_attention_npu_  (single d-chunk, exercises all 4 kernels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"b{case[0]}-h{case[1]}-n{case[2]}-d{case[3]}-e{case[4]}-{str(case[5]).split('.')[-1]}")
        for case in [
            # Small seq, basic sanity
            (1, 4, 32, 128, 64, torch.bfloat16),
            # n < BLOCK (256), not aligned
            (1, 4, 100, 128, 128, torch.bfloat16),
            # float16 dtype
            (1, 4, 256, 128, 128, torch.float16),
            # float32 dtype
            (1, 4, 128, 128, 128, torch.float32),
            # batch > 1
            (2, 4, 128, 128, 64, torch.bfloat16),
            # d = 64 (smaller head dim)
            (1, 4, 128, 64, 64, torch.bfloat16),
            # e != d
            (1, 4, 128, 64, 128, torch.bfloat16),
        ]
    ],
)
def test_lightning_attention_npu_single_chunk(b, h, n, d, e, dtype):
    """Test lightning_attention_npu_ against naive PyTorch reference.

    This exercises all 4 triton kernels through the _attention.apply path.
    All n values are <= BLOCK (256) to ensure single-block correctness.
    """
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((b, h, n, d), dtype, device)
    k = _randn((b, h, n, d), dtype, device)
    v = _randn((b, h, n, e), dtype, device)
    ed = _rand_decay(h, device).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=device)

    # NOTE: Must clone kv_history before the triton call because
    # _fwd_kv_reduce modifies it in-place.  The naive reference must
    # receive the ORIGINAL (pre-modification) value.
    o_triton, kv_triton = lightning_attention_npu_(q, k, v, ed, kv_history.clone())
    o_ref, _ = _naive_triton_lightning_attention(q, k, v, ed, kv_history)

    torch.testing.assert_close(
        o_triton.float().cpu(),
        o_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    assert kv_triton.shape == (b, h, (n + BLOCK_SIZE - 1) // BLOCK_SIZE + 1, d, e)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    ("b", "h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"b{case[0]}-h{case[1]}-n{case[2]}-d{case[3]}-e{case[4]}-{str(case[5]).split('.')[-1]}")
        for case in [
            (1, 4, 256, 128, 128, torch.bfloat16),
            (2, 4, 128, 128, 64, torch.bfloat16),
        ]
    ],
)
def test_lightning_attention_npu_single_chunk_with_kv_history(b, h, n, d, e, dtype):
    """Test lightning_attention_npu_ with non-zero initial KV history."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((b, h, n, d), dtype, device)
    k = _randn((b, h, n, d), dtype, device)
    v = _randn((b, h, n, e), dtype, device)
    ed = _rand_decay(h, device).view(1, h, 1, 1)
    kv_history = _randn((b, h, d, e), torch.float32, device)

    o_triton, kv_triton = lightning_attention_npu_(q, k, v, ed, kv_history.clone())
    o_ref, kv_ref = _naive_triton_lightning_attention(q, k, v, ed, kv_history)

    torch.testing.assert_close(
        o_triton.float().cpu(),
        o_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_triton[:, :, -1, :, :].cpu(),
        kv_ref[:, :, -1, :, :].cpu(),
        rtol=rtol,
        atol=atol,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests for multi-block sequences  (n > BLOCK)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"b{case[0]}-h{case[1]}-n{case[2]}-d{case[3]}-e{case[4]}-{str(case[5]).split('.')[-1]}")
        for case in [
            # n > BLOCK, not aligned
            (1, 4, 300, 128, 128, torch.bfloat16),
            # Larger n, production path promotes q/k/v to float32.
            (1, 4, 768, 128, 128, torch.float32),
        ]
    ],
)
def test_lightning_attention_npu_multi_block(b, h, n, d, e, dtype):
    """Test lightning_attention_npu_ with multi-block sequences (n > BLOCK).

    This exercises the _fwd_kv_parallel, _fwd_kv_reduce, and
    _fwd_none_diag_kernel in the multi-block path.
    Uses small decay rates so inter-block numerical differences
    remain within tolerance.
    """
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((b, h, n, d), dtype, device)
    k = _randn((b, h, n, d), dtype, device)
    v = _randn((b, h, n, e), dtype, device)
    # Use small decay rates to keep errors within tolerance
    ed = _rand_decay(h, device, scale=0.01)
    ed = ed.view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=device)

    o_triton, kv_triton = lightning_attention_npu_(q, k, v, ed, kv_history.clone())
    o_ref, kv_ref = _naive_triton_lightning_attention(q, k, v, ed, kv_history)

    torch.testing.assert_close(
        o_triton.float().cpu(),
        o_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_triton[:, :, -1, :, :].cpu(),
        kv_ref[:, :, -1, :, :].cpu(),
        rtol=rtol,
        atol=atol,
    )

    # Also verify the output is well-formed
    assert not torch.isnan(o_triton).any(), "Output contains NaN values"
    assert not torch.isinf(o_triton).any(), "Output contains Inf values"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests for lightning_attention_npu  (full function with d-dimension chunking)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("b", "h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"b{case[0]}-h{case[1]}-n{case[2]}-d{case[3]}-e{case[4]}-{str(case[5]).split('.')[-1]}")
        for case in [
            # d=128 -> single chunk (m=128)
            (1, 4, 128, 128, 128, torch.bfloat16),
            # d=64  -> single chunk (m=64)
            (1, 4, 128, 64, 64, torch.bfloat16),
            # n not aligned to BLOCK, single-block
            (1, 4, 100, 128, 128, torch.bfloat16),
            # n == BLOCK, production path promotes q/k/v to float32.
            (1, 4, 256, 128, 128, torch.float32),
        ]
    ],
)
def test_lightning_attention_npu(b, h, n, d, e, dtype):
    """Test lightning_attention_npu (with d-dimension chunking) against naive reference."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((b, h, n, d), dtype, device)
    k = _randn((b, h, n, d), dtype, device)
    v = _randn((b, h, n, e), dtype, device)
    ed = _rand_decay(h, device)

    # Triton output
    o_triton, kv_triton = lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=None)

    # Naive reference output
    o_ref, kv_ref = _naive_lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=None)

    torch.testing.assert_close(
        o_triton.float().cpu(),
        o_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_triton[:, :, -1, :, :].cpu(),
        kv_ref[:, :, -1, :, :].cpu(),
        rtol=rtol,
        atol=atol,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    ("b", "h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"b{case[0]}-h{case[1]}-n{case[2]}-d{case[3]}-e{case[4]}-{str(case[5]).split('.')[-1]}")
        for case in [
            # Production path promotes q/k/v to float32 while cache is float32.
            (1, 4, 128, 128, 128, torch.float32),
            (1, 4, 256, 64, 128, torch.bfloat16),
        ]
    ],
)
def test_lightning_attention_npu_with_kv_history(b, h, n, d, e, dtype):
    """Test lightning_attention_npu with pre-existing KV history."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((b, h, n, d), dtype, device)
    k = _randn((b, h, n, d), dtype, device)
    v = _randn((b, h, n, e), dtype, device)
    ed = _rand_decay(h, device)
    kv_history = _randn((b, h, d, e), torch.float32, device)

    o_triton, kv_triton = lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=kv_history.clone())
    o_ref, kv_ref = _naive_lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=kv_history)

    torch.testing.assert_close(
        o_triton.float().cpu(),
        o_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_triton[:, :, -1, :, :].cpu(),
        kv_ref[:, :, -1, :, :].cpu(),
        rtol=rtol,
        atol=atol,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests for AscendLightningAttentionKernel.jit_linear_forward_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"h{case[0]}-n{case[1]}-d{case[2]}-e{case[3]}-{str(case[4]).split('.')[-1]}")
        for case in [
            (4, 128, 128, 128, torch.bfloat16),
            (8, 256, 128, 128, torch.float32),
            (4, 100, 128, 128, torch.bfloat16),
            (4, 256, 64, 64, torch.bfloat16),
            (4, 128, 128, 128, torch.float16),
        ]
    ],
)
def test_ascend_lightning_attention_kernel_prefix(h, n, d, e, dtype):
    """Test AscendLightningAttentionKernel.jit_linear_forward_prefix."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    # jit_linear_forward_prefix receives one sequence in [h, n, d] layout.
    q = _randn((h, n, d), dtype, device)
    k = _randn((h, n, d), dtype, device)
    v = _randn((h, n, e), dtype, device)

    slope_rate = _rand_decay(h, device)
    kv_caches = torch.zeros(h, d, e, dtype=torch.float32, device=device)

    # Triton output (clone all shared tensors)
    kv_caches_triton = kv_caches.clone()
    out_triton = AscendLightningAttentionKernel.jit_linear_forward_prefix(
        q.clone(), k.clone(), v.clone(), kv_caches_triton, slope_rate.clone(), block_size=256
    )

    # Naive reference output
    out_ref, kv_caches_ref = _naive_jit_linear_forward_prefix(q, k, v, kv_caches, slope_rate, block_size=256)

    torch.testing.assert_close(
        out_triton.float().cpu(),
        out_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_caches_triton.cpu(),
        kv_caches_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    ("h", "n", "d", "e", "dtype"),
    [
        pytest.param(*case, id=f"h{case[0]}-n{case[1]}-d{case[2]}-e{case[3]}-{str(case[4]).split('.')[-1]}")
        for case in [
            # Production path promotes q/k/v to float32 while cache is float32.
            (4, 128, 128, 128, torch.float32),
        ]
    ],
)
def test_ascend_lightning_attention_kernel_prefix_with_history(h, n, d, e, dtype):
    """Test AscendLightningAttentionKernel.jit_linear_forward_prefix with
    non-zero initial kv_caches."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    rtol, atol = _get_tolerances(dtype)

    q = _randn((h, n, d), dtype, device)
    k = _randn((h, n, d), dtype, device)
    v = _randn((h, n, e), dtype, device)
    slope_rate = _rand_decay(h, device)
    kv_caches = _randn((h, d, e), torch.float32, device)

    kv_caches_triton = kv_caches.clone()
    out_triton = AscendLightningAttentionKernel.jit_linear_forward_prefix(
        q.clone(), k.clone(), v.clone(), kv_caches_triton, slope_rate.clone(), block_size=256
    )

    out_ref, kv_caches_ref = _naive_jit_linear_forward_prefix(q, k, v, kv_caches, slope_rate, block_size=256)

    torch.testing.assert_close(
        out_triton.float().cpu(),
        out_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        kv_caches_triton.cpu(),
        kv_caches_ref.cpu(),
        rtol=rtol,
        atol=atol,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    ("h", "n", "d", "e"),
    [
        pytest.param(*case, id=f"h{case[0]}-n{case[1]}-d{case[2]}-e{case[3]}")
        for case in [
            # Multi-block sequence via prefix kernel
            (4, 512, 128, 128),
        ]
    ],
)
def test_ascend_lightning_attention_kernel_prefix_multi_block(h, n, d, e):
    """Smoke test the prefix wrapper with multi-block sequences."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"
    dtype = torch.bfloat16

    q = _randn((h, n, d), dtype, device)
    k = _randn((h, n, d), dtype, device)
    v = _randn((h, n, e), dtype, device)

    # Small decay rates for multi-block tolerance
    slope_rate = _rand_decay(h, device, scale=0.01)
    kv_caches = torch.zeros(h, d, e, dtype=torch.float32, device=device)

    kv_caches_triton = kv_caches.clone()
    out_triton = AscendLightningAttentionKernel.jit_linear_forward_prefix(
        q.clone(), k.clone(), v.clone(), kv_caches_triton, slope_rate.clone(), block_size=256
    )

    assert out_triton.shape == (n, h * e)
    assert not torch.isnan(out_triton).any(), "Output contains NaN values"
    assert not torch.isinf(out_triton).any(), "Output contains Inf values"
    assert kv_caches_triton.shape == (h, d, e)
    assert not torch.isnan(kv_caches_triton).any(), "KV cache contains NaN values"
    assert not torch.isinf(kv_caches_triton).any(), "KV cache contains Inf values"
    assert not torch.allclose(kv_caches_triton, kv_caches), "KV cache should be updated"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Output shape and basic sanity tests
# ---------------------------------------------------------------------------


def test_lightning_attention_npu_output_shapes():
    """Verify output tensor shapes match expected shapes."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"

    b, h, n, d, e = 1, 4, 256, 128, 128
    q = _randn((b, h, n, d), torch.bfloat16, device)
    k = _randn((b, h, n, d), torch.bfloat16, device)
    v = _randn((b, h, n, e), torch.bfloat16, device)
    ed = _rand_decay(h, device)

    o, kv = lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=None)

    assert o.shape == (b, h, n, e), f"Expected output shape {(b, h, n, e)}, got {o.shape}"
    assert kv.dim() == 5, f"Expected kv_return to be 5D, got {kv.dim()}D"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_ascend_kernel_prefix_output_shape():
    """Verify AscendLightningAttentionKernel.jit_linear_forward_prefix output shape."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"

    h, n, d, e = 4, 256, 128, 128
    q = _randn((h, n, d), torch.bfloat16, device)
    k = _randn((h, n, d), torch.bfloat16, device)
    v = _randn((h, n, e), torch.bfloat16, device)
    slope_rate = _rand_decay(h, device)
    kv_caches = torch.zeros(h, d, e, dtype=torch.float32, device=device)

    out = AscendLightningAttentionKernel.jit_linear_forward_prefix(q, k, v, kv_caches, slope_rate, block_size=256)

    expected_shape = (n, h * e)
    assert out.shape == expected_shape, f"Expected output shape {expected_shape}, got {out.shape}"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_lightning_attention_output_no_nan():
    """Verify the output does not contain NaN or Inf values."""
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"

    b, h, n, d, e = 1, 4, 256, 128, 128
    q = _randn((b, h, n, d), torch.bfloat16, device)
    k = _randn((b, h, n, d), torch.bfloat16, device)
    v = _randn((b, h, n, e), torch.bfloat16, device)
    ed = _rand_decay(h, device).view(1, h, 1, 1)

    o, _ = lightning_attention_npu_(q, k, v, ed, torch.zeros(b, h, d, e, dtype=torch.float32, device=device))

    assert not torch.isnan(o).any(), "Output contains NaN values"
    assert not torch.isinf(o).any(), "Output contains Inf values"

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_lightning_attention_causal_property():
    """Verify causal property: output at position t should not depend on
    keys/values at positions > t.

    If we modify V at positions after t, the output at position t should
    remain unchanged.  This tests the diagonal kernel's causal mask.
    Uses float32 for numerical precision.
    """
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"

    b, h, n, d, e = 1, 4, 128, 64, 64
    q = _randn((b, h, n, d), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    k = _randn((b, h, n, d), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    v = _randn((b, h, n, e), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    ed = _rand_decay(h, device).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=device)

    # Output with original V
    o_orig, _ = lightning_attention_npu_(q, k, v, ed, kv_history.clone())

    # Scramble V at positions after t=50
    v_scrambled = v.clone()
    v_scrambled[:, :, 50:, :] = _randn(v_scrambled[:, :, 50:, :].shape, torch.float32, device)

    # Output with scrambled V (should be identical at positions 0..49)
    o_scrambled, _ = lightning_attention_npu_(q, k, v_scrambled, ed, kv_history.clone())

    # Output at positions 0..49 should be unchanged
    torch.testing.assert_close(
        o_orig[:, :, :50, :].cpu(),
        o_scrambled[:, :, :50, :].cpu(),
        rtol=1e-5,
        atol=1e-5,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_lightning_attention_decay_effect():
    """Verify that different decay rates produce different outputs.

    With a larger decay rate, the output is more dominated by recent tokens.
    This test verifies the decay mechanism is active, using the naive
    reference as ground truth since both should produce valid outputs.
    """
    torch.manual_seed(42)
    init_device_properties_triton()
    device = "npu"

    b, h, n, d, e = 1, 4, 128, 64, 64
    q = _randn((b, h, n, d), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    k = _randn((b, h, n, d), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    v = _randn((b, h, n, e), torch.float32, device, scale=SEMANTIC_INPUT_SCALE)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=device)

    # Small decay rate (slow decay, long memory)
    ed_small = torch.full((1, h, 1, 1), 0.01, dtype=torch.float32, device=device)
    o_small, _ = lightning_attention_npu_(q, k, v, ed_small, kv_history.clone())

    # Large decay rate (fast decay, short memory)
    ed_large = torch.full((1, h, 1, 1), 1.0, dtype=torch.float32, device=device)
    o_large, _ = lightning_attention_npu_(q, k, v, ed_large, kv_history.clone())

    # Verify both produce valid (non-NaN, non-Inf) outputs
    assert not torch.isnan(o_small).any(), "Small decay output contains NaN"
    assert not torch.isinf(o_small).any(), "Small decay output contains Inf"
    assert not torch.isnan(o_large).any(), "Large decay output contains NaN"
    assert not torch.isinf(o_large).any(), "Large decay output contains Inf"

    # The outputs should differ because different decay rates produce
    # different attention distributions.
    assert not torch.allclose(o_small, o_large, rtol=1e-3, atol=1e-3), (
        "Different decay rates should produce different outputs"
    )

    # Verify both outputs match naive reference
    o_ref_small, _ = _naive_triton_lightning_attention(q, k, v, ed_small, kv_history)
    o_ref_large, _ = _naive_triton_lightning_attention(q, k, v, ed_large, kv_history)

    torch.testing.assert_close(
        o_small.cpu(),
        o_ref_small.cpu(),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        o_large.cpu(),
        o_ref_large.cpu(),
        rtol=1e-3,
        atol=1e-3,
    )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
