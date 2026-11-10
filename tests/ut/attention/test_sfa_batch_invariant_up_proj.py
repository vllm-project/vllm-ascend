# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Batch invariance of the SFA q/kv up-projection.

``npu_transpose_batchmatmul`` and its ``torch.bmm`` fallback do not round
identically, so a guard that picks between them based on the *packed* token
count makes a row's bits depend on how many unrelated rows share the launch.
These tests pin the fix in ``sfa_v1._transpose_batchmatmul_on_grid``: the fused
op is always used, and every launch stays on a fixed token grid well below the
operator's operand limit.
"""

import torch

from vllm_ascend.attention import sfa_v1


class _RecordingTBMM:
    """Row-independent stand-in for ``torch_npu.npu_transpose_batchmatmul``.

    Row ``i`` of the output only depends on row ``i`` of the input, which is the
    property the real fused kernel provides (and the property the batch-invariant
    path relies on).
    """

    def __init__(self) -> None:
        self.token_dims: list[int] = []

    def __call__(self, x: torch.Tensor, w: torch.Tensor, **kwargs):
        self.token_dims.append(int(x.shape[0]))
        return x[:, :, : w.shape[-1]].clone()


def _make_inputs(num_tokens: int, num_heads: int = 2, dim: int = 4):
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(num_tokens, num_heads, dim, generator=gen)
    w = torch.randn(num_heads, dim, dim, generator=gen)
    return x, w


def test_chunk_stays_under_the_operand_limit():
    assert 0 < sfa_v1.TRANSPOSE_BMM_CHUNK <= sfa_v1.TRANSPOSE_BMM_MAX_SUPPORTED_DIM


def test_every_launch_uses_the_fixed_grid(monkeypatch):
    fake = _RecordingTBMM()
    monkeypatch.setattr(sfa_v1.torch_npu, "npu_transpose_batchmatmul", fake)
    chunk = sfa_v1.TRANSPOSE_BMM_CHUNK

    # One launch below the grid, and a packed batch that used to fall back to bmm.
    for num_tokens in (chunk - 1, chunk, chunk + 1, 65712):
        fake.token_dims.clear()
        x, w = _make_inputs(num_tokens)
        out = sfa_v1._transpose_batchmatmul_on_grid(x, w)
        assert out.shape == (num_tokens, x.shape[1], w.shape[-1])
        assert max(fake.token_dims) <= chunk, fake.token_dims
        assert len(fake.token_dims) == (num_tokens + chunk - 1) // chunk, fake.token_dims


def test_a_row_is_unaffected_by_the_packed_batch(monkeypatch):
    fake = _RecordingTBMM()
    monkeypatch.setattr(sfa_v1.torch_npu, "npu_transpose_batchmatmul", fake)
    chunk = sfa_v1.TRANSPOSE_BMM_CHUNK

    solo_tokens = chunk // 2
    x_solo, w = _make_inputs(solo_tokens)
    solo = sfa_v1._transpose_batchmatmul_on_grid(x_solo, w)

    # Same leading rows, but now packed with 2x more rows: crosses the old
    # 65536-token guard and used to switch to torch.bmm.
    x_packed, _ = _make_inputs(solo_tokens * 3 + 7)
    x_packed[:solo_tokens] = x_solo
    packed = sfa_v1._transpose_batchmatmul_on_grid(x_packed, w)

    assert torch.equal(packed[:solo_tokens], solo)


def test_method_uses_the_fused_grid_not_the_bmm_fallback(monkeypatch):
    """The projection must stay on the fused op for bf16, whatever the token count."""
    fake = _RecordingTBMM()
    monkeypatch.setattr(sfa_v1.torch_npu, "npu_transpose_batchmatmul", fake)
    chunk = sfa_v1.TRANSPOSE_BMM_CHUNK

    def _forbid_bmm(*args, **kwargs):
        raise AssertionError("q up-proj fell back to torch.bmm for a bf16 input")

    monkeypatch.setattr(torch, "bmm", _forbid_bmm)

    impl = object.__new__(sfa_v1.AscendSFAImpl)
    impl.local_num_heads = 2
    impl.qk_head_dim = 8
    impl.qk_nope_head_dim = 4
    impl.qk_rope_head_dim = 4
    impl.W_UK_T = torch.zeros(2, 4, 4, dtype=torch.bfloat16)
    impl.q_proj = lambda x: (torch.zeros(x.shape[0], impl.local_num_heads * impl.qk_head_dim, dtype=torch.bfloat16),)

    num_tokens = chunk + 10
    ql_nope, q_pe = impl._q_proj_and_k_up_proj(torch.zeros(num_tokens, impl.local_num_heads * impl.qk_head_dim))

    assert fake.token_dims == [chunk, num_tokens - chunk], fake.token_dims
    assert ql_nope.shape == (num_tokens, 2, 4)
    assert q_pe.shape == (num_tokens, 2, 4)
