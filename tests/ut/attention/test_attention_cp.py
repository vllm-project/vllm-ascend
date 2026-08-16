#
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

"""Unit tests for the non-uniform BSND helpers used by GQA decode DCP.

``pad_decode_query_to_bsnd`` and ``_compact_bsnd_out`` are the two halves of
the non-uniform speculative-decode path in
``AscendAttentionDCPImpl._forward_decode_dcp``: the first converts a packed
TND query into the BSND layout ``npu_fused_infer_attention_score`` expects,
the second strips the padding back out of the kernel output.  Both are pure
tensor bookkeeping, so they run on CPU without an NPU.
"""

import torch

from vllm_ascend.attention.context_parallel.attention_cp import (
    _compact_bsnd_out,
    pad_decode_query_to_bsnd,
)

NUM_HEADS = 2
HEAD_SIZE = 4


def _packed_query(
    query_lens: list[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a TND query whose every element is unique and non-zero.

    Values start at 1 so that "this slot is padding" can be asserted as
    "this slot is exactly 0".
    """
    num_tokens = sum(query_lens)
    return torch.arange(
        1,
        num_tokens * NUM_HEADS * HEAD_SIZE + 1,
        dtype=dtype,
    ).reshape(num_tokens, NUM_HEADS, HEAD_SIZE)


def _flat_bsnd(num_decodes: int, max_q: int, last_dim: int) -> torch.Tensor:
    """Build a ``(num_decodes * max_q, NUM_HEADS, last_dim)`` kernel output."""
    num_rows = num_decodes * max_q
    return torch.arange(
        1,
        num_rows * NUM_HEADS * last_dim + 1,
        dtype=torch.float32,
    ).reshape(num_rows, NUM_HEADS, last_dim)


def test_pad_decode_query_keeps_shape_and_per_request_content() -> None:
    query_lens = [3, 1, 2]
    query = _packed_query(query_lens)

    padded = pad_decode_query_to_bsnd(query, query_lens, max_q=3)

    assert padded.shape == (3, 3, NUM_HEADS, HEAD_SIZE)
    offset = 0
    for i, query_len in enumerate(query_lens):
        torch.testing.assert_close(padded[i, :query_len], query[offset : offset + query_len])
        offset += query_len


def test_pad_decode_query_zero_fills_padding_slots() -> None:
    query_lens = [3, 1, 2]
    query = _packed_query(query_lens)

    padded = pad_decode_query_to_bsnd(query, query_lens, max_q=3)

    for i, query_len in enumerate(query_lens):
        tail = padded[i, query_len:]
        torch.testing.assert_close(tail, torch.zeros_like(tail))


def test_pad_decode_query_skips_requests_without_tokens() -> None:
    """A zero-length request occupies a row but consumes no input token."""
    query_lens = [2, 0, 3]
    query = _packed_query(query_lens)

    padded = pad_decode_query_to_bsnd(query, query_lens, max_q=3)

    torch.testing.assert_close(padded[0, :2], query[:2])
    torch.testing.assert_close(padded[1], torch.zeros_like(padded[1]))
    torch.testing.assert_close(padded[2, :3], query[2:5])


def test_compact_bsnd_out_selects_valid_rows() -> None:
    decode_q_lens = [3, 1, 2]
    max_q = 3
    attn_out = _flat_bsnd(len(decode_q_lens), max_q, HEAD_SIZE)
    attn_lse = _flat_bsnd(len(decode_q_lens), max_q, 1)

    out, lse = _compact_bsnd_out(attn_out, attn_lse, decode_q_lens, max_q)

    valid_rows = torch.tensor([0, 1, 2, 3, 6, 7])
    assert out.shape == (6, NUM_HEADS, HEAD_SIZE)
    assert lse.shape == (6, NUM_HEADS, 1)
    torch.testing.assert_close(out, attn_out[valid_rows])
    torch.testing.assert_close(lse, attn_lse[valid_rows])


def test_compact_bsnd_out_drops_padding_garbage() -> None:
    """Padding slots are never written by the kernel and must not leak out."""
    decode_q_lens = [3, 1, 2]
    max_q = 3
    attn_out = _flat_bsnd(len(decode_q_lens), max_q, HEAD_SIZE)
    attn_lse = _flat_bsnd(len(decode_q_lens), max_q, 1)
    padding_rows = torch.tensor([4, 5, 8])
    attn_out[padding_rows] = torch.nan
    attn_lse[padding_rows] = torch.inf

    out, lse = _compact_bsnd_out(attn_out, attn_lse, decode_q_lens, max_q)

    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()


def test_compact_bsnd_out_follows_forward_layout_transform() -> None:
    """Reproduce the reshapes ``_forward_decode_dcp`` applies before compacting.

    The kernel returns ``out`` as ``(B, S, N, D)`` but ``lse`` as
    ``(B, N, S, 1)``; the caller transposes the latter so that both are
    row-indexed by ``batch * max_q + step`` before compaction.  This test pins
    that ordering down: dropping the transpose swaps the head and step axes
    and silently corrupts the DCP lse merge.
    """
    decode_q_lens = [2, 1]
    max_q = 2
    num_decodes = len(decode_q_lens)
    kernel_out = torch.arange(
        num_decodes * max_q * NUM_HEADS * HEAD_SIZE,
        dtype=torch.float32,
    ).reshape(num_decodes, max_q, NUM_HEADS, HEAD_SIZE)
    kernel_lse = torch.arange(
        num_decodes * NUM_HEADS * max_q,
        dtype=torch.float32,
    ).reshape(num_decodes, NUM_HEADS, max_q, 1)

    flat_out = kernel_out.view(-1, kernel_out.shape[2], kernel_out.shape[3])
    flat_lse = kernel_lse.transpose(1, 2).reshape(-1, kernel_lse.shape[1], 1)
    out, lse = _compact_bsnd_out(flat_out, flat_lse, decode_q_lens, max_q)

    expected_out = torch.cat([kernel_out[0, :2], kernel_out[1, :1]], dim=0)
    expected_lse = torch.cat(
        [kernel_lse[0, :, :2].transpose(0, 1), kernel_lse[1, :, :1].transpose(0, 1)],
        dim=0,
    )
    torch.testing.assert_close(out, expected_out)
    torch.testing.assert_close(lse, expected_lse)


def test_pad_then_compact_round_trip() -> None:
    """Padding and compaction are inverses for an identity "attention"."""
    query_lens = [3, 1, 2]
    max_q = 3
    query = _packed_query(query_lens)

    padded = pad_decode_query_to_bsnd(query, query_lens, max_q)
    flat = padded.view(-1, NUM_HEADS, HEAD_SIZE)
    flat_lse = torch.zeros(flat.shape[0], NUM_HEADS, 1)
    out, _ = _compact_bsnd_out(flat, flat_lse, query_lens, max_q)

    torch.testing.assert_close(out, query)
