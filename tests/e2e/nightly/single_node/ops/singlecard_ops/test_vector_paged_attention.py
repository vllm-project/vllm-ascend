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
#
"""Correctness of the VectorPagedAttention custom operator.

The reference is computed in fp32 from the same pages the operator reads, so a
mismatch is the kernel's and not the layout's.
"""

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

HEAD_DIM = 64


def _reference(query, key_cache, value_cache, block_table, seq_lens, scale):
    """Paged single-query attention, fp32, straight from the definition."""
    batch, num_heads, head_dim = query.shape
    block_size = key_cache.shape[1]
    out = torch.empty(batch, num_heads, head_dim, dtype=torch.float32)
    for b in range(batch):
        length = int(seq_lens[b])
        rows = [(int(block_table[b, pos // block_size]), pos % block_size) for pos in range(length)]
        keys = torch.stack([key_cache[blk, off] for blk, off in rows])
        values = torch.stack([value_cache[blk, off] for blk, off in rows])
        keys = keys.view(length, num_heads, head_dim).float()
        values = values.view(length, num_heads, head_dim).float()
        for h in range(num_heads):
            scores = (keys[:, h, :] * query[b, h].float()).sum(-1) * scale
            weights = torch.softmax(scores, dim=0)
            out[b, h] = (weights.unsqueeze(-1) * values[:, h, :]).sum(0)
    return out


def _build(batch, num_heads, block_size, max_blocks, lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    num_blocks = batch * max_blocks + 3
    shape = (num_blocks, block_size, num_heads * HEAD_DIM)
    query = torch.randn(batch, num_heads, HEAD_DIM, generator=generator).bfloat16()
    key_cache = torch.randn(shape, generator=generator).bfloat16()
    value_cache = torch.randn(shape, generator=generator).bfloat16()
    # Deliberately non-contiguous page ids: the kernel must follow the table.
    block_table = torch.arange(batch * max_blocks, dtype=torch.int32)
    block_table = block_table.flip(0).reshape(batch, max_blocks) % num_blocks
    seq_lens = torch.tensor(lengths, dtype=torch.int32)
    return query, key_cache, value_cache, block_table, seq_lens


@pytest.mark.parametrize(
    "batch, num_heads, block_size, max_blocks, lengths",
    [
        (1, 12, 128, 4, [1]),  # a single token
        (1, 12, 128, 4, [137]),  # mid-page
        (1, 12, 128, 4, [512]),  # exactly the declared capacity
        (1, 12, 16, 8, [128]),  # exactly full pages
        (1, 12, 32, 8, [200]),  # a smaller page
        (2, 12, 128, 4, [137, 512]),  # unequal lengths in one step
        (4, 8, 64, 8, [7, 64, 65, 300]),  # several requests, several page counts
        (1, 1, 8, 16, [61]),  # one head, the smallest page
    ],
)
def test_matches_fp32_reference(batch, num_heads, block_size, max_blocks, lengths):
    query, key_cache, value_cache, block_table, seq_lens = _build(
        batch, num_heads, block_size, max_blocks, lengths, seed=1234
    )
    scale = HEAD_DIM**-0.5
    expected = _reference(query, key_cache, value_cache, block_table, seq_lens, scale)

    actual = torch.ops._C_ascend.npu_vector_paged_attention(
        query.npu(),
        key_cache.npu(),
        value_cache.npu(),
        block_table.npu(),
        seq_lens.npu(),
        num_kv_heads=num_heads,
        scale=scale,
    )

    # The kernel accumulates in fp32 and rounds once on the way out, so the
    # only difference from the reference should be that final rounding.
    torch.testing.assert_close(actual.cpu().float(), expected.bfloat16().float(), atol=0.0, rtol=0.0)


def test_accepts_a_four_dimensional_cache():
    batch, num_heads, block_size, max_blocks = 1, 12, 128, 4
    query, key_cache, value_cache, block_table, seq_lens = _build(
        batch, num_heads, block_size, max_blocks, [137], seed=7
    )
    scale = HEAD_DIM**-0.5
    flat = torch.ops._C_ascend.npu_vector_paged_attention(
        query.npu(),
        key_cache.npu(),
        value_cache.npu(),
        block_table.npu(),
        seq_lens.npu(),
        num_kv_heads=num_heads,
        scale=scale,
    )
    shape = (key_cache.shape[0], block_size, num_heads, HEAD_DIM)
    nested = torch.ops._C_ascend.npu_vector_paged_attention(
        query.npu(),
        key_cache.view(shape).npu(),
        value_cache.view(shape).npu(),
        block_table.npu(),
        seq_lens.npu(),
        num_kv_heads=num_heads,
        scale=scale,
    )
    torch.testing.assert_close(flat.cpu(), nested.cpu(), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda kw: kw.update(head_dim=32), "head_dim"),
        (lambda kw: kw.update(num_kv_heads=6), "multi-head only"),
        (lambda kw: kw.update(block_size=48), "power of two"),
        (lambda kw: kw.update(dtype=torch.float16), "bfloat16"),
        (lambda kw: kw.update(batch=8, num_heads=12), "vector core count"),
    ],
)
def test_rejects_outside_its_declared_domain(mutate, message):
    """Out-of-domain shapes must fail in the adapter, not during capture."""
    kw = dict(
        batch=1, num_heads=12, head_dim=HEAD_DIM, block_size=128, max_blocks=4, dtype=torch.bfloat16, num_kv_heads=None
    )
    mutate(kw)
    batch, num_heads, head_dim = kw["batch"], kw["num_heads"], kw["head_dim"]
    block_size, max_blocks = kw["block_size"], kw["max_blocks"]
    num_kv_heads = kw["num_kv_heads"] if kw["num_kv_heads"] else num_heads
    num_blocks = batch * max_blocks + 3
    query = torch.randn(batch, num_heads, head_dim).to(kw["dtype"]).npu()
    cache = torch.randn(num_blocks, block_size, num_kv_heads * head_dim)
    cache = cache.to(kw["dtype"]).npu()
    block_table = torch.zeros(batch, max_blocks, dtype=torch.int32).npu()
    seq_lens = torch.full((batch,), 8, dtype=torch.int32).npu()

    with pytest.raises(RuntimeError, match=message):
        torch.ops._C_ascend.npu_vector_paged_attention(
            query,
            cache,
            cache,
            block_table,
            seq_lens,
            num_kv_heads=num_kv_heads,
            scale=head_dim**-0.5,
        )
