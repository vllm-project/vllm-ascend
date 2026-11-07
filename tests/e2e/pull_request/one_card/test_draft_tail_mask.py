# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("shape", [(2, 9, 4, 2, 128, 1), (8, 8, 4, 1, 256, 2)])
@pytest.mark.parametrize("invalid_value", [100.0, float("nan"), float("inf")])
def test_draft_tail_mask_matches_exact_attention(causal, window, strided, shape, invalid_value):
    """Compare paged BSND with FP32 attention over only the valid KV prefix.

    Repeated shapes change lengths; permuted pages catch incorrect addressing.
    Unread tail values include NaN/Inf, which a score mask alone cannot isolate.
    """
    torch.manual_seed(16271)
    batch, query_len, heads, kv_heads, head_size, blocks_per_req = shape
    block_size = 128
    capacity = blocks_per_req * block_size
    query = torch.randn(batch, query_len, heads, head_size, dtype=torch.bfloat16)
    key = torch.randn(batch, capacity, kv_heads, head_size, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    device_query = query.to("npu")
    if strided:
        query_backing = torch.zeros(batch, query_len, heads, 2 * head_size, dtype=query.dtype, device="npu")
        device_query = query_backing[..., :head_size]
        device_query.copy_(query.to("npu"))
    num_blocks = batch * blocks_per_req
    page_order = torch.randperm(num_blocks)
    table_columns = blocks_per_req if blocks_per_req == 1 else 320
    block_table = torch.zeros(batch, table_columns, dtype=torch.int32)
    block_table[:, :blocks_per_req] = page_order.reshape(batch, blocks_per_req).to(torch.int32)
    block_table = block_table.to("npu")
    impl = SimpleNamespace(
        sinks=None,
        sliding_window=window,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_size=head_size,
        scale=head_size**-0.5,
    )
    length_pair = [31, 55] if blocks_per_req == 1 else [137, 191]
    for pair in (length_pair, length_pair[::-1]):
        lengths = pair * (batch // 2)
        step_key, step_value = key.clone(), value.clone()
        bounds = [length + query_len - 1 for length in lengths]
        for req, length in enumerate(lengths):
            step_key[req, length:] = invalid_value
            step_value[req, length:] = invalid_value
        key_pages = torch.empty_like(step_key.reshape(num_blocks, block_size, kv_heads, head_size))
        value_pages = torch.empty_like(key_pages)
        key_pages[page_order] = step_key.reshape_as(key_pages)
        value_pages[page_order] = step_value.reshape_as(value_pages)
        backing = torch.stack((key_pages, value_pages), dim=1).to("npu")
        device_key, device_value = backing[:, 0].flatten(2), backing[:, 1].flatten(2)
        if not strided:
            device_key, device_value = device_key.contiguous(), device_value.contiguous()
        before = backing.clone()
        metadata = SimpleNamespace(
            draft_kv_upper_bound=True,
            draft_query_lens=[query_len] * batch,
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
            draft_tail_mask_cache={},
            causal=causal,
        )
        output = torch.empty(batch * query_len, heads, head_size, dtype=query.dtype, device="npu")
        result = AscendAttentionBackendImpl._forward_draft_tail_masked(
            impl,
            device_query.reshape_as(output),
            device_key,
            device_value,
            metadata,
            block_table,
            block_size,
            bounds,
            batch * query_len,
            output,
        )
        assert result is not None
        references = []
        for req, length in enumerate(lengths):
            keys = step_key[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            values = step_value[req, :length].float().repeat_interleave(heads // kv_heads, dim=1).permute(1, 0, 2)
            queries = query[req].float().permute(1, 0, 2)
            scores = queries @ keys.transpose(1, 2) * impl.scale
            last = torch.arange(query_len) + length - query_len if causal else torch.full((query_len,), length - 1)
            visible = torch.arange(length).view(1, length) <= last.view(query_len, 1)
            if window is not None:
                visible &= torch.arange(length).view(1, length) >= last.view(query_len, 1) - window
            scores.masked_fill_(~visible.unsqueeze(0), float("-inf"))
            references.append((scores.softmax(-1) @ values).permute(1, 0, 2))
        reference = torch.stack(references)
        actual = output.cpu().float().reshape_as(reference)
        # BF16 operator versus FP32 reference: fixed before running.
        relative_l2_tolerance = 0.02
        assert torch.isfinite(actual).all()
        assert (actual - reference).norm() / reference.norm() < relative_l2_tolerance
        torch.testing.assert_close(backing, before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("page_padding", [1, 3])
@pytest.mark.parametrize("inner_stride", [1, 2])
@pytest.mark.parametrize("batch", [2, 8])
def test_draft_kv_gather_is_bit_exact(dtype, page_padding, inner_stride, batch):
    """Exercise zero length, page boundaries, sentinels and strided source rows."""
    from vllm_ascend.ops.triton.draft_kv_gather import gather_draft_kv

    torch.manual_seed(16271)
    page_size, pages = 128, 3
    width = 65 if batch == 2 else 256
    count = batch * pages
    backing = torch.randn(count, page_padding, page_size, width * inner_stride, dtype=dtype)
    other = torch.randn_like(backing)
    key, value = backing[:, 0, :, ::inner_stride], other[:, 0, :, ::inner_stride]
    key[0, 0, 0] = -0.0
    order = torch.randperm(count).reshape(batch, pages)
    lens = [0, 257] if batch == 2 else [0, 1, 127, 128, 129, 255, 256, 384]
    # Reuse identical tensor shapes while changing both lengths and page IDs.
    for lengths in (lens, lens[::-1]):
        table = order.clone().to(torch.int32)
        source_key, source_value = key.clone(), value.clone()
        expected_key = torch.zeros(batch, pages * page_size, width, dtype=dtype)
        expected_value = torch.zeros_like(expected_key)
        for req, length in enumerate(lengths):
            for pos, page in enumerate(order[req]):
                valid = max(0, min(page_size, length - pos * page_size))
                expected_key[req, pos * page_size : pos * page_size + valid] = key[page, :valid]
                expected_value[req, pos * page_size : pos * page_size + valid] = value[page, :valid]
                source_key[page, valid:] = float("nan")
                source_value[page, valid:] = float("inf")
                if valid == 0:
                    table[req, pos] = -99
        device_backing, device_other = backing.to("npu"), other.to("npu")
        device_key = device_backing[:, 0, :, ::inner_stride]
        device_value = device_other[:, 0, :, ::inner_stride]
        device_key.copy_(source_key.to("npu"))
        device_value.copy_(source_value.to("npu"))
        before_key, before_value = device_backing.clone(), device_other.clone()
        length_storage = torch.zeros(batch * 2, dtype=torch.int32, device="npu")
        length_storage[::2] = torch.tensor(lengths, dtype=torch.int32, device="npu")
        table_storage = torch.zeros(batch, pages * 2, dtype=torch.int32, device="npu")
        table_storage[:, ::2] = table.to("npu")
        out_key, out_value, out_table = gather_draft_kv(
            device_key, device_value, table_storage[:, ::2], length_storage[::2], batch, pages
        )
        integer_dtype = torch.int32 if dtype == torch.float32 else torch.int16
        for actual, expected in [(out_key, expected_key), (out_value, expected_value)]:
            torch.testing.assert_close(
                actual.cpu().reshape_as(expected).view(integer_dtype), expected.view(integer_dtype), rtol=0, atol=0
            )
        torch.testing.assert_close(out_table.cpu(), torch.arange(count, dtype=torch.int32).reshape(batch, pages))
        torch.testing.assert_close(device_backing, before_key, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(device_other, before_value, rtol=0, atol=0, equal_nan=True)
