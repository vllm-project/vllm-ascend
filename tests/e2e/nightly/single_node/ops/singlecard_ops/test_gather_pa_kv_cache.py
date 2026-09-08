# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _make_cache(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype)
    return torch.randn(shape, dtype=torch.float32).to(dtype)


def _golden(
    cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_offset: torch.Tensor | None,
    is_seq_lens_cumsum: bool,
) -> torch.Tensor:
    block_size = cache.size(1)
    if is_seq_lens_cumsum:
        lengths = seq_lens[1:] - seq_lens[:-1]
    else:
        lengths = seq_lens

    gathered = []
    for batch_id, length in enumerate(lengths.tolist()):
        table_offset = 0 if seq_offset is None else int(seq_offset[batch_id]) // block_size
        block_count = (length + block_size - 1) // block_size
        block_ids = block_tables[batch_id, table_offset : table_offset + block_count].to(torch.int64)
        gathered.append(cache.index_select(0, block_ids).flatten(0, 1)[:length])
    return torch.cat(gathered)


def _to_pa_nz(cache: torch.Tensor) -> torch.Tensor:
    num_blocks, block_size, token_size = cache.shape
    assert cache.dtype == torch.int8
    assert token_size % 32 == 0
    return cache.view(num_blocks, block_size, token_size // 32, 32).permute(0, 2, 1, 3).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int8])
@pytest.mark.parametrize("is_seq_lens_cumsum", [False, True])
def test_gather_pa_kv_cache(dtype: torch.dtype, is_seq_lens_cumsum: bool):
    num_blocks = 12
    block_size = 4
    num_heads = 2
    key_head_size = 64
    value_head_size = 32

    # Keep only every second physical block to exercise stride-aware cache access.
    key_storage = _make_cache((num_blocks * 2, block_size, num_heads, key_head_size), dtype)
    value_storage = _make_cache((num_blocks * 2, block_size, num_heads, value_head_size), dtype)
    key_cache = key_storage[::2]
    value_cache = value_storage[::2]
    key_cache_npu = key_storage.npu()[::2]
    value_cache_npu = value_storage.npu()[::2]
    assert not key_cache_npu.is_contiguous()
    assert not value_cache_npu.is_contiguous()

    if is_seq_lens_cumsum:
        block_tables = torch.tensor([[9, 2, 7, 4], [1, 8, 3, 6]], dtype=torch.int32)
        seq_lens = torch.tensor([0, 6, 15], dtype=torch.int32)
        seq_offset = None
    else:
        block_tables = torch.tensor([[0, 9, 2, 7], [1, 8, 3, 6]], dtype=torch.int32)
        seq_lens = torch.tensor([6, 9], dtype=torch.int32)
        seq_offset = torch.tensor([block_size, 0], dtype=torch.int32)

    expected_key = _golden(key_cache, block_tables, seq_lens, seq_offset, is_seq_lens_cumsum)
    expected_value = _golden(value_cache, block_tables, seq_lens, seq_offset, is_seq_lens_cumsum)
    key = torch.empty_like(expected_key, device="npu")
    value = torch.empty_like(expected_value, device="npu")

    torch.ops._C_ascend.gather_pa_kv_cache(
        key_cache_npu,
        value_cache_npu,
        block_tables.npu(),
        seq_lens.npu(),
        key,
        value,
        seq_offset=None if seq_offset is None else seq_offset.npu(),
        is_seq_lens_cumsum=is_seq_lens_cumsum,
    )

    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)


@pytest.mark.parametrize("is_seq_lens_cumsum", [False, True])
def test_gather_pa_kv_cache_pa_nz(is_seq_lens_cumsum: bool):
    num_blocks = 12
    block_size = 4
    key_token_size = 96
    value_token_size = 160
    key_cache = _make_cache((num_blocks, block_size, key_token_size), torch.int8)
    value_cache = _make_cache((num_blocks, block_size, value_token_size), torch.int8)

    # PA_NZ is [num_blocks, token_size // 32, block_size, 32].
    key_nz = _to_pa_nz(key_cache).repeat_interleave(2, dim=0).npu()[::2]
    value_nz = _to_pa_nz(value_cache).repeat_interleave(2, dim=0).npu()[::2]
    assert not key_nz.is_contiguous()
    assert not value_nz.is_contiguous()

    if is_seq_lens_cumsum:
        block_tables = torch.tensor([[9, 2, 7, 4], [1, 8, 3, 6]], dtype=torch.int32)
        seq_lens = torch.tensor([0, 6, 15], dtype=torch.int32)
        seq_offset = None
    else:
        block_tables = torch.tensor([[0, 9, 2, 7], [1, 8, 3, 6]], dtype=torch.int32)
        seq_lens = torch.tensor([6, 9], dtype=torch.int32)
        seq_offset = torch.tensor([block_size, 0], dtype=torch.int32)

    expected_key = _golden(key_cache, block_tables, seq_lens, seq_offset, is_seq_lens_cumsum)
    expected_value = _golden(value_cache, block_tables, seq_lens, seq_offset, is_seq_lens_cumsum)
    key = torch.empty_like(expected_key, device="npu")
    value = torch.empty_like(expected_value, device="npu")

    torch.ops._C_ascend.gather_pa_kv_cache(
        key_nz,
        value_nz,
        block_tables.npu(),
        seq_lens.npu(),
        key,
        value,
        seq_offset=None if seq_offset is None else seq_offset.npu(),
        cache_mode="PA_NZ",
        is_seq_lens_cumsum=is_seq_lens_cumsum,
    )

    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)


def test_gather_pa_kv_cache_pa_nz_large_token():
    # More than 4095 data blocks exercises the PA_NZ two-part GM-to-UB copy.
    num_blocks = 5
    block_size = 2
    key_token_size = 32 * 4100
    value_token_size = 32 * 4097
    key_cache = _make_cache((num_blocks, block_size, key_token_size), torch.int8)
    value_cache = _make_cache((num_blocks, block_size, value_token_size), torch.int8)
    block_tables = torch.tensor([[4, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([3], dtype=torch.int32)
    expected_key = _golden(key_cache, block_tables, seq_lens, None, False)
    expected_value = _golden(value_cache, block_tables, seq_lens, None, False)
    key = torch.empty_like(expected_key, device="npu")
    value = torch.empty_like(expected_value, device="npu")

    torch.ops._C_ascend.gather_pa_kv_cache(
        _to_pa_nz(key_cache).npu(),
        _to_pa_nz(value_cache).npu(),
        block_tables.npu(),
        seq_lens.npu(),
        key,
        value,
        cache_mode="PA_NZ",
    )

    torch.testing.assert_close(key.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value.cpu(), expected_value, rtol=0, atol=0)


def test_gather_pa_kv_cache_rejects_unaligned_token_size():
    key_cache = torch.zeros((2, 4, 1, 3), dtype=torch.float16, device="npu")
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    seq_lens = torch.ones((1,), dtype=torch.int32, device="npu")
    key = torch.empty((1, 1, 3), dtype=torch.float16, device="npu")
    value = torch.empty_like(key)

    with pytest.raises(RuntimeError, match="key token size in bytes must be aligned"):
        torch.ops._C_ascend.gather_pa_kv_cache(
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            key,
            value,
        )


def test_gather_pa_kv_cache_pa_nz_rejects_non_int8():
    key_cache = torch.zeros((2, 1, 4, 16), dtype=torch.float16, device="npu")
    value_cache = torch.zeros_like(key_cache)
    block_tables = torch.zeros((1, 1), dtype=torch.int32, device="npu")
    seq_lens = torch.ones((1,), dtype=torch.int32, device="npu")
    key = torch.empty((1, 16), dtype=torch.float16, device="npu")
    value = torch.empty_like(key)

    with pytest.raises(RuntimeError, match="PA_NZ mode only supports int8"):
        torch.ops._C_ascend.gather_pa_kv_cache(
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            key,
            value,
            cache_mode="PA_NZ",
        )
