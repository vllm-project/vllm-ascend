import pytest
import torch

from vllm_ascend.attention.fia_kv_layout import fia_kv_cache_views


@pytest.mark.parametrize("gap", [2, 4])
@pytest.mark.parametrize("num_blocks", [1, 5])
def test_fia_views_preserve_pages_and_storage_offsets(gap, num_blocks):
    block_size, hidden = 4, 8
    page = block_size * hidden
    # Include a preceding layer and an exact-sized backing ending at V.
    prefix = 3 * page
    backing = torch.arange(prefix + num_blocks * gap * page, dtype=torch.float32)
    shape = (num_blocks, block_size, hidden)
    strides = (gap * page, hidden, 1)
    key = backing.as_strided(shape, strides, prefix)
    value = backing.as_strided(shape, strides, prefix + page)
    dense_key, dense_value, factor = fia_kv_cache_views(key, value)
    assert factor == gap
    for original, dense in [(key, dense_key), (value, dense_value)]:
        assert dense.is_contiguous()
        assert dense.data_ptr() == original.data_ptr()
        assert dense.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        ids = torch.tensor([num_blocks - 1, 0])
        torch.testing.assert_close(dense[ids * factor], original[ids])
    # A cache write becomes visible immediately without refreshing/copying the view.
    key[-1].fill_(7)
    value[-1].fill_(9)
    assert torch.all(dense_key[(num_blocks - 1) * factor] == 7)
    assert torch.all(dense_value[(num_blocks - 1) * factor] == 9)
    if num_blocks > 1:
        before = dense_key[0].clone()
        # Reassign another physical block to a different cache group.
        backing[prefix + gap * page : prefix + 2 * gap * page].fill_(-1)
        torch.testing.assert_close(dense_key[0], before)


def test_fia_views_keep_unsupported_layouts():
    for key, value in [
        (torch.empty(3, 4, 8), torch.empty(3, 4, 8)),
        (torch.empty(3, 8, 4).transpose(1, 2), torch.empty(3, 8, 4).transpose(1, 2)),
        (torch.empty(3, 4, 8), torch.empty(3, 4, 16)),
        (torch.empty(0, 4, 8), torch.empty(0, 4, 8)),
    ]:
        k, v, factor = fia_kv_cache_views(key, value)
        assert k is key and v is value and factor == 1
