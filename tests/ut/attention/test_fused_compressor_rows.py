from vllm_ascend.attention.context_parallel.fused_compressor_utils import (
    get_fused_compressor_row_info,
)


def test_fused_rows_allow_padded_output_with_valid_slot_rows():
    info = get_fused_compressor_row_info((100, 4096), (26, 64), (25, 2), 4)
    assert info.expected_rows == 26
    assert info.valid_rows == 25
    assert info.safe


def test_fused_rows_reject_slot_mapping_overrun():
    info = get_fused_compressor_row_info((100, 4096), (26, 64), (27, 2), 4)
    assert not info.safe
    assert "exceed" in info.reason


def test_fused_rows_reject_zero_valid_rows():
    info = get_fused_compressor_row_info((100, 4096), (26, 64), (0, 2), 128)
    assert info.expected_rows == 26
    assert info.valid_rows == 0
    assert not info.safe


def test_fused_rows_support_bsh_shape():
    info = get_fused_compressor_row_info((2, 130, 4096), (2, 33, 64), (64, 2), 4)
    assert info.expected_rows == 66
    assert info.valid_rows == 64
    assert info.safe


def test_fused_cache_layout_is_rank4_single_head():
    # The row helper is independent of cache layout; this test documents the
    # runtime layout accepted by the Python fused guard.
    cache_shape = (128, 128, 1, 512)
    assert len(cache_shape) == 4
    assert cache_shape[2] == 1
    assert cache_shape[3] == 512


def test_fused_rows_reject_bad_slot_shape_and_x_rank():
    assert not get_fused_compressor_row_info((100, 4096), (26, 64), (25,), 4).safe
    assert not get_fused_compressor_row_info((100,), (26, 64), (25, 2), 4).safe
