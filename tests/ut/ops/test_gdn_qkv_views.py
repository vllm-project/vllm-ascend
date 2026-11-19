import pytest
import torch

from vllm_ascend.ops.gdn import _rearrange_decode_qkv


@pytest.mark.parametrize("tokens", [0, 1, 4, 8])
def test_decode_qkv_views_preserve_storage(tokens):
    packed = torch.randn(tokens, 8192, dtype=torch.bfloat16)
    parts = packed.split((2048, 2048, 4096), dim=-1)
    actual = _rearrange_decode_qkv(packed, 2048, 4096, 128, 128, allow_strided=True)
    for got, part, heads in zip(actual, parts, (16, 16, 32)):
        assert got.shape == (1, tokens, heads, 128)
        assert got.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
        assert got.storage_offset() == part.storage_offset()
        if tokens > 1:
            assert got.stride()[-3:] == (8192, 128, 1)
        torch.testing.assert_close(got.squeeze(0), part.view(tokens, heads, 128))


def test_decode_qkv_views_reject_inner_stride():
    backing = torch.randn(4, 16384, dtype=torch.bfloat16)
    packed = backing[:, ::2]
    actual = _rearrange_decode_qkv(packed, 2048, 4096, 128, 128, allow_strided=True)
    for got in actual:
        assert got.is_contiguous()


@pytest.mark.parametrize(
    "installed,expected", [("26.10.0+main.dev2ef0735", True), ("26.9.1", False), ("26.10.0", False)]
)
def test_view_abi_requires_verified_wheel(monkeypatch, installed, expected):
    from vllm_ascend.ops import gdn

    gdn._supports_recurrent_qkv_views.cache_clear()
    monkeypatch.setattr(gdn, "version", lambda _: installed)
    try:
        assert gdn._supports_recurrent_qkv_views() is expected
    finally:
        gdn._supports_recurrent_qkv_views.cache_clear()
