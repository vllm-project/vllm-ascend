from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.flash_mla import (
    build_flash_mla_metadata,
    flash_mla_metadata_size,
    run_flash_mla,
)


def _fake_ops(metadata_op=None, attention_op=None):
    return SimpleNamespace(
        flash_mla_with_kvcache_metadata=metadata_op,
        flash_mla_with_kvcache=attention_op,
    )


def test_flash_mla_metadata_size_is_4k_aligned():
    size = flash_mla_metadata_size(batch_size=3)
    assert size == 8192
    assert size % 4096 == 0


def test_build_flash_mla_metadata_copies_into_stable_buffer():
    generated = torch.arange(8, dtype=torch.int32)
    metadata_op = MagicMock(return_value=generated)
    output_buffer = torch.empty_like(generated)
    cache_seqlens = torch.tensor([8, 9], dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)

    with patch("vllm_ascend.ops.flash_mla._flash_mla_ops", return_value=_fake_ops(metadata_op=metadata_op)):
        result = build_flash_mla_metadata(
            cache_seqlens,
            8,
            1,
            cu_seqlens_q=cu_seqlens_q,
            head_dim_qk=576,
            head_dim_v=512,
            layout_q="TND",
            output_buffer=output_buffer,
        )

    assert result is output_buffer
    assert torch.equal(result, generated)
    metadata_op.assert_called_once_with(
        cache_seqlens,
        8,
        1,
        cu_seqlens_q=cu_seqlens_q,
        head_dim_qk=576,
        head_dim_v=512,
        layout_q="TND",
    )


def test_build_flash_mla_metadata_rejects_wrong_buffer_shape():
    metadata_op = MagicMock(return_value=torch.empty(8, dtype=torch.int32))
    with (
        patch(
            "vllm_ascend.ops.flash_mla._flash_mla_ops",
            return_value=_fake_ops(metadata_op=metadata_op),
        ),
        pytest.raises(ValueError, match="buffer shape mismatch"),
    ):
        build_flash_mla_metadata(
            torch.ones(2, dtype=torch.int32),
            8,
            1,
            output_buffer=torch.empty(7, dtype=torch.int32),
        )


def test_run_flash_mla_forwards_prebuilt_metadata():
    query = torch.randn(2, 8, 576)
    kv_cache = torch.randn(4, 128, 1, 576)
    metadata = torch.empty(4096, dtype=torch.int32)
    expected = (torch.randn(2, 8, 512), torch.empty(1))
    attention_op = MagicMock(return_value=expected)

    with patch(
        "vllm_ascend.ops.flash_mla._flash_mla_ops",
        return_value=_fake_ops(attention_op=attention_op),
    ):
        result = run_flash_mla(
            query,
            kv_cache,
            metadata=metadata,
            layout_q="TND",
            layout_kv="PA_BBND",
        )

    assert result is expected
    attention_op.assert_called_once_with(
        query,
        kv_cache,
        metadata=metadata,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
