import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.ops.flash_attn import (
    build_flash_attn_metadata,
    flash_attn_metadata_size,
    run_flash_attn,
)


def _fake_cann_modules(metadata_op=None, attention_op=None):
    package = ModuleType("cann_ops_transformer")
    ops = ModuleType("cann_ops_transformer.ops")
    ops.flash_attn_metadata = metadata_op
    ops.flash_attn = attention_op
    package.ops = ops
    return {"cann_ops_transformer": package, "cann_ops_transformer.ops": ops}


def test_flash_attn_metadata_size_is_4k_aligned():
    size = flash_attn_metadata_size(batch_size=3, num_heads_kv=2)
    assert size == 12288
    assert size % 4096 == 0


def test_build_flash_attn_metadata_copies_into_stable_buffer():
    generated = torch.arange(8, dtype=torch.int32)
    metadata_op = MagicMock(return_value=generated)
    output_buffer = torch.empty_like(generated)

    with patch.dict(sys.modules, _fake_cann_modules(metadata_op=metadata_op)):
        result = build_flash_attn_metadata(
            8,
            2,
            128,
            output_buffer=output_buffer,
            batch_size=2,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
        )

    assert result is output_buffer
    assert torch.equal(result, generated)
    metadata_op.assert_called_once_with(8, 2, 128, batch_size=2, max_seqlen_q=-1, max_seqlen_kv=-1)


def test_build_flash_attn_metadata_rejects_wrong_buffer_shape():
    metadata_op = MagicMock(return_value=torch.empty(8, dtype=torch.int32))
    with (
        patch.dict(sys.modules, _fake_cann_modules(metadata_op=metadata_op)),
        pytest.raises(ValueError, match="buffer shape mismatch"),
    ):
        build_flash_attn_metadata(8, 2, 128, output_buffer=torch.empty(7, dtype=torch.int32))


def test_run_flash_attn_forwards_prebuilt_metadata():
    query = torch.randn(2, 8, 128)
    key = torch.randn(2, 2, 128)
    value = torch.randn_like(key)
    metadata = torch.empty(4096, dtype=torch.int32)
    expected = (torch.randn_like(query), torch.empty(1))
    attention_op = MagicMock(return_value=expected)

    with patch.dict(sys.modules, _fake_cann_modules(attention_op=attention_op)):
        result = run_flash_attn(query, key, value, metadata=metadata, layout_q="TND")

    assert result is expected
    attention_op.assert_called_once_with(query, key, value, metadata=metadata, layout_q="TND")
