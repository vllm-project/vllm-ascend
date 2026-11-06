# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import li_quant_cache


class TensorMetadata:
    """Exercise the metadata guard without reading any device values."""

    def __init__(self, shape, dtype, device="npu", index=0, contiguous=True):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device, index=index)
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return math.prod(self.shape)

    def view(self, *shape):
        assert self.contiguous
        return TensorMetadata((self.numel() // shape[1], shape[1]), self.dtype)


@pytest.fixture
def arguments():
    return [
        TensorMetadata((6, 128), torch.bfloat16),
        TensorMetadata((6,), torch.int64),
        TensorMetadata((16, 128, 1, 128), torch.int8),
        TensorMetadata((16, 128, 1, 1), torch.float16),
    ]


@pytest.fixture
def operator(monkeypatch):
    op = Mock(return_value=("key", "scale"))
    # Keep the older-build test independent of whether a real extension has
    # registered the operator in this test process's dispatcher.
    api = SimpleNamespace(
        bfloat16=torch.bfloat16,
        float16=torch.float16,
        int8=torch.int8,
        int32=torch.int32,
        int64=torch.int64,
        ops=SimpleNamespace(_C_ascend=SimpleNamespace(li_quant_cache=op)),
    )
    monkeypatch.setattr(li_quant_cache, "torch", api)
    return op


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_dispatch_preserves_paged_cache_storage(arguments, operator, dtype, slot_dtype):
    arguments[0].dtype, arguments[1].dtype = dtype, slot_dtype
    assert li_quant_cache.try_li_quant_cache(*arguments) == ("key", "scale")
    x, slots, key_cache, scale_cache = operator.call_args.args
    assert x is arguments[0] and slots is arguments[1]
    assert key_cache.shape == (2048, 128) and key_cache.dtype == torch.int8
    assert scale_cache.shape == (2048, 1) and scale_cache.dtype == torch.float16


@pytest.mark.parametrize(
    "index,replacement",
    [
        (0, TensorMetadata((0, 128), torch.bfloat16)),
        (0, TensorMetadata((5, 128), torch.bfloat16)),
        (0, TensorMetadata((6, 64), torch.bfloat16)),
        (0, TensorMetadata((6, 128), torch.float32)),
        (0, TensorMetadata((6, 128), torch.bfloat16, contiguous=False)),
        (1, TensorMetadata((6, 1), torch.int64)),
        (1, TensorMetadata((6,), torch.int16)),
        (1, TensorMetadata((6,), torch.int64, contiguous=False)),
        (2, TensorMetadata((0, 128), torch.int8)),
        (2, TensorMetadata((2048, 128), torch.int8, index=1)),
        (2, TensorMetadata((2048, 128), torch.int8, device="cpu")),
        (2, TensorMetadata((2048, 128), torch.int8, contiguous=False)),
        (2, TensorMetadata((2048, 64), torch.int8)),
        (2, TensorMetadata((2048, 128), torch.float16)),
        (3, TensorMetadata((2047, 1), torch.float16)),
        (3, TensorMetadata((2048, 1), torch.float32)),
        (3, TensorMetadata((2048, 1), torch.float16, contiguous=False)),
    ],
)
def test_unsupported_metadata_never_launches(arguments, operator, index, replacement):
    arguments[index] = replacement
    assert li_quant_cache.try_li_quant_cache(*arguments) is None
    operator.assert_not_called()


def test_missing_build_falls_back(arguments, operator):
    del li_quant_cache.torch.ops._C_ascend.li_quant_cache
    assert li_quant_cache.try_li_quant_cache(*arguments) is None
    operator.assert_not_called()


def test_operator_error_is_not_retried(arguments, operator):
    operator.side_effect = RuntimeError("kernel failed")
    with pytest.raises(RuntimeError, match="kernel failed"):
        li_quant_cache.try_li_quant_cache(*arguments)
    operator.assert_called_once()
