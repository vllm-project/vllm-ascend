# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.models.qwen4_exp.lightning_indexer import (
    is_qsa_lightning_cache_compatible,
)


@pytest.mark.parametrize("page_size", [192, 384, 768])
def test_accepts_dynamic_page_size(page_size: int) -> None:
    cache = torch.empty((2, page_size, 1, 128), dtype=torch.bfloat16)

    assert is_qsa_lightning_cache_compatible(cache)


def test_accepts_axis0_noncontiguous_cache() -> None:
    storage = torch.empty((4, 768, 1, 128), dtype=torch.bfloat16)
    cache = storage[::2]

    assert cache.shape == (2, 768, 1, 128)
    assert cache.stride() == (196608, 128, 128, 1)
    assert not cache.is_contiguous()
    assert is_qsa_lightning_cache_compatible(cache)


def test_accepts_runtime_singleton_head_stride() -> None:
    storage = torch.empty(2 * 768 * 128, dtype=torch.bfloat16)
    cache = torch.as_strided(
        storage,
        size=(2, 768, 1, 128),
        stride=(98304, 128, 98304, 1),
    )

    assert cache.stride() == (98304, 128, 98304, 1)
    assert cache.is_contiguous()
    assert is_qsa_lightning_cache_compatible(cache)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 768, 1, 128),
        (2, 0, 1, 128),
        (2, 768, 2, 128),
        (2, 768, 1, 64),
    ],
)
def test_rejects_invalid_cache_shape(shape: tuple[int, ...]) -> None:
    cache = torch.empty(shape, dtype=torch.bfloat16)

    assert not is_qsa_lightning_cache_compatible(cache)
