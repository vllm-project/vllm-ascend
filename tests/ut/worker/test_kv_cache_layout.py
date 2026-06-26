# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker import kv_cache_layout


class HNDBackend:
    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        return (0, 1, 3, 2, 4)


def test_hnd_layout_rejected_on_non_a5() -> None:
    with (
        patch.object(kv_cache_layout, "get_kv_cache_layout", return_value="HND"),
        patch.object(kv_cache_layout, "get_ascend_device_type", return_value=AscendDeviceType.A2),
        pytest.raises(RuntimeError, match="only supported on Ascend A5"),
    ):
        kv_cache_layout.get_kv_cache_stride_order(HNDBackend, (2, 2, 3, 4, 5))


def test_hnd_layout_allowed_on_a5() -> None:
    with (
        patch.object(kv_cache_layout, "get_kv_cache_layout", return_value="HND"),
        patch.object(kv_cache_layout, "get_ascend_device_type", return_value=AscendDeviceType.A5),
    ):
        assert kv_cache_layout.get_kv_cache_stride_order(HNDBackend, (2, 2, 3, 4, 5)) == (0, 1, 3, 2, 4)


def test_hnd_split_cache_view_uses_a5_stride_order() -> None:
    raw_tensor = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32)

    with (
        patch.object(kv_cache_layout, "get_kv_cache_layout", return_value="HND"),
        patch.object(kv_cache_layout, "get_ascend_device_type", return_value=AscendDeviceType.A5),
    ):
        cache = kv_cache_layout.view_split_kv_cache_with_stride_order(
            raw_tensor,
            torch.float32,
            (2, 2, 3, 4, 5),
            (2, 3, 4, 5),
            HNDBackend,
        )

    assert cache.shape == (2, 3, 4, 5)
    assert cache.stride() == (60, 5, 15, 1)
