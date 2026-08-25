# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager

_HCCL_TEARDOWN_ENABLED = False


@contextmanager
def snapshot_hccl_teardown(enabled: bool) -> Iterator[None]:
    """Enable snapshot-specific HCCL teardown for the current cleanup."""
    global _HCCL_TEARDOWN_ENABLED
    previous = _HCCL_TEARDOWN_ENABLED
    _HCCL_TEARDOWN_ENABLED = enabled
    try:
        yield
    finally:
        _HCCL_TEARDOWN_ENABLED = previous


def is_snapshot_hccl_teardown_enabled() -> bool:
    return _HCCL_TEARDOWN_ENABLED
