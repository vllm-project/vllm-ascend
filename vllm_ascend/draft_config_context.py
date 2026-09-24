# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Scoped state for draft ``VllmConfig`` reconstruction.

Some external draft loaders retain the target ``model_config`` and pass the
actual draft config separately to ``get_model``. A ``ContextVar`` lets
configuration validation distinguish those reconstructed draft configs from
the original target config without relying on speculative method names.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DRAFT_CONFIG_METHOD: ContextVar[str | None] = ContextVar(
    "ascend_draft_config_method",
    default=None,
)


@contextmanager
def draft_config_loading(method: str) -> Iterator[None]:
    """Mark ``VllmConfig`` reconstruction performed by a draft loader."""
    token = _DRAFT_CONFIG_METHOD.set(method)
    try:
        yield
    finally:
        _DRAFT_CONFIG_METHOD.reset(token)


def is_draft_config_loading() -> bool:
    """Return whether the current context is reconstructing a draft config."""
    return _DRAFT_CONFIG_METHOD.get() is not None


def get_draft_config_loading_method() -> str | None:
    """Return the speculative method owning the current loader context."""
    return _DRAFT_CONFIG_METHOD.get()
