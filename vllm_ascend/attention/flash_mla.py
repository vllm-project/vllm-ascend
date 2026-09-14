# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


@lru_cache
def _get_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load packaged CANN 9.2 FlashMLA operators after device selection.

    The custom ``.run`` installs the operator binaries, while the
    ``cann_ops_transformer`` wheel registers their PyTorch dispatchers. Keep
    the dependency lazy so configurations without FlashMLA remain unaffected.
    """
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return (
            namespace.flash_mla_with_kvcache,
            namespace.flash_mla_with_kvcache_metadata,
        )
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "A5 FlashMLA requires flash_mla_with_kvcache and "
            "flash_mla_with_kvcache_metadata from a matching CANN 9.2 "
            "cann_ops_transformer custom package and Python wheel."
        ) from exc


def ensure_flash_mla_ops_loaded() -> None:
    """Fail during FlashMLA initialization when the external ABI is absent."""
    _get_flash_mla_ops()


def flash_mla_with_kvcache_metadata(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Dispatch metadata generation through the external CANN package."""
    _, metadata_op = _get_flash_mla_ops()
    return metadata_op(*args, **kwargs)


def flash_mla_with_kvcache(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch FlashMLA through the external CANN package."""
    attention_op, _ = _get_flash_mla_ops()
    return attention_op(*args, **kwargs)
