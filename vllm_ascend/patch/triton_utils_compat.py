# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import sys
from importlib import import_module
from types import ModuleType
from typing import Any

_GLUON_MODULE_NAME = "triton.experimental.gluon"
_GLUON_LANGUAGE_MODULE_NAME = f"{_GLUON_MODULE_NAME}.language"


class _UnavailableGluonModule(ModuleType):
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            f"Triton Gluon attribute {name!r} is unavailable on Ascend. "
            "Gluon is only imported by vLLM for ROCm Inkling kernels."
        )


def _unsupported_aggregate(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(
        "Triton aggregate is unavailable on Ascend. It is only imported by vLLM for ROCm Inkling kernels."
    )


def install_triton_utils_compat() -> None:
    """Provide the ROCm-only Triton symbols imported globally by vLLM."""
    try:
        triton_experimental = import_module("triton.experimental")
        triton_core = import_module("triton.language.core")
    except ImportError:
        # 310P intentionally runs without Triton.
        return

    try:
        import_module(_GLUON_LANGUAGE_MODULE_NAME)
    except ImportError:
        gluon_language = _UnavailableGluonModule(_GLUON_LANGUAGE_MODULE_NAME)
        gluon = _UnavailableGluonModule(_GLUON_MODULE_NAME)
        gluon.language = gluon_language

        sys.modules[_GLUON_MODULE_NAME] = gluon
        sys.modules[_GLUON_LANGUAGE_MODULE_NAME] = gluon_language
        triton_experimental.gluon = gluon

    if not hasattr(triton_core, "_aggregate"):
        triton_core._aggregate = _unsupported_aggregate
