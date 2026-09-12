"""Compatibility helpers for Triton's experimental Gluon module hierarchy."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from types import ModuleType
from typing import Any, cast

from packaging.version import InvalidVersion, Version

_REAL_GLUON_MIN_VERSION = Version("3.6")


def _triton_version() -> Version | None:
    try:
        return Version(importlib.metadata.version("triton"))
    except (importlib.metadata.PackageNotFoundError, InvalidVersion):
        return None


def _install_legacy_gluon_stubs() -> None:
    """Install the complete hierarchy required by Triton before 3.6."""

    experimental = sys.modules.get("triton.experimental")
    if experimental is None:
        experimental = ModuleType("triton.experimental")
        experimental.__path__ = []
        sys.modules["triton.experimental"] = experimental

    gluon = sys.modules.get("triton.experimental.gluon")
    if gluon is None:
        gluon = ModuleType("triton.experimental.gluon")
        gluon.__path__ = []
        sys.modules["triton.experimental.gluon"] = gluon

    language = sys.modules.get("triton.experimental.gluon.language")
    if language is None:
        language = ModuleType("triton.experimental.gluon.language")
        sys.modules["triton.experimental.gluon.language"] = language

    cast(Any, experimental).gluon = gluon
    cast(Any, gluon).language = language


def ensure_gluon_compatibility() -> None:
    """Use real Gluon on Triton 3.6+, otherwise retain the legacy stub.

    Triton 3.6 ships a Gluon implementation compatible with its own runtime.
    Shadowing it with the historical empty vLLM-Ascend stub breaks native
    specialization during graph-mode compilation. Older Triton dependency
    lines either do not provide Gluon or provide one incompatible with their
    core, so they still require the complete compatibility hierarchy.
    """

    version = _triton_version()
    if version is not None and version >= _REAL_GLUON_MIN_VERSION:
        # Deliberately propagate failures from inside a real modern Gluon
        # package: replacing a broken installation with empty modules would
        # hide an ABI/package error until the first compiled kernel.
        importlib.import_module("triton.experimental.gluon")
        importlib.import_module("triton.experimental.gluon.language")
        return

    _install_legacy_gluon_stubs()
