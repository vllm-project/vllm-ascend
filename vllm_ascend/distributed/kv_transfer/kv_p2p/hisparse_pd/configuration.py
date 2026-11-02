# SPDX-License-Identifier: Apache-2.0
"""HiSparse P/D's bounded MemFabric URMA configuration, before library loading."""

from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.metadata
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
URMA_SWAP_ENV = "MF_HYBM_URMA_SWAP_SPACE_SIZE"
DEPRECATED_URMA_SWAP_ENV = "HYBM_URMA_SWAP_SPACE_SIZE"
DEFAULT_URMA_SWAP_MIB = 16
MAX_URMA_SWAP_MIB = 64
SUPPORTED_MEMFABRIC_VERSION = "1.3.0"


def parse_urma_swap_mib(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("urma_swap_space_mib must be a canonical positive integer")
    text = str(value)
    if len(text) > 2 or re.fullmatch("[1-9][0-9]*", text) is None:
        raise ValueError("urma_swap_space_mib must be a canonical positive integer")
    result = int(text)
    if not 1 <= result <= MAX_URMA_SWAP_MIB:
        raise ValueError(f"urma_swap_space_mib must be within 1..{MAX_URMA_SWAP_MIB} MiB")
    return result


def validate_transport_options(extra):
    protocol = extra.get("memfabric_transfer_protocol")
    if protocol != "device_urma":
        raise ValueError("HiSparse P/D requires explicit memfabric_transfer_protocol='device_urma'")
    return (protocol, parse_urma_swap_mib(extra.get("urma_swap_space_mib", DEFAULT_URMA_SWAP_MIB)))


_preload_receipt = None


def configure_urma_staging(value):
    """Set the library's existing MiB option before any MemFabric dlopen.

    MemFabric 1.3.0 captures the environment in a global string at dlopen;
    Initialize later converts that saved string to bytes. An already-loaded
    core cannot be repaired by changing os.environ, even if no engine exists.
    """
    global _preload_receipt
    mib = parse_urma_swap_mib(value)
    if DEPRECATED_URMA_SWAP_ENV in os.environ:
        raise ValueError("Remove the deprecated HYBM_URMA_SWAP_SPACE_SIZE before starting HiSparse P/D")
    inherited = os.environ.get(URMA_SWAP_ENV)
    if inherited is not None and inherited != str(mib):
        raise ValueError(f"Inherited {URMA_SWAP_ENV} conflicts with the explicit HiSparse P/D MiB setting")
    if _preload_receipt is not None:
        if _preload_receipt["pid"] != os.getpid():
            raise RuntimeError("An inherited preload receipt cannot qualify a different process")
        if _preload_receipt["swap_mib"] != mib or inherited != str(mib):
            raise ValueError("URMA configuration changed after its verified pre-load setup")
        library = Path(_preload_receipt["library_path"])
        if hashlib.sha256(library.read_bytes()).hexdigest() != _preload_receipt["library_sha256"]:
            raise RuntimeError("MemFabric library changed after verified pre-load setup")
        return deepcopy(_preload_receipt)
    imported = any(
        (
            name == "memfabric_hybrid"
            or name.startswith("memfabric_hybrid.")
            or name in ("_pymf_transfer", "_pymf_hybrid", "_pymf_acc_offload")
            for name in tuple(sys.modules)
        )
    )
    try:
        mapped = "libmf_hybm_core.so" in Path("/proc/self/maps").read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(
            "Cannot verify that the MemFabric core is unloaded before configuring HiSparse P/D"
        ) from error
    if imported or mapped:
        raise RuntimeError(
            "MemFabric was already imported or its core library is mapped. Restart the worker with the URMA setting configured before any MemFabric import; changing the environment after dlopen cannot take effect."
        )
    package = importlib.metadata.distribution("memfabric-hybrid")
    if package.version != SUPPORTED_MEMFABRIC_VERSION:
        raise ValueError(f"HiSparse P/D URMA configuration currently requires MemFabric {SUPPORTED_MEMFABRIC_VERSION}")
    library = Path(package.locate_file("memfabric_hybrid/lib/libmf_hybm_core.so"))
    library_sha256 = hashlib.sha256(library.read_bytes()).hexdigest()
    os.environ[URMA_SWAP_ENV] = str(mib)
    logger.info(
        "HiSparse P/D device_urma staging configured before library load: %s MiB (%s bytes), MemFabric %s, core SHA256 %s",
        mib,
        mib * 1024 * 1024,
        package.version,
        library_sha256,
    )
    _preload_receipt = {
        "pid": os.getpid(),
        "library_path": str(library.resolve()),
        "verified_before_dlopen": True,
        "package_version": package.version,
        "library_sha256": library_sha256,
        "swap_mib": mib,
        "swap_bytes": mib * 1024 * 1024,
        "configured_before_library_load": True,
    }
    return deepcopy(_preload_receipt)
