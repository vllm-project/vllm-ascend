# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compile immutable Ascend C specializations before graph capture."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import vllm.envs as vllm_envs

from vllm_ascend.ops.statetree.config import validate_tiles

_KERNEL_ARCH = "dav-3510"


def get_kernel_source() -> Path:
    """Resolve wheel resources or the canonical source in an editable checkout."""
    packaged = Path(__file__).with_name("kernels") / "statetree.asc"
    if packaged.is_file():
        return packaged
    checkout = Path(__file__).resolve().parents[3] / "csrc" / "statetree" / "statetree.asc"
    if checkout.is_file():
        return checkout
    raise FileNotFoundError("StateTree kernel source is missing; reinstall vllm-ascend with its kernel resources")


@dataclass(frozen=True)
class _CompiledKernel:
    library: ctypes.CDLL
    launch: Any
    decode: Any


@cache
def build_kernel(
    batch_tile: int = 16,
    n_tile: int = 128,
    k_tile: int = 256,
    escape_slots: int = 8,
) -> _CompiledKernel:
    """Compile and cache one immutable Ascend C specialization."""
    validate_tiles(batch_tile, n_tile, k_tile, escape_slots)
    compiler = shutil.which("bisheng")
    if compiler is None:
        raise RuntimeError("StateTree requires bisheng; source the CANN environment before starting vLLM")
    source = get_kernel_source()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cache_dir = Path(vllm_envs.VLLM_CACHE_ROOT) / "ascend" / "statetree"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / (f"kernel_{source_hash[:12]}_{batch_tile}_{n_tile}_{k_tile}_{escape_slots}.so")
    lock_path = output.with_suffix(".so.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.exists():
            temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
            try:
                command = [
                    compiler,
                    str(source),
                    "-o",
                    str(temporary),
                    f"--npu-arch={_KERNEL_ARCH}",
                    "-O3",
                    "-shared",
                    "-fPIC",
                    f"-DST_BM={batch_tile}",
                    f"-DST_BN={n_tile}",
                    f"-DST_BK={k_tile}",
                    f"-DST_ES={escape_slots}",
                ]
                subprocess.run(command, check=True)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)

    library = ctypes.CDLL(str(output))
    launch = library.statetree_launch
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint32] * 3
    launch.restype = ctypes.c_int
    decode = library.statetree_decode_launch
    decode.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 3
    decode.restype = ctypes.c_int
    return _CompiledKernel(library, launch, decode)
