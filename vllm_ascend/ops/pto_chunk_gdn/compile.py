#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Bisheng JIT compilation for the PTO GDN megakernel on Ascend NPU.

The megakernel is compiled on first use and cached under
``vllm_ascend/ops/pto_chunk_gdn/kernels/compiled_lib/``.
Re-compilation is triggered when the C++ source mtime changes.

Environment variables:
    PTO_LIB_PATH            Path to pto-isa header directory (contains ``include/``).
                            Auto-detected from ``csrc/third_party/pto-isa`` in the
                            package source tree, then ``/sources/pto-isa`` fallback.
    ASCEND_TOOLKIT_HOME     Ascend toolkit root (required).
    GDN_NPU_DEVICE          NPU device for ``cube_core_num`` query (default ``npu:0``).
    VERBOSE_COMPILE         Set to ``1`` to print the full bisheng command.
    PTO_DYNAMIC_EXTRA_FLAGS Extra flags appended to every bisheng invocation.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file's installed location
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_VLLM_ASCEND_DIR = _THIS_DIR.parent.parent           # vllm_ascend/
_PACKAGE_ROOT = _VLLM_ASCEND_DIR.parent              # site-packages root

# C++ sources live in csrc/pto_chunk_gdn/ inside the source tree.
# In editable installs (pip install -e .) _PACKAGE_ROOT == repo root.
_CSRC_PTO = _PACKAGE_ROOT / "csrc" / "pto_chunk_gdn"
if not _CSRC_PTO.is_dir():
    _CSRC_PTO = _THIS_DIR / "csrc"

KERNELS_PTO: str = str(_CSRC_PTO)
KERNEL_INCLUDE: str = str(_CSRC_PTO / "include")

# Compiled .so cache — writable at runtime inside the installed package.
_COMPILED_DIR = _THIS_DIR / "kernels" / "compiled_lib"
COMPILED_DIR: str = str(_COMPILED_DIR)

_DRIVER_INC = "/usr/local/Ascend/driver/kernel/inc"

ASCEND_TOOLKIT_HOME: str = (
    os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH", "")
)
if not ASCEND_TOOLKIT_HOME:
    raise RuntimeError(
        "ASCEND_TOOLKIT_HOME (or ASCEND_HOME_PATH) must be set to the Ascend toolkit root."
    )


def _resolve_pto_lib_path() -> str:
    if "PTO_LIB_PATH" in os.environ:
        return os.environ["PTO_LIB_PATH"]
    submodule = _PACKAGE_ROOT / "csrc" / "third_party" / "pto-isa"
    if (submodule / "include").is_dir():
        os.environ["PTO_LIB_PATH"] = str(submodule)
        return str(submodule)
    fallback = "/sources/pto-isa"
    if os.path.isdir(os.path.join(fallback, "include")):
        os.environ["PTO_LIB_PATH"] = fallback
        return fallback
    return ASCEND_TOOLKIT_HOME


PTO_LIB_PATH: str = _resolve_pto_lib_path()

# ---------------------------------------------------------------------------
# Hardware: query cube_core_num
# ---------------------------------------------------------------------------
_npu_dev = os.environ.get("GDN_NPU_DEVICE", "npu:0")
try:
    BLOCK_DIM: int = int(
        getattr(torch.npu.get_device_properties(_npu_dev), "cube_core_num", 20)
    )
except (RuntimeError, AssertionError):
    BLOCK_DIM = 24


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def _common_flags(
    *,
    num_heads: int,
    key_heads: int,
    hidden_size: int,
    chunk_size: int,
) -> list[str]:
    flags = [
        "-fPIC", "-shared", "-xcce", "-DMEMORY_BASE", "-O2", "-std=gnu++17",
        "--cce-aicore-arch=dav-c220",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-Wno-macro-redefined", "-Wno-ignored-attributes",
        f"-I{KERNEL_INCLUDE}",
        f"-I{os.path.join(PTO_LIB_PATH, 'include')}",
        f"-I{ASCEND_TOOLKIT_HOME}/include",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/runtime",
        f"-I{ASCEND_TOOLKIT_HOME}/pkg_inc/profiling",
        f"-DGDN_H={num_heads}",
        f"-DGDN_HG={key_heads}",
        f"-DGDN_D={hidden_size}",
        f"-DGDN_C={chunk_size}",
    ]
    if os.path.isdir(_DRIVER_INC):
        flags.append(f"-I{_DRIVER_INC}")
    extra = os.environ.get("PTO_DYNAMIC_EXTRA_FLAGS", "").split()
    flags.extend(extra)
    return flags


@lru_cache(maxsize=None)
def compile_mega_kernel(
    *,
    num_heads: int = 16,
    key_heads: int | None = None,
    hidden_size: int = 128,
    chunk_size: int = 128,
    cpp_mtime_ns: int = 0,
) -> str:
    """Compile the fused PTO GDN megakernel and return the ``.so`` path.

    Args:
        num_heads:   Number of value heads H.
        key_heads:   Number of Q/K heads Hg (GQA; defaults to H if None).
        hidden_size: Head dimension D.
        chunk_size:  Chunk size C (must be 128).
        cpp_mtime_ns: Source file mtime for cache invalidation.

    Returns:
        Absolute path to the compiled ``.so`` file.
    """
    kh = key_heads if key_heads is not None else num_heads
    os.makedirs(COMPILED_DIR, exist_ok=True)
    cpp_path = os.path.join(KERNELS_PTO, "mega_kernel.cpp")
    lib_path = os.path.join(
        COMPILED_DIR,
        f"mega_kernel_H{num_heads}_Hg{kh}_D{hidden_size}_C{chunk_size}.so",
    )
    flags = _common_flags(
        num_heads=num_heads, key_heads=kh, hidden_size=hidden_size, chunk_size=chunk_size
    )
    cmd = ["bisheng", *flags, cpp_path, "-o", lib_path]
    if os.environ.get("VERBOSE_COMPILE"):
        print("compile:", " ".join(cmd))
    import logging
    logging.getLogger(__name__).info(
        "[pto_chunk_gdn] Compiling mega_kernel H=%d Hg=%d D=%d C=%d …",
        num_heads, kh, hidden_size, chunk_size,
    )
    subprocess.run(cmd, check=True, timeout=600)
    return lib_path
