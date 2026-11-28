# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build and launch the CATLASS QSFA kernel for the A3 MTP5 benchmark."""

import argparse
import ctypes
import os
import shlex
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

DEFAULT_LIBRARY = Path(__file__).resolve().with_name("libqsfa_catlass_a3.so")
_SOURCE = DEFAULT_LIBRARY.with_name("qsfa_catlass_a3.cpp")
_DEFAULT_CANN_PATH = Path("/usr/local/Ascend/ascend-toolkit/latest")
_CORES = 20
_QUERY_TOKENS = 6
_HEADS = 64
_LATENT_CHANNELS = 512
_QUERY_CHANNELS = 576
_PAGE_SIZE = 128
_PAGES_PER_REQUEST = 1024
_PACKED_WIDTH = 656
_TOPK = 2048
_PANEL_SHAPE = (_CORES, 2, 128, 512)


def _build(catlass_include: Path, cann_path: Path, output: Path) -> None:
    compiler = cann_path / "compiler" / "ccec_compiler" / "bin" / "bisheng"
    if not compiler.is_file():
        compiler = shutil.which("bisheng") or compiler
    command = [
        str(compiler),
        "-xasc",
        "--npu-arch=dav-2201",
        "-DCATLASS_ARCH=2201",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "--shared",
        "--cce-auto-sync=off",
        "-mllvm",
        "-cce-vf-remove-membar=false",
        "-mllvm",
        "-cce-aicore-hoist-movemask=false",
        f"-I{catlass_include}",
        f"-I{cann_path / 'include'}",
        f"-I{cann_path / 'include/experiment/msprof'}",
        f"-I{cann_path / 'include/experiment/runtime'}",
        f"-I{cann_path / 'pkg_inc'}",
        f"-I{cann_path / 'pkg_inc/runtime'}",
        f"-I{cann_path / 'pkg_inc/profiling'}",
        str(_SOURCE),
        f"-L{cann_path / 'lib64'}",
        "-lruntime",
        "-lascendcl",
        "-lm",
        "-ltiling_api",
        "-lplatform",
        "-lc_sec",
        "-ldl",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as build_dir:
        library = Path(build_dir) / output.name
        command += ["-o", str(library)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        library.replace(output)


@cache
def _load_launch(library_path: Path):
    if not library_path.is_file():
        raise FileNotFoundError(f"Build {library_path} with catlass_qsfa.py before running the benchmark.")
    launch = ctypes.CDLL(str(library_path)).qsfa_catlass_launch
    launch.argtypes = [ctypes.c_void_p] * 14 + [ctypes.c_int32, ctypes.c_float]
    launch.restype = ctypes.c_int
    return launch


def qsfa_a3(
    query: "Tensor",
    kv: "Tensor",
    indices: "Tensor",
    page_table: "Tensor",
    query_ends: "Tensor",
    kv_lengths: "Tensor",
    scale: float,
    *,
    library_path: Path = DEFAULT_LIBRARY,
) -> "Tensor":
    """Launch the fixed 128K, Top-2048, TP1/MTP5 fixture on the current NPU stream."""
    import torch

    batch = query_ends.numel()
    if not 1 <= batch <= 80:
        raise ValueError("CATLASS A3 MTP5 supports fixture batches from 1 to 80")
    rows = _QUERY_TOKENS * batch
    device = query.device
    if device.type != "npu":
        raise ValueError("CATLASS A3 MTP5 requires NPU tensors")
    expected = (
        (query, (rows, _HEADS, _QUERY_CHANNELS), torch.bfloat16),
        (kv, (batch * _PAGES_PER_REQUEST, _PAGE_SIZE, 1, _PACKED_WIDTH), torch.int8),
        (indices, (rows, 1, _TOPK), torch.int32),
        (page_table, (batch, _PAGES_PER_REQUEST), torch.int32),
        (query_ends, (batch,), torch.int32),
        (kv_lengths, (batch,), torch.int32),
    )
    for tensor, shape, dtype in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Expected contiguous {dtype} tensor on {device} with shape {shape}")

    launch = _load_launch(Path(library_path).resolve())
    output = torch.empty((rows, _HEADS, _LATENT_CHANNELS), dtype=torch.bfloat16, device=device)
    kv_nz = torch.empty((_CORES, 4, 36, 512, 16), dtype=torch.bfloat16, device=device)
    scores = torch.empty(_PANEL_SHAPE, dtype=torch.float32, device=device)
    probs = torch.empty(_PANEL_SHAPE, dtype=torch.bfloat16, device=device)
    partials = torch.empty(_PANEL_SHAPE, dtype=torch.float32, device=device)
    numerator = torch.empty(_PANEL_SHAPE, dtype=torch.float32, device=device)
    valid = torch.empty((_CORES, 4, 2, 128), dtype=torch.int32, device=device)
    tensors = (
        query,
        kv,
        indices,
        page_table,
        query_ends,
        kv_lengths,
        output,
        kv_nz,
        scores,
        probs,
        partials,
        numerator,
        valid,
    )
    stream = torch.npu.current_stream(device)
    status = launch(stream.npu_stream, *(tensor.data_ptr() for tensor in tensors), rows, float(scale))
    if status:
        raise RuntimeError(f"Could not initialize the CATLASS A3 cross-core sync base (runtime status {status})")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catlass-include", type=Path, required=True, help="CATLASS C++ repository include directory")
    parser.add_argument(
        "--cann-path",
        type=Path,
        default=os.environ.get("ASCEND_HOME_PATH", _DEFAULT_CANN_PATH),
        help="CANN toolkit root",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_LIBRARY, help="output shared library path")
    args = parser.parse_args()
    if not (args.catlass_include / "catlass/catlass.hpp").is_file():
        parser.error("--catlass-include must point to the CATLASS C++ include directory")
    _build(args.catlass_include.resolve(), args.cann_path.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
