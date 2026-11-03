# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the accuracy probe; no vLLM build or installation is needed."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import load


def build(args: argparse.Namespace) -> None:
    cann = args.cann_home.resolve()
    header_roots = (
        cann / "include",
        cann / "opp/built-in/op_impl/ai_core/tbe/op_api/include",
    )
    if args.op_include is not None:
        header_roots = (args.op_include.resolve(), *header_roots)
    headers = [path for root in header_roots if root.is_dir() for path in root.rglob("aclnn_quant_all_reduce.h")]
    if not headers:
        raise FileNotFoundError(
            "Missing SDK header aclnn_quant_all_reduce.h. Locate it with find under your CANN install, "
            "then pass --op-include DIR. This tool compiles against the installed ABI; it does not guess it."
        )
    npu_root = Path(torch_npu.__file__).resolve().parent
    library_dirs = [
        path
        for path in (
            cann / "lib64",
            cann / "arm64-linux/lib64",
            cann / "aarch64-linux/lib64",
            cann / "x86_64-linux/lib64",
            cann / "opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64",
            npu_root / "lib",
        )
        if path.is_dir()
    ]
    include_dirs = [
        path
        for path in (
            cann / "include",
            cann / "arm64-linux/include",
            cann / "aarch64-linux/include",
            cann / "x86_64-linux/include",
            headers[0].parent,
            headers[0].parent.parent,
            npu_root / "include",
            npu_root / "include/third_party/hccl/inc",
            npu_root.parent,
        )
        if path.is_dir()
    ]
    op_library = next(
        (
            directory / name
            for name in ("libopapi_transformer.so", "libopapi.so")
            for directory in library_dirs
            if (directory / name).is_file()
        ),
        None,
    )
    if op_library is None:
        raise FileNotFoundError(f"No libopapi_transformer.so or libopapi.so under {cann}")
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"SDK header: {headers[0]}\nOperator library: {op_library}", flush=True)
    library = load(
        name="a5_quant_comm_probe",
        sources=[str(Path(__file__).with_name("quant_comm.cpp"))],
        extra_include_paths=list(dict.fromkeys(map(str, include_dirs))),
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=[
            *[f"-L{path}" for path in library_dirs],
            *[f"-Wl,-rpath,{path}" for path in library_dirs],
            str(op_library),
            "-lnnopbase",
            "-lascendcl",
            "-ltorch_npu",
        ],
        build_directory=str(build_dir),
        is_python_module=False,
        verbose=True,
    )
    manifest = build_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "library": str(library),
                "op_library": str(op_library),
                "sdk_header": str(headers[0]),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Built and loaded successfully. Manifest: {manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-home", type=Path, required=True, help="e.g. /usr/local/Ascend/cann-9.2.0")
    parser.add_argument("--op-include", type=Path, help="directory containing aclnn_quant_all_reduce.h")
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/a5_quant_comm_probe"))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
