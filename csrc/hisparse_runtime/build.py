# SPDX-License-Identifier: Apache-2.0
"""Build the HiSparse native libraries for an explicit CANN target.

Run in the matching A5 CANN 9.2 / torch 2.10 / torch-npu 2.10 environment.
Outputs stay in --output; installing into a serving environment is explicit.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npu-arch", required=True, help="SIMT-capable bisheng target architecture")
    parser.add_argument(
        "--resolver-ub-bytes", type=int, required=True, help="Usable per-AIV UB bytes, including descriptors"
    )
    parser.add_argument("--install", action="store_true", help="Install built libraries into this checkout")
    args = parser.parse_args()
    if args.resolver_ub_bytes <= 512:
        parser.error("--resolver-ub-bytes must leave space beyond the 512-byte descriptor")
    import torch

    spec = importlib.util.find_spec("torch_npu")
    if spec is None or spec.origin is None:
        raise RuntimeError("torch_npu must be installed in this build environment")
    npu = Path(spec.origin).parent
    torch_root = Path(torch.__file__).parent
    source = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            str(args.cann_root / "bin/bisheng"),
            "-x",
            "asc",
            f"--npu-arch={args.npu_arch}",
            f"-DHISPARSE_RESOLVER_UB_BYTES={args.resolver_ub_bytes}",
            "-O2",
            "-shared",
            "-fPIC",
            "-std=c++17",
            str(source / "resolver.asc"),
            "-o",
            str(output / "hisparse_resolver60103_args.so"),
        ],
        check=True,
    )
    common = [
        "g++",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-O2",
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "-I" + str(npu / "include"),
        "-I" + str(torch_root / "include"),
        "-I" + str(torch_root / "include/torch/csrc/api/include"),
        "-I" + str(args.cann_root / "include"),
        "-L" + str(torch_root / "lib"),
        "-L" + str(npu / "lib"),
        "-L" + str(args.cann_root / "lib64"),
    ]
    subprocess.run(
        common
        + [
            str(source / "torch_binding.cpp"),
            "-o",
            str(output / "torch_binding.so"),
            "-L" + str(output),
            "-l:hisparse_resolver60103_args.so",
            "-Wl,-rpath,$ORIGIN",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-ltorch_npu",
            "-lascendcl",
        ],
        check=True,
    )

    subprocess.run(
        common
        + [
            str(source / "batch_copy.cpp"),
            str(source / "host_registration.cpp"),
            "-o",
            str(output / "hisparse_pta_host2511.so"),
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-ltorch_npu",
            "-lascendcl",
            "-ldl",
        ],
        check=True,
    )

    if args.install:
        import shutil

        package = source.parents[1] / "vllm_ascend"
        for name in ("hisparse_resolver60103_args.so", "torch_binding.so", "hisparse_pta_host2511.so"):
            shutil.copy2(output / name, package / "ops" / name)
        shutil.copy2(
            output / "hisparse_pta_host2511.so",
            package / "distributed/kv_transfer/kv_p2p/hisparse_pd/hisparse_pta_host2511.so",
        )


if __name__ == "__main__":
    main()
