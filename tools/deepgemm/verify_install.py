# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Verify the bundled DeepGEMM extension and its JIT inputs after installation."""

from pathlib import Path


def main() -> None:
    import torch_npu  # noqa: F401 -- initialize the extension dependencies

    from vllm_ascend.third_party import deep_gemm as package

    package_dir = Path(package.__file__).resolve().parent
    import vllm_ascend

    expected_dir = Path(vllm_ascend.__file__).resolve().parent / "third_party" / "deep_gemm"
    extension_path = Path(package._C.__file__).resolve()
    if package_dir != expected_dir or extension_path.parent != package_dir or not callable(package.fp8_fp4_mega_moe):
        raise RuntimeError(f"Invalid bundled DeepGEMM extension: {extension_path}")

    if not any((package_dir / "include" / "deep_gemm").rglob("*.hpp")):
        raise RuntimeError(f"Missing DeepGEMM JIT headers in {package_dir}")
    print(f"Verified bundled DeepGEMM at {extension_path}")


if __name__ == "__main__":
    main()
