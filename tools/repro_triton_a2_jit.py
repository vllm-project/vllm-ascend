#!/usr/bin/env python3
"""Reproduce Triton Ascend JIT compilation failures without starting vLLM.

Examples:
    python tools/repro_triton_a2_jit.py --kernel vector-add
    python tools/repro_triton_a2_jit.py --kernel gumbel

The script creates a fresh Triton cache directory by default so that an
existing compiled kernel cannot hide a compiler/toolchain mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("vector-add", "gumbel"),
        default="vector-add",
        help="Kernel to JIT compile. vector-add isolates Triton itself; "
        "gumbel follows the failing vLLM sampler path.",
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Triton cache directory. Defaults to a new directory under /tmp.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size used by the Gumbel reproducer.",
    )
    return parser.parse_args()


ARGS = parse_args()
CACHE_DIR = ARGS.cache_dir or Path(tempfile.mkdtemp(prefix="triton-a2-jit-"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# This must be configured before importing Triton.
os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR.resolve())

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_environment() -> None:
    print("=== Environment ===")
    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__}")
    print(f"torch_npu={distribution_version('torch-npu')}")
    print(f"triton={distribution_version('triton')}")
    print(f"triton_ascend={distribution_version('triton-ascend')}")
    print(f"triton_module={triton.__file__}")
    print(f"triton_cache={CACHE_DIR.resolve()}")

    owners = importlib.metadata.packages_distributions().get("triton", [])
    print(f"triton_namespace_owners={owners}")

    triton_root = Path(triton.__file__).resolve().parent
    hivmc_candidates = list(triton_root.glob("backends/ascend/**/hivmc"))
    if not hivmc_candidates:
        print("hivmc=NOT_FOUND")
        return

    for hivmc in hivmc_candidates:
        print(f"hivmc={hivmc}")
        print(f"hivmc_sha256={sha256(hivmc)}")
        result = subprocess.run(
            [str(hivmc), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        help_text = result.stdout + result.stderr
        print(f"hivmc_help_exit_code={result.returncode}")
        print(f"hivmc_has_target={'target' in help_text}")
        print(
            "hivmc_has_enable_triton_kernel_compile="
            f"{'enable-triton-kernel-compile' in help_text}"
        )


def run_vector_add() -> None:
    size = 4096
    block_size = 1024
    x = torch.rand(size, device="npu", dtype=torch.float32)
    y = torch.rand(size, device="npu", dtype=torch.float32)
    output = torch.empty_like(x)
    grid = (triton.cdiv(size, block_size),)

    print("=== Launch vector-add Triton kernel ===")
    vector_add_kernel[grid](
        x,
        y,
        output,
        size,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(output, x + y, rtol=1e-3, atol=1e-3)


def run_gumbel() -> None:
    # Import only in this mode so vector-add remains independent of vLLM.
    from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

    num_tokens = 1
    logits = torch.randn(
        num_tokens,
        ARGS.vocab_size,
        dtype=torch.float32,
        device="npu",
    )
    expanded_idx_mapping = torch.zeros(
        num_tokens,
        dtype=torch.int32,
        device="npu",
    )
    temperature = torch.ones(num_tokens, dtype=torch.float32, device="npu")
    seed = torch.tensor([12345], dtype=torch.int64, device="npu")
    pos = torch.arange(num_tokens, dtype=torch.int32, device="npu")

    print("=== Launch vLLM Ascend Gumbel Triton kernel ===")
    sampled = gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
    )
    torch.npu.synchronize()
    print(f"sampled={sampled.cpu().tolist()}")


def main() -> int:
    print_environment()
    torch.npu.set_device(ARGS.device)
    print(f"npu_device={ARGS.device}")
    print(f"npu_name={torch.npu.get_device_name(ARGS.device)}")

    try:
        if ARGS.kernel == "vector-add":
            run_vector_add()
        else:
            run_gumbel()
    except Exception:
        print("JIT_REPRO_RESULT=FAIL", file=sys.stderr)
        traceback.print_exc()
        return 1

    print("JIT_REPRO_RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
