# SPDX-License-Identifier: Apache-2.0
"""A3 parity coverage for the official InplacePartialRotaryMul migration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

LEGACY_ROOT_ENV = "VLLM_ASCEND_LEGACY_ROOT"
REPO_ROOT = Path(__file__).parents[3]


def _run_operator(root: Path, custom_opp_path: str | None, output_path: Path, dtype: str) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not existing_pythonpath else f"{root}:{existing_pythonpath}"
    if custom_opp_path is None:
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    else:
        env["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run-inplace-partial-rotary-mul", str(output_path), dtype],
        cwd=root,
        env=env,
        check=True,
        timeout=180,
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_a3_official_inplace_partial_rotary_mul_matches_legacy_custom(tmp_path: Path, dtype: str) -> None:
    legacy_root_value = os.environ.get(LEGACY_ROOT_ENV)
    if not legacy_root_value:
        pytest.skip(f"set {LEGACY_ROOT_ENV} to an A3-built checkout to run operator parity")

    legacy_root = Path(legacy_root_value).resolve()
    legacy_opp = legacy_root / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "custom_transformer"
    if not legacy_opp.is_dir():
        pytest.fail(f"legacy A3 custom OPP is missing: {legacy_opp}")

    legacy_output = tmp_path / f"legacy-{dtype}.pt"
    official_output = tmp_path / f"official-{dtype}.pt"
    _run_operator(legacy_root, str(legacy_opp), legacy_output, dtype)
    _run_operator(REPO_ROOT, None, official_output, dtype)

    legacy = torch.load(legacy_output, map_location="cpu", weights_only=True)
    official = torch.load(official_output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(official, legacy, rtol=0, atol=0)


def _run_inplace_partial_rotary_mul(output_path: Path, dtype: str) -> None:
    import torch_npu  # noqa: F401  # Registers the NPU dispatch key.
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch.manual_seed(20260910)
    tensor_dtype = getattr(torch, dtype)
    x = torch.randn((2, 1, 4, 128), dtype=tensor_dtype, device="npu")
    cos = torch.randn((2, 1, 1, 64), dtype=tensor_dtype, device="npu")
    sin = torch.randn((2, 1, 1, 64), dtype=tensor_dtype, device="npu")

    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        import vllm_ascend.vllm_ascend_C  # noqa: F401  # Registers _C_ascend bindings.

        torch.ops._C_ascend.inplace_partial_rotary_mul(x, cos, sin, rotary_mode="interleave", partial_slice=[64, 128])
    else:
        inplace_partial_rotary_mul(x, cos, sin, rotary_mode="interleave", partial_slice=[64, 128])
    torch.npu.synchronize()
    torch.save(x.cpu(), output_path)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--run-inplace-partial-rotary-mul":
        raise SystemExit("this module is a pytest test or an internal parity runner")
    _run_inplace_partial_rotary_mul(Path(sys.argv[2]), sys.argv[3])
