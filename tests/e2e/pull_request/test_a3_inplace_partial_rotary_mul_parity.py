# SPDX-License-Identifier: Apache-2.0
"""A3 parity coverage for the official InplacePartialRotaryMul migration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

LEGACY_ROOT_ENV = "VLLM_ASCEND_LEGACY_ROOT"
REPO_ROOT = Path(__file__).parents[3]


def _run_operator(
    root: Path, custom_opp_path: str | None, output_path: Path, dtype: str, shape: tuple[int, ...]
) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root) if not existing_pythonpath else f"{root}:{existing_pythonpath}"
    if custom_opp_path is None:
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    else:
        env["ASCEND_CUSTOM_OPP_PATH"] = custom_opp_path

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-inplace-partial-rotary-mul",
            str(output_path),
            dtype,
            json.dumps(shape),
        ],
        cwd=root,
        env=env,
        check=True,
        timeout=180,
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("shape", [(2, 1, 4, 128), (16, 1, 64, 128)])
def test_a3_official_inplace_partial_rotary_mul_matches_legacy_custom(
    tmp_path: Path, dtype: str, shape: tuple[int, ...]
) -> None:
    legacy_root_value = os.environ.get(LEGACY_ROOT_ENV)
    if not legacy_root_value:
        pytest.skip(f"set {LEGACY_ROOT_ENV} to an A3-built checkout to run operator parity")
    assert legacy_root_value is not None

    legacy_root = Path(legacy_root_value).resolve()
    legacy_opp = legacy_root / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "custom_transformer"
    if not legacy_opp.is_dir():
        pytest.fail(f"legacy A3 custom OPP is missing: {legacy_opp}")

    legacy_output = tmp_path / f"legacy-{dtype}.pt"
    official_output = tmp_path / f"official-{dtype}.pt"
    _run_operator(legacy_root, str(legacy_opp), legacy_output, dtype, shape)
    _run_operator(REPO_ROOT, None, official_output, dtype, shape)

    legacy = torch.load(legacy_output, map_location="cpu", weights_only=True)
    official = torch.load(official_output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(official, legacy, rtol=0, atol=0)


@torch.inference_mode()
def _run_inplace_partial_rotary_mul(output_path: Path, dtype: str, shape: tuple[int, ...]) -> None:
    import torch_npu  # noqa: F401  # Registers the NPU dispatch key.
    import vllm_ascend.vllm_ascend_C as binding  # type: ignore[import-untyped]

    package_root = Path.cwd().resolve() / "vllm_ascend"
    assert Path(binding.__file__).resolve().is_relative_to(package_root), binding.__file__
    for vendor in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep):
        if vendor:
            assert Path(vendor).resolve().is_relative_to(package_root), vendor

    torch.manual_seed(20260910)
    tensor_dtype = getattr(torch, dtype)
    original = torch.randn(shape, dtype=tensor_dtype)
    # Use the same angle for each interleaved pair and seed inputs on the CPU.
    angles = torch.rand((shape[0], 1, 1, 32)).repeat_interleave(2, dim=-1)
    x = original.npu()
    cos = angles.cos().to(tensor_dtype).npu()
    sin = angles.sin().to(tensor_dtype).npu()

    # Both model paths retain this binding; the rebuilt OPP selects the provider.
    torch.ops._C_ascend.inplace_partial_rotary_mul(x, cos, sin, rotary_mode="interleave", partial_slice=[64, 128])
    torch.npu.synchronize()
    output = x.cpu()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[..., :64], original[..., :64], rtol=0, atol=0)
    assert not torch.equal(output[..., 64:], original[..., 64:])
    torch.save(output, output_path)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--run-inplace-partial-rotary-mul":
        raise SystemExit("this module is a pytest test or an internal parity runner")
    _run_inplace_partial_rotary_mul(Path(sys.argv[2]), sys.argv[3], tuple(json.loads(sys.argv[4])))
