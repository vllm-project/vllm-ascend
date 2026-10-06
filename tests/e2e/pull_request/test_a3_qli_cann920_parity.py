# SPDX-License-Identifier: Apache-2.0
"""A3 CANN 9.2.0 QLI/metadata parity and operator-call latency."""

from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parents[3]
LEGACY_ROOT_ENV = "VLLM_ASCEND_LEGACY_ROOT"


def _check_cann_version() -> str:
    home = os.environ.get("ASCEND_HOME_PATH")
    if not home:
        raise RuntimeError("ASCEND_HOME_PATH must point to the CANN 9.2.0 final installation")
    root = Path(home).resolve()
    candidates = [root / "version.info", root / "opp" / "version.info"]
    override = os.environ.get("CANN_920_VERSION_INFO")
    if override:
        candidates.insert(0, Path(override))
    version_file = next((path for path in candidates if path.is_file()), None)
    if version_file is None:
        raise RuntimeError(f"CANN version.info not found under {root}")
    version = version_file.read_text(errors="replace")
    if not re.search(r"(?im)^Version\s*=\s*9\.2\.0\s*$", version):
        raise RuntimeError(f"Expected CANN 9.2.0 final, got {version_file}: {version[:300]}")
    if "9.2.0-beta" in str(root) or "9.2.0-beta" in version:
        raise RuntimeError("CANN beta installation is not acceptable")
    for name in ("ASCEND_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
        value = os.environ.get(name, "")
        if "cann-9.1" in value or "9.2.0-beta" in value:
            raise RuntimeError(f"Mixed CANN installation in {name}: {value}")
    return str(version_file)


@pytest.mark.parametrize(
    ("version", "accepted"),
    [("9.2.0", True), ("9.2.0-beta.2", False), ("9.1.0", False)],
)
def test_cann_version_guard(tmp_path: Path, monkeypatch, version: str, accepted: bool) -> None:
    (tmp_path / "version.info").write_text(f"Version={version}\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.delenv("CANN_920_VERSION_INFO", raising=False)
    for name in ("ASCEND_OPP_PATH", "LD_LIBRARY_PATH", "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    if accepted:
        assert _check_cann_version() == str(tmp_path / "version.info")
    else:
        with pytest.raises(RuntimeError, match="Expected CANN 9.2.0 final"):
            _check_cann_version()


def test_cann_version_guard_rejects_mixed_library_path(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "version.info").write_text("Version=9.2.0\n")
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path))
    monkeypatch.delenv("CANN_920_VERSION_INFO", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/cann-9.1.0/lib64")
    with pytest.raises(RuntimeError, match="Mixed CANN installation"):
        _check_cann_version()


def _run(root: Path, output: Path, mode: str, strided: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    if mode == "legacy":
        opp = root / "vllm_ascend" / "_cann_ops_custom" / "vendors" / "custom_transformer"
        if not opp.is_dir():
            pytest.fail(f"legacy custom OPP missing: {opp}")
        env["ASCEND_CUSTOM_OPP_PATH"] = str(opp)
    else:
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--run", mode, str(output), str(int(strided))],
        cwd=root,
        env=env,
        check=True,
        timeout=600,
    )


@pytest.mark.parametrize("strided", [False, True], ids=["paged-contiguous", "paged-axis0-strided"])
def test_qli_metadata_and_topk_cann920_parity_and_latency(tmp_path: Path, strided: bool) -> None:
    legacy_root_name = os.environ.get(LEGACY_ROOT_ENV)
    if not legacy_root_name:
        pytest.skip(f"set {LEGACY_ROOT_ENV} to an A3-built checkout before the QLI migration")
    _check_cann_version()
    legacy_root = Path(legacy_root_name).resolve()
    legacy_output = tmp_path / "legacy.pt"
    official_output = tmp_path / "official.pt"
    _run(legacy_root, legacy_output, "legacy", strided)
    _run(REPO_ROOT, official_output, "official", strided)

    legacy = torch.load(legacy_output, map_location="cpu", weights_only=True)
    official = torch.load(official_output, map_location="cpu", weights_only=True)
    assert legacy["key_stride"] == official["key_stride"]
    assert legacy["scale_stride"] == official["scale_stride"]
    torch.testing.assert_close(official["metadata"], legacy["metadata"], rtol=0, atol=0)
    torch.testing.assert_close(official["indices"], legacy["indices"], rtol=0, atol=0)
    for op in ("metadata", "qli", "combined"):
        old_us, new_us = legacy["latency_us"][op], official["latency_us"][op]
        print(f"{op}: legacy {old_us:.3f} us, official {new_us:.3f} us, delta {(new_us / old_us - 1) * 100:+.1f}%")


def _latency_us(call, iterations: int = 1000) -> float:
    for _ in range(20):
        call()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(samples)


def _run_on_npu(mode: str, output: Path, strided: bool) -> None:
    version_file = _check_cann_version()
    import torch_npu  # noqa: F401

    from vllm_ascend.device.hardware import AscendDeviceType, device_type_from_runtime_soc

    soc = torch_npu.npu.get_soc_version()
    if device_type_from_runtime_soc(soc) != AscendDeviceType.A3:
        raise RuntimeError(f"This comparison requires A3, got SoC version {soc}")

    if mode == "legacy":
        import vllm_ascend.vllm_ascend_C as extension

        if not Path(extension.__file__).resolve().is_relative_to(Path.cwd()):
            raise RuntimeError(f"legacy binding is from the wrong checkout: {extension.__file__}")
        metadata_op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata
        qli_op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2
    else:
        from cann_ops_transformer import ops

        metadata_op = ops.quant_lightning_indexer_metadata
        qli_op = ops.quant_lightning_indexer

    torch.manual_seed(20260924)
    seq_lens = torch.tensor([2049, 2050], dtype=torch.int32, device="npu")
    compressed_lens = torch.div(seq_lens, 4, rounding_mode="floor")
    residual_lens = torch.remainder(seq_lens, 4)
    query = torch.randint(-127, 128, (3, 64, 128), dtype=torch.int8).npu()
    weights = torch.rand((3, 64), dtype=torch.float16).npu()
    query_scale = torch.rand((3, 64), dtype=torch.float16).add_(0.01).npu()
    key = torch.randint(-127, 128, (10, 128, 1, 128), dtype=torch.int8).npu()
    key_scale = torch.rand((10, 128, 1), dtype=torch.float16).add_(0.01).npu()
    if strided:
        key_storage = torch.empty((20, 128, 1, 128), dtype=torch.int8, device="npu")
        scale_storage = torch.empty((20, 128, 1), dtype=torch.float16, device="npu")
        key_storage[::2].copy_(key)
        scale_storage[::2].copy_(key_scale)
        key, key_scale = key_storage[::2], scale_storage[::2]
    cu_seqlens_q = torch.tensor([0, 1, 3], dtype=torch.int32, device="npu")
    block_table = torch.tensor([[7, 2, 6, 0, 9], [1, 4, 8, 3, 5]], dtype=torch.int32, device="npu")
    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=compressed_lens,
        cmp_residual_k=residual_lens,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=4,
    )
    metadata_kwargs = dict(common, batch_size=2, max_seqlen_q=2, max_seqlen_k=513)
    if mode == "legacy":
        metadata_kwargs["device"] = "npu"

    def make_metadata():
        return metadata_op(64, 1, 128, 512, 2, **metadata_kwargs)

    metadata = make_metadata()
    qli_kwargs = dict(common, block_table=block_table, metadata=metadata, return_value=0)

    def select_topk():
        return qli_op(query, key, weights, query_scale, key_scale, 512, 2, **qli_kwargs)[0]

    indices = select_topk()
    torch.npu.synchronize()
    result = {
        "version_file": version_file,
        "key_stride": tuple(key.stride()),
        "scale_stride": tuple(key_scale.stride()),
        "metadata": metadata.cpu(),
        "indices": indices.cpu(),
        "latency_us": {
            "metadata": _latency_us(make_metadata),
            "qli": _latency_us(select_topk),
            "combined": _latency_us(
                lambda: qli_op(
                    query, key, weights, query_scale, key_scale, 512, 2, **dict(qli_kwargs, metadata=make_metadata())
                )
            ),
        },
    }
    torch.save(result, output)


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "--run":
        raise SystemExit("run with pytest")
    _run_on_npu(sys.argv[2], Path(sys.argv[3]), bool(int(sys.argv[4])))
