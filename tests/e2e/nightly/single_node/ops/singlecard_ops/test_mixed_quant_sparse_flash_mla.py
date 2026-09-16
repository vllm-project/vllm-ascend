# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Run the original MQSMLA C8 accuracy case through the vLLM Ascend binding."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from vllm_ascend.device.device_config import is_950


def test_mixed_quant_sparse_flash_mla_original_c8(tmp_path: Path) -> None:
    if not is_950():
        pytest.skip("MixedQuantSparseFlashMla requires Ascend 950 (A5)")

    source_dir = Path(__file__).resolve().parents[1] / "mixed_quant_sparse_flash_mla"
    repository = Path(__file__).resolve().parents[6]
    report = tmp_path / "original-c8.xml"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (repository, environment.get("PYTHONPATH", "")) if path
    )
    environment["MQSMLA_RESULT_SAVE_PATH"] = str(tmp_path / "original-c8.xlsx")
    environment["MQSMLA_BATCH_CONSISTENCY"] = "auto"
    environment["SAVE_PT"] = "0"
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(source_dir / "test_mixed_quant_sparse_flash_mla_single.py"),
        "-c",
        str(source_dir / "pytest.ini"),
        f"--confcutdir={source_dir}",
        f"--rootdir={source_dir}",
        f"--junitxml={report}",
        "-rA",
        "-s",
        "-v",
        "-m",
        "ci",
        "-W",
        "ignore::UserWarning",
        "-W",
        "ignore::DeprecationWarning",
    ]
    result = subprocess.run(
        command,
        cwd=source_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = result.stdout + result.stderr
    (tmp_path / "original-c8.log").write_text(output, encoding="utf-8")
    print(output)
    assert result.returncode == 0, f"Original C8 accuracy pytest failed ({result.returncode})"
    assert report.exists(), "Original C8 accuracy pytest produced no result report"
    cases = ET.parse(report).getroot().findall(".//testcase")
    assert len(cases) == 1, "Expected the one original decode_first baseline case"
    assert "CSA_decode_TND_BF16_PA_BBND_FP8_E4M3FN_B48_S12_S28192_D512_K1024" in cases[0].get("name", "")
    assert all(case.find(tag) is None for case in cases for tag in ("failure", "error", "skipped")), (
        "The original C8 accuracy case must pass without skipping"
    )


def test_mixed_quant_sparse_flash_mla_rope0_c8(tmp_path: Path) -> None:
    if not is_950():
        pytest.skip("MixedQuantSparseFlashMla requires Ascend 950 (A5)")

    source_dir = Path(__file__).resolve().parents[1] / "mixed_quant_sparse_flash_mla"
    repository = Path(__file__).resolve().parents[6]
    report = tmp_path / "rope0-c8.xml"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (repository, environment.get("PYTHONPATH", "")) if path
    )
    command = [
        sys.executable, "-m", "pytest",
        str(source_dir / "test_mixed_quant_sparse_flash_mla_rope0.py"),
        "-c", str(source_dir / "pytest.ini"),
        f"--confcutdir={source_dir}", f"--rootdir={source_dir}",
        f"--junitxml={report}", "-rA", "-s", "-v", "-m", "ci",
        "-W", "ignore::UserWarning", "-W", "ignore::DeprecationWarning",
    ]
    result = subprocess.run(
        command, cwd=source_dir, env=environment, capture_output=True, text=True, timeout=1800
    )
    output = result.stdout + result.stderr
    (tmp_path / "rope0-c8.log").write_text(output, encoding="utf-8")
    print(output)
    assert result.returncode == 0, f"RoPE0 C8 accuracy pytest failed ({result.returncode})"
    assert report.exists(), "RoPE0 C8 accuracy pytest produced no result report"
    cases = ET.parse(report).getroot().findall(".//testcase")
    assert len(cases) == 10, "Expected all ten D448/RoPE0 accuracy cases"
    assert all("test_mixed_quant_sparse_flash_mla_rope0[" in case.get("name", "") for case in cases)
    assert all(case.find(tag) is None for case in cases for tag in ("failure", "error", "skipped")), (
        "All RoPE0 C8 accuracy cases must pass without skipping"
    )
