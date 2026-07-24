#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""Hardware detection utilities for pull-request E2E tests.

Device detection uses ``npu-smi`` via a one-shot subprocess at first
call, mirroring ``setup.py``'s ``get_chip_type()`` logic.  The main
process never imports ``torch_npu``, so fork-based multiprocessing
(e.g. shared ``Counter``) works without side effects.
"""

from __future__ import annotations

import subprocess
import sys

_CACHED_SOC_VERSION: str | None = None

# SOC version → device type mapping, matching setup.py's gen_build_info().
_SOC_TO_DEVICE: dict[str, str] = {
    "910b": "A2",
    "910c": "A3",
    "310p": "_310P",
    "ascend910b1": "A2",
    "ascend910b2": "A2",
    "ascend910b2c": "A2",
    "ascend910b3": "A2",
    "ascend910b4": "A2",
    "ascend910b4-1": "A2",
    "ascend910_9391": "A3",
    "ascend910_9381": "A3",
    "ascend910_9372": "A3",
    "ascend910_9392": "A3",
    "ascend910_9382": "A3",
    "ascend910_9362": "A3",
    "ascend310p1": "_310P",
    "ascend310p3": "_310P",
    "ascend310p5": "_310P",
    "ascend310p7": "_310P",
    "ascend310p3vir01": "_310P",
    "ascend310p3vir02": "_310P",
    "ascend310p3vir04": "_310P",
    "ascend310p3vir08": "_310P",
}


def _get_device_type(soc_version: str) -> str | None:
    """Map a SOC version string (e.g. ``"ascend910_9362"``) to a
    device type (``"A2"``, ``"A3"``, ``"310P"``, ``"A5"``).
    Returns ``None`` for unknown SOC versions."""
    if "ascend950" in soc_version:
        return "A5"
    device = _SOC_TO_DEVICE.get(soc_version)
    if device is not None:
        return device.lstrip("_")
    return None


def detect_device_type() -> str:
    """Return the cached Ascend NPU chip type.

    Derives the type from the detailed SOC version via
    ``_SOC_TO_DEVICE``.  Returns ``"A2"``, ``"A3"``, ``"310P"``,
    ``"A5"``, or ``"unknown"``.
    """
    soc_version = detect_soc_version()
    if soc_version == "unknown":
        return "unknown"
    device = _get_device_type(soc_version)
    return device if device is not None else "unknown"


def detect_npu_count() -> int:
    """Return the number of visible Ascend NPU devices.

    Uses ``torch_npu.npu.device_count()`` via a one-shot subprocess
    (``ASCEND_RT_VISIBLE_DEVICES`` is already reflected by the NPU
    driver in that call).  Returns 0 on failure (e.g. CPU-only machine).
    """
    try:
        probe = "import torch_npu; print(torch_npu.npu.device_count())"
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            count = int(result.stdout.strip())
            if count > 0:
                return count
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass

    return 0


def get_value_from_lines(lines: list[str], key: str) -> str:
    for line in lines:
        line = " ".join(line.split())
        if key in line:
            return line.split(":")[-1].strip()
    return ""


def detect_soc_version() -> str:
    """Return the detailed SOC version string (e.g. ``"ascend910_9362"``).

    Uses ``npu-smi info -t board -i 0 -o json`` via a one-shot subprocess,
    with the same parsing logic as ``setup.py``'s ``get_chip_type()``.

    Returns ``"unknown"`` if detection fails.
    """
    global _CACHED_SOC_VERSION

    if _CACHED_SOC_VERSION is not None:
        return _CACHED_SOC_VERSION

    try:
        # Get NPU ID
        npu_info_lines = subprocess.check_output(["npu-smi", "info", "-l"]).decode().strip().split("\n")
        npu_id = int(get_value_from_lines(npu_info_lines, "NPU ID"))

        # Stage 1: query board info without -c flag
        board_info_lines = (
            subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id)]).decode().strip().split("\n")
        )

        # Check if Chip Name exists (Ascend950 includes it directly)
        chip_name = get_value_from_lines(board_info_lines, "Chip Name")

        # Stage 2: query with -c flag only if Chip Name not found (A2/A3/310P)
        if not chip_name:
            chip_info_lines = (
                subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(npu_id), "-c", "0"])
                .decode()
                .strip()
                .split("\n")
            )
        else:
            # Ascend950 already has complete info
            chip_info_lines = board_info_lines

        # Extract required fields
        chip_name = get_value_from_lines(chip_info_lines, "Chip Name")
        chip_type = get_value_from_lines(chip_info_lines, "Chip Type")
        npu_name = get_value_from_lines(chip_info_lines, "NPU Name")

        if "310" in chip_name:
            # 310P case
            assert chip_type
            _CACHED_SOC_VERSION = (chip_type + chip_name).lower()
        elif "910" in chip_name:
            if chip_type:
                # A2 case
                assert not npu_name
                _CACHED_SOC_VERSION = (chip_type + chip_name).lower()
            else:
                # A3 case
                assert npu_name
                _CACHED_SOC_VERSION = (chip_name + "_" + npu_name).lower()
        elif "950" in chip_name:
            assert npu_name
            _CACHED_SOC_VERSION = (chip_name + "_" + npu_name).lower()
        elif "a2g3" in chip_name.lower():
            _CACHED_SOC_VERSION = "ascend910b1"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Get chip info failed: {e}")
    except FileNotFoundError:
        raise RuntimeError("npu-smi not found. Ensure Ascend NPU drivers are installed and npu-smi is in PATH.")

    if _CACHED_SOC_VERSION is None:
        _CACHED_SOC_VERSION = "unknown"
    return _CACHED_SOC_VERSION
