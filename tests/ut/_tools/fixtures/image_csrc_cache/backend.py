# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Minimal network-free editable backend for a real CPU compiler/L1 roundtrip."""

import ctypes
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    name = "csrc_fixture-0.0.0.dist-info"
    destination = Path(metadata_directory) / name
    destination.mkdir()
    (destination / "METADATA").write_text("Metadata-Version: 2.1\nName: csrc-fixture\nVersion: 0.0.0\n")
    return name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    root = Path.cwd()
    output = root / "vllm_ascend"
    output.mkdir(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "csrc/scripts/build_cache.py",
            "run",
            "--cache-root",
            os.environ["VLLM_ASCEND_BUILD_CACHE_DIR"],
            "--domain",
            "third_party",
            "--unit",
            "transport-fixture",
            "--output-dir",
            str(output),
            "--environment-profile",
            "host-cxx",
            "--prepared-input",
            str(root / "hello.cpp"),
            "--recipe-value",
            "shared-library-fixture",
            "--environment-tool",
            "g++",
            "--normalize-path",
            str(root),
            "--artifact-include",
            "_C.so",
            "--",
            "g++",
            "-shared",
            "-nostdlib",
            "-fPIC",
            str(root / "hello.cpp"),
            "-o",
            str(output / "_C.so"),
        ],
        check=True,
    )
    native = ctypes.CDLL(str(output / "_C.so"))
    assert native.cache_fixture_value() in (42, 43)
    name = "csrc_fixture-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(Path(wheel_directory) / name, "w") as wheel:
        wheel.writestr(
            "csrc_fixture-0.0.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: csrc-fixture\nVersion: 0.0.0\n"
        )
        wheel.writestr(
            "csrc_fixture-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("csrc_fixture-0.0.0.dist-info/RECORD", "")
    return name
