# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the actual C++ loader with CPU-only ELF libraries, not NPU mocks."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SYMBOLS = (
    "aclnnQuantLightningIndexerV2GetWorkspaceSize",
    "aclnnQuantLightningIndexerV2",
    "aclnnQuantLightningIndexerV2MetadataGetWorkspaceSize",
    "aclnnQuantLightningIndexerV2Metadata",
)
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def loader(tmp_path):
    compiler = shutil.which("c++")
    if sys.platform != "linux" or compiler is None:
        pytest.skip("requires Linux ELF loader and a C++ compiler, but no NPU")
    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include "csrc/aclnn_torch_adapter/cann_op_api.h"\n'
        "#include <iostream>\n"
        "int main() {\n"
        '  dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL);\n'
        "  try {\n"
        + "".join(
            f'    std::cout << reinterpret_cast<int(*)()>(vllm_ascend::GetCannQliOpApiFuncAddr("{symbol}"))() << " ";\n'
            for symbol in SYMBOLS
        )
        + "  } catch (const std::exception& e) { std::cerr << e.what(); return 2; }\n"
        "}\n"
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT),
            str(source),
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    def build_library(name, symbols, value):
        source = tmp_path / f"{name}.cpp"
        source.write_text("\n".join(f'extern "C" int {symbol}() {{ return {value}; }}' for symbol in symbols))
        subprocess.run(
            [compiler, "-shared", "-fPIC", str(source), "-o", str(tmp_path / name)],
            check=True,
            capture_output=True,
        )

    def run():
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(tmp_path)
        env["ASCEND_CUSTOM_OPP_PATH"] = str(tmp_path)
        env.pop("LD_PRELOAD", None)
        return subprocess.run([str(executable)], env=env, capture_output=True, text=True, check=False)

    # Deliberately preloaded globally with every conflicting symbol.
    build_library("libcust_opapi.so", SYMBOLS, 99)
    for name in ("libopapi_transformer.so", "libopapi.so"):
        build_library(name, ["unrelated"], 0)
    return build_library, run


@pytest.mark.parametrize("official", ["libopapi_transformer.so", "libopapi.so"])
def test_official_provider_ignores_custom_symbols(loader, official):
    build, run = loader
    build(official, SYMBOLS, 7)
    result = run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["7"] * 4


def test_transformer_preferred_when_both_complete(loader):
    build, run = loader
    build("libopapi_transformer.so", SYMBOLS, 7)
    build("libopapi.so", SYMBOLS, 8)
    result = run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["7"] * 4


def test_incomplete_component_falls_back_as_a_group(loader):
    build, run = loader
    build("libopapi_transformer.so", SYMBOLS[:2], 7)
    build("libopapi.so", SYMBOLS, 8)
    result = run()
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["8"] * 4


@pytest.mark.parametrize("missing", SYMBOLS)
def test_missing_symbol_never_falls_back_to_custom_or_mixes_libraries(loader, missing):
    build, run = loader
    build("libopapi_transformer.so", [symbol for symbol in SYMBOLS if symbol != missing], 7)
    build("libopapi.so", [missing], 8)
    result = run()
    assert result.returncode == 2
    assert "No compatible official CANN operator library" in result.stderr
    assert missing in result.stderr


def test_no_official_symbols_fail_closed(loader):
    build, run = loader
    # Stub both names so the test cannot pick up a system CANN installation.
    for name in ("libopapi_transformer.so", "libopapi.so"):
        build(name, ["unrelated"], 0)
    result = run()
    assert result.returncode == 2
    assert "missing aclnnQuantLightningIndexerV2" in result.stderr
