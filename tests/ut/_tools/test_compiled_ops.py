# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from .build_cache_test_utils import REPO_ROOT

_CMAKE = "cmake"
pytestmark = pytest.mark.skipif(shutil.which(_CMAKE) is None, reason="CMake is required")
_WRITERS = ("common", "open_mc2", "rty", "rty_soc", "rty_mc2")


def _write_project(root: Path, writer: str) -> None:
    cmake_dir = REPO_ROOT / "csrc/cmake"
    collector = cmake_dir / "compiled_ops.cmake"
    root.mkdir(parents=True)
    (root / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.22)
project(compiled_ops_regression LANGUAGES CXX)
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
set(PKG_NAME fixture)
set(ASCEND_OP_NAME "")
set(ASCEND_CANN_PACKAGE_PATH "${{CMAKE_CURRENT_SOURCE_DIR}}/cann")
set(BUILD_OPEN_PROJECT ON)
if(EXISTS "{collector}")
    include("{collector}")
endif()
include("{cmake_dir}/func.cmake")
if("{writer}" STREQUAL "open_mc2")
    include("{cmake_dir}/variables.cmake")
    include("{cmake_dir}/obj_func.cmake")
else()
    include("{cmake_dir}/rty_obj_func.cmake")
endif()
foreach(name intf_pub_cxx17 intf_pub_cxx14 dlog_headers alog_headers slog_headers)
    add_library(${{name}} INTERFACE)
endforeach()
set(SELECTED "alpha;beta" CACHE STRING "Selected fixture operators")
foreach(name IN LISTS SELECTED)
    add_subdirectory(operators/${{name}}/op_host)
endforeach()
function(export_manifest)
    if(COMMAND get_compiled_ops)
        get_compiled_ops(COMPILED_OPS COMPILED_OP_DIRS)
    endif()
    file(WRITE "${{CMAKE_BINARY_DIR}}/manifest.txt" "${{COMPILED_OPS}}\n${{COMPILED_OP_DIRS}}\n")
endfunction()
export_manifest()
""",
        encoding="utf-8",
    )
    calls = {
        "common": "add_op_to_compiled_list()",
        "open_mc2": "add_mc2_modules_sources()",
        "rty": "add_modules_sources()",
        "rty_soc": "add_modules_sources_with_soc()",
        "rty_mc2": "set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})\nadd_mc2_modules_sources()",
    }
    for name in ("alpha", "beta"):
        directory = root / "operators" / name / "op_host"
        directory.mkdir(parents=True)
        (directory / "CMakeLists.txt").write_text(calls[writer] + "\n", encoding="utf-8")


def _configure(root: Path, selected: list[str], extra_args: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    result = subprocess.run(
        [_CMAKE, "-S", str(root), "-B", str(root / "build"), f"-DSELECTED={';'.join(selected)}", *extra_args],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = (root / "build/manifest.txt").read_text().splitlines()
    return [part for part in lines[0].split(";") if part], [part for part in lines[1].split(";") if part]


def _expected_dirs(root: Path, names: list[str]) -> list[str]:
    return [str(root / "operators" / name) for name in names]


@pytest.mark.parametrize("writer", _WRITERS)
def test_reconfigure_replaces_selected_operators(tmp_path, writer):
    root = tmp_path / "project with spaces"
    _write_project(root, writer)
    for selected in (["alpha", "beta"], ["alpha", "beta"], ["beta"], []):
        names, directories = _configure(root, selected)
        assert names == selected
        assert directories == _expected_dirs(root, selected)


@pytest.mark.parametrize("writer", _WRITERS)
def test_legacy_cache_does_not_survive_configuration(tmp_path, writer):
    root = tmp_path / "project"
    _write_project(root, writer)
    names, directories = _configure(
        root,
        ["beta"],
        ("-DCOMPILED_OPS=removed_operator", "-DCOMPILED_OP_DIRS=/removed/operator"),
    )
    assert names == ["beta"]
    assert directories == _expected_dirs(root, ["beta"])
    cache = (root / "build/CMakeCache.txt").read_text()
    assert "COMPILED_OPS:" not in cache
    assert "COMPILED_OP_DIRS:" not in cache


def test_collection_stays_with_its_csrc_directory(tmp_path):
    collector = REPO_ROOT / "csrc/cmake/compiled_ops.cmake"
    root = tmp_path / "parent"
    root.mkdir()
    for name in ("first", "second"):
        child = root / name
        child.mkdir()
        (child / "CMakeLists.txt").write_text(
            f"""
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
include("{collector}")
record_compiled_op({name}_op "${{CMAKE_CURRENT_SOURCE_DIR}}/{name}_op")
""",
            encoding="utf-8",
        )
    (root / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.22)
project(directory_isolation LANGUAGES NONE)
add_subdirectory(first)
add_subdirectory(second)
foreach(name first second)
    set(OPS_TRANSFORMER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${name}")
    get_compiled_ops(operators directories)
    if(NOT operators STREQUAL "${name}_op")
        message(FATAL_ERROR "Another project's operators leaked into ${name}: ${operators}")
    endif()
    if(NOT directories STREQUAL "${OPS_TRANSFORMER_DIR}/${name}_op")
        message(FATAL_ERROR "Incorrect directory for ${name}: ${directories}")
    endif()
endforeach()
""",
        encoding="utf-8",
    )
    result = subprocess.run([_CMAKE, "-S", str(root), "-B", str(root / "build")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_duplicate_registration_keeps_paired_order(tmp_path):
    collector = REPO_ROOT / "csrc/cmake/compiled_ops.cmake"
    root = tmp_path / "project"
    root.mkdir()
    (root / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.22)
project(registration_order LANGUAGES NONE)
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
include("{collector}")
record_compiled_op(zeta "${{OPS_TRANSFORMER_DIR}}/zeta")
record_compiled_op(alpha "${{OPS_TRANSFORMER_DIR}}/alpha")
record_compiled_op(zeta "${{OPS_TRANSFORMER_DIR}}/zeta")
get_compiled_ops(names directories)
if(NOT names STREQUAL "zeta;alpha")
    message(FATAL_ERROR "Duplicate registration changed the order: ${{names}}")
endif()
if(NOT directories STREQUAL "${{OPS_TRANSFORMER_DIR}}/zeta;${{OPS_TRANSFORMER_DIR}}/alpha")
    message(FATAL_ERROR "Names and directories are no longer paired: ${{directories}}")
endif()
""",
        encoding="utf-8",
    )
    result = subprocess.run([_CMAKE, "-S", str(root), "-B", str(root / "build")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_dependency_lookup_reads_current_collection(tmp_path):
    cmake_dir = REPO_ROOT / "csrc/cmake"
    root = tmp_path / "project"
    root.mkdir()
    existing = root / "operators/alpha"
    existing.mkdir(parents=True)
    (existing / "CMakeLists.txt").write_text('message(FATAL_ERROR "Completed dependency was added again")\n')
    pending = root / "operators/beta/op_host"
    pending.mkdir(parents=True)
    (pending.parent / "CMakeLists.txt").write_text("add_subdirectory(op_host)\n")
    (pending / "CMakeLists.txt").write_text("add_op_to_compiled_list()\n")
    (root / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.22)
project(dependency_collection LANGUAGES NONE)
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
include("{cmake_dir}/compiled_ops.cmake")
include("{cmake_dir}/func.cmake")
set(OPS_CATEGORY_LIST operators)
set(ASCEND_OP_NAME requested)
set(COMPILED_OPS stale_snapshot)
record_compiled_op(alpha "${{OPS_TRANSFORMER_DIR}}/operators/alpha")
add_dependent_ops("alpha;beta;beta")
get_compiled_ops(names directories)
if(NOT names STREQUAL "alpha;beta")
    message(FATAL_ERROR "Dependencies did not use the current collection: ${{names}}")
endif()
""",
        encoding="utf-8",
    )
    result = subprocess.run([_CMAKE, "-S", str(root), "-B", str(root / "build")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_distinct_operator_directories_are_retained(tmp_path):
    collector = REPO_ROOT / "csrc/cmake/compiled_ops.cmake"
    root = tmp_path / "project"
    root.mkdir()
    (root / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.22)
project(distinct_directories LANGUAGES NONE)
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
include("{collector}")
record_compiled_op(shared_name "${{OPS_TRANSFORMER_DIR}}/first/shared_name")
record_compiled_op(shared_name "${{OPS_TRANSFORMER_DIR}}/second/shared_name")
get_compiled_ops(names directories)
list(LENGTH names name_count)
list(LENGTH directories directory_count)
if(NOT name_count EQUAL 2 OR NOT directory_count EQUAL 2)
    message(FATAL_ERROR "A distinct source directory was omitted: ${{directories}}")
endif()
""",
        encoding="utf-8",
    )
    result = subprocess.run([_CMAKE, "-S", str(root), "-B", str(root / "build")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_kernel_source_copy_uses_current_collection(tmp_path):
    cmake_dir = REPO_ROOT / "csrc/cmake"
    root = tmp_path / "project"
    root.mkdir()
    for name in ("active", "stale"):
        directory = root / "operators" / name / "op_kernel"
        directory.mkdir(parents=True)
        (directory / f"{name}.cpp").write_text(f"// {name} kernel source\n")
    (root / "CMakeLists.txt").write_text(
        f"""
cmake_minimum_required(VERSION 3.22)
project(kernel_source_collection LANGUAGES NONE)
set(OPS_TRANSFORMER_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}")
include("{cmake_dir}/compiled_ops.cmake")
include("{cmake_dir}/gen_ops_info.cmake")
set(COMPILED_OPS stale)
set(COMPILED_OP_DIRS "${{OPS_TRANSFORMER_DIR}}/operators/stale")
record_compiled_op(active "${{OPS_TRANSFORMER_DIR}}/operators/active")
set(ASCEND_KERNEL_SRC_DST "${{CMAKE_BINARY_DIR}}/copied")
set(ASCEND_GRAPH_CONF_DST "${{CMAKE_BINARY_DIR}}/graph")
set(GRAPH_PLUGIN_NAME fixture)
add_library(fixture_proto_headers INTERFACE)
function(gen_aclnn_with_opdef)
    add_custom_target(opbuild_custom_gen_aclnn_all)
endfunction()
gen_ops_info_and_python()
""",
        encoding="utf-8",
    )
    for command in (
        [_CMAKE, "-S", str(root), "-B", str(root / "build")],
        [_CMAKE, "--build", str(root / "build"), "--target", "ascendc_kernel_src_copy"],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    copied = root / "build/copied"
    assert (copied / "active/active.cpp").read_text() == "// active kernel source\n"
    assert not (copied / "stale").exists()
