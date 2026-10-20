# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Exercise the production CMake codegen rules without a CANN installation."""

import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _section(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end, source.index(start))]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class TestCustomOpCodegen(unittest.TestCase):
    def test_schema_change_reaches_interfaces_and_kernel_wrapper(self):
        generators = ["Unix Makefiles"]
        if shutil.which("ninja"):
            generators.append("Ninja")
        for generator in generators:
            with self.subTest(generator=generator), tempfile.TemporaryDirectory() as temporary:
                self._check_incremental_build(Path(temporary), generator)

    def _check_incremental_build(self, root: Path, generator: str):
        source = root / "source"
        build = root / "build"
        scripts = source / "cmake" / "scripts" / "util"
        scripts.mkdir(parents=True)
        (scripts / "opbuild.py").write_text(
            textwrap.dedent("""\
                import ctypes
                import os
                import sys
                from pathlib import Path

                library = ctypes.CDLL(sys.argv[1])
                count = library.schema_inputs()
                output = Path(sys.argv[2])
                output.mkdir(parents=True, exist_ok=True)
                project = os.environ["OPS_PROJECT_NAME"]
                stem = {"aclnn": "aclnn_default", "aclnnInner": "aclnnInner_inner",
                        "aclnnExc": "exc_proto"}[project]
                (output / (stem + ".cpp")).write_text(str(count))
                if project == "aclnn":
                    (output / (stem + ".h")).write_text(str(count))
                (output / "aic-test-ops-info.ini").write_text(str(count))
                """),
        )
        (scripts / "ascendc_impl_build.py").write_text(
            textwrap.dedent("""\
                import sys
                from pathlib import Path

                output = Path(sys.argv[5]) / "dynamic"
                output.mkdir(parents=True, exist_ok=True)
                inputs = Path(sys.argv[6])
                for name, directory in (("default", inputs), ("inner", inputs / "inner"),
                                        ("exc", inputs / "exc")):
                    value = (directory / "aic-test-ops-info.ini").read_text()
                    (output / (name + ".py")).write_text(value)
                """),
        )
        (scripts / "parse_ini_to_json.py").write_text(
            "import sys\nfrom pathlib import Path\n"
            "Path(sys.argv[4]).write_text(','.join(Path(p).read_text() for p in sys.argv[1:4]))\n",
        )
        custom_build = (REPO_ROOT / "csrc/cmake/custom_build.cmake").read_text()
        functions = (REPO_ROOT / "csrc/cmake/func.cmake").read_text()
        opbuild = _section(
            custom_build,
            "# ------------------------------------------------ opbuild",
            "# ------------------------------------------------ generate adapt py",
        )
        adapt = _section(
            custom_build, "add_custom_target(generate_transformer_adapt_py", "foreach (_op_name ${OP_LIST})"
        )
        info = _section(functions, "function(add_ops_info_target)", "endfunction()") + "endfunction()\n"
        copy = _section(
            functions, "            set(DYNAMIC_PY_FILE", "            add_custom_command(OUTPUT ${OP_BIN_OUT_DIR}"
        )

        targets = {"default": "op_host_aclnn", "inner": "op_host_aclnnInner", "exc": "op_host_aclnnExc"}
        cmake = textwrap.dedent(f'''\
            cmake_minimum_required(VERSION 3.20)
            project(codegen_regression CXX)
            set(BUILD_OPEN_PROJECT ON)
            set(ASCEND_COMPUTE_UNIT test)
            set(HI_PYTHON "{sys.executable}")
            set(OP_BUILD_TOOL "{sys.executable}" "{scripts}/opbuild.py")
            set(base_aclnn_binary_dir "${{CMAKE_BINARY_DIR}}/autogen")
            set(ASCEND_AUTOGEN_DIR "${{base_aclnn_binary_dir}}")
            set(ASCEND_IMPL_OUT_DIR "${{CMAKE_BINARY_DIR}}/impl")
            set(ASCENDC_CMAKE_UTIL_DIR "{scripts}")
            set(CUSTOM_DIR "${{CMAKE_BINARY_DIR}}/custom")
            set(generate_aclnn_srcs "${{base_aclnn_binary_dir}}/aclnn_default.cpp")
            set(generate_aclnn_headers "${{base_aclnn_binary_dir}}/aclnn_default.h")
            set(generate_aclnn_inner_srcs "${{base_aclnn_binary_dir}}/inner/aclnnInner_inner.cpp")
            set(generate_exclude_proto_srcs "${{base_aclnn_binary_dir}}/exc/exc_proto.cpp")
            ''')
        for name, target in targets.items():
            (source / f"{name}.cpp").write_text('extern "C" int schema_inputs() { return 14; }\n')
            cmake += f"add_library({target} SHARED {name}.cpp)\n"
        cmake += opbuild + adapt + info
        cmake += "add_custom_target(generate_ops_info ALL)\nadd_ops_info_target(COMPUTE_UNIT test)\n"
        cmake += 'set(OP_SRC_OUT_DIR "${CMAKE_BINARY_DIR}/kernel")\nfile(MAKE_DIRECTORY "${OP_SRC_OUT_DIR}")\n'
        for name in targets:
            cmake += f"set(op_file {name})\nset(op_type {name})\nset(OP_TARGET_NAME {name})\n" + copy
        cmake += "add_custom_target(kernel_wrappers ALL DEPENDS default_py_copy inner_py_copy exc_py_copy)\n"
        (source / "CMakeLists.txt").write_text(cmake)

        self._run("cmake", "-G", generator, "-S", str(source), "-B", str(build))
        self._run("cmake", "--build", str(build), "--parallel", "8")
        self._check_outputs(build, {name: 14 for name in targets})

        # Rebuild in place, exactly as pip's editable build does after an ABI change.
        expected = {}
        for offset, name in enumerate(targets):
            expected[name] = 15 + offset
            (source / f"{name}.cpp").write_text(
                f'extern "C" int schema_inputs() {{ return {expected[name]}; }}\n',
            )
        self._run("cmake", "--build", str(build), "--parallel", "8")
        self._check_outputs(build, expected)

        unchanged = [build / "autogen/aclnn_default.cpp", build / "kernel/default.py"]
        mtimes = [path.stat().st_mtime_ns for path in unchanged]
        self._run("cmake", "--build", str(build), "--parallel", "8")
        self._check_outputs(build, expected)
        self.assertEqual(mtimes, [path.stat().st_mtime_ns for path in unchanged])

    def _check_outputs(self, build: Path, expected: dict[str, int]):
        for name, value in expected.items():
            self.assertEqual((build / f"kernel/{name}.py").read_text(), str(value))
        self.assertEqual((build / "autogen/aclnn_default.h").read_text(), str(expected["default"]))
        info = build / "autogen/aic-test-ops-info.json"
        self.assertEqual(info.read_text(), ",".join(str(value) for value in expected.values()))
        installed = build / "custom/op_impl/ai_core/tbe/config/test/aic-test-ops-info.json"
        self.assertEqual(installed.read_text(), info.read_text())

    def _run(self, *command: str):
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
