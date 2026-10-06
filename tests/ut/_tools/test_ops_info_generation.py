# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real CMake ops-info rule without a CANN/NPU dependency."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("cmake") and shutil.which("ninja"), "CMake and Ninja are required")
class TestOpsInfoGeneration(unittest.TestCase):
    def test_incremental_schema_and_deleted_custom_copy(self):
        root = Path(__file__).resolve().parents[3]
        cmake_source = (root / "csrc/cmake/func.cmake").read_text(encoding="utf-8")
        start = cmake_source.index("function(add_ops_info_target)")
        end = cmake_source.index("endfunction()", start) + len("endfunction()")
        rule = cmake_source[start:end]
        parser = root / "csrc/cmake/scripts/util/parse_ini_to_json.py"
        for built_in in (False, True):
            with self.subTest(built_in=built_in), tempfile.TemporaryDirectory() as scratch:
                project = Path(scratch)
                autogen = project / "autogen"
                autogen.mkdir()
                ini = autogen / "aic-ascend950-ops-info.ini"
                ini.write_text(self._schema(1), encoding="utf-8")
                for child in ("inner", "exc"):
                    (autogen / child).mkdir()
                    (autogen / child / ini.name).write_text(
                        self._schema(1).replace("[TestSchema]", f"[{child.title()}Schema]"), encoding="utf-8"
                    )
                cmakelists = f"""cmake_minimum_required(VERSION 3.20)
project(ops_info_test NONE)
set(HI_PYTHON "{Path(sys.executable).as_posix()}")
set(ASCENDC_CMAKE_UTIL_DIR "{parser.parent.as_posix()}")
set(ASCEND_AUTOGEN_DIR "{autogen.as_posix()}")
set(base_aclnn_binary_dir "{autogen.as_posix()}")
set(CUSTOM_DIR "{project.as_posix()}/custom")
set(ENABLE_BUILT_IN {"ON" if built_in else "OFF"})
add_custom_target(opbuild_gen_default)
add_custom_target(opbuild_gen_inner)
add_custom_target(opbuild_gen_exc)
add_custom_target(generate_ops_info)
{rule}
add_ops_info_target(COMPUTE_UNIT ascend950)
"""
                (project / "CMakeLists.txt").write_text(cmakelists, encoding="utf-8")
                build = project / "build"
                self._run("cmake", "-S", str(project), "-B", str(build), "-G", "Ninja")
                name = "aic-ascend950-ops-info" + ("-transformer" if built_in else "") + ".json"
                generated = autogen / name
                custom = project / "custom/op_impl/ai_core/tbe/config/ascend950" / name
                self._run("cmake", "--build", str(build), "--target", "generate_ops_info")
                self._assert_outputs(generated, custom, 1)
                # Reproduce switching from an old schema to a new operator ABI
                # without touching CMakeLists or deleting the generated JSON.
                ini.write_text(self._schema(3), encoding="utf-8")
                self._run("cmake", "--build", str(build), "--target", "generate_ops_info")
                self._assert_outputs(generated, custom, 3)
                # A smaller operator set/schema must not retain trailing bytes
                # from the previous, longer JSON on the open file descriptor.
                ini.write_text(self._schema(1), encoding="utf-8")
                self._run("cmake", "--build", str(build), "--target", "generate_ops_info")
                self._assert_outputs(generated, custom, 1)
                custom.unlink()
                self._run("cmake", "--build", str(build), "--target", "generate_ops_info")
                self._assert_outputs(generated, custom, 1)

    @staticmethod
    def _schema(outputs):
        fields = ["[TestSchema]", "opFile.value=test_schema", "opInterface.value=test_schema"]
        for index in range(outputs):
            fields.extend(
                [
                    f"output{index}.name=out{index}",
                    f"output{index}.paramType=required",
                    f"output{index}.dtype=float16",
                    f"output{index}.format=ND",
                ]
            )
        return "\n".join(fields) + "\n"

    def _assert_outputs(self, generated, custom, expected):
        self.assertEqual(generated.read_bytes(), custom.read_bytes())
        schema = json.loads(generated.read_text())["TestSchema"]
        self.assertEqual(len([key for key in schema if key.startswith("output")]), expected)

    @staticmethod
    def _run(*args):
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise AssertionError(f"{args}:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
