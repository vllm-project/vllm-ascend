# SPDX-License-Identifier: Apache-2.0
"""Compile the WY compatibility header without CANN or device execution."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HEADER = (
    Path(__file__).resolve().parents[3] / "csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compat_310p.h"
)


class WYCompatibilityTests(unittest.TestCase):
    def compile_and_run(self, definitions, body):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("g++ or clang++ required for compatibility checks")
        source = f"""
#include <cstdint>
#define __CCE_KT_TEST__
#define PIPE_MTE3 3
{definitions}
#include "{HEADER.as_posix()}"
static_assert(PIPE_FIX == PIPE_MTE3);
int main() {{ {body} }}
"""
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "compat"
            result = subprocess.run(
                [compiler, "-std=c++17", "-x", "c++", "-o", str(executable), "-"],
                input=source,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(executable)], text=True, capture_output=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_bisheng_native_bfloat16_is_not_redeclared(self):
        self.compile_and_run(
            "#define __CCE_AICORE__ 200\n#define __BISHENG_CCEC__\ntypedef float bfloat16_t;",
            "return AscendC::ToFloat(bfloat16_t(1.25f)) == 1.25f ? 0 : 1;",
        )

    def test_legacy_compiler_retains_fallback(self):
        self.compile_and_run(
            "#define __CCE_AICORE__ 200",
            "return AscendC::ToFloat(bfloat16_t(1.25f)) == 0.f ? 0 : 1;",
        )

    def test_predeclared_legacy_type_retains_conversion_and_alias(self):
        self.compile_and_run(
            """
#define __CCE_AICORE__ 200
#define __bfloat16_t_defined
typedef float bfloat16_t;
int LoadDataWithSparseCal() { return 0; }
""",
            "return AscendC::ToFloat(bfloat16_t(1.25f)) == 1.25f ? LoadDataWithSparse() : 1;",
        )

    def test_other_architectures_do_not_enable_310p_shims(self):
        for architecture in (220, 310, None):
            with self.subTest(architecture=architecture):
                definitions = "" if architecture is None else f"#define __CCE_AICORE__ {architecture}\n"
                self.compile_and_run(
                    definitions
                    + """
typedef float bfloat16_t;
namespace AscendC { float ToFloat(bfloat16_t value) { return value; } }
int LoadDataWithSparse() { return 0; }
""",
                    "return AscendC::ToFloat(bfloat16_t(1.25f)) == 1.25f ? LoadDataWithSparse() : 1;",
                )


if __name__ == "__main__":
    unittest.main()
