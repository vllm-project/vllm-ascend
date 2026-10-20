# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regression for the actual C++ helper used by AICPU and AICore.

Can also run directly without importing vLLM, Torch, or the NPU plugin:
python tests/ut/attention/test_dsa_dcp_causal_bounds.py
"""

import pathlib
import shutil
import subprocess
import tempfile
import unittest


class TestDsaDcpCausalBounds(unittest.TestCase):
    def test_local_causal_prefix(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A C++ compiler is required for the native boundary regression")
        root = pathlib.Path(__file__).resolve().parents[3]
        source = root / "csrc/attention/common/tests/test_dcp_causal.cpp"
        with tempfile.TemporaryDirectory() as directory:
            binary = pathlib.Path(directory) / "test_dcp_causal"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                check=True,
            )
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
