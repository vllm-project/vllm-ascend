# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check custom-op source completeness without requiring a CANN toolchain."""

import unittest
from pathlib import Path

CSRC = Path(__file__).resolve().parents[2] / "csrc"


class TestCustomOpSources(unittest.TestCase):
    def test_torch_adapter_headers_exist(self):
        binding = (CSRC / "torch_binding.cpp").read_text(encoding="utf-8")
        includes = (line.split('"')[1] for line in binding.splitlines() if line.startswith('#include "'))
        headers = [header for header in includes if header.endswith("_torch_adpt.h")]
        self.assertTrue(headers, "No custom-op adapter headers found")
        missing = [header for header in headers if not (CSRC / header).is_file()]
        self.assertEqual(missing, [], f"Missing custom-op adapter headers: {missing}")

    def test_selected_custom_ops_have_source_directories(self):
        build = (CSRC / "build_aclnn.sh").read_text(encoding="utf-8")
        arrays = [part.split(")", 1)[0] for part in build.split("CUSTOM_OPS_ARRAY=(")[1:]]
        self.assertTrue(arrays, "No custom-op build lists found")
        selected = {
            line.strip()[1:-1]
            for array in arrays
            for line in array.splitlines()
            if line.strip().startswith('"') and line.strip().endswith('"')
        }
        self.assertTrue(selected, "No selected custom ops found")
        # Match resolve_op_dir's fallback search depth of three under csrc.
        directories = {path.name for pattern in ("*", "*/*", "*/*/*") for path in CSRC.glob(pattern) if path.is_dir()}
        missing = sorted(selected - directories)
        self.assertEqual(missing, [], f"Selected custom ops have no source directory: {missing}")


if __name__ == "__main__":
    unittest.main()
