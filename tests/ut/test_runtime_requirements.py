# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from collections import Counter
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeRequirementsTest(unittest.TestCase):
    def test_runtime_source_contains_only_unique_pep508_requirements(self):
        names = []
        for line in (REPO_ROOT / "requirements/runtime.txt").read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                requirement = Requirement(line)
                names.append(requirement.name.lower().replace("_", "-"))
        self.assertTrue(names)
        self.assertEqual(len(names), len(set(names)))

    def test_setup_entry_point_resolves_the_runtime_source(self):
        # Load only the existing reader; importing setup.py would probe CANN
        # and invoke setuptools rather than testing dependency consumption.
        source = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
        reader = next(
            node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "get_requirements"
        )
        module = ast.Module(body=[reader], type_ignores=[])
        namespace: dict[str, Any] = {"get_path": lambda filename: str(REPO_ROOT / filename)}
        exec(compile(module, str(REPO_ROOT / "setup.py"), "exec"), namespace)
        resolved = namespace["get_requirements"]()
        expected = (REPO_ROOT / "requirements/runtime.txt").read_text(encoding="utf-8").splitlines()

        def normalize(lines):
            return Counter(str(Requirement(line)) for line in lines if line.strip() and not line.startswith("#"))

        self.assertEqual(normalize(resolved), normalize(expected))


if __name__ == "__main__":
    unittest.main()
