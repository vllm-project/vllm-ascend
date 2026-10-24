# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from tools.export_requirements import export_requirements

REPO_ROOT = Path(__file__).resolve().parents[2]


class ProjectDependenciesTest(unittest.TestCase):
    def test_compatibility_export_is_current(self):
        export_requirements(REPO_ROOT, check=True)
        self.assertFalse((REPO_ROOT / "requirements/runtime.txt").exists())

    def test_export_preserves_direct_constraints_and_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\ndependencies = ["example>=1,<3", \'other[one]; python_version < "3.12"\']\n',
                encoding="utf-8",
            )
            export_requirements(root)
            self.assertEqual(
                (root / "requirements.txt").read_text().splitlines()[2:],
                ["example>=1,<3", 'other[one]; python_version < "3.12"'],
            )
            export_requirements(root, check=True)
            (root / "requirements.txt").write_text("example==99\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Regenerate"):
                export_requirements(root, check=True)

    def test_invalid_or_dynamic_declarations_are_rejected(self):
        for project in (
            'dynamic = ["dependencies"]',
            'dependencies = ["not a valid requirement"]',
            'dependencies = ["example\\n--no-deps"]',
        ):
            with self.subTest(project=project), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "pyproject.toml").write_text("[project]\n" + project + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    export_requirements(root)


if __name__ == "__main__":
    unittest.main()
