import importlib.metadata
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from check_lint_dependencies import dependencies_satisfied


class TestLintDependencies(unittest.TestCase):
    def check_report(self, report, returncode=0):
        def run(command, **kwargs):
            self.assertIn("--dry-run", command)
            self.assertIn("--no-index", command)
            self.assertNotIn("--no-deps", command)
            if report is not None:
                path = Path(command[command.index("--report") + 1])
                path.write_text(json.dumps(report), encoding="utf-8")
            return subprocess.CompletedProcess(command, returncode, stdout="", stderr="dependency mismatch")

        with patch("check_lint_dependencies.subprocess.run", side_effect=run):
            return dependencies_satisfied(Path("requirements-dev.txt"))

    def test_matching_environment_skips_installation(self):
        self.assertTrue(self.check_report({"install": []}))

    def test_planned_install_does_not_count_as_satisfied(self):
        self.assertFalse(self.check_report({"install": [{"metadata": {"name": "xlite"}}]}))

    def test_missing_or_incompatible_dependency_requires_installation(self):
        self.assertFalse(self.check_report(None, returncode=1))

    def test_success_without_a_report_requires_installation(self):
        self.assertFalse(self.check_report(None))

    def test_incomplete_or_invalid_report_requires_installation(self):
        for report in ({}, {"install": None}, {"install": "unexpected"}):
            with self.subTest(report=report):
                self.assertFalse(self.check_report(report))

    def test_real_pip_accepts_recursive_requirements_and_ignores_inactive_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            version = importlib.metadata.version("pip")
            (root / "nested.txt").write_text(f"pip=={version}\n", encoding="utf-8")
            requirements = root / "requirements.txt"
            requirements.write_text(
                "-r nested.txt\nlint-check-nonexistent-package; python_version < '0'\n", encoding="utf-8"
            )
            self.assertTrue(dependencies_satisfied(requirements))

    def test_real_pip_rejects_installed_version_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text("pip==0.0.0\n", encoding="utf-8")
            self.assertFalse(dependencies_satisfied(requirements))

    def test_real_pip_checks_transitive_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "lint_check_fixture-1.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: lint-check-fixture\nVersion: 1.0\n"
                "Requires-Dist: lint-check-nonexistent-package\n",
                encoding="utf-8",
            )
            requirements = root / "requirements.txt"
            requirements.write_text("lint-check-fixture==1.0\n", encoding="utf-8")
            pythonpath = str(root) + os.pathsep + os.environ.get("PYTHONPATH", "")
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                self.assertFalse(dependencies_satisfied(requirements))


if __name__ == "__main__":
    unittest.main()
