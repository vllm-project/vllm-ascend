# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.dependency_compatibility import compare_reports, main, prepare_probe


class DependencyCompatibilityTest(unittest.TestCase):
    def prepare(self, contents, packages="[]", candidates="[]", source="requirements.txt"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "requirements.txt").write_text(contents, encoding="utf-8")
        return root, prepare_probe(root, source, packages, candidates, root / "results")

    def test_generic_selection_and_candidate_replacement(self):
        root, plan = self.prepare("packaging\npsutil==6.0.0\n", '["psutil"]', '["psutil>=6,<7", "attrs==24.2.0"]')
        self.assertEqual(plan["baseline"], ["psutil==6.0.0"])
        self.assertEqual(plan["candidate"], ["psutil<7,>=6", "attrs==24.2.0"])
        self.assertEqual((root / "requirements.txt").read_text(), "packaging\npsutil==6.0.0\n")

    def test_recursive_includes_and_url_fragments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("-r common.txt\n", encoding="utf-8")
            (root / "common.txt").write_text(
                "example @ https://example.org/example.whl#sha256=abc # provenance\n", encoding="utf-8"
            )
            plan = prepare_probe(root, "requirements.txt", "[]", "[]", root / "results")
            self.assertEqual(plan["baseline"], ["example @ https://example.org/example.whl#sha256=abc"])

    def test_build_requirements_from_pyproject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[build-system]\nrequires = ["setuptools>=64", "packaging"]\n', encoding="utf-8"
            )
            plan = prepare_probe(root, "pyproject.toml", '["packaging"]', "[]", root / "results")
            self.assertEqual(plan["baseline"], ["packaging"])

    def test_static_runtime_requirements_from_pyproject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pyproject.toml").write_text(
                '[project]\ndependencies = ["example>=1"]\n[build-system]\nrequires = ["build-only>=2"]\n',
                encoding="utf-8",
            )
            plan = prepare_probe(root, "pyproject.toml:project.dependencies", "[]", "[]", root / "results")
            self.assertEqual(plan["baseline"], ["example>=1"])

    def test_invalid_inputs_are_rejected(self):
        for packages, candidates in (
            ('"packaging"', "[]"),
            ('["missing"]', "[]"),
            ('["packaging", "Packaging"]', "[]"),
            ("[]", '["packaging>=24", "packaging<27"]'),
            ("[]", '["packaging @ https://example.org/package.whl"]'),
            ("[]", '["--no-deps"]'),
        ):
            with self.subTest(packages=packages, candidates=candidates), self.assertRaises(ValueError):
                self.prepare("packaging\n", packages, candidates)

    def test_candidate_preserves_markers_and_extras(self):
        with self.assertRaisesRegex(ValueError, "markers and extras"):
            self.prepare('example[one]>=1; python_version < "3.13"\n', candidates='["example>=2"]')

    def test_multiline_candidate_cannot_inject_pip_options(self):
        for declaration in ('example>=1; os_name == "other\n--no-deps"', r'example>=1; os_name == "other\n--no-deps"'):
            with self.subTest(declaration=declaration), self.assertRaisesRegex(ValueError, "single line"):
                self.prepare("packaging\n", candidates=json.dumps([declaration]))

    def test_include_cycle_and_outside_repository_are_rejected(self):
        for contents in ("-r requirements.txt\n", "-r ../outside.txt\n"):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                self.prepare(contents)

    def test_unsupported_pip_options_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported pip option"):
            self.prepare("--extra-index-url https://example.org\npackaging\n")

    def test_installed_graph_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile, packages in (
                ("baseline", [("packaging", "24.2"), ("old", "1")]),
                ("candidate", [("Packaging", "26.3"), ("new", "2")]),
            ):
                (root / profile).mkdir()
                report = {"installed": [{"metadata": {"name": name, "version": version}} for name, version in packages]}
                (root / profile / "installed.json").write_text(json.dumps(report), encoding="utf-8")
                (root / profile / "pip-check.log").write_text("No broken requirements found.\n", encoding="utf-8")
                (root / profile / "success").touch()
            result = compare_reports(root)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["changed"], {"packaging": {"baseline": "24.2", "candidate": "26.3"}})
            self.assertEqual(result["added"], {"new": "2"})
            self.assertEqual(result["removed"], {"old": "1"})

    def test_partial_install_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result = compare_reports(Path(directory))
            self.assertEqual(result["status"], "incomplete")

    def test_comparison_cli_fails_for_incomplete_results(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("sys.argv", ["probe", "compare", "--output", directory]),
            patch("builtins.print"),
            self.assertRaises(SystemExit) as failure,
        ):
            main()
        self.assertEqual(failure.exception.code, 1)

    def test_existing_profile_results_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text("packaging\n", encoding="utf-8")
            (root / "results" / "baseline").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "fresh output"):
                prepare_probe(root, "requirements.txt", "[]", "[]", root / "results")


if __name__ == "__main__":
    unittest.main()
