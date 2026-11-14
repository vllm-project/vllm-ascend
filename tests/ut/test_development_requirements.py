# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_requirements(filename):
    requirements = []
    for line in (REPO_ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        parts = line.split(maxsplit=1)
        if parts and parts[0] == "-r":
            if len(parts) != 2:
                raise ValueError(f"Missing requirements include path in {filename}")
            requirements.extend(read_requirements(parts[1]))
        elif line and not line.startswith("-"):
            requirements.append(Requirement(line))
    return requirements


class DevelopmentRequirementsTest(unittest.TestCase):
    def test_development_constraint_can_narrow_unconstrained_runtime(self):
        with patch(f"{__name__}.read_requirements", return_value=[Requirement("ray"), Requirement("ray>=2.47.1")]):
            self.test_no_redundant_development_requirements()

    def test_redundant_entries_are_rejected(self):
        for entries in (("ray>=2", "ray"), ("pytest-mock", "pytest_mock"), ("ray>=2", "ray>=2")):
            with (
                self.subTest(entries=entries),
                patch(f"{__name__}.read_requirements", return_value=[Requirement(entry) for entry in entries]),
                self.assertRaisesRegex(AssertionError, "Redundant"),
            ):
                self.test_no_redundant_development_requirements()

    def test_pip_options_are_not_distribution_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "common.txt").write_text("numpy\n", encoding="utf-8")
            (root / "dev.txt").write_text(
                "-r common.txt\n-c constraints.txt\n--extra-index-url https://example.org\npytest\n",
                encoding="utf-8",
            )
            with patch(f"{__name__}.REPO_ROOT", root):
                self.assertEqual([str(req) for req in read_requirements("dev.txt")], ["numpy", "pytest"])

    def test_missing_include_path_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dev.txt").write_text("-r \n", encoding="utf-8")
            with (
                patch(f"{__name__}.REPO_ROOT", root),
                self.assertRaisesRegex(ValueError, "Missing requirements include path"),
            ):
                read_requirements("dev.txt")

    def test_no_redundant_development_requirements(self):
        requirements = read_requirements("requirements-dev.txt")
        # Distinct constraints, extras, and platform markers may legitimately
        # require multiple entries for one distribution.
        for index, requirement in enumerate(requirements):
            for other in requirements[:index]:
                if (
                    canonicalize_name(requirement.name) == canonicalize_name(other.name)
                    and requirement.extras == other.extras
                    and requirement.marker == other.marker
                    and requirement.url == other.url
                ):
                    self.assertTrue(
                        requirement.specifier and (not other.specifier or requirement.specifier != other.specifier),
                        f"Redundant development requirement: {requirement} overlaps {other}",
                    )


if __name__ == "__main__":
    unittest.main()
