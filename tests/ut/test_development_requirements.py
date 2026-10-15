# SPDX-License-Identifier: Apache-2.0

import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[2]


def read_requirements(filename):
    requirements = []
    for line in (REPO_ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("-r "):
            requirements.extend(read_requirements(line.split()[1]))
        elif line:
            requirements.append(Requirement(line))
    return requirements


class DevelopmentRequirementsTest(unittest.TestCase):
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
                        requirement.specifier and other.specifier and requirement.specifier != other.specifier,
                        f"Redundant development requirement: {requirement} overlaps {other}",
                    )


if __name__ == "__main__":
    unittest.main()
