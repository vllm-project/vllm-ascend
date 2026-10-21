# SPDX-License-Identifier: Apache-2.0
"""Export static runtime declarations for legacy pip/CI preinstall commands."""

import argparse
import sys
from pathlib import Path

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def export_requirements(root: Path, *, check: bool = False) -> None:
    with (root / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file).get("project", {})
    dependencies = project.get("dependencies")
    if (
        not isinstance(dependencies, list)
        or any(not isinstance(line, str) for line in dependencies)
        or "dependencies" in project.get("dynamic", [])
    ):
        raise ValueError("pyproject.toml must declare static project.dependencies")
    for line in dependencies:
        if "\n" in line or "\r" in line:
            raise ValueError("Each dependency must be a single PEP 508 requirement")
        Requirement(line)
    text = (
        "# Generated from pyproject.toml [project].dependencies; do not edit.\n"
        "# Regenerate: python -m tools.export_requirements\n" + "\n".join(dependencies) + "\n"
    )
    destination = root / "requirements.txt"
    if check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
            raise ValueError("Regenerate requirements.txt with: python -m tools.export_requirements")
    else:
        destination.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        export_requirements(args.root, check=args.check)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
