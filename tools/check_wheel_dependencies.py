# SPDX-License-Identifier: Apache-2.0
"""Compare a built wheel's runtime requirements with the setup.py input."""

import argparse
import zipfile
from collections import Counter
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def requirement_key(line):
    requirement = Requirement(line)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        str(requirement.marker) if requirement.marker else "",
        requirement.url,
    )


def read_requirements(path, root):
    """Resolve includes relative to the repository root, as setup.py does."""
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for index, character in enumerate(line):
            if character == "#" and index > 0 and line[index - 1].isspace():
                line = line[:index]
                break
        line = line.strip()
        if line.startswith("-r "):
            requirements.extend(read_requirements(root / line.split()[1], root))
        elif line and not line.startswith(("#", "--")):
            requirements.append(requirement_key(line))
    return requirements


def check_wheel_dependencies(wheel, requirements):
    expected = Counter(read_requirements(requirements, requirements.parent))
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    if canonicalize_name(metadata.get("Name", "")) != "vllm-ascend":
        raise ValueError("Expected a vllm-ascend wheel")
    actual = Counter(requirement_key(line) for line in metadata.get_all("Requires-Dist", []))
    if actual != expected:
        raise ValueError(f"Missing requirements: {expected - actual}\nUnexpected requirements: {actual - expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--requirements", type=Path, default=Path(__file__).resolve().parents[1] / "requirements.txt")
    args = parser.parse_args()
    check_wheel_dependencies(args.wheel, args.requirements)
    print(f"Runtime requirements match: {args.wheel}")


if __name__ == "__main__":
    main()
