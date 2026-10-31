"""Check the installed lint dependencies without querying package indexes.

Exit zero only when pip resolves the requirements without planning any installs.
Missing/incompatible packages or an inconclusive check request the normal install.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def dependencies_satisfied(requirements: Path) -> bool:
    # Reuse pip's handling of recursive requirements, markers and transitive
    # dependencies. A dry run does not change the environment; --no-index keeps
    # this fast path independent of package-index availability.
    with tempfile.TemporaryDirectory(prefix="lint-dependencies-") as directory:
        report = Path(directory) / "report.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "--dry-run",
                "--no-index",
                "--no-input",
                "--quiet",
                "--report",
                str(report),
                "-r",
                str(requirements),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("Dependency check could not confirm a matching environment; installing requirements.")
            print(result.stderr.strip())
            return False
        try:
            installs = json.loads(report.read_text(encoding="utf-8"))["install"]
        except (OSError, ValueError, KeyError, TypeError):
            print("Dependency check produced no valid report; installing requirements.")
            return False
        if not isinstance(installs, list) or installs:
            print("Dependency check requires package changes; installing requirements.")
            return False
    print("Installed dependencies satisfy the requirements; skipping installation.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    args = parser.parse_args()
    return 0 if dependencies_satisfied(args.requirements) else 1


if __name__ == "__main__":
    sys.exit(main())
