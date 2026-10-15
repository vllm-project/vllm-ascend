# SPDX-License-Identifier: Apache-2.0
"""Prepare dependency experiments and compare normally installed package graphs."""

import argparse
import json
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SOURCES = ("requirements.txt", "requirements-dev.txt", "requirements-lint.txt", "pyproject.toml")


def read_requirements(path, root, active=()):
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Requirements include must stay within the repository")
    if path in active:
        raise ValueError(f"Requirements include cycle: {path}")
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for index, character in enumerate(line):
            if character == "#" and index > 0 and line[index - 1].isspace():
                line = line[:index]
                break
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if parts[0] == "-r":
            if len(parts) != 2:
                raise ValueError("Missing requirements include path")
            requirements.extend(read_requirements(path.parent / parts[1], root, (*active, path)))
        elif line.startswith("-"):
            raise ValueError(f"Unsupported pip option in probe source: {line}")
        else:
            requirements.append(Requirement(line))
    return requirements


def parse_list(value):
    parsed = json.loads(value)
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError("Expected a JSON array of strings")
    if any("\n" in item or "\r" in item for item in parsed):
        raise ValueError("Each JSON entry must be a single line")
    return parsed


def prepare_probe(root, source, packages_json, candidates_json, output):
    if any((output / profile).exists() for profile in ("baseline", "candidate")):
        raise ValueError("Use a fresh output directory to avoid reusing profile results")
    if source not in SOURCES:
        raise ValueError(f"Unsupported dependency source: {source}")
    if source == "pyproject.toml":
        with (root / source).open("rb") as file:
            requirements = [Requirement(line) for line in tomllib.load(file)["build-system"]["requires"]]
    else:
        requirements = read_requirements(root / source, root)
    selected = []
    for name in parse_list(packages_json):
        requirement = Requirement(name)
        if requirement.specifier or requirement.extras or requirement.marker or requirement.url:
            raise ValueError("Selected packages must be plain distribution names")
        selected.append(canonicalize_name(requirement.name))
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate selected packages")
    available = {canonicalize_name(requirement.name) for requirement in requirements}
    if missing := set(selected) - available:
        raise ValueError(f"Selected packages not declared by {source}: {sorted(missing)}")
    baseline = [req for req in requirements if not selected or canonicalize_name(req.name) in selected]
    if not baseline:
        raise ValueError("Probe baseline must contain at least one dependency")
    candidates = [Requirement(line) for line in parse_list(candidates_json)]
    names = [canonicalize_name(req.name) for req in candidates]
    if len(set(names)) != len(names):
        raise ValueError("Use one candidate declaration per distribution")
    for requirement in candidates:
        if requirement.url or not requirement.specifier:
            raise ValueError("Candidates must use version constraints, not URLs or pip options")
        for original in baseline:
            if canonicalize_name(original.name) == canonicalize_name(requirement.name) and (
                original.extras != requirement.extras or original.marker != requirement.marker
            ):
                raise ValueError("Candidates replacing direct declarations must preserve markers and extras")
    candidate = [req for req in baseline if canonicalize_name(req.name) not in names] + candidates
    if any("\n" in str(req) or "\r" in str(req) for req in (*baseline, *candidate)):
        raise ValueError("Each rendered requirement must be a single line")
    plan = {
        "source": source,
        "selected_packages": selected,
        "baseline": [str(req) for req in baseline],
        "candidate": [str(req) for req in candidate],
        "scope": "Selected direct dependencies and their resolved transitive graph",
    }
    output.mkdir(parents=True, exist_ok=True)
    for profile in ("baseline", "candidate"):
        (output / f"{profile}.txt").write_text("\n".join(plan[profile]) + "\n", encoding="utf-8")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def compare_reports(output):
    result: dict[str, object]
    missing = [
        profile
        for profile in ("baseline", "candidate")
        if not (output / profile / "success").is_file() or not (output / profile / "installed.json").is_file()
    ]
    if missing:
        result = {"status": "incomplete", "unsuccessful_profiles": missing}
    else:
        graphs = {}
        for profile in ("baseline", "candidate"):
            report = json.loads((output / profile / "installed.json").read_text(encoding="utf-8"))
            graphs[profile] = {
                canonicalize_name(item["metadata"]["name"]): item["metadata"]["version"] for item in report["installed"]
            }
        baseline, candidate = graphs["baseline"], graphs["candidate"]
        result = {
            "status": "complete",
            "added": {name: candidate[name] for name in sorted(candidate.keys() - baseline.keys())},
            "removed": {name: baseline[name] for name in sorted(baseline.keys() - candidate.keys())},
            "changed": {
                name: {"baseline": baseline[name], "candidate": candidate[name]}
                for name in sorted(baseline.keys() & candidate.keys())
                if baseline[name] != candidate[name]
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", choices=SOURCES, default="requirements.txt")
    prepare.add_argument("--packages-json", default="[]")
    prepare.add_argument("--candidates-json", default="[]")
    prepare.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    prepare.add_argument("--output", type=Path, default=Path("compatibility-results"))
    compare = subparsers.add_parser("compare")
    compare.add_argument("--output", type=Path, default=Path("compatibility-results"))
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_probe(args.root, args.source, args.packages_json, args.candidates_json, args.output)
    else:
        result = compare_reports(args.output)
    print(json.dumps(result, indent=2))
    if args.command == "compare" and result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
