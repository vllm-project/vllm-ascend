"""Inventory every declared operation in a hardware-specific Docker build."""

from pathlib import Path

import regex as re
import tomllib

from tools.bisect import git_ops
from tools.bisect.build_recipe import BuildRecipe


def _requirements(repo: Path, commit: str, path: str, seen: set[str]) -> list[dict[str, str]]:
    if path in seen:
        return []
    seen.add(path)
    content = git_ops.file_at_commit(repo, commit, path)
    if content is None:
        return [{"source": path, "requirement": "<unresolved: file missing>"}]
    entries = []
    for raw in content.splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        include = re.match(r"^(?:-r|--requirement)\s+(.+)$", line)
        if include:
            child = (Path(path).parent / include.group(1).strip()).as_posix()
            entries.extend(_requirements(repo, commit, child, seen))
        else:
            entries.append({"source": path, "requirement": line})
    return entries


def compile_manifest(repo: Path, commit: str, recipe: BuildRecipe) -> dict[str, object]:
    """Keep every active instruction; extract known installs without dropping opaque work."""
    operations = []
    unresolved = [
        "Base image contents and packages are external to these Dockerfiles",
        "Docker build-arg overrides are not available from source defaults",
    ]
    for layer, path, steps in (
        ("base", recipe.root_dockerfile, recipe.root_chain),
        ("nightly", recipe.nightly_dockerfile, recipe.nightly_chain),
    ):
        for index, step in enumerate(steps, start=1):
            command = step.command
            operation = {
                "layer": layer,
                "source": path,
                "step": index,
                "instruction": step.instruction,
                "command": command,
                "arg_inputs": dict(step.inputs),
                "pip_commands": [
                    match.group(0).strip()
                    for match in re.finditer(
                        r"\b(?:python3?\s+-m\s+)?pip(?:3)?\s+(?:install|uninstall)\b[^;&|]*",
                        command,
                    )
                ],
                "apt_commands": [
                    match.group(0).strip() for match in re.finditer(r"\bapt-get\s+install\b[^;&|]*", command)
                ],
                "git_commands": [
                    match.group(0).strip()
                    for match in re.finditer(r"\bgit\s+(?:clone|fetch|checkout)\b[^;&|]*", command)
                ],
                "external_commands": [
                    match.group(0).strip() for match in re.finditer(r"\b(?:wget|mfcli|bash)\s+[^;&|]*", command)
                ],
            }
            operations.append(operation)
            if step.instruction == "RUN" and re.search(r"\b(?:if|case|for|while)\b", command):
                unresolved.append(f"{layer} step {index}: shell branches require evaluation for the build target")
            if re.search(r"pip install -e /vllm-workspace/vllm(?:/|\s)", command):
                unresolved.append(
                    f"{layer} step {index}: vLLM source and extras dependencies come from an external checkout"
                )
            if "benchmark" in command and "pip install -e ." in command:
                unresolved.append(f"{layer} step {index}: AISBench package dependencies come from an external checkout")
            if "tools/build_rust.sh" in command:
                unresolved.append(f"{layer} step {index}: vLLM Rust build script comes from an external checkout")
            if "mfcli kernel install" in command:
                unresolved.append(f"{layer} step {index}: mfcli installs SoC-specific kernels at build time")
            if step.instruction == "COPY":
                unresolved.append(f"{layer} step {index}: COPY input depends on the Docker build context")
            if operation["apt_commands"]:
                unresolved.append(
                    f"{layer} step {index}: apt package versions are resolved by the build-time repository"
                )
            if operation["git_commands"] and "git clone" in command:
                unresolved.append(f"{layer} step {index}: cloned repository contents require the selected Git revision")
            if any(" install " in item for item in operation["pip_commands"]):
                unresolved.append(
                    f"{layer} step {index}: complete pip dependency closure requires a build-time resolver report"
                )

    project = tomllib.loads(recipe.pyproject)
    requirements = {
        "base": _requirements(repo, commit, "requirements.txt", set()),
        "nightly": _requirements(repo, commit, "requirements-nightly.txt", set()),
        "build_system": project.get("build-system", {}).get("requires", []),
    }
    return {"operations": operations, "requirements": requirements, "unresolved": sorted(set(unresolved))}
