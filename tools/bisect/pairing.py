"""Describe the declared and observed dependencies of a bisect candidate."""

import importlib.metadata
from pathlib import Path

import regex as re
from packaging.requirements import InvalidRequirement, Requirement

from tools.bisect import git_ops
from tools.bisect.build_manifest import compile_manifest
from tools.bisect.build_recipe import BuildRecipe


def _installed(requirement: str) -> str:
    try:
        parsed = Requirement(requirement)
    except InvalidRequirement:
        return "unparsed"
    try:
        version = importlib.metadata.version(parsed.name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"
    if parsed.marker and not parsed.marker.evaluate():
        return f"{version} (marker inactive)"
    if parsed.specifier and not parsed.specifier.contains(version):
        return f"{version} (constraint mismatch)"
    return version


def pairing_details(repo: Path, commit: str, recipe: BuildRecipe) -> list[str]:
    """Return a complete declaration inventory; observed versions are informational."""
    lines = ["[pairing:detail] Declared dependencies and locally installed versions:"]
    manifest = compile_manifest(repo, commit, recipe)
    for layer, steps in (("base", recipe.root_chain), ("nightly", recipe.nightly_chain)):
        for index, step in enumerate(steps, start=1):
            inputs = ", ".join(f"{name}={value}" for name, value in step.inputs) or "none"
            lines.append(
                f"[pairing:chain] {layer} step {index} {step.instruction}: {step.command} | ARG inputs: {inputs}"
            )
    for layer in ("base", "nightly"):
        for item in manifest["requirements"][layer]:
            lines.append(
                f"[pairing:detail] {item['source']}: {item['requirement']} | "
                f"installed={_installed(item['requirement'])}"
            )
    for requirement in manifest["requirements"]["build_system"]:
        lines.append(f"[pairing:build-only] pyproject.toml: {requirement} | host-installed={_installed(requirement)}")
    for operation in manifest["operations"]:
        for kind in ("pip_commands", "apt_commands", "git_commands", "external_commands"):
            for command in operation[kind]:
                lines.append(f"[pairing:operation] {operation['layer']} step {operation['step']} {command}")
    lines.extend(f"[pairing:unresolved] {item}" for item in manifest["unresolved"])

    for path in (recipe.root_dockerfile, recipe.nightly_dockerfile):
        content = git_ops.file_at_commit(repo, commit, path) or ""
        from_instruction = next((line.strip() for line in content.splitlines() if line.startswith("FROM ")), "missing")
        lines.append(f"[pairing:docker] {path}: FROM {from_instruction}")
        for raw in content.splitlines():
            match = re.match(r"^ARG\s+([A-Z][A-Z0-9_]*)=(.*)$", raw)
            if match and any(
                token in match.group(1)
                for token in (
                    "VERSION",
                    "DATE",
                    "TAG",
                    "COMMIT",
                    "INDEX_URL",
                    "REPO",
                    "OBS_BASE",
                    "OBS_PREFIX",
                    "RELEASE",
                )
            ):
                lines.append(f"[pairing:docker] {path}: {match.group(1)}={match.group(2).strip()}")

    lines.extend(f"[pairing:docker] runtime-wheel={wheel}" for wheel in recipe.runtime_wheels)
    if recipe.flash_attention_ref:
        lines.append(f"[pairing:docker] flash-attention-npu={recipe.flash_attention_ref}")
    lines.append(f"[pairing:docker] AISBench={recipe.ais_bench_url}@{recipe.ais_bench_tag}")
    benchmark = repo / "benchmark"
    if (benchmark / ".git").exists():
        try:
            actual_benchmark = git_ops.current_commit(benchmark)
        except (OSError, git_ops.GitError):
            actual_benchmark = "unavailable"
    else:
        actual_benchmark = "not a git checkout"
    lines.append(f"[pairing:observed] AISBench source={actual_benchmark}")
    cann_version = Path("/usr/local/Ascend/ascend-toolkit/latest/version.info")
    if not cann_version.is_file():
        cann_version = Path("/usr/local/Ascend/ascend-toolkit/version.info")
    if cann_version.is_file():
        lines.append(f"[pairing:observed] CANN version file={cann_version}")
        lines.extend(
            f"[pairing:observed] CANN {line.strip()}"
            for line in cann_version.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip().startswith(("Version=", "version="))
        )
    lines.append(
        "[pairing:detail] Installed versions above are observations; temporary isolated build dependencies "
        "and the running CANN image are not verified here."
    )
    return lines
