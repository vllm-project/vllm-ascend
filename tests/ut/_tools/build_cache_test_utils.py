# Copyright (c) 2026 Huawei Technologies Co., Ltd.

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE = REPO_ROOT / "csrc" / "scripts" / "build_cache.py"


def build_cache_command(
    *,
    cache_root: Path,
    domain: str,
    unit: str,
    output_dir: Path,
    environment_profile: str,
    prepared_inputs: Sequence[Path],
    recipe_values: Sequence[str],
    environment_values: Sequence[str],
    environment_tools: Sequence[str],
    build_command: Sequence[str],
    normalize_paths: Sequence[Path] = (),
    artifact_includes: Sequence[str] = (),
    soc: str | None = None,
    operator: str | None = None,
    action: str | None = None,
    operator_source: Path | None = None,
    publish_dir: Path | None = None,
    publish_state_dir: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ENGINE),
        "run",
        "--cache-root",
        str(cache_root),
        "--domain",
        domain,
        "--unit",
        unit,
        "--output-dir",
        str(output_dir),
        "--environment-profile",
        environment_profile,
    ]
    for path in prepared_inputs:
        command.extend(["--prepared-input", str(path)])
    for value in recipe_values:
        command.extend(["--recipe-value", value])
    for value in environment_values:
        command.extend(["--environment-value", value])
    for tool in environment_tools:
        command.extend(["--environment-tool", tool])
    for path in normalize_paths:
        command.extend(["--normalize-path", str(path)])
    for pattern in artifact_includes:
        command.extend(["--artifact-include", pattern])

    custom_values = {
        "--soc": soc,
        "--operator": operator,
        "--action": action,
        "--operator-source": operator_source,
        "--publish-dir": publish_dir,
        "--publish-state-dir": publish_state_dir,
    }
    for option, value in custom_values.items():
        if value is not None:
            command.extend([option, str(value)])

    command.append("--")
    command.extend(build_command)
    return command


def run_command(
    command: Sequence[str],
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if env:
        environment.update(env)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
