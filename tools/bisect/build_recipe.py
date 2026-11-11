# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Read the two Docker build layers used by a nightly bisect candidate."""

import platform
from dataclasses import dataclass
from pathlib import Path

import regex as re

from tools.bisect import git_ops

ROOT_DOCKERFILES = {
    "a2": "Dockerfile",
    "a3": "Dockerfile.a3",
    "310p": "Dockerfile.310p",
    "a5": "Dockerfile.a5",
}
NIGHTLY_DOCKERFILE_DIR = ".github/workflows/dockerfiles"


@dataclass(frozen=True)
class BuildStep:
    instruction: str
    command: str
    inputs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BuildRecipe:
    """Inputs that determine the candidate's base and nightly Python environment."""

    soc: str
    root_dockerfile: str
    nightly_dockerfile: str
    cann_image: str
    vllm_ref: str
    pytorch_index_url: str
    ascend_index_url: str
    nightly_index_url: str
    ais_bench_url: str
    ais_bench_tag: str
    requirements: str
    pyproject: str
    requirements_nightly: str
    root_build_steps: str
    nightly_build_steps: str
    runtime_wheels: tuple[str, ...]
    flash_attention_url: str | None
    flash_attention_ref: str | None
    root_chain: tuple[BuildStep, ...] = ()
    nightly_chain: tuple[BuildStep, ...] = ()


def _read(repo: Path, commit: str, path: str) -> str:
    content = git_ops.file_at_commit(repo, commit, path)
    if content is None:
        raise ValueError(f"build recipe file missing at {commit[:12]}: {path}")
    return content


def _args(dockerfile: str) -> dict[str, str]:
    values = {}
    for line in dockerfile.splitlines():
        match = re.match(r"^ARG\s+([A-Z][A-Z0-9_]*)=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"')
    return values


def _build_chain(dockerfile: str) -> tuple[BuildStep, ...]:
    """Translate active Dockerfile instructions and their effective ARG inputs."""
    args: dict[str, str] = {}
    instructions: list[str] = []
    pending = ""
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pending += (" " if pending else "") + line.rstrip("\\").strip()
        if line.endswith("\\"):
            continue
        instructions.append(pending)
        pending = ""
    if pending:
        instructions.append(pending)

    steps = []
    for line in instructions:
        kind, _, command = line.partition(" ")
        if kind == "ARG":
            name, separator, value = command.partition("=")
            if separator:
                args[name] = value.strip().strip('"')
            continue
        references = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", command))
        references.update(re.findall(r"\$([A-Z][A-Z0-9_]*)\b", command))
        inputs = tuple(sorted((name, args[name]) for name in references if name in args))
        if kind == "RUN" and "flash-attention-npu" in command:
            command = re.sub(r"git checkout [0-9a-f]{40}", "git checkout <flash-attention-ref>", command)
        steps.append(BuildStep(kind, command, inputs))
    return tuple(steps)


def _build_steps(dockerfile: str, variable_args: tuple[str, ...]) -> str:
    """Keep active build instructions, excluding separately handled ARG defaults."""
    lines = []
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if any(stripped.startswith(f"ARG {name}=") for name in variable_args):
            continue
        if stripped.startswith(("MOONCAKE_WHL=", "FLA_WHL=", "FLA_SHA256=", "git checkout ")):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def _runtime_wheels(dockerfile: str, args: dict[str, str]) -> tuple[str, ...]:
    arch = "aarch64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"
    wheels = []
    mooncake = re.search(rf'MOONCAKE_WHL="([^"]*{arch}\.whl)"', dockerfile)
    if mooncake and args.get("MOONCAKE_OBS_PREFIX"):
        wheels.append(f"{args['MOONCAKE_OBS_PREFIX']}/{mooncake.group(1)}")
    fla = re.search(rf'FLA_WHL="([^"]*{arch}\.whl)"', dockerfile)
    fla_url = re.search(r'URL="(https?://[^\"]*\$\{FLA_WHL\})"', dockerfile)
    if fla and fla_url:
        filename = fla.group(1).replace("${FLA_VERSION}", args.get("FLA_VERSION", ""))
        wheels.append(fla_url.group(1).replace("${FLA_WHL}", filename))
    return tuple(wheels)


def normalize_build_soc(soc: str) -> str:
    """Map device variants to a Dockerfile family without changing case identity."""
    target = soc.lower()
    if target in ("910b", "ascend910b", "ascend910b1") or target.startswith("a2-"):
        target = "a2"
    elif target in ("950", "ascend950") or target.startswith("a5-"):
        target = "a5"
    elif target.startswith("a3-"):
        target = "a3"
    elif target.startswith("310p-"):
        target = "310p"
    if target not in ROOT_DOCKERFILES:
        raise ValueError(f"unsupported bisect SoC {soc!r}")
    return target


def recipe_at_commit(repo: Path, commit: str, soc: str, os_variant: str | None = None) -> BuildRecipe:
    """Resolve build inputs from the selected chip's Dockerfiles at one commit."""
    target = normalize_build_soc(soc)

    if os_variant is None:
        os_release = Path("/etc/os-release")
        os_variant = os_release.read_text(encoding="utf-8") if os_release.exists() else ""
    root_path = ROOT_DOCKERFILES[target]
    if "openeuler" in os_variant.lower():
        root_path += ".openEuler"
    nightly_path = f"{NIGHTLY_DOCKERFILE_DIR}/Dockerfile.nightly.{target}"
    root = _read(repo, commit, root_path)
    nightly = _read(repo, commit, nightly_path)
    root_args = _args(root)
    nightly_args = _args(nightly)
    base = re.search(r"^FROM\s+(\S+)", root, re.MULTILINE)
    if base is None:
        raise ValueError(f"base image missing from {root_path} at {commit[:12]}")
    vllm_ref = root_args.get("VLLM_COMMIT") or root_args.get("VLLM_TAG")
    if not vllm_ref:
        raise ValueError(f"vLLM ref missing from {root_path} at {commit[:12]}")
    flash_attention = re.search(r"^\s+git checkout ([0-9a-f]{40})\s*&&", root, re.MULTILINE)
    flash_attention_url = re.search(r"^\s+git clone (https?://\S+) /tmp/flash-attention-npu", root, re.MULTILINE)

    return BuildRecipe(
        soc=target,
        root_dockerfile=root_path,
        nightly_dockerfile=nightly_path,
        cann_image=base.group(1),
        vllm_ref=vllm_ref,
        pytorch_index_url=root_args.get("PYTORCH_INDEX_URL", ""),
        ascend_index_url=root_args.get("ASCEND_INDEX_URL", ""),
        nightly_index_url=nightly_args.get("PIP_EXTRA_INDEX_URL", ""),
        ais_bench_url=nightly_args.get("AIS_BENCH_URL", ""),
        ais_bench_tag=nightly_args.get("AIS_BENCH_TAG", ""),
        requirements=_read(repo, commit, "requirements.txt"),
        pyproject=_read(repo, commit, "pyproject.toml"),
        requirements_nightly=git_ops.file_at_commit(repo, commit, "requirements-nightly.txt") or "",
        root_build_steps=_build_steps(
            root,
            (
                "VLLM_TAG",
                "VLLM_COMMIT",
                "PYTORCH_INDEX_URL",
                "ASCEND_INDEX_URL",
                "MOONCAKE_OBS_PREFIX",
                "FLA_VERSION",
            ),
        ),
        nightly_build_steps=_build_steps(
            nightly,
            ("PIP_EXTRA_INDEX_URL", "AIS_BENCH_TAG", "AIS_BENCH_URL"),
        ),
        runtime_wheels=_runtime_wheels(root, root_args),
        flash_attention_url=flash_attention_url.group(1) if flash_attention_url else None,
        flash_attention_ref=flash_attention.group(1) if flash_attention else None,
        root_chain=_build_chain(root),
        nightly_chain=_build_chain(nightly),
    )
