# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Recover explicit target-image replay steps from Docker image history."""

import shlex
from dataclasses import dataclass

import regex as re

from tools.bisect.build_analyzer.replay_cleanup import protect_replay_cleanup


@dataclass(frozen=True)
class ImageReplayStep:
    kind: str
    command: str
    content_image: str | None = None
    content_path: str | None = None
    destination: str | None = None


def target_stage_plan(history: list[dict], reference: str) -> dict:
    """Docker history input is newest first; never infer missing RUN commands."""
    ordered = list(reversed(history))
    start = next((i for i, item in enumerate(ordered) if item.get("Comment") == "buildkit.dockerfile.v0"), None)
    blockers = []
    warnings = []
    steps = []
    if start is None:
        return {"status": "blocked", "actions": [], "blockers": ["No explicit build boundary in image history"]}
    if any(item.get("Comment") != "buildkit.dockerfile.v0" for item in ordered[start:]):
        blockers.append("Image history has an ambiguous final-stage boundary")
    for item in ordered[start:]:
        text = item.get("CreatedBy", "").removesuffix(" # buildkit")
        kind, _, body = text.partition(" ")
        if kind == "COPY":
            words = shlex.split(body)
            if len(words) != 2 or not words[-1].startswith("/"):
                blockers.append(f"Cannot recover COPY destination: {text}")
                continue
            destination = words[-1]
            steps.append(ImageReplayStep("copy-from-target-image", text, reference, destination, destination))
        elif kind == "RUN":
            match = re.fullmatch(r"(?:\|\d+ (.*?))?(/bin/\S+) -c (.*)", body, flags=re.DOTALL)
            if match is None:
                blockers.append(f"Cannot recover RUN command: {text}")
                continue
            assignments, shell, command = match.groups()
            command, cleanup_warnings = protect_replay_cleanup(command)
            warnings.extend(cleanup_warnings)
            prefix = ["env", *shlex.split(assignments)] if assignments else []
            steps.append(ImageReplayStep("run", shlex.join([*prefix, shell, "-c", command])))
        elif kind in {"ARG", "ENV", "SHELL", "ENTRYPOINT", "CMD", "WORKDIR", "USER", "LABEL"}:
            steps.append(ImageReplayStep(kind.lower(), body))
        else:
            blockers.append(f"Unsupported or missing image-history instruction: {text}")
    return {
        "status": "blocked" if blockers else "reviewable",
        "actions": steps,
        "blockers": blockers,
        "warnings": warnings,
    }
