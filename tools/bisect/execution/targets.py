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
"""Validated bindings from participant identity to an existing execution channel."""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExecutionTarget:
    name: str
    channel: str
    repository: str
    python: str
    settings: tuple[tuple[str, str], ...] = ()

    def option(self, key: str) -> str:
        return dict(self.settings)[key]


@dataclass(frozen=True)
class ExecutionTargets:
    nodes: tuple[ExecutionTarget, ...]
    prepare_timeout_seconds: float = 7200.0

    @classmethod
    def load(cls, path: Path) -> "ExecutionTargets":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or set(data) - {"nodes", "prepare_timeout_seconds"}:
            raise ValueError("Invalid execution target document")
        items = data.get("nodes")
        if not isinstance(items, list) or not items:
            raise ValueError("nodes must be a nonempty explicit list")
        timeout = data.get("prepare_timeout_seconds", 7200)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("prepare_timeout_seconds must be finite and positive")
        nodes = []
        identities = set()
        names = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each node must be an object")
            channel = item.get("channel")
            required = {
                "local": set(),
                "kubernetes": {"context", "namespace", "pod", "container", "pod_uid", "container_id"},
                "ray": {"address", "node_id"},
            }
            if channel not in required:
                raise ValueError(f"Unsupported explicit execution channel: {channel!r}")
            keys = {"name", "channel", "repository", "python"} | required[channel]
            if set(item) != keys or not all(
                isinstance(value, str) and value and "\x00" not in value for value in item.values()
            ):
                raise ValueError(f"Node fields must be exactly {sorted(keys)} with nonempty string values")
            name = item["name"]
            if not name.replace("-", "").replace("_", "").isalnum() or name in names:
                raise ValueError("Node names must be unique simple identifiers")
            # One participant per environment, even if repository paths differ.
            if channel == "local":
                identity = (channel,)
            elif channel == "kubernetes":
                identity = (channel, item["context"], item["pod_uid"], item["container"])
            else:
                identity = (channel, item["address"], item["node_id"])
            if identity in identities:
                raise ValueError("The same execution environment cannot participate twice")
            if not (item["repository"].startswith("/") or Path(item["repository"]).is_absolute()):
                raise ValueError("Target repository paths must be absolute")
            identities.add(identity)
            names.add(name)
            nodes.append(
                ExecutionTarget(
                    name,
                    channel,
                    item["repository"],
                    item["python"],
                    tuple(sorted((key, item[key]) for key in required[channel])),
                )
            )
        return cls(tuple(nodes), float(timeout))

    @property
    def identity(self) -> str:
        content = [(node.name, node.channel, node.repository, node.python, node.settings) for node in self.nodes]
        return hashlib.sha256(json.dumps((content, self.prepare_timeout_seconds)).encode()).hexdigest()
