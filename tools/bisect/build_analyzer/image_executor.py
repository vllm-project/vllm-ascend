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

"""Execute explicit target-image replay actions after validating all copies."""

import shlex
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PurePosixPath

from tools.bisect.build_analyzer.base_apply import _current_entry, apply_base_delta, validate_base_application
from tools.bisect.build_analyzer.base_image import BaseImageDelta, BaseImageSnapshot

_REPORT_INTERVAL_S = 5.0


class ImageReplayExecutor:
    def __init__(
        self, actions, resolver, architecture, environment, root=Path("/"), report: Callable[[str], None] | None = None
    ):
        self.environment = environment
        self.root = root
        self.cwd = root
        self.copies = {}
        self.report = report
        snapshots = {}
        for action in actions:
            if not action.kind.startswith("image-"):
                continue
            kind = action.kind.removeprefix("image-")
            if kind not in {
                "copy-from-target-image",
                "run",
                "env",
                "arg",
                "shell",
                "workdir",
                "user",
                "label",
                "cmd",
                "entrypoint",
            }:
                raise RuntimeError(f"unsupported target-image action: {kind}")
            if kind == "user" and action.command not in {"root", "0", "0:0"}:
                raise RuntimeError(f"cannot change replay process user: {action.command}")
            if kind == "env" and (
                not shlex.split(action.command)
                or any("=" not in assignment for assignment in shlex.split(action.command))
            ):
                raise RuntimeError(f"unsupported ENV instruction: {action.command}")
            if kind == "workdir" and (
                not PurePosixPath(action.command).is_absolute() or ".." in PurePosixPath(action.command).parts
            ):
                raise RuntimeError(f"unsupported WORKDIR: {action.command}")
            if kind != "copy-from-target-image":
                continue
            if not action.content_image or not action.artifact or action.artifact != action.destination:
                raise RuntimeError("COPY needs an exact target-image path and matching destination")
            path = PurePosixPath(action.destination)
            if not path.is_absolute() or ".." in path.parts or str(path) == "/":
                raise RuntimeError(f"unsafe COPY destination: {path}")
            if action.content_image not in snapshots:
                self._report(f"[base-image:copy-prepare] action={action.id} resolving={action.content_image}")
                snapshots[action.content_image] = resolver.resolve(action.content_image, architecture)
            snapshot = snapshots[action.content_image]
            prefix = str(path).lstrip("/")
            entries = tuple(
                entry for entry in snapshot.files if entry[0] == prefix or entry[0].startswith(prefix + "/")
            )
            if not entries:
                raise RuntimeError(f"target image is missing COPY content: {path}")
            # Docker COPY merges a directory; it does not delete unrelated live files.
            self._report(
                f"[base-image:copy-prepare] action={action.id} destination={path} "
                f"image_digest={snapshot.digest} target_entries={len(entries)}"
            )
            live_entries = []
            last_report = time.monotonic()
            for index, (name, _, _) in enumerate(entries, 1):
                value = _current_entry(root / name)
                if value is not None:
                    live_entries.append((name, *value))
                if time.monotonic() - last_report >= _REPORT_INTERVAL_S:
                    self._report(f"[base-image:copy-prepare] action={action.id} inspected={index}/{len(entries)}")
                    last_report = time.monotonic()
            live = tuple(live_entries)
            before = BaseImageSnapshot(
                "live COPY destination", "live", snapshot.architecture, live, (), snapshot.archive
            )
            after = replace(snapshot, files=entries, config=())
            previous = {name: (kind, value) for name, kind, value in live}
            changed = tuple(name for name, kind, value in entries if previous.get(name) != (kind, value))
            delta = BaseImageDelta(before, after, changed, ())
            reasons = validate_base_application(delta, root)
            if reasons:
                raise RuntimeError("COPY preflight failed: " + "; ".join(reasons[:10]))
            self.copies[action.id] = delta

    def execute(self, action, run):
        kind = action.kind.removeprefix("image-")
        if kind == "copy-from-target-image":
            delta = self.copies[action.id]
            started = time.monotonic()
            self._report(
                f"[base-image:copy-start] action={action.id} destination={action.destination} "
                f"image_digest={delta.after.digest} changed_entries={len(delta.changed_files)}"
            )
            try:
                apply_base_delta(delta, self.root, report=self._report)
            except Exception:
                self._report(f"[base-image:copy-failed] action={action.id} destination={action.destination}")
                raise
            self._report(
                f"[base-image:copy-done] action={action.id} destination={action.destination} "
                f"changed_entries={len(delta.changed_files)} "
                f"elapsed_s={time.monotonic() - started:.2f}"
            )
            self._report_versions(delta)
        elif kind == "env":
            assignments = shlex.split(action.command)
            if not assignments or any("=" not in assignment for assignment in assignments):
                raise RuntimeError(f"unsupported ENV instruction: {action.command}")
            for assignment in assignments:
                name, _, value = assignment.partition("=")
                self.environment[name] = value
        elif kind == "workdir":
            path = PurePosixPath(action.command)
            if not path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsupported WORKDIR: {path}")
            self.cwd = self.root / str(path).lstrip("/")
            self.cwd.mkdir(parents=True, exist_ok=True)
        elif kind == "run":
            command = "cd " + shlex.quote(self.cwd.as_posix()) + " && " + action.command
            run(["/bin/bash", "-c", command], self.environment)
        # ARG is embedded in recovered RUN commands. SHELL is also embedded.
        # CMD/ENTRYPOINT describe container startup; this tool starts pytest explicitly.
        elif kind not in {"arg", "shell", "user", "label", "cmd", "entrypoint"}:
            raise RuntimeError(f"unsupported target-image action: {kind}")

    def _report(self, message: str) -> None:
        if self.report is not None:
            self.report(message)

    def _report_versions(self, delta: BaseImageDelta) -> None:
        seen = set()
        for name, kind, _ in delta.after.files:
            path = self.root / name
            if (
                kind != "file"
                or path.name not in {"version.info", "ascend_toolkit_install.info"}
                or not name.startswith("usr/local/Ascend/")
            ):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                content = path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                self._report(f"[base-image:version] path={path} unavailable: {exc}")
            else:
                self._report(f"[base-image:version] path={path}\n{content}")
