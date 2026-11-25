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
"""Validate whether a base-image delta can be applied inside a live container."""

import hashlib
import json
import os
import shutil
import stat
import sys
import tarfile
import tempfile
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path, PurePosixPath

from tools.bisect.build_analyzer.base_image import BaseImageDelta

_COPY_PROGRESS_INTERVAL_S = 5.0


def _current_entry(path: Path) -> tuple[str, str] | None:
    if not path.is_symlink() and not path.exists():
        return None
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return "symlink", f"{metadata.st_uid}:{metadata.st_gid}:{os.readlink(path)}"
    if stat.S_ISDIR(metadata.st_mode):
        return "directory", f"{stat.S_IMODE(metadata.st_mode):o}:{metadata.st_uid}:{metadata.st_gid}"
    if stat.S_ISREG(metadata.st_mode):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return "file", f"{stat.S_IMODE(metadata.st_mode):o}:{metadata.st_uid}:{metadata.st_gid}:{digest.hexdigest()}"
    return "special", f"{metadata.st_mode}"


def _loaded_files() -> set[Path]:
    """Include every visible process, not only the bisect controller."""
    loaded = {Path(sys.executable).resolve()}
    process_root = Path("/proc")
    if not process_root.is_dir():
        return loaded
    for process in process_root.iterdir():
        if not process.name.isdecimal():
            continue
        for mapped_file in (process / "maps",):
            try:
                lines = mapped_file.read_text(encoding="utf-8", errors="replace").splitlines()
            except (OSError, PermissionError):
                continue
            for line in lines:
                name = line.split(maxsplit=5)[-1]
                if name.startswith("/"):
                    loaded.add(Path(name.removesuffix(" (deleted)")))
        with suppress(OSError, RuntimeError):
            loaded.add((process / "exe").resolve(strict=True))
    return loaded


def _mount_points() -> set[Path]:
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return set()
    points: set[Path] = set()
    for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) > 4:
            points.add(Path(fields[4].replace("\\040", " ")))
    return points


def validate_base_application(delta: BaseImageDelta, root: Path) -> tuple[str, ...]:
    """Return every reason a rootfs delta cannot safely touch this container.

    The application scope is CANN's user-space installation. Changes elsewhere still
    appear in the image diff but cannot be replayed in a running container.
    """
    old_files = {name: (kind, value) for name, kind, value in delta.before.files}
    new_files = {name: (kind, value) for name, kind, value in delta.after.files}
    reasons: list[str] = []
    if root == Path("/") and "/cann:" in delta.after.reference:
        for required in (
            Path("/dev/davinci_manager"),
            Path("/dev/devmm_svm"),
            Path("/usr/local/Ascend/driver/version.info"),
        ):
            if not required.exists():
                reasons.append(f"{required}: required host driver/device resource is unavailable")
    active_files = _loaded_files() if root == Path("/") else set()
    mount_points = _mount_points() if root == Path("/") else set()
    for setting in delta.changed_config:
        if setting not in {"Env", "Labels"}:
            reasons.append(f"image configuration {setting}: cannot be changed inside a running container")
    for name in delta.changed_files:
        parts = PurePosixPath(name).parts
        allowed = (parts[:2] == ("usr", "local") and len(parts) >= 3) or parts[:2] == ("etc", "Ascend")
        if not allowed:
            reasons.append(f"{name}: outside the live-container CANN application scope")
            continue
        path = root.joinpath(*parts)
        if path in active_files or path.resolve() in active_files:
            reasons.append(f"{name}: loaded by a running process")
            continue
        if any(parent.is_symlink() for parent in path.parents if parent != root and root in parent.parents):
            reasons.append(f"{name}: a parent path is a symbolic link")
            continue
        if root == Path("/") and any(
            candidate in mount_points or os.path.ismount(candidate)
            for candidate in (path, *path.parents)
            if candidate != root
        ):
            reasons.append(f"{name}: path belongs to a mounted filesystem")
            continue
        before = old_files.get(name)
        after = new_files.get(name)
        if before is not None and before[0] == "directory" and (after is None or after[0] != "directory"):
            reasons.append(f"{name}: removing a directory may remove files written by later layers")
            continue
        if after is not None and after[0] in {"special", "hardlink"}:
            reasons.append(f"{name}: target entry type {after[0]} is not supported for live replacement")
            continue
        if after is not None and after[0] == "directory" and before is not None and before[0] != "directory":
            reasons.append(f"{name}: file-to-directory replacement is not supported")
            continue
        if before is not None and before[0] in {"special", "hardlink"}:
            reasons.append(f"{name}: existing {before[0]} cannot be replaced")
            continue
        current = _current_entry(path)
        if current != before and current != after:
            reasons.append(f"{name}: live file differs from both base images; an upper layer may own it")
    return tuple(reasons)


def target_environment(delta: BaseImageDelta, inherited: dict[str, str]) -> dict[str, str]:
    """Apply image Env to a child process without changing the bisect controller."""
    environment = dict(inherited)
    old_config = dict(delta.before.config)
    image_config = dict(delta.after.config)
    old_assignments = json.loads(old_config.get("Env", "[]")) or []
    new_assignments = json.loads(image_config.get("Env", "[]")) or []
    new_names = {assignment.partition("=")[0] for assignment in new_assignments}
    for assignment in old_assignments:
        name = assignment.partition("=")[0]
        if name not in new_names:
            environment.pop(name, None)
    for assignment in new_assignments:
        name, separator, value = assignment.partition("=")
        if separator:
            environment[name] = value
    return environment


def apply_base_delta(
    delta: BaseImageDelta,
    root: Path,
    report: Callable[[str], None] | None = None,
) -> None:
    """Replace eligible files transactionally; never overwrite upper-layer writes."""
    blockers = validate_base_application(delta, root)
    if blockers:
        raise RuntimeError("base image cannot be applied: " + "; ".join(blockers[:10]))
    target_files = {name: (kind, value) for name, kind, value in delta.after.files}
    last_report = time.monotonic()

    def progress(phase: str, completed: int, *, final: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if report is not None and (completed == 0 or final or now - last_report >= _COPY_PROGRESS_INTERVAL_S):
            report(f"[base-image:copy-progress] phase={phase} entries={completed}/{len(delta.changed_files)}")
            last_report = now

    root = root.resolve()
    backup_parent = root / "usr" / "local"
    backup_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bisect-base-", dir=backup_parent) as temporary:
        backup = Path(temporary)
        staged = backup / "staged"
        staged.mkdir()
        replacements: list[tuple[Path, Path, bool]] = []
        new_directories: list[Path] = []
        with tarfile.open(delta.after.archive, "r") as archive:
            members = {member.name.removeprefix("./").rstrip("/"): member for member in archive}
            for index, name in enumerate(delta.changed_files):
                progress("stage", index)
                target = root.joinpath(*PurePosixPath(name).parts)
                target_kind = target_files.get(name, (None, ""))[0]
                if target_kind == "directory":
                    if not target.exists():
                        new_directories.append(target)
                    continue
                if target_kind is None:
                    continue
                member = members.get(name)
                if member is None:
                    raise RuntimeError(f"target image archive is missing {name}")
                destination = staged / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if target_kind == "file":
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"target image archive cannot read {name}")
                    with destination.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    destination.chmod(member.mode)
                    if os.name != "nt":
                        os.chown(destination, member.uid, member.gid)
                elif target_kind == "symlink":
                    destination.symlink_to(member.linkname)
                    if os.name != "nt":
                        os.lchown(destination, member.uid, member.gid)
            progress("stage", len(delta.changed_files), final=True)
        old_metadata: list[tuple[Path, int, int, int]] = []
        try:
            for directory in sorted(new_directories, key=lambda path: len(path.parts)):
                directory.mkdir(parents=True, exist_ok=True)
            for index, name in enumerate(
                sorted(delta.changed_files, key=lambda value: (-len(PurePosixPath(value).parts), value))
            ):
                progress("apply", index)
                target = root.joinpath(*PurePosixPath(name).parts)
                target_kind = target_files.get(name, (None, ""))[0]
                if target_kind == "directory":
                    if target.exists() and target not in new_directories:
                        metadata = target.stat()
                        old_metadata.append((target, stat.S_IMODE(metadata.st_mode), metadata.st_uid, metadata.st_gid))
                    member = members.get(name)
                    if member is not None:
                        target.chmod(member.mode)
                        if os.name != "nt":
                            os.chown(target, member.uid, member.gid)
                    continue
                previous = backup / "previous" / name
                previous.parent.mkdir(parents=True, exist_ok=True)
                existed = target.exists() or target.is_symlink()
                if existed:
                    target.replace(previous)
                replacements.append((target, previous, existed))
                if target_kind is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (staged / name).replace(target)
            progress("apply", len(delta.changed_files), final=True)
            for name in sorted(delta.changed_files, key=lambda value: (-len(PurePosixPath(value).parts), value)):
                if name not in target_files:
                    directory = root.joinpath(*PurePosixPath(name).parts)
                    if directory.is_dir():
                        directory.rmdir()
        except Exception:
            for target, previous, existed in reversed(replacements):
                if target.is_symlink() or target.is_file():
                    target.unlink()
                if existed:
                    previous.replace(target)
            for directory, mode, uid, gid in old_metadata:
                if directory.exists():
                    directory.chmod(mode)
                    if os.name != "nt":
                        os.chown(directory, uid, gid)
            for directory in sorted(new_directories, key=lambda path: -len(path.parts)):
                if directory.exists():
                    directory.rmdir()
            raise
