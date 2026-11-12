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
"""Resolve a base image only when a build comparison needs its contents."""

import hashlib
import shlex
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


@dataclass(frozen=True)
class BaseImageSnapshot:
    reference: str
    digest: str
    architecture: str
    files: tuple[tuple[str, str, str], ...]
    config: tuple[tuple[str, str], ...]
    archive: Path


@dataclass(frozen=True)
class BaseImageDelta:
    before: BaseImageSnapshot
    after: BaseImageSnapshot
    changed_files: tuple[str, ...]
    changed_config: tuple[str, ...]


class BaseImageError(RuntimeError):
    """A base image cannot be resolved or inspected reliably."""


class ImageResolver(Protocol):
    def resolve(self, reference: str, architecture: str) -> BaseImageSnapshot: ...


def from_reference(value: str) -> str:
    """Return the image operand from a resolved Dockerfile FROM instruction."""
    words = shlex.split(value)
    images = [word for word in words if not word.startswith("--")]
    if not images or images[0].startswith("<unknown>"):
        raise BaseImageError(f"unresolved FROM image: {value}")
    return images[0]


def _manifest(archive: Path) -> tuple[tuple[str, str, str], ...]:
    files: dict[str, tuple[str, str]] = {}
    with tarfile.open(archive, "r") as stream:
        for member in stream:
            name = member.name.removeprefix("./").rstrip("/")
            if not name:
                continue
            if member.name.startswith("/") or ".." in PurePosixPath(name).parts:
                raise BaseImageError(f"unsafe path in image archive: {member.name}")
            if member.isfile():
                reader = stream.extractfile(member)
                if reader is None:
                    raise BaseImageError(f"cannot read image file {name}")
                digest = hashlib.sha256()
                while chunk := reader.read(1024 * 1024):
                    digest.update(chunk)
                files[name] = ("file", f"{member.mode:o}:{member.uid}:{member.gid}:{digest.hexdigest()}")
            elif member.issym():
                files[name] = ("symlink", f"{member.uid}:{member.gid}:{member.linkname}")
            elif member.islnk():
                files[name] = ("hardlink", f"{member.uid}:{member.gid}:{member.linkname}")
            elif member.isdir():
                files[name] = ("directory", f"{member.mode:o}:{member.uid}:{member.gid}")
            else:
                files[name] = ("special", f"{member.type!r}:{member.mode:o}:{member.uid}:{member.gid}")
    return tuple((name, *value) for name, value in sorted(files.items()))


def compare_base_images(before: BaseImageSnapshot, after: BaseImageSnapshot) -> BaseImageDelta:
    old_files = {name: (kind, value) for name, kind, value in before.files}
    new_files = {name: (kind, value) for name, kind, value in after.files}
    old_config = dict(before.config)
    new_config = dict(after.config)
    changed_files = tuple(
        sorted(name for name in old_files.keys() | new_files.keys() if old_files.get(name) != new_files.get(name))
    )
    changed_config = tuple(
        sorted(name for name in old_config.keys() | new_config.keys() if old_config.get(name) != new_config.get(name))
    )
    return BaseImageDelta(
        before,
        after,
        changed_files,
        changed_config,
    )


def compare_base_references(
    before: str,
    after: str,
    architecture: str,
    resolver: ImageResolver,
) -> BaseImageDelta | None:
    """Keep the common same-FROM bisect path free of registry requests."""
    if before == after:
        return None
    return compare_base_images(resolver.resolve(before, architecture), resolver.resolve(after, architecture))
