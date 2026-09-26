# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Private persistent caches for the local interface analyzer.

The cache uses pickle for Python ASTs and analyzer dataclasses. Pickle is not a
safe interchange format: only files created below this tool-owned directory are
read. Never copy cache files from an untrusted source into that directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import platform
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

CACHE_NAMESPACE = "vllm-interface-contracts"
CACHE_SCHEMA_VERSION = 4
_CACHE_MAGIC = "vllm-interface-contracts-private-pickle"
_LOCK_WAIT_SECONDS = 30.0
_STALE_LOCK_SECONDS = 600.0


def require_private_cache_path(path: Path) -> None:
    """Reject observed links/reparse points before accessing trusted cache data.

    This is a path-boundary check, not a sandbox against concurrent malicious
    directory replacement. Cache parents must remain private to the user.
    A pickle envelope is not authentication; never import untrusted caches.
    """
    absolute = Path(os.path.abspath(path))
    for candidate in (*reversed(absolute.parents), absolute):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise OSError(f"unsafe cache path: linked or reparse-point entry {candidate}")
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise OSError(f"unsafe cache path: hard-linked file {candidate}")


def default_cache_dir() -> Path:
    """Return a user-local parent directory for analyzer-owned cache data."""

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base)
    base = os.environ.get("XDG_CACHE_HOME")
    return Path(base) if base else Path.home() / ".cache"


def normalized_repo_path(root: Path) -> str:
    return os.path.normcase(str(root.resolve()))


def python_identity() -> dict[str, str]:
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag or "unknown",
    }


def build_identity(
    *,
    component: str,
    repo_root: Path,
    commit_sha: str,
    analyzer_version: str,
    component_schema: int,
    config: Mapping[str, object],
    source_fingerprint: str = "clean",
) -> dict[str, object]:
    """Build the complete, reviewable identity used for one cache component."""

    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "component_schema_version": component_schema,
        "component": component,
        "repo_root": normalized_repo_path(repo_root),
        "commit_sha": commit_sha,
        "analyzer_version": analyzer_version,
        "python": python_identity(),
        "config": dict(sorted(config.items())),
        "source_fingerprint": source_fingerprint,
    }


def git_source_state(repo_root: Path, pathspec: str) -> tuple[bool, str]:
    """Return whether committed-source caching is safe for one package tree."""

    import subprocess

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                pathspec,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return False, f"source state unavailable: {type(error).__name__}: {error}"
    dirty_lines = tuple(line for line in result.stdout.splitlines() if line.strip())
    if dirty_lines:
        digest = hashlib.sha256("\n".join(dirty_lines).encode()).hexdigest()
        return False, f"uncommitted source changes ({len(dirty_lines)}, status={digest[:12]})"
    return True, "clean"


@dataclass
class CacheResult:
    component: str
    enabled: bool
    status: str
    commit_sha: str | None = None
    key: str | None = None
    path: str | None = None
    reason: str | None = None
    load_seconds: float = 0.0
    write_seconds: float = 0.0
    saved_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "enabled": self.enabled,
            "status": self.status,
            "commit_sha": self.commit_sha,
            "key": self.key,
            "path": self.path,
            "reason": self.reason,
            "load_seconds": round(self.load_seconds, 6),
            "write_seconds": round(self.write_seconds, 6),
            "saved_seconds": round(self.saved_seconds, 6),
        }


class PersistentCache:
    """Read and write only analyzer-owned cache entries below one root."""

    def __init__(self, root: Path | None, *, enabled: bool = True):
        self.root = (root.resolve() / CACHE_NAMESPACE) if root is not None else None
        self.enabled = enabled and self.root is not None
        self.events: list[CacheResult] = []

    def _entry_path(self, component: str, identity: Mapping[str, object]) -> tuple[str, Path]:
        if self.root is None:
            raise ValueError("cache root is unavailable")
        serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        key = hashlib.sha256(serialized.encode()).hexdigest()
        component_dir = self.root / f"schema-{CACHE_SCHEMA_VERSION}" / component
        path = component_dir / f"{key}.pickle"
        root_value = os.path.normcase(os.path.abspath(self.root))
        path_value = os.path.normcase(os.path.abspath(path))
        if os.path.commonpath((root_value, path_value)) != root_value:
            raise ValueError("cache entry escaped the configured cache directory")
        return key, path

    def load(
        self,
        component: str,
        identity: Mapping[str, object],
        *,
        validator: Callable[[object], bool] | None = None,
    ) -> tuple[object | None, CacheResult]:
        commit = str(identity.get("commit_sha") or "") or None
        if not self.enabled:
            result = CacheResult(component, False, "disabled", commit_sha=commit)
            self.events.append(result)
            return None, result
        key, path = self._entry_path(component, identity)
        result = CacheResult(component, True, "miss", commit_sha=commit, key=key, path=str(path))
        started = time.perf_counter()
        try:
            require_private_cache_path(path)
        except OSError as error:
            result.status = "bypassed"
            result.reason = str(error)
            result.load_seconds = time.perf_counter() - started
            self.events.append(result)
            return None, result
        if path.is_file():
            try:
                with path.open("rb") as stream:
                    envelope = pickle.load(stream)  # noqa: S301 - tool-owned cache directory only.
                if not isinstance(envelope, dict) or envelope.get("magic") != _CACHE_MAGIC:
                    raise ValueError("cache envelope is not owned by this analyzer")
                if envelope.get("identity") != dict(identity):
                    raise ValueError("cache identity does not match")
                payload = envelope.get("payload")
                if validator is not None and not validator(payload):
                    raise ValueError("cache payload failed validation")
                result.status = "hit"
                build_seconds = envelope.get("build_seconds", 0.0)
                if isinstance(build_seconds, (float, int)):
                    result.saved_seconds = max(0.0, float(build_seconds))
                result.load_seconds = time.perf_counter() - started
                result.saved_seconds = max(0.0, result.saved_seconds - result.load_seconds)
                self.events.append(result)
                return payload, result
            except Exception as error:
                result.status = "corrupt"
                result.reason = f"{type(error).__name__}: {error}"
                with contextlib.suppress(OSError):
                    require_private_cache_path(path)
                    path.unlink()
        result.load_seconds = time.perf_counter() - started
        self.events.append(result)
        return None, result

    @contextlib.contextmanager
    def _write_lock(self, path: Path) -> Iterator[bool]:
        lock = path.with_suffix(path.suffix + ".lock")
        deadline = time.monotonic() + _LOCK_WAIT_SECONDS
        acquired = False
        while time.monotonic() < deadline:
            require_private_cache_path(lock)
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                acquired = True
                break
            except FileExistsError:
                with contextlib.suppress(OSError):
                    if time.time() - lock.stat().st_mtime > _STALE_LOCK_SECONDS:
                        lock.unlink()
                        continue
                time.sleep(0.05)
        try:
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    require_private_cache_path(lock)
                    lock.unlink()

    def store(
        self,
        component: str,
        identity: Mapping[str, object],
        payload: object,
        *,
        build_seconds: float,
        result: CacheResult | None = None,
    ) -> CacheResult:
        commit = str(identity.get("commit_sha") or "") or None
        if not self.enabled:
            return result or CacheResult(component, False, "disabled", commit_sha=commit)
        key, path = self._entry_path(component, identity)
        current = result or CacheResult(component, True, "miss", commit_sha=commit, key=key, path=str(path))
        started = time.perf_counter()
        temporary_path: Path | None = None
        try:
            require_private_cache_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            require_private_cache_path(path)
            with self._write_lock(path) as acquired:
                if not acquired:
                    current.status = "write_skipped"
                    current.reason = "another process held the cache write lock"
                    return current
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=path.parent,
                    prefix=f".{key}-",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    pickle.dump(
                        {
                            "magic": _CACHE_MAGIC,
                            "identity": dict(identity),
                            "build_seconds": build_seconds,
                            "payload": payload,
                        },
                        stream,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                require_private_cache_path(path)
                os.replace(temporary_path, path)
                temporary_path = None
                if current.status == "corrupt":
                    current.status = "corrupt_rebuilt"
        except Exception as error:
            current.status = "write_error"
            current.reason = f"{type(error).__name__}: {error}"
        finally:
            if temporary_path is not None:
                with contextlib.suppress(OSError):
                    require_private_cache_path(temporary_path)
                    temporary_path.unlink()
            current.write_seconds = time.perf_counter() - started
        return current

    def clear(self) -> bool:
        """Remove only the configured analyzer-owned cache directory."""

        if not self.enabled or self.root is None:
            return False
        require_private_cache_path(self.root)
        if not self.root.exists():
            return False
        marker_parts = {CACHE_NAMESPACE, f"schema-{CACHE_SCHEMA_VERSION}"}
        if not marker_parts.intersection(self.root.parts):
            raise ValueError("refusing to clear a cache directory outside the analyzer namespace")
        for directory, folders, files in os.walk(self.root, followlinks=False):
            for name in (*folders, *files):
                require_private_cache_path(Path(directory) / name)
        shutil.rmtree(self.root)
        return True
