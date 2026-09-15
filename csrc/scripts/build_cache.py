#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Local content-addressed build cache for vLLM-Ascend.
#
# Cache equivalence:
#   prepared_input_hash       - what is compiled?
#   recipe_hash               - how is it compiled?
#   compiler_environment_hash - what compiles it?
#
# The cache engine owns no static operator or third-party unit list.

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Sequence

SCHEMA_VERSION = 3
ARTIFACT_MODEL_BY_DOMAIN = {"third_party": 1, "custom_operator": 2}
PUBLISH_STATE_SCHEMA = 1
CHUNK_SIZE = 1024 * 1024
EVENT_LOG_ENV = "VLLM_ASCEND_BUILD_CACHE_EVENT_LOG"
LOCK_POLL_SECONDS = 0.05
LOCK_WAIT_EVENT_SECONDS = 1.0
DEFAULT_ENTRY_LOCK_TIMEOUT_SECONDS = 60.0
DEFAULT_ACTION_LOCK_TIMEOUT_SECONDS = 120.0
DEFAULT_PUBLISH_LOCK_TIMEOUT_SECONDS = 120.0
DEFAULT_GENERIC_LOCK_TIMEOUT_SECONDS = 120.0
DEFAULT_EXCLUDES = (
    ".git",
    ".git/**",
    "__pycache__",
    "__pycache__/**",
    "*.pyc",
    "*.pyo",
    "*.done",
    "*.log",
    "*.tmp",
)

_TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".py",
    ".sh",
    ".cmake",
    ".json",
    ".ini",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
}


class _LockTimeoutError(Exception):
    def __init__(
        self,
        *,
        kind: str,
        path: Path,
        waited_seconds: float,
    ) -> None:
        self.kind = kind
        self.path = path
        self.waited_seconds = waited_seconds
        super().__init__(
            f"{kind} lock timeout after {waited_seconds:.3f}s: {path}"
        )


class _IndexLockBusy(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, parsed)


def _emit_event(event: str, **fields) -> None:
    # Best-effort JSONL telemetry; it must never affect build correctness.
    event_log = os.environ.get(EVENT_LOG_ENV)
    if not event_log:
        return

    payload = {
        "timestamp_ns": time.time_ns(),
        "pid": os.getpid(),
        "event": event,
        **fields,
    }
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    path = Path(event_log).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o644,
        )
        try:
            # One append/write per event. There is deliberately no telemetry
            # lock: observability must never serialize compilation.
            os.write(fd, encoded)
        finally:
            os.close(fd)
    except OSError:
        pass


def _measure_phase(
    phase: str,
    function,
    *args,
    event_fields: dict | None = None,
    **kwargs,
):
    fields = event_fields or {}
    _emit_event("phase_start", phase=phase, **fields)
    start = time.monotonic()
    try:
        return function(*args, **kwargs)
    finally:
        _emit_event(
            "phase_end",
            phase=phase,
            seconds=round(time.monotonic() - start, 6),
            **fields,
        )


def _lock_kind(path: Path) -> str:
    if path.name == "cache_index.json.lock":
        return "index"
    if path.name == ".publish.lock":
        return "publish"
    if path.parent.name == ".action_locks":
        return "action"
    return "generic"


def _lock_timeout_seconds(kind: str) -> float:
    if kind == "entry":
        return _env_float(
            "VLLM_ASCEND_BUILD_CACHE_ENTRY_LOCK_TIMEOUT_SECONDS",
            DEFAULT_ENTRY_LOCK_TIMEOUT_SECONDS,
        )
    if kind == "action":
        return _env_float(
            "VLLM_ASCEND_BUILD_CACHE_ACTION_LOCK_TIMEOUT_SECONDS",
            DEFAULT_ACTION_LOCK_TIMEOUT_SECONDS,
        )
    if kind == "publish":
        return _env_float(
            "VLLM_ASCEND_BUILD_CACHE_PUBLISH_LOCK_TIMEOUT_SECONDS",
            DEFAULT_PUBLISH_LOCK_TIMEOUT_SECONDS,
        )
    return _env_float(
        "VLLM_ASCEND_BUILD_CACHE_LOCK_TIMEOUT_SECONDS",
        DEFAULT_GENERIC_LOCK_TIMEOUT_SECONDS,
    )


def _acquire_timed_lock(
    stream,
    path: Path,
    *,
    kind: str,
    timeout_seconds: float,
) -> float:
    start = time.monotonic()
    next_wait_event = LOCK_WAIT_EVENT_SECONDS

    while True:
        try:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            waited = time.monotonic() - start
            if waited >= LOCK_WAIT_EVENT_SECONDS:
                _emit_event(
                    "lock_acquired",
                    lock=kind,
                    path=str(path),
                    waited_seconds=round(waited, 6),
                )
            return waited
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise

        waited = time.monotonic() - start
        if waited >= timeout_seconds:
            _emit_event(
                "lock_timeout",
                lock=kind,
                path=str(path),
                waited_seconds=round(waited, 6),
            )
            raise _LockTimeoutError(
                kind=kind,
                path=path,
                waited_seconds=waited,
            )

        if waited >= next_wait_event:
            _emit_event(
                "lock_wait",
                lock=kind,
                path=str(path),
                waited_seconds=round(waited, 6),
            )
            next_wait_event += 5.0

        time.sleep(
            min(
                LOCK_POLL_SECONDS,
                max(0.0, timeout_seconds - waited),
            )
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(records: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(records):
        name_bytes = name.encode("utf-8")
        value_bytes = value.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(value_bytes).to_bytes(8, "big"))
        digest.update(value_bytes)
    return digest.hexdigest()


def _is_excluded(relative_path: str, excludes: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(relative_path, pattern) for pattern in excludes)


def _hash_prepared_inputs(
    paths: Sequence[Path],
    excludes: Sequence[str],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for input_index, raw_path in enumerate(paths):
        path = raw_path.resolve()
        label = f"input[{input_index}]"

        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError(f"prepared input does not exist: {path}")

        if path.is_symlink():
            target = os.readlink(path)
            records.append((label, f"symlink:{target}"))
            manifest.append({"path": label, "kind": "symlink", "target": target})
            continue

        if path.is_file():
            file_hash = _sha256_file(path)
            logical_path = f"{label}/{path.name}"
            records.append((logical_path, file_hash))
            manifest.append(
                {"path": logical_path, "kind": "file", "sha256": file_hash}
            )
            continue

        for child in sorted(
            path.rglob("*"),
            key=lambda item: item.relative_to(path).as_posix(),
        ):
            relative = child.relative_to(path).as_posix()
            if _is_excluded(relative, excludes):
                continue

            logical_path = f"{label}/{relative}"
            if child.is_symlink():
                target = os.readlink(child)
                records.append((logical_path, f"symlink:{target}"))
                manifest.append(
                    {"path": logical_path, "kind": "symlink", "target": target}
                )
            elif child.is_file():
                file_hash = _sha256_file(child)
                records.append((logical_path, file_hash))
                manifest.append(
                    {"path": logical_path, "kind": "file", "sha256": file_hash}
                )

    return _canonical_hash(records), manifest


def _looks_text(path: Path, data: bytes) -> bool:
    if path.name == "CMakeLists.txt" or path.suffix.lower() in _TEXT_SUFFIXES:
        return b"\x00" not in data
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _git_tracked_files(source_dir: Path, repo_root: Path) -> list[Path] | None:
    try:
        relative = source_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None

    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "ls-files",
                "-z",
                "--",
                relative.as_posix(),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return None

    if proc.returncode != 0:
        return None

    files: list[Path] = []
    for raw in proc.stdout.split(b"\x00"):
        if not raw:
            continue
        files.append(repo_root / os.fsdecode(raw))
    return files


def _hash_operator_text(
    source_dir: Path,
    repo_root: Path | None,
) -> tuple[str, list[dict]]:
    source_dir = source_dir.resolve()
    candidates = (
        _git_tracked_files(source_dir, repo_root)
        if repo_root is not None
        else None
    )

    if candidates is None:
        candidates = [path for path in source_dir.rglob("*") if path.is_file()]

    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for path in sorted(candidates, key=lambda item: str(item)):
        if not path.exists() and not path.is_symlink():
            continue
        try:
            relative = path.resolve().relative_to(source_dir).as_posix()
        except ValueError:
            # A tracked symlink may resolve outside source_dir. Use the tracked
            # path identity rather than the resolved target path.
            try:
                relative = path.relative_to(source_dir).as_posix()
            except ValueError:
                continue

        if path.is_symlink():
            target = os.readlink(path)
            records.append((relative, f"symlink:{target}"))
            manifest.append(
                {"path": relative, "kind": "symlink", "target": target}
            )
            continue

        data = path.read_bytes()
        if not _looks_text(path, data):
            continue

        digest = _sha256_bytes(data)
        records.append((relative, digest))
        manifest.append({"path": relative, "sha256": digest})

    return _canonical_hash(records), manifest


def _normalize_text(text: str, normalize_paths: Sequence[Path]) -> str:
    normalized = text.replace("\\", "/")
    replacements: list[tuple[str, str]] = []

    for index, path in enumerate(normalize_paths):
        try:
            value = str(path.resolve()).replace("\\", "/")
        except FileNotFoundError:
            value = str(path.absolute()).replace("\\", "/")
        replacements.append((value.rstrip("/"), f"<PATH_{index}>"))

    for source, replacement in sorted(
        replacements,
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if source:
            normalized = normalized.replace(source, replacement)

    return normalized


def _hash_recipe(
    recipe_files: Sequence[Path],
    recipe_values: Sequence[str],
    command: Sequence[str],
    normalize_paths: Sequence[Path],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = []
    manifest: list[dict] = []

    for index, raw_path in enumerate(recipe_files):
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"recipe file does not exist: {path}")

        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            value_hash = _sha256_bytes(data)
            kind = "binary"
        else:
            normalized = _normalize_text(text, normalize_paths)
            value_hash = _sha256_bytes(normalized.encode("utf-8"))
            kind = "text"

        records.append((f"recipe_file[{index}]", value_hash))
        manifest.append(
            {
                "index": index,
                "name": path.name,
                "kind": kind,
                "sha256": value_hash,
            }
        )

    for index, value in enumerate(recipe_values):
        normalized = _normalize_text(value, normalize_paths)
        records.append((f"recipe_value[{index}]", normalized))
        manifest.append(
            {
                "index": index,
                "kind": "value",
                "value": normalized,
            }
        )

    normalized_command = [
        _normalize_text(str(token), normalize_paths) for token in command
    ]
    records.append(
        (
            "original_command",
            json.dumps(
                normalized_command,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    )
    manifest.append({"kind": "command", "argv": normalized_command})

    return _canonical_hash(records), manifest


def _command_version(argv: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    output = proc.stdout.strip()
    return output or None


def _resolve_tool(tool: str) -> str | None:
    path = Path(tool).expanduser()
    if path.is_absolute() or os.sep in tool:
        return str(path) if path.exists() else None
    return shutil.which(tool)


def _candidate_cann_metadata_files() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))

    opp_path = os.environ.get("ASCEND_OPP_PATH")
    if opp_path:
        roots.append(Path(opp_path).parent)

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "version.info",
                root / "ascend_toolkit_install.info",
                root.parent / "version.info",
            ]
        )
    candidates.append(Path("/etc/ascend_install.info"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _hash_compiler_environment(
    profile: str,
    environment_files: Sequence[Path],
    environment_values: Sequence[str],
    environment_tools: Sequence[str],
) -> tuple[str, list[dict]]:
    records: list[tuple[str, str]] = [
        ("profile", profile),
        ("system", platform.system()),
        ("machine", platform.machine()),
    ]
    manifest: list[dict] = [
        {"name": "profile", "value": profile},
        {"name": "system", "value": platform.system()},
        {"name": "machine", "value": platform.machine()},
    ]

    default_tools: list[str] = []
    if profile == "host-cxx":
        default_tools = [] if environment_tools else ["cc", "c++", "cmake"]
    elif profile == "ascendc":
        default_tools = [] if environment_tools else ["bisheng", "ccec"]
        environment_files = list(environment_files) + _candidate_cann_metadata_files()
    else:
        raise ValueError(f"unknown environment profile: {profile}")

    seen_tools: set[str] = set()
    for index, tool in enumerate(list(environment_tools) + default_tools):
        resolved = _resolve_tool(tool)
        if resolved is None:
            continue
        canonical = str(Path(resolved).resolve())
        if canonical in seen_tools:
            continue
        seen_tools.add(canonical)

        output = _command_version([canonical, "--version"])
        if output is None:
            continue

        # The absolute installation path is intentionally not hashed. The
        # executable's reported identity/version is the semantic signal.
        tool_name = Path(canonical).name
        records.append((f"tool[{index}]:{tool_name}", output))
        manifest.append(
            {
                "name": f"tool[{index}]",
                "tool": tool_name,
                "version": output,
            }
        )

    seen_files: set[str] = set()
    for raw_path in environment_files:
        path = raw_path.expanduser()
        if not path.is_file():
            continue
        digest = _sha256_file(path)
        identity = path.name
        dedupe_key = f"{identity}:{digest}"
        if dedupe_key in seen_files:
            continue
        seen_files.add(dedupe_key)
        records.append((f"metadata:{identity}", digest))
        manifest.append(
            {
                "name": f"metadata:{identity}",
                "sha256": digest,
            }
        )

    for index, value in enumerate(environment_values):
        records.append((f"value[{index}]", value))
        manifest.append({"name": f"value[{index}]", "value": value})

    return _canonical_hash(records), manifest


def _artifact_signature(path: Path) -> tuple:
    # lstat() describes the directory entry itself instead of following a
    # symlink. A symlink may be a required build product (protobuf's `protoc`
    # is one such case), so it must participate in artifact discovery.
    stat = path.lstat()
    if path.is_symlink():
        return (
            "symlink",
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            os.readlink(path),
        )
    return ("file", stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _snapshot(root: Path) -> dict[str, tuple]:
    snapshot: dict[str, tuple] = {}
    if not root.exists():
        return snapshot

    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.is_file():
            snapshot[path.relative_to(root).as_posix()] = _artifact_signature(path)
    return snapshot


def _internal_symlink_target(root: Path, link: Path) -> Path | None:
    target = Path(os.readlink(link))
    if target.is_absolute():
        candidate = target
    else:
        candidate = link.parent / target

    root_abs = Path(os.path.abspath(root))
    candidate_abs = Path(os.path.abspath(os.path.normpath(candidate)))
    try:
        relative = candidate_abs.relative_to(root_abs)
    except ValueError:
        return None
    return root_abs / relative


def _expand_internal_symlink_targets(
    root: Path,
    selected: set[str],
) -> set[str]:
    # Caching only `protoc -> ascend_protoc` would restore a broken link
    # after deleting the transient build tree. Include the in-tree target
    # closure even when the target does not match ARTIFACT_INCLUDE.
    root = Path(os.path.abspath(root))
    expanded = set(selected)
    pending = list(selected)

    while pending:
        relative = pending.pop()
        path = root / relative
        if not path.is_symlink():
            continue

        target = _internal_symlink_target(root, path)
        if target is None:
            # External links belong to the environment. Preserve the link
            # itself but do not import external files into this cache entry.
            continue
        if not (target.is_symlink() or target.is_file()):
            raise RuntimeError(
                f"symlink artifact target does not exist: {path} -> "
                f"{os.readlink(path)}"
            )

        target_relative = target.relative_to(root).as_posix()
        if target_relative not in expanded:
            expanded.add(target_relative)
            pending.append(target_relative)

    return expanded


def _collect_artifacts(
    root: Path,
    before: dict[str, tuple],
    include_patterns: Sequence[str],
) -> list[str]:
    if not root.exists():
        return []

    if include_patterns:
        selected: set[str] = set()
        for path in root.rglob("*"):
            if not (path.is_symlink() or path.is_file()):
                continue
            relative = path.relative_to(root).as_posix()
            if any(
                fnmatch.fnmatch(relative, pattern)
                for pattern in include_patterns
            ):
                selected.add(relative)
        return sorted(_expand_internal_symlink_targets(root, selected))

    after = _snapshot(root)
    selected = {
        relative
        for relative, signature in after.items()
        if before.get(relative) != signature
    }
    return sorted(_expand_internal_symlink_targets(root, selected))


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    if not safe:
        raise ValueError(f"invalid empty cache path component from {value!r}")
    return safe


def _entry_path(
    cache_root: Path,
    domain: str,
    unit: str,
    final_key: str,
    operator_text_hash: str | None,
    soc: str | None,
    operator: str | None,
    action: str | None,
) -> Path:
    if domain == "third_party":
        return cache_root / "third_party" / _safe_component(unit) / final_key

    if not soc or not operator or not action or not operator_text_hash:
        raise ValueError(
            "custom_operator cache requires --soc, --operator, --action, "
            "and --operator-source"
        )

    return (
        cache_root
        / "custom_operator"
        / _safe_component(soc)
        / _safe_component(operator)
        / operator_text_hash
        / _safe_component(action)
        / final_key
    )


def _load_json(path: Path, default: dict) -> dict:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temp_name)


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        kind = _lock_kind(path)
        if kind == "index":
            try:
                fcntl.flock(
                    stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                _emit_event(
                    "index_skipped",
                    lock="index",
                    path=str(path),
                    reason="lock_busy",
                )
                raise _IndexLockBusy(str(path)) from None
        else:
            _acquire_timed_lock(
                stream,
                path,
                kind=kind,
                timeout_seconds=_lock_timeout_seconds(kind),
            )
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _file_lock_or_error(path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a+")
    except OSError as exc:
        yield exc
        return

    with stream:
        try:
            _acquire_timed_lock(
                stream,
                path,
                kind="entry",
                timeout_seconds=_lock_timeout_seconds("entry"),
            )
        except (OSError, _LockTimeoutError) as exc:
            yield exc
            return
        try:
            yield None
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _expected_artifact_model(domain: str) -> int:
    try:
        return ARTIFACT_MODEL_BY_DOMAIN[domain]
    except KeyError as exc:
        raise ValueError(f"unknown cache domain: {domain}") from exc


def _validate_artifact(root: Path, artifact: dict) -> bool:
    path = root / artifact["path"]
    kind = artifact.get("kind", "file")

    if kind == "symlink":
        return path.is_symlink() and os.readlink(path) == artifact.get("target")

    if kind != "file":
        return False
    return (
        path.is_file()
        and not path.is_symlink()
        and _sha256_file(path) == artifact.get("sha256")
    )


def _describe_artifacts(root: Path, artifact_paths: Sequence[str]) -> list[dict]:
    artifacts: list[dict] = []
    for relative in sorted(set(artifact_paths)):
        source = root / relative
        if source.is_symlink():
            artifacts.append(
                {
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(source),
                }
            )
            continue
        if source.is_file():
            artifacts.append(
                {
                    "path": relative,
                    "kind": "file",
                    "sha256": _sha256_file(source),
                }
            )
    if not artifacts:
        raise RuntimeError(f"no cacheable artifacts found in {root}")
    return artifacts


def _validate_entry(entry: Path, final_key: str, domain: str) -> dict | None:
    manifest_path = entry / "manifest.json"
    artifact_root = entry / "artifacts"
    if not manifest_path.is_file() or not artifact_root.is_dir():
        return None

    manifest = _load_json(manifest_path, {})
    if manifest.get("schema") != SCHEMA_VERSION:
        return None
    if manifest.get("final_key") != final_key:
        return None
    if manifest.get("domain") != domain:
        return None

    # Existing schema-3 third-party entries predate artifact_model and are
    # still safe: their symlink-aware serialization is already complete.
    # Existing schema-3 custom-operator entries are *not* safe because they
    # were discovered by diffing a shared output directory under parallel
    # compilation. Treat missing artifact_model as the legacy model (1), so
    # only custom operators are invalidated by the new isolated model (2).
    artifact_model = manifest.get("artifact_model", 1)
    if artifact_model != _expected_artifact_model(domain):
        return None

    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        return None
    try:
        valid = all(
            _validate_artifact(artifact_root, artifact)
            for artifact in artifacts
        )
    except OSError:
        return None
    if not valid:
        return None
    return manifest


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.build-cache-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        shutil.copy2(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _atomic_create_symlink(target: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.build-cache-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink()
    try:
        temp_path.symlink_to(target)
        os.replace(temp_path, destination)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _atomic_publish_artifact(
    source_root: Path,
    destination_root: Path,
    artifact: dict,
) -> None:
    source = source_root / artifact["path"]
    destination = destination_root / artifact["path"]
    kind = artifact.get("kind", "file")

    if kind == "file":
        _atomic_copy_file(source, destination)
        return
    if kind == "symlink":
        _atomic_create_symlink(artifact["target"], destination)
        return
    raise RuntimeError(f"unsupported artifact kind: {kind!r}")


def _restore_entry(entry: Path, output_dir: Path, manifest: dict) -> None:
    artifact_root = entry / "artifacts"

    # Files first, then symlinks, so relative symlinks are valid immediately.
    for artifact in manifest["artifacts"]:
        if artifact.get("kind", "file") == "file":
            _atomic_publish_artifact(artifact_root, output_dir, artifact)
    for artifact in manifest["artifacts"]:
        if artifact.get("kind") == "symlink":
            _atomic_publish_artifact(artifact_root, output_dir, artifact)


def _save_entry(
    entry: Path,
    output_dir: Path,
    artifacts: Sequence[dict],
    manifest_base: dict,
) -> None:
    entry.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{entry.name}.tmp-", dir=str(entry.parent))
    )
    try:
        artifact_root = temp_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)

        serialized: list[dict] = []
        for artifact in artifacts:
            source = output_dir / artifact["path"]
            destination = artifact_root / artifact["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)

            if artifact.get("kind", "file") == "symlink":
                destination.symlink_to(artifact["target"])
            else:
                shutil.copy2(source, destination)
            serialized.append(dict(artifact))

        manifest = dict(manifest_base)
        manifest["schema"] = SCHEMA_VERSION
        manifest["artifacts"] = serialized
        manifest["created_at"] = int(time.time())
        _atomic_write_json(temp_dir / "manifest.json", manifest)

        if not all(_validate_artifact(artifact_root, artifact) for artifact in serialized):
            raise RuntimeError("artifact verification failed before cache publish")

        if entry.exists():
            shutil.rmtree(entry)
        os.replace(temp_dir, entry)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _reset_private_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _publish_action_state_path(state_dir: Path, action_identity: str) -> Path:
    digest = _sha256_bytes(action_identity.encode("utf-8"))
    return state_dir / "actions" / f"{digest}.json"


def _load_publish_state(path: Path, default: dict) -> dict:
    payload = _load_json(path, default)
    if payload.get("schema") != PUBLISH_STATE_SCHEMA:
        return default
    return payload


def _publish_action_artifacts(
    *,
    source_root: Path,
    publish_dir: Path,
    publish_state_dir: Path,
    action_identity: str,
    artifacts: Sequence[dict],
) -> None:
    publish_dir.mkdir(parents=True, exist_ok=True)
    publish_state_dir.mkdir(parents=True, exist_ok=True)

    lock_path = publish_state_dir / ".publish.lock"
    owners_path = publish_state_dir / "owners.json"
    action_state_path = _publish_action_state_path(
        publish_state_dir,
        action_identity,
    )

    with _file_lock(lock_path):
        owners = _load_publish_state(
            owners_path,
            {
                "schema": PUBLISH_STATE_SCHEMA,
                "initialized": False,
                "owners": {},
            },
        )

        # The first isolated-model publisher owns initialization of the shared
        # operator output directory. This removes stale files left by the
        # legacy shared-snapshot model without racing other actions: all real
        # compilation now happens in private stage directories, and this short
        # bootstrap is protected by the shared publish lock.
        if not owners.get("initialized", False):
            if publish_dir.exists():
                shutil.rmtree(publish_dir)
            publish_dir.mkdir(parents=True, exist_ok=True)
            owners["initialized"] = True

        action_state = _load_publish_state(
            action_state_path,
            {
                "schema": PUBLISH_STATE_SCHEMA,
                "action": action_identity,
                "artifacts": [],
            },
        )

        owner_map = owners.setdefault("owners", {})
        previous = {
            artifact["path"]: artifact
            for artifact in action_state.get("artifacts", [])
        }
        current = {artifact["path"]: artifact for artifact in artifacts}

        # Remove artifacts that this action published in an earlier invocation
        # but no longer produces. The owner table lives in the disposable build
        # tree, so it cannot affect another checkout/workspace.
        for relative, old_artifact in previous.items():
            if relative in current:
                continue
            owner = owner_map.get(relative)
            if not owner or owner.get("action") != action_identity:
                continue
            destination = publish_dir / relative
            if destination.is_symlink() or destination.is_file():
                if not _validate_artifact(publish_dir, old_artifact):
                    raise RuntimeError(
                        "published artifact changed outside its owning action: "
                        f"{destination}"
                    )
                destination.unlink(missing_ok=True)
            elif destination.exists():
                raise RuntimeError(
                    f"published artifact became a directory: {destination}"
                )
            owner_map.pop(relative, None)

        # Publish each action's exact private outputs into the shared directory.
        # The publish lock serializes this short operation while compilation
        # itself remains fully parallel in private stage directories.
        ordered = sorted(
            artifacts,
            key=lambda item: (item.get("kind") == "symlink", item["path"]),
        )
        for artifact in ordered:
            relative = artifact["path"]
            owner = owner_map.get(relative)
            if owner and owner.get("action") != action_identity:
                raise RuntimeError(
                    "custom-operator artifact ownership collision: "
                    f"{relative} owned by {owner.get('action')}, "
                    f"requested by {action_identity}"
                )
            _atomic_publish_artifact(source_root, publish_dir, artifact)
            owner_map[relative] = {
                "action": action_identity,
                "artifact": dict(artifact),
            }

        owners["schema"] = PUBLISH_STATE_SCHEMA
        action_state = {
            "schema": PUBLISH_STATE_SCHEMA,
            "action": action_identity,
            "artifacts": [dict(item) for item in artifacts],
        }
        _atomic_write_json(owners_path, owners)
        _atomic_write_json(action_state_path, action_state)


def _safe_update_index(*args, **kwargs) -> None:
    try:
        _update_index(*args, **kwargs)
    except _IndexLockBusy:
        # Index metadata is observational only. A busy index must never
        # serialize or delay the build/cache correctness path.
        return
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"[build-cache] WARNING index update failed: {exc}",
            flush=True,
        )
        _emit_event(
            "warning",
            component="index",
            message=str(exc),
        )


def _run_build_command(
    args: argparse.Namespace,
    command: Sequence[str],
) -> tuple[int, float]:
    environment = _parse_set_env(args.set_env)
    start = time.monotonic()
    process = subprocess.run(
        list(command),
        cwd=args.working_directory,
        env=environment,
        check=False,
        close_fds=False,
    )
    return process.returncode, time.monotonic() - start


def _logical_status(
    domain: str,
    previous_key: str | None,
    current_key: str,
    previous_operator_text_hash: str | None,
    operator_text_hash: str | None,
) -> str:
    if previous_key is None:
        return "NEW"
    if domain == "custom_operator":
        return (
            "UNCHANGED"
            if previous_operator_text_hash == operator_text_hash
            else "MODIFIED"
        )
    return "UNCHANGED" if previous_key == current_key else "MODIFIED"


def _update_index(
    cache_root: Path,
    domain: str,
    unit: str,
    final_key: str,
    prepared_input_hash: str,
    recipe_hash: str,
    environment_hash: str,
    cache_status: str,
    operator_text_hash: str | None,
    soc: str | None,
    operator: str | None,
    action: str | None,
) -> None:
    domain_root = cache_root / domain
    index_path = domain_root / "cache_index.json"
    lock_path = domain_root / "cache_index.json.lock"

    with _file_lock(lock_path):
        index = _load_json(
            index_path,
            {"schema": SCHEMA_VERSION, "units": {}},
        )
        if index.get("schema") != SCHEMA_VERSION:
            index = {"schema": SCHEMA_VERSION, "units": {}}

        if domain == "third_party":
            unit_key = unit
        else:
            unit_key = f"{soc}/{operator}/{action}"

        state = index["units"].setdefault(unit_key, {})
        previous_key = state.get("current_key")
        previous_operator_text_hash = state.get("operator_text_hash")
        logical_status = _logical_status(
            domain,
            previous_key,
            final_key,
            previous_operator_text_hash,
            operator_text_hash,
        )

        state["previous_key"] = previous_key
        state["current_key"] = final_key
        state["operator_text_hash"] = operator_text_hash
        state["prepared_input_hash"] = prepared_input_hash
        state["recipe_hash"] = recipe_hash
        state["compiler_environment_hash"] = environment_hash
        state["logical_status"] = logical_status
        state["cache_status"] = cache_status
        state["updated_at"] = int(time.time())

        history = state.setdefault("history", {})
        history[final_key] = {
            "operator_text_hash": operator_text_hash,
            "prepared_input_hash": prepared_input_hash,
            "recipe_hash": recipe_hash,
            "compiler_environment_hash": environment_hash,
            "logical_status": logical_status,
            "cache_status": cache_status,
            "updated_at": int(time.time()),
        }

        _atomic_write_json(index_path, index)


def _parse_set_env(values: Sequence[str]) -> dict[str, str]:
    environment = os.environ.copy()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set-env expects NAME=VALUE, got {value!r}")
        name, content = value.split("=", 1)
        environment[name] = content
    return environment


def run(args: argparse.Namespace) -> int:
    cache_root = Path(args.cache_root).expanduser().resolve()
    event_fields = {
        "domain": args.domain,
        "unit": args.unit,
        "action": args.action,
        "soc": args.soc,
    }
    _emit_event("action_start", **event_fields)
    prepared_inputs = [Path(value) for value in args.prepared_input]
    recipe_files = [Path(value) for value in args.recipe_file]
    environment_files = [Path(value) for value in args.environment_file]
    normalize_paths = [Path(value) for value in args.normalize_path]
    output_dir = Path(args.output_dir).resolve()

    publish_dir: Path | None = None
    publish_state_dir: Path | None = None
    if args.domain == "custom_operator":
        if not args.operator_source:
            raise ValueError("custom_operator cache requires --operator-source")
        if not args.publish_dir or not args.publish_state_dir:
            raise ValueError(
                "custom_operator cache requires --publish-dir and "
                "--publish-state-dir for action-isolated output"
            )
        publish_dir = Path(args.publish_dir).resolve()
        publish_state_dir = Path(args.publish_state_dir).resolve()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    command = list(args.command)
    if not command:
        raise ValueError("missing build command after --")

    excludes = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    prepared_input_hash, prepared_manifest = _measure_phase(
        "hash_prepared_inputs",
        _hash_prepared_inputs,
        prepared_inputs,
        excludes,
        event_fields=event_fields,
    )

    operator_text_hash: str | None = None
    operator_text_manifest: list[dict] = []
    if args.domain == "custom_operator":
        repo_root = Path(args.repo_root) if args.repo_root else None
        operator_text_hash, operator_text_manifest = _measure_phase(
            "hash_operator_text",
            _hash_operator_text,
            Path(args.operator_source),
            repo_root,
            event_fields=event_fields,
        )

    recipe_hash, recipe_manifest = _measure_phase(
        "hash_recipe",
        _hash_recipe,
        recipe_files,
        args.recipe_value,
        command,
        normalize_paths,
        event_fields=event_fields,
    )
    environment_hash, environment_manifest = _measure_phase(
        "hash_compiler_environment",
        _hash_compiler_environment,
        args.environment_profile,
        environment_files,
        args.environment_value,
        args.environment_tool,
        event_fields=event_fields,
    )
    final_key = _canonical_hash(
        [
            ("prepared_input_hash", prepared_input_hash),
            ("recipe_hash", recipe_hash),
            ("compiler_environment_hash", environment_hash),
        ]
    )

    entry = _entry_path(
        cache_root=cache_root,
        domain=args.domain,
        unit=args.unit,
        final_key=final_key,
        operator_text_hash=operator_text_hash,
        soc=args.soc,
        operator=args.operator,
        action=args.action,
    )
    lock_name = _sha256_bytes(str(entry).encode("utf-8"))
    entry_lock = cache_root / args.domain / ".locks" / f"{lock_name}.lock"

    manifest_base = {
        "domain": args.domain,
        "unit": args.unit,
        "soc": args.soc,
        "operator": args.operator,
        "action": args.action,
        "final_key": final_key,
        "artifact_model": _expected_artifact_model(args.domain),
        "operator_text_hash": operator_text_hash,
        "prepared_input_hash": prepared_input_hash,
        "recipe_hash": recipe_hash,
        "compiler_environment_hash": environment_hash,
        "operator_text_inputs": operator_text_manifest,
        "prepared_inputs": prepared_manifest,
        "recipe": recipe_manifest,
        "compiler_environment": environment_manifest,
    }

    def update_index(cache_status: str) -> None:
        _safe_update_index(
            cache_root,
            args.domain,
            args.unit,
            final_key,
            prepared_input_hash,
            recipe_hash,
            environment_hash,
            cache_status,
            operator_text_hash,
            args.soc,
            args.operator,
            args.action,
        )

    def publish_custom(artifacts: Sequence[dict]) -> None:
        assert publish_dir is not None
        assert publish_state_dir is not None
        action_identity = f"{args.unit}/{args.action}"
        _measure_phase(
            "publish",
            _publish_action_artifacts,
            source_root=output_dir,
            publish_dir=publish_dir,
            publish_state_dir=publish_state_dir,
            action_identity=action_identity,
            artifacts=artifacts,
            event_fields=event_fields,
        )

    def build_without_cache(reason: str) -> int:
        print(
            f"[build-cache] BYPASS domain={args.domain} unit={args.unit} "
            f"reason={reason}",
            flush=True,
        )
        _emit_event(
            "cache_result",
            status="BYPASS",
            reason=reason,
            **event_fields,
        )
        if args.domain == "custom_operator":
            _reset_private_output(output_dir)
            returncode, _ = _run_build_command(args, command)
            if returncode != 0:
                return returncode
            paths = _collect_artifacts(output_dir, {}, args.artifact_include)
            artifacts = _describe_artifacts(output_dir, paths)
            publish_custom(artifacts)
            return 0

        before = _snapshot(output_dir)
        returncode, _ = _run_build_command(args, command)
        if returncode != 0:
            return returncode
        # No cache save in bypass mode; third-party products already live in
        # output_dir and downstream can continue.
        _collect_artifacts(output_dir, before, args.artifact_include)
        return 0

    # Same action + same build tree must never reset the same private stage
    # concurrently, even if the persistent cache itself is unavailable.
    if args.domain == "custom_operator":
        assert publish_state_dir is not None
        action_identity = f"{args.unit}/{args.action}"
        action_lock_name = _sha256_bytes(action_identity.encode("utf-8"))
        action_lock = (
            publish_state_dir
            / ".action_locks"
            / f"{action_lock_name}.lock"
        )
    else:
        action_lock = None

    @contextlib.contextmanager
    def action_scope():
        if action_lock is None:
            yield
        else:
            with _file_lock(action_lock):
                yield

    with action_scope():
        with _file_lock_or_error(entry_lock) as lock_error:
            if lock_error is not None:
                return build_without_cache(
                    f"cache lock unavailable: {lock_error}"
                )

            manifest = _measure_phase(
                "validate_entry",
                _validate_entry,
                entry,
                final_key,
                args.domain,
                event_fields=event_fields,
            )
            if manifest is not None:
                try:
                    if args.domain == "custom_operator":
                        _reset_private_output(output_dir)
                    _measure_phase(
                        "restore_entry",
                        _restore_entry,
                        entry,
                        output_dir,
                        manifest,
                        event_fields=event_fields,
                    )
                    if args.domain == "custom_operator":
                        publish_custom(manifest["artifacts"])
                    print(
                        f"[build-cache] HIT domain={args.domain} "
                        f"unit={args.unit} key={final_key}",
                        flush=True,
                    )
                    _emit_event(
                        "cache_result",
                        status="HIT",
                        key=final_key,
                        **event_fields,
                    )
                    update_index("HIT")
                    return 0
                except (OSError, RuntimeError) as exc:
                    print(
                        f"[build-cache] WARNING restore failed; rebuilding "
                        f"domain={args.domain} unit={args.unit} "
                        f"key={final_key}: {exc}",
                        flush=True,
                    )
                    _emit_event(
                        "warning",
                        component="restore",
                        key=final_key,
                        message=str(exc),
                        **event_fields,
                    )
                    if args.domain == "custom_operator":
                        _reset_private_output(output_dir)

            print(
                f"[build-cache] MISS domain={args.domain} unit={args.unit} "
                f"key={final_key}",
                flush=True,
            )
            _emit_event(
                "cache_result",
                status="MISS",
                key=final_key,
                **event_fields,
            )

            if args.domain == "custom_operator":
                _reset_private_output(output_dir)
                before: dict[str, tuple] = {}
            else:
                before = _snapshot(output_dir)

            returncode, elapsed = _run_build_command(args, command)
            if returncode != 0:
                return returncode

            paths = _collect_artifacts(
                output_dir,
                before,
                args.artifact_include,
            )
            artifacts = _describe_artifacts(output_dir, paths)

            if args.domain == "custom_operator":
                # Publishing is part of build correctness and therefore must
                # succeed. Cache persistence remains an optimization.
                publish_custom(artifacts)

            cache_saved = False
            try:
                _save_entry(
                    entry,
                    output_dir,
                    artifacts,
                    {
                        **manifest_base,
                        "build_seconds": elapsed,
                    },
                )
                cache_saved = True
                print(
                    f"[build-cache] SAVED domain={args.domain} "
                    f"unit={args.unit} key={final_key} "
                    f"artifacts={len(artifacts)} "
                    f"build_seconds={elapsed:.3f}",
                    flush=True,
                )
                _emit_event(
                    "cache_result",
                    status="SAVED",
                    key=final_key,
                    artifacts=len(artifacts),
                    build_seconds=round(elapsed, 6),
                    **event_fields,
                )
            except (OSError, RuntimeError) as exc:
                print(
                    f"[build-cache] WARNING save failed; build result kept "
                    f"domain={args.domain} unit={args.unit} "
                    f"key={final_key}: {exc}",
                    flush=True,
                )
                _emit_event(
                    "warning",
                    component="save",
                    key=final_key,
                    message=str(exc),
                    **event_fields,
                )

            update_index("MISS_BUILT" if cache_saved else "MISS_UNCACHED")
            return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vLLM-Ascend local build cache"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--cache-root", required=True)
    run_parser.add_argument(
        "--domain",
        required=True,
        choices=("third_party", "custom_operator"),
    )
    run_parser.add_argument("--unit", required=True)
    run_parser.add_argument("--soc")
    run_parser.add_argument("--operator")
    run_parser.add_argument("--action")
    run_parser.add_argument("--operator-source")
    run_parser.add_argument("--repo-root")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--publish-dir")
    run_parser.add_argument("--publish-state-dir")
    run_parser.add_argument("--prepared-input", action="append", default=[])
    run_parser.add_argument("--recipe-file", action="append", default=[])
    run_parser.add_argument("--recipe-value", action="append", default=[])
    run_parser.add_argument(
        "--environment-profile",
        choices=("ascendc", "host-cxx"),
        required=True,
    )
    run_parser.add_argument("--environment-file", action="append", default=[])
    run_parser.add_argument("--environment-value", action="append", default=[])
    run_parser.add_argument("--environment-tool", action="append", default=[])
    run_parser.add_argument("--normalize-path", action="append", default=[])
    run_parser.add_argument("--artifact-include", action="append", default=[])
    run_parser.add_argument("--exclude", action="append", default=[])
    run_parser.add_argument("--set-env", action="append", default=[])
    run_parser.add_argument("--working-directory")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.subcommand != "run":
        parser.error(f"unsupported subcommand: {args.subcommand}")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return run(args)
    except _LockTimeoutError as exc:
        print(
            f"[build-cache] ERROR lock timeout kind={exc.kind} "
            f"waited_seconds={exc.waited_seconds:.3f} path={exc.path}",
            flush=True,
        )
        _emit_event(
            "error",
            error="lock_timeout",
            lock=exc.kind,
            path=str(exc.path),
            waited_seconds=round(exc.waited_seconds, 6),
        )
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
