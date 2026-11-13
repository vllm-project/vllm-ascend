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
"""Read OCI/Docker images with the Python standard library, on demand."""

import hashlib
import http.client
import json
import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path, PurePosixPath
from uuid import uuid4

import regex as re

from tools.bisect.build_analyzer.base_image import BaseImageError, BaseImageSnapshot, _manifest

_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_CHUNK_SIZE = 1024 * 1024
_BLOB_RETRIES = 8
logger = logging.getLogger(__name__)


def _image_address(reference: str) -> tuple[str, str, str]:
    first, separator, remainder = reference.partition("/")
    if not separator or not ("." in first or ":" in first or first == "localhost"):
        registry, repository = "registry-1.docker.io", reference
        if "/" not in repository:
            repository = "library/" + repository
    else:
        registry, repository = first, remainder
    if "@" in repository:
        repository, tag = repository.rsplit("@", 1)
    elif ":" in repository.rsplit("/", 1)[-1]:
        repository, tag = repository.rsplit(":", 1)
    else:
        tag = "latest"
    if not registry or not repository or not tag:
        raise BaseImageError(f"invalid image reference: {reference}")
    return registry, repository, tag


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _check_digest(digest: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise BaseImageError(f"unsupported or invalid image digest: {digest}")


def _safe_name(name: str) -> str:
    name = name.removeprefix("./").rstrip("/")
    if name.startswith("/") or ".." in PurePosixPath(name).parts:
        raise BaseImageError(f"unsafe image path: {name}")
    return name


class RegistryImageResolver:
    """Resolve selected-platform image layers and cache a flattened rootfs."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._tokens: dict[tuple[str, str], str] = {}

    def resolve_history(self, reference: str, architecture: str) -> tuple[str, list[dict]]:
        """Read target history without downloading filesystem layers."""
        registry, repository, tag = _image_address(reference)
        manifest, digest = self._json(registry, repository, f"manifests/{tag}")
        if "manifests" in manifest:
            arch = "arm64" if architecture == "aarch64" else "amd64"
            descriptor = next(
                (
                    item
                    for item in manifest["manifests"]
                    if item.get("platform", {}).get("architecture") == arch
                    and item.get("platform", {}).get("os") == "linux"
                ),
                None,
            )
            if descriptor is None:
                raise BaseImageError(f"image {reference} has no linux/{arch} manifest")
            digest = descriptor["digest"]
            manifest, _ = self._json(registry, repository, f"manifests/{digest}", digest)
        config = json.loads(self._blob(registry, repository, manifest["config"]["digest"]).read_bytes())
        history = [
            {"CreatedBy": item.get("created_by", ""), "Comment": item.get("comment", "")}
            for item in reversed(config.get("history", []))
        ]
        return f"{registry}/{repository}@{digest}", history

    def _open(self, registry: str, repository: str, route: str, extra_headers: dict[str, str] | None = None):
        url = f"https://{registry}/v2/{repository}/{route}"
        headers = {"Accept": _ACCEPT, **(extra_headers or {})}
        token = self._tokens.get((registry, repository))
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            return urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as exc:
            challenge = exc.headers.get("WWW-Authenticate", "")
            if exc.code != 401 or not challenge.lower().startswith("bearer "):
                raise BaseImageError(
                    f"registry request failed: {registry}/{repository}/{route}: HTTP {exc.code}"
                ) from exc
            parameters = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
            realm = parameters.get("realm")
            if not realm or not realm.startswith("https://"):
                raise BaseImageError(f"registry authentication challenge is unsupported: {registry}") from exc
            query = {name: parameters[name] for name in ("service", "scope") if name in parameters}
            token_url = realm + ("&" if "?" in realm else "?") + urllib.parse.urlencode(query)
            try:
                with urllib.request.urlopen(token_url, timeout=120) as response:
                    credentials = json.load(response)
            except (OSError, ValueError) as token_error:
                raise BaseImageError(f"cannot obtain registry token for {registry}") from token_error
            token = credentials.get("token") or credentials.get("access_token")
            if not token:
                raise BaseImageError(f"registry did not return a token for {registry}")
            self._tokens[(registry, repository)] = token
            request = urllib.request.Request(url, headers={**headers, "Authorization": f"Bearer {token}"})
            try:
                return urllib.request.urlopen(request, timeout=120)
            except urllib.error.HTTPError as retry_error:
                raise BaseImageError(
                    f"registry request failed after authentication: {registry}/{repository}/{route}: "
                    f"HTTP {retry_error.code}"
                ) from retry_error
        except OSError as exc:
            raise BaseImageError(f"cannot reach registry {registry}: {exc}") from exc

    def _json(self, registry: str, repository: str, route: str, expected: str | None = None) -> tuple[dict, str]:
        with closing(self._open(registry, repository, route)) as response:
            data = response.read()
        digest = _digest(data)
        if expected is not None and digest != expected:
            raise BaseImageError(f"registry returned an unexpected digest for {route}: {digest}")
        try:
            return json.loads(data), digest
        except ValueError as exc:
            raise BaseImageError(f"registry returned invalid JSON for {route}") from exc

    def _blob(self, registry: str, repository: str, digest: str) -> Path:
        _check_digest(digest)
        path = self.cache_dir / "blobs" / digest.removeprefix("sha256:")
        if path.is_file():
            return path
        logger.info("[image-content] downloading blob %s", digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        for attempt in range(_BLOB_RETRIES):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with closing(self._open(registry, repository, f"blobs/{digest}", headers)) as response:
                    status = response.status
                    if offset and status != 206:
                        temporary.unlink(missing_ok=True)
                        offset = 0
                    content_range = response.headers.get("Content-Range", "")
                    expected_total = None
                    if status == 206:
                        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                        if match is None or int(match.group(1)) != offset:
                            raise BaseImageError(f"invalid blob range response: {digest}")
                        expected_total = int(match.group(3))
                    elif response.headers.get("Content-Length"):
                        expected_total = int(response.headers["Content-Length"])
                    with temporary.open("ab" if offset else "wb") as output:
                        while chunk := response.read(_CHUNK_SIZE):
                            output.write(chunk)
                if expected_total is not None and temporary.stat().st_size < expected_total:
                    raise http.client.IncompleteRead(b"", expected_total - temporary.stat().st_size)
                checksum = hashlib.sha256()
                with temporary.open("rb") as stream:
                    while chunk := stream.read(_CHUNK_SIZE):
                        checksum.update(chunk)
                if f"sha256:{checksum.hexdigest()}" != digest:
                    raise BaseImageError(f"image blob checksum mismatch: {digest}")
                temporary.replace(path)
                logger.info("[image-content] cached blob %s (%d bytes)", digest, path.stat().st_size)
                return path
            except (OSError, http.client.IncompleteRead) as exc:
                if attempt == _BLOB_RETRIES - 1:
                    raise BaseImageError(
                        f"image blob download failed after {_BLOB_RETRIES} attempts: {digest}"
                    ) from exc
                time.sleep(min(attempt + 1, 5))
        raise AssertionError("unreachable")

    def _flatten(self, registry: str, repository: str, layers: list[dict], destination: Path) -> None:
        entries: dict[str, tuple[tarfile.TarInfo, Path | None]] = {}
        files_dir = destination.parent / f"files-{uuid4().hex}"
        files_dir.mkdir(exist_ok=True)
        for descriptor in layers:
            logger.info("[image-content] unpacking layer %s", descriptor["digest"])
            media_type = descriptor.get("mediaType", "")
            if "zstd" in media_type or "nondistributable" in media_type:
                raise BaseImageError(f"unsupported image layer format: {media_type}")
            layer = self._blob(registry, repository, descriptor["digest"])
            try:
                stream = tarfile.open(layer, "r:*")  # noqa: SIM115 - closed by the with statement below
            except tarfile.TarError as exc:
                raise BaseImageError(f"cannot read image layer {descriptor['digest']}: {exc}") from exc
            with stream:
                members = [(_safe_name(member.name), member) for member in stream]
                # OCI whiteouts erase entries from lower layers, regardless of
                # the order of entries within the current layer tar.
                for name, _ in members:
                    filename = PurePosixPath(name).name
                    parent = str(PurePosixPath(name).parent)
                    if filename == ".wh..wh..opq":
                        prefix = "" if parent == "." else parent + "/"
                        for previous in tuple(entries):
                            if previous.startswith(prefix):
                                del entries[previous]
                    elif filename.startswith(".wh."):
                        removed = str(PurePosixPath(parent) / filename.removeprefix(".wh."))
                        for previous in tuple(entries):
                            if previous == removed or previous.startswith(removed + "/"):
                                del entries[previous]
                for name, member in members:
                    if not name or PurePosixPath(name).name.startswith(".wh."):
                        continue
                    body_path = None
                    if member.isfile():
                        source = stream.extractfile(member)
                        if source is None:
                            raise BaseImageError(f"cannot read image file {name}")
                        body_path = files_dir / hashlib.sha256(name.encode("utf-8")).hexdigest()
                        with body_path.open("wb") as output:
                            shutil.copyfileobj(source, output, _CHUNK_SIZE)
                    entries[name] = member, body_path
        temporary = destination.with_suffix(".tmp")
        logger.info("[image-content] writing flattened archive (%d entries)", len(entries))
        try:
            with tarfile.open(temporary, "w") as output:
                for name, (member, body_path) in sorted(entries.items()):
                    member.name = name
                    if body_path is None:
                        output.addfile(member)
                    else:
                        with body_path.open("rb") as body:
                            output.addfile(member, body)
            temporary.replace(destination)
            shutil.rmtree(files_dir)
        finally:
            temporary.unlink(missing_ok=True)

    def resolve(self, reference: str, architecture: str) -> BaseImageSnapshot:
        registry, repository, tag = _image_address(reference)
        platform = f"linux/{'arm64' if architecture == 'aarch64' else 'amd64'}"
        expected = tag if tag.startswith("sha256:") else None
        if expected is not None:
            _check_digest(expected)
        manifest, digest = self._json(registry, repository, f"manifests/{tag}", expected)
        if "manifests" in manifest:
            descriptor = next(
                (
                    item
                    for item in manifest["manifests"]
                    if item.get("platform", {}).get("os") == "linux"
                    and item.get("platform", {}).get("architecture") == platform.split("/")[1]
                ),
                None,
            )
            if descriptor is None:
                raise BaseImageError(f"image {reference} has no {platform} manifest")
            digest = descriptor["digest"]
            _check_digest(digest)
            manifest, _ = self._json(registry, repository, f"manifests/{digest}", digest)
        directory = self.cache_dir / digest.removeprefix("sha256:") / platform.replace("/", "-")
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "rootfs.tar"
        config_path = directory / "config.json"
        manifest_path = directory / "manifest.json"
        if not config_path.is_file():
            config_path.write_bytes(self._blob(registry, repository, manifest["config"]["digest"]).read_bytes())
        if not archive.is_file():
            self._flatten(registry, repository, manifest["layers"], archive)
        if not manifest_path.is_file():
            logger.info("[image-content] indexing flattened archive %s", archive)
            manifest_path.write_text(json.dumps(_manifest(archive)), encoding="utf-8")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8")).get("config", {})
            files = tuple(tuple(item) for item in json.loads(manifest_path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise BaseImageError(f"invalid cached image metadata for {reference}") from exc
        settings = tuple((name, json.dumps(value, sort_keys=True)) for name, value in sorted(config.items()))
        return BaseImageSnapshot(reference, digest, platform, files, settings, archive)
