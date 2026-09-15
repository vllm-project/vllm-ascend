#!/usr/bin/env python3
"""Download selected missing ModelScope safetensors shards.

The script obtains the repository manifest first, so it does not guess file
sizes or silently download a shard that is not present in the requested model.
Downloads are written to ``*.part`` files and resumed with HTTP Range requests
when the connection is interrupted.

Example:
    python tools/download_missing_safetensors.py --output-dir /data/model

Use ``--verify-sha256`` when an existing file must be checked.  Hashing the
33 large shards can take a considerable amount of time and disk bandwidth.
Use ``--verify-existing`` to verify every already downloaded shard without
starting any download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ID = "Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp"
DEFAULT_REVISION = "master"
MANIFEST_URL = "https://modelscope.cn/api/v1/models/{model_id}/repo/files"
RESOLVE_URL = "https://modelscope.cn/models/{model_id}/resolve/{revision}/{path}"
TOTAL_SHARDS = 70
MISSING_SHARDS = (
    9,
    12,
    14,
    16,
    17,
    18,
    20,
    23,
    25,
    28,
    29,
    31,
    32,
    34,
    35,
    37,
    39,
    42,
    45,
    48,
    51,
    53,
    54,
    56,
    57,
    59,
    60,
    62,
    64,
    65,
    67,
    69,
    70,
)
CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "vllm-ascend-missing-shards/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing the model shards (created if necessary).",
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION, help="ModelScope revision (default: master).")
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Verify completed and existing files against the manifest SHA-256.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify already downloaded shards only; do not download anything. "
        "Missing shards are reported but do not cause failure.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Attempts per shard after a network failure (default: 5).",
    )
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=8,
        help="Streaming chunk size in MiB (default: 8).",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files and sizes without downloading.")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.chunk_size_mib < 1:
        parser.error("--chunk-size-mib must be at least 1")
    return args


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to fetch ModelScope manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("Code") != 200:
        raise RuntimeError(f"unexpected ModelScope manifest response: {payload!r}")
    return payload


def get_manifest(revision: str) -> dict[str, dict[str, Any]]:
    query = urllib.parse.urlencode({"Revision": revision, "Recursive": "true"})
    url = MANIFEST_URL.format(model_id=MODEL_ID) + "?" + query
    payload = request_json(url)
    data = payload.get("Data")
    files = data.get("Files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        raise RuntimeError("ModelScope manifest does not contain Data.Files")

    manifest: dict[str, dict[str, Any]] = {}
    for item in files:
        if not isinstance(item, dict) or item.get("Type") != "blob":
            continue
        path = item.get("Path")
        if isinstance(path, str):
            manifest[path] = item
    return manifest


def shard_name(shard_number: int) -> str:
    return f"quant_model_weights-{shard_number:05d}-of-{TOTAL_SHARDS:05d}.safetensors"


def sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        chunk = stream.read(chunk_size)
        while chunk:
            digest.update(chunk)
            chunk = stream.read(chunk_size)
    return digest.hexdigest()


def validate_existing(path: Path, expected_size: int, expected_sha256: str, verify: bool, chunk_size: int) -> bool:
    if not path.exists():
        return False
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{path} already exists with size {actual_size}, expected {expected_size}; "
            "remove or move it before retrying"
        )
    if verify:
        actual_sha256 = sha256_file(path, chunk_size)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"{path} has SHA-256 {actual_sha256}, expected {expected_sha256}; "
                "remove or move it before retrying"
            )
    return True


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{size} B"


def verify_existing_shards(
    output_dir: Path,
    manifest: dict[str, dict[str, Any]],
    chunk_size: int,
) -> int:
    """Verify all standard model shards that are present in ``output_dir``.

    Missing shards are expected while a partial model download is in progress,
    so they are reported separately and do not affect the return code.  A
    present shard with an incorrect size or digest is a verification failure.
    """
    checked = 0
    missing = 0
    failed = 0
    for number in range(1, TOTAL_SHARDS + 1):
        name = shard_name(number)
        path = output_dir / name
        item = manifest.get(name)
        if item is None:
            raise RuntimeError(f"shard is not present in ModelScope manifest: {name}")
        if not path.exists():
            missing += 1
            print(f"MISSING {name}", flush=True)
            continue

        checked += 1
        expected_size = item["Size"]
        expected_sha256 = item["Sha256"]
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            failed += 1
            print(
                f"SIZE-MISMATCH {name}: got {actual_size}, expected {expected_size}",
                flush=True,
            )
            continue

        print(f"CHECK {name} ({format_size(expected_size)})", flush=True)
        actual_sha256 = sha256_file(path, chunk_size)
        if actual_sha256 != expected_sha256:
            failed += 1
            print(
                f"SHA256-MISMATCH {name}: got {actual_sha256}, expected {expected_sha256}",
                flush=True,
            )
        else:
            print(f"OK {name}: {actual_sha256}", flush=True)

    print(
        f"Verification summary: checked={checked}, missing={missing}, failed={failed}",
        flush=True,
    )
    return 1 if failed else 0


def download_shard(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    revision: str,
    retries: int,
    chunk_size: int,
    verify: bool,
) -> None:
    if validate_existing(path, expected_size, expected_sha256, verify, chunk_size):
        print(f"SKIP {path.name}: already complete ({format_size(expected_size)})", flush=True)
        return

    part_path = path.with_name(path.name + ".part")
    if part_path.exists() and part_path.stat().st_size > expected_size:
        raise RuntimeError(f"{part_path} is larger than the expected shard; remove it before retrying")

    encoded_model = "/".join(urllib.parse.quote(part, safe="") for part in MODEL_ID.split("/"))
    encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in path.name.split("/"))
    url = RESOLVE_URL.format(
        model_id=encoded_model,
        revision=urllib.parse.quote(revision, safe=""),
        path=encoded_path,
    )

    for attempt in range(1, retries + 1):
        offset = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range returns 200; restart the .part
                # file in that case instead of appending a duplicate prefix.
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                mode = "ab" if append else "wb"
                downloaded = offset
                last_report = time.monotonic()
                with part_path.open(mode) as output:
                    chunk = response.read(chunk_size)
                    while chunk:
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 5:
                            percent = downloaded * 100 / expected_size
                            print(
                                f"  {path.name}: {percent:6.2f}% "
                                f"({format_size(downloaded)}/{format_size(expected_size)})",
                                flush=True,
                            )
                            last_report = now
                        chunk = response.read(chunk_size)
                if downloaded != expected_size:
                    raise RuntimeError(f"download ended at {downloaded} bytes, expected {expected_size}")
            if verify:
                actual_sha256 = sha256_file(part_path, chunk_size)
                if actual_sha256 != expected_sha256:
                    raise RuntimeError(
                        f"SHA-256 mismatch: got {actual_sha256}, expected {expected_sha256}"
                    )
            os.replace(part_path, path)
            print(f"DONE {path.name}: {format_size(expected_size)}", flush=True)
            return
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            if attempt == retries:
                raise RuntimeError(f"failed to download {path.name} after {retries} attempts: {exc}") from exc
            delay = min(60, 2 ** (attempt - 1))
            print(f"  attempt {attempt}/{retries} failed for {path.name}: {exc}; retrying in {delay}s", flush=True)
            time.sleep(delay)


def main() -> int:
    args = parse_args()
    chunk_size = args.chunk_size_mib * 1024 * 1024
    manifest = get_manifest(args.revision)
    if len(manifest) < TOTAL_SHARDS:
        raise RuntimeError(f"manifest contains {len(manifest)} files, expected at least {TOTAL_SHARDS} shards")

    if args.verify_existing:
        return verify_existing_shards(args.output_dir, manifest, chunk_size)

    selected: list[tuple[str, dict[str, Any]]] = []
    for number in MISSING_SHARDS:
        name = shard_name(number)
        item = manifest.get(name)
        if item is None:
            raise RuntimeError(f"missing shard is not present in ModelScope manifest: {name}")
        size = item.get("Size")
        sha256 = item.get("Sha256")
        if not isinstance(size, int) or size <= 0 or not isinstance(sha256, str) or len(sha256) != 64:
            raise RuntimeError(f"invalid size or SHA-256 metadata for {name}: {item!r}")
        selected.append((name, item))

    total_size = sum(item["Size"] for _, item in selected)
    print(
        f"ModelScope: {MODEL_ID}@{args.revision}\n"
        f"Selected shards: {len(selected)}\n"
        f"Total download: {format_size(total_size)}\n"
        f"Output directory: {args.output_dir}",
        flush=True,
    )
    for name, item in selected:
        print(f"  {name}  {format_size(item['Size'])}", flush=True)
    if args.dry_run:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, item in selected:
        download_shard(
            args.output_dir / name,
            item["Size"],
            item["Sha256"],
            args.revision,
            args.retries,
            chunk_size,
            args.verify_sha256,
        )
    print("All selected shards are complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Existing .part files are kept for resume.", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
