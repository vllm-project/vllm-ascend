# SPDX-License-Identifier: Apache-2.0
"""Prepare a pinned, selectively downloaded checkpoint for a nightly gate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from .identity import checkpoint_identity, digest
from .profiles import get_profile
from .reducer import (
    execute_plan,
    load_source_selective,
    plan_reduction,
    required_shards,
    resolve_index_filename,
    verify_reduced,
)
from .safetensors_io import sha256_of_file


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked_file(root: Path, name: str, expected: dict) -> Path:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError(f"Unsafe source file: {name}")
    target = root / name
    if not target.is_file() or target.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing/wrong size source file: {target}")
    if sha256_of_file(str(target)) != expected["sha256"]:
        raise ValueError(f"Source checksum mismatch: {target}")
    return target


def download_file(root: Path, name: str, descriptor: dict) -> None:
    expected = descriptor["files"][name]
    if (root / name).exists():
        checked_file(root, name, expected)
        return
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name:
        raise ValueError(f"Unsafe source file: {name}")
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    url = f"https://modelscope.cn/models/{descriptor['model']}/resolve/{descriptor['revision']}/" + urllib.parse.quote(
        name
    )
    print(f"Fetching {name} ({expected['bytes']} bytes)", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vllm-ascend-glm-reduced/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024**2)
        if partial.stat().st_size != expected["bytes"] or sha256_of_file(str(partial)) != expected["sha256"]:
            raise ValueError(f"Downloaded source checksum mismatch: {name}")
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def prepare_checkpoint(
    descriptor_path: str | Path,
    source_dir: str | Path,
    cache_dir: str | Path,
    *,
    profile_name: str = "glm-moe-dsa",
    layers: int = 8,
    download: bool = False,
) -> tuple[Path, str]:
    descriptor = read_json(descriptor_path)
    profile = get_profile(profile_name)
    index_name = resolve_index_filename(descriptor["files"])
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    recipe = {"source": descriptor, "profile": profile_name, "layers": layers}
    output = cache / digest(recipe)
    if output.exists():
        report = verify_reduced(str(output))
        manifest = read_json(output / "reduction_manifest.json")
        source = manifest["source"]
        if (
            not report["ok"]
            or report["profile"] != profile_name
            or report["keep_layers"] != layers
            or source["config_sha256"] != descriptor["files"]["config.json"]["sha256"]
            or source["index_sha256"] != descriptor["files"][index_name]["sha256"]
            or source["quant_description_sha256"] != descriptor["files"]["quant_model_description.json"]["sha256"]
            or not manifest["reduction"]["keep_mtp"]
        ):
            raise ValueError(f"Cached reduction failed verification: {report}")
        return output, checkpoint_identity(output)

    source_root = Path(source_dir)
    source_root.mkdir(parents=True, exist_ok=True)
    metadata = ("config.json", index_name)
    for name in metadata:
        if download:
            download_file(source_root, name, descriptor)
        checked_file(source_root, name, descriptor["files"][name])
    needed = required_shards(str(source_root / metadata[0]), str(source_root / metadata[1]), profile, layers)["shards"]
    # Copy tokenizer/config/quantization auxiliaries; download only required main shards.
    auxiliary = [name for name in descriptor["files"] if not name.startswith("quant_model_weights-")]
    selected = sorted(set(needed + auxiliary))
    if download:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda name: download_file(source_root, name, descriptor), selected))
    for name in selected:
        checked_file(source_root, name, descriptor["files"][name])
    source = load_source_selective(str(source_root), profile, layers)
    # Provider cache bookkeeping is machine-local, not part of the checkpoint.
    # Only copy auxiliaries whose bytes were verified against the pinned source.
    source.aux_files = [name for name in source.aux_files if name in descriptor["files"]]
    plan = plan_reduction(source, profile, keep_layers=layers)
    execute_plan(plan, source, str(output))
    report = verify_reduced(str(output))
    if not report["ok"]:
        raise ValueError(f"Reduction verification failed: {report}")
    return output, checkpoint_identity(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Pinned source descriptor JSON")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--download", action="store_true", help="Explicit provisioning only; nightly stays offline")
    args = parser.parse_args()
    output, identity = prepare_checkpoint(args.source, args.source_dir, args.cache_dir, download=args.download)
    print(json.dumps({"model": str(output), "checkpoint_id": identity}))


if __name__ == "__main__":
    main()
