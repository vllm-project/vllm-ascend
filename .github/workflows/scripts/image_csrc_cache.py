#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Run L1 identity/build operations in the actual compiler image, without host mounts.

Docker cp works with the sibling Docker daemon used by container jobs; a bind
mount of GITHUB_WORKSPACE would refer to a different path on that daemon's host.
No credentials, Docker socket, or NPU devices are passed to these containers.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from types import MappingProxyType

SOURCE = "/vllm-workspace/vllm-ascend"
CACHE = "/tmp/csrc-l1-cache"
IMAGE_BUILD_JOBS = 4
SOCS = MappingProxyType(
    {
        "a2": "ascend910b1",
        "910b": "ascend910b1",
        "a3": "ascend910_9391",
        "310p": "ascend310p1",
        "a5": "ascend950dt_9582",
        "950": "ascend950dt_9582",
    }
)


def docker(*args: str, capture: bool = False) -> str:
    result = subprocess.run(["docker", *args], check=True, text=True, stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


@contextlib.contextmanager
def container(image: str, command: list[str]):
    identity = docker(
        "create",
        "--name",
        f"va-cpu-csrc-l1-{uuid.uuid4().hex[:12]}",
        "--network",
        "host",
        "--entrypoint",
        command[0],
        image,
        *command[1:],
        capture=True,
    )
    try:
        yield identity
    finally:
        docker("rm", "--force", identity)


def start(identity: str, *, capture: bool = False) -> str:
    output = docker("start", "--attach", identity, capture=capture)
    status = docker("wait", identity, capture=True)
    if status != "0":
        raise RuntimeError(f"compiler container exited with status {status}")
    return output


def snapshot_key(image: str, engine: Path, arguments: list[str]) -> str:
    with container(image, ["python3", "/tmp/csrc-build-cache.py", "snapshot-key", *arguments]) as identity:
        docker("cp", str(engine), f"{identity}:/tmp/csrc-build-cache.py")
        return start(identity, capture=True)


def source_archive(source: Path, destination: Path) -> None:
    # Only tracked project files (including initialized submodules), never local
    # credentials, restored L1, or arbitrary files under the workspace.
    files = (
        subprocess.run(
            ["git", "-C", str(source), "ls-files", "--recurse-submodules", "-z"],
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    with tarfile.open(destination, "w") as archive:
        for name in files:
            if name and (source / name).is_file():
                archive.add(source / name, arcname=name, recursive=False)


def build(args: argparse.Namespace) -> None:
    source = Path(args.source_root).resolve()
    cache = Path(args.cache_dir).resolve()
    if cache != source / "csrc" / "build_cache":
        raise ValueError("image L1 transport must use the source workspace's csrc/build_cache directory")
    cache.mkdir(parents=True, exist_ok=True)
    soc = SOCS.get(args.soc_version.lower(), args.soc_version.lower())
    # Match the existing image install's dependency resolution, or the nightly
    # consumer's --no-deps install against its already prepared runtime image.
    install = "python3 -m pip install -e ."
    uninstall = ""
    if args.nightly:
        uninstall = "python3 -m pip uninstall -y vllm-ascend || true"
        # Nightly consumers install uv at job setup; it is not guaranteed to be
        # baked into the image. Keep their dependency filtering and system mode.
        install = """python3 -m pip install uv uc-manager
export UV_SYSTEM_PYTHON=1
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_EXTRA_INDEX_URL="$PIP_EXTRA_INDEX_URL https://triton-ascend.osinfra.cn/pypi/simple"
uv pip install -r requirements-nightly.txt
grep -vE '^(torch|torch-npu|torchvision|torchaudio|triton-ascend|memfabric_hybrid|memcache_hybrid)[=<>!]' \\
  requirements.txt > /tmp/requirements-runtime.txt || true
uv pip install -r /tmp/requirements-runtime.txt
uv pip install -e . --no-deps"""
    command = f"""set -euo pipefail
{uninstall}
rm -rf {SOURCE}
mkdir -p {SOURCE} {CACHE}
tar -xf /tmp/csrc-source.tar -C {SOURCE}
cp -a /tmp/csrc-seed/. {CACHE}/
rm -f {CACHE}/.updated
export SOC_VERSION={soc}
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS={IMAGE_BUILD_JOBS}
export VLLM_BATCH_INVARIANT={0 if args.nightly or soc.startswith("ascend310p") else 1}
export VLLM_ASCEND_BUILD_CACHE_DIR={CACHE}
export VLLM_ASCEND_BUILD_CACHE_EVENT_LOG=/tmp/csrc-l1-events.jsonl
export PIP_EXTRA_INDEX_URL="${{ASCEND_INDEX_URL:-https://repo.huaweicloud.com/ascend/repos/pypi}} ${{PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu/}}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd {SOURCE}
echo '[csrc-l1] source install start'
SECONDS=0
{install}
printf '[csrc-l1] source install complete: %ss\\n' "$SECONDS"
find vllm_ascend -maxdepth 1 -name '*.so' -type f | grep -q .
"""
    with tempfile.TemporaryDirectory(prefix="csrc-image-") as temporary:
        temp = Path(temporary)
        source_archive(source, temp / "source.tar")
        with container(args.image, ["bash", "-c", command]) as identity:
            docker("cp", str(temp / "source.tar"), f"{identity}:/tmp/csrc-source.tar")
            docker("cp", str(cache), f"{identity}:/tmp/csrc-seed")
            start(identity)
            event_log = os.environ.get("VLLM_ASCEND_BUILD_CACHE_EVENT_LOG")
            if event_log:
                try:
                    docker("cp", f"{identity}:/tmp/csrc-l1-events.jsonl", event_log)
                except subprocess.CalledProcessError:
                    print("::notice::Compiler image did not emit inner L1 telemetry.")
            # Replace rather than merge: otherwise entries pruned by build.sh
            # would survive in the old host seed and re-enter the next snapshot.
            updated = temp / "updated-cache"
            docker("cp", f"{identity}:{CACHE}", str(updated))
            shutil.rmtree(cache)
            shutil.copytree(updated, cache, symlinks=True)
            if args.artifacts:
                package = temp / "package"
                docker("cp", f"{identity}:{SOURCE}/vllm_ascend", str(package))
                publish_artifacts(package, source / "vllm_ascend")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with Path(output).open("a", encoding="utf-8") as stream:
            stream.write("built=true\n")


def publish_artifacts(package: Path, destination: Path) -> None:
    libraries = list(package.glob("*.so"))
    if not libraries:
        raise ValueError("compiler image did not produce native shared libraries")
    destination.mkdir(parents=True, exist_ok=True)
    for path in [*libraries, *(package / name for name in ("_cann_ops_custom", "lib", "include"))]:
        if not path.exists():
            continue
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--soc-version", required=True, choices=sorted(SOCS))
    parser.add_argument("--nightly", action="store_true")
    parser.add_argument("--artifacts", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
