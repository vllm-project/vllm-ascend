# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Build the pinned NPU operator with its own setup.py, then vendor its wheel.

This is a build-time tool, not a serving entrypoint. Source overrides must be
clean checkouts of the pinned revision (including recursive dependencies).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REVISION = "3a35326271b069b8b5bf1a754435acd6cc91575b"
REPOSITORY = "https://github.com/deepseek-ai/DeepGEMM-Ascend.git"
SUBMODULES = ("third-party/deep-jit", "third-party/magic_enum")


def run(*args, cwd=None, env=None):
    subprocess.run(args, cwd=cwd, env=env, check=True)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def prepare_source(source):
    if not source.exists():
        source.mkdir(parents=True)
        run("git", "init", str(source))
        run("git", "-C", str(source), "fetch", "--depth=1", REPOSITORY, REVISION)
        run("git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD")
    if git(source, "rev-parse", "HEAD") != REVISION:
        raise RuntimeError(f"DeepGEMM-Ascend source must be at {REVISION}")
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("DeepGEMM-Ascend source has tracked modifications")
    run("git", "-C", str(source), "submodule", "update", "--init", "--recursive", *SUBMODULES)
    for line in git(source, "submodule", "status", "--recursive", *SUBMODULES).splitlines():
        if line.startswith(("+", "-", "U")):
            raise RuntimeError(f"Unpinned dependency: {line}")


def vendor_wheel(wheel, destination):
    """Install only the package payload; never install its top-level dist-info."""
    with zipfile.ZipFile(wheel) as archive:
        members = [name for name in archive.namelist() if name.startswith("deep_gemm/") and not name.endswith("/")]
        if not any(Path(name).name.startswith("_C.") and name.endswith(".so") for name in members):
            raise RuntimeError("DeepGEMM wheel does not contain its native NPU extension")
        if not any(name.startswith("deep_gemm/include/deep_gemm/") for name in members):
            raise RuntimeError("DeepGEMM wheel does not contain device JIT headers")
        for name in members:
            relative = Path(name).relative_to("deep_gemm")
            if ".." in relative.parts:
                raise RuntimeError(f"Invalid wheel member: {name}")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    # Upstream captures DG_JIT_* values during wheel construction. Never ship
    # the build machine's cache/debug settings to a serving environment.
    (destination / "envs.py").write_text("persistent_envs = {}\n", encoding="utf-8")


def fix_device_includes(destination):
    """Two explicit include fixes for the pinned CANN 9.2 device headers.

    common.hpp uses type traits, not utility. Avoid pulling in constrained
    std::pair overloads unsupported by this Bisheng frontend. Communication
    diagnostics must include their declarations instead of relying on transitive
    includes. No standard-library shadow headers or feature-macro overrides.
    Remove these replacements when the pinned upstream incorporates the fixes.
    """
    replacements = (
        ("include/deep_gemm/common.hpp", "#include <utility>", "#include <type_traits>"),
        (
            "include/deep_gemm/comm/mega_moe.hpp",
            "#include <simt_api/device_functions.h>",
            "#include <simt_api/device_functions.h>\n"
            "#include <utils/debug/asc_printf.h>\n#include <utils/debug/asc_assert.h>",
        ),
    )
    hashes = {}
    for name, old, new in replacements:
        path = destination / name
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise RuntimeError(f"Pinned device include fix no longer applies to {name}")
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def build(source, work_dir, destination):
    prepare_source(source)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Select the image's supported host compiler only for the operator build.
    # Do not change the serving process or other framework compiler settings.
    env.setdefault("CXX", shutil.which("g++-13") or "g++")
    env.setdefault("CC", shutil.which("gcc-13") or "gcc")
    for key in tuple(env):
        if key.startswith("DG_JIT_"):
            del env[key]
    # DeepInfer builds operators through their native setup.py. The selected
    # compiler and Torch ABI are owned by that build, not CUDA CMake rules.
    with tempfile.TemporaryDirectory(prefix="wheel-", dir=work_dir) as temporary:
        wheel_dir = Path(temporary)
        run(
            sys.executable,
            "setup.py",
            "build",
            "--build-base",
            str(wheel_dir / "build"),
            "bdist_wheel",
            "--dist-dir",
            str(wheel_dir),
            cwd=source,
            env=env,
        )
        wheels = list(wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one DeepGEMM wheel, found {len(wheels)}")
        vendor_wheel(wheels[0], destination)
    device_header_hashes = fix_device_includes(destination)
    shutil.copyfile(source / "LICENSE", destination / "LICENSE")
    licenses = destination / "licenses"
    licenses.mkdir(exist_ok=True)
    for submodule in SUBMODULES:
        for license_file in (source / submodule).glob("LICENSE*"):
            if license_file.is_file():
                shutil.copyfile(license_file, licenses / f"{Path(submodule).name}-{license_file.name}")
    import torch
    import torch_npu

    manifest = {
        "platform": "ascend",
        "repository": REPOSITORY,
        "revision": REVISION,
        "submodules": git(source, "submodule", "status", "--recursive", *SUBMODULES).splitlines(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "python": sys.version.split()[0],
        "device_header_fixes": device_header_hashes,
    }
    (destination / "ascend_build.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    build(args.source.resolve(), args.work_dir.resolve(), args.destination.resolve())
