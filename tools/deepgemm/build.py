# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Build the native Ascend extension, Python modules and JIT headers for CMake."""

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def native_build_env():
    """Select a C++20 toolchain for this extension, not the serving process."""
    env = os.environ.copy()
    toolset = Path("/opt/openEuler/gcc-toolset-14/root/usr/bin")
    cann = Path(env.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"))
    sdk_compiler = cann / "tools/hcc/bin/aarch64-target-linux-gnu-g++"
    if not env.get("CXX"):
        if (toolset / "g++").is_file():
            env["CXX"] = str(toolset / "g++")
        elif compiler := shutil.which("g++-14") or shutil.which("g++-13"):
            env["CXX"] = compiler
        elif platform.machine() == "aarch64" and sdk_compiler.is_file():
            env["CXX"] = str(sdk_compiler)
        else:
            # The manylinux toolset exposes its native GCC as plain g++.
            env["CXX"] = "g++"
    if Path(env["CXX"]).resolve() == sdk_compiler.resolve():
        # HCC is a cross toolchain: add host elfutils after its standard
        # headers and hide its statically linked libstdc++ from Torch.
        multiarch = sysconfig.get_config_var("MULTIARCH")
        multiarch_lib = Path("/usr/lib") / (multiarch or "")
        host_lib = multiarch_lib if (multiarch_lib / "libdw.so").is_file() else Path("/usr/lib64")
        for key, flags in (
            ("CFLAGS", "-idirafter /usr/include"),
            ("LDFLAGS", f"-L{host_lib} -static-libstdc++ -Wl,--exclude-libs,libstdc++.a"),
        ):
            env[key] = f"{env.get(key, '')} {flags}".strip()
    for key in tuple(env):
        if key.startswith("DG_JIT_"):
            del env[key]
    return env


def build(source, build_dir):
    subprocess.run(
        [sys.executable, "setup.py", "build", "--build-base", str(build_dir), "--build-lib", str(build_dir / "python")],
        cwd=source,
        env=native_build_env(),
        check=True,
    )
    package = build_dir / "python/deep_gemm"
    shutil.copy2(source / "LICENSE", package / "LICENSE")
    licenses = package / "licenses"
    licenses.mkdir(exist_ok=True)
    for name in ("deep-jit", "magic_enum"):
        for license_file in (source / "third-party" / name).glob("LICENSE*"):
            shutil.copy2(license_file, licenses / f"{name}-{license_file.name}")


if __name__ == "__main__":
    build(*(Path(arg).resolve() for arg in sys.argv[1:]))
