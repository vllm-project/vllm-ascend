#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/setup.py
#

import importlib.util
import logging
import os
import subprocess
import sys
from sysconfig import get_paths
from typing import Dict, List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools_scm import get_version


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def check_or_set_default_env(cmake_args,
                             env_name,
                             env_variable,
                             default_path=""):
    if env_variable is None:
        logging.warning(
            f"No {env_name} found in your environment, pleause try to set {env_name} "
            "if you customize the installation path of this library, otherwise default "
            "path will be adapted during build this project")
        logging.warning(f"Set default {env_name}: {default_path}")
        env_variable = default_path
    else:
        logging.info(f"Found existing {env_name}: {env_variable}")
    # cann package seems will check this environments in cmake, need write this env variable back.
    if env_name == "ASCEND_HOME_PATH":
        os.environ["ASCEND_HOME_PATH"] = env_variable
    cmake_args += [f"-D{env_name}={env_variable}"]
    return cmake_args


envs = load_module_from_path("envs",
                             os.path.join(ROOT_DIR, "vllm_ascend", "envs.py"))


class CMakeExtension(Extension):

    def __init__(self,
                 name: str,
                 cmake_lists_dir: str = ".",
                 **kwargs) -> None:
        super().__init__(name, sources=[], py_limited_api=False, **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class custom_build_info(build_py):

    def run(self):
        soc_version = envs.SOC_VERSION
        if not soc_version:
            raise ValueError(
                "SOC version is not set. Please set SOC_VERSION environment variable."
            )
        if "310" in soc_version and not envs.COMPILE_CUSTOM_KERNELS:
            raise ValueError(
                "SOC version 310 only supports custom kernels. Please set COMPILE_CUSTOM_KERNELS=1 to enable custom kernels."
            )

        package_dir = os.path.join(ROOT_DIR, "vllm_ascend", "_build_info.py")
        with open(package_dir, "w+") as f:
            f.write('# Auto-generated file\n')
            f.write(f"__soc_version__ = '{soc_version}'\n")
            f.write(
                f"__sleep_mode_enabled__ = {envs.COMPILE_CUSTOM_KERNELS}\n")
        logging.info(
            f"Generated _build_info.py with SOC version: {soc_version}")
        super().run()


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: Dict[str, bool] = {}

    #
    # Determine number of compilation jobs
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()
        num_jobs = max(1, num_jobs)

        return num_jobs

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        source_dir = os.path.abspath(ROOT_DIR)
        python_executable = sys.executable
        cmake_args = ["cmake"]
        # Default use release mode to compile the csrc code
        # Turbo now support compiled with Release, Debug and RelWithDebugInfo
        if envs.CMAKE_BUILD_TYPE is None or envs.CMAKE_BUILD_TYPE not in [
                "Debug",
                "Release",
                "RelWithDebugInfo",
        ]:
            envs.CMAKE_BUILD_TYPE = "Release"
        cmake_args += [f"-DCMAKE_BUILD_TYPE={envs.CMAKE_BUILD_TYPE}"]
        # Default dump the compile commands for lsp
        cmake_args += ["-DCMAKE_EXPORT_COMPILE_COMMANDS=1"]
        if envs.CXX_COMPILER is not None:
            cmake_args += [f"-DCMAKE_CXX_COMPILER={envs.CXX_COMPILER}"]
        if envs.C_COMPILER is not None:
            cmake_args += [f"-DCMAKE_C_COMPILER={envs.C_COMPILER}"]
        if envs.VERBOSE:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        # find ASCEND_HOME_PATH
        check_or_set_default_env(
            cmake_args,
            "ASCEND_HOME_PATH",
            envs.ASCEND_HOME_PATH,
            "/usr/local/Ascend/ascend-toolkit/latest",
        )

        # find PYTHON_EXECUTABLE
        check_or_set_default_env(cmake_args, "PYTHON_EXECUTABLE",
                                 sys.executable)

        # find PYTHON_INCLUDE_PATH
        check_or_set_default_env(cmake_args, "PYTHON_INCLUDE_PATH",
                                 get_paths()["include"])

        # ccache and ninja can not be applied at ascendc kernels now

        try:
            # if pybind11 is installed via pip
            pybind11_cmake_path = (subprocess.check_output(
                [python_executable, "-m", "pybind11",
                 "--cmakedir"]).decode().strip())
        except subprocess.CalledProcessError as e:
            # else specify pybind11 path installed from source code on CI container
            raise RuntimeError(f"CMake configuration failed: {e}")

        install_path = os.path.join(ROOT_DIR, self.build_lib)
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = os.path.join(ROOT_DIR, "vllm_ascend")
        # add CMAKE_INSTALL_PATH
        cmake_args += [f"-DCMAKE_INSTALL_PREFIX={install_path}"]

        cmake_args += [f"-DCMAKE_PREFIX_PATH={pybind11_cmake_path}"]

        cmake_args += [f"-DSOC_VERSION={envs.SOC_VERSION}"]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        torch_npu_command = "python3 -m pip show torch-npu | grep '^Location:' | awk '{print $2}'"
        try:
            torch_npu_path = subprocess.check_output(
                torch_npu_command, shell=True).decode().strip()
            torch_npu_path += "/torch_npu"
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Retrieve torch version version failed: {e}")

        # add TORCH_NPU_PATH
        cmake_args += [f"-DTORCH_NPU_PATH={torch_npu_path}"]

        build_tool = []
        # TODO(ganyi): ninja and ccache support for ascend c auto codegen. now we can only use make build
        # if which('ninja') is not None:
        #     build_tool += ['-G', 'Ninja']
        # Default build tool to whatever cmake picks.

        cmake_args += [source_dir]
        logging.info(f"cmake config command: {cmake_args}")
        try:
            subprocess.check_call(cmake_args, cwd=self.build_temp)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"CMake configuration failed: {e}")

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        if not envs.COMPILE_CUSTOM_KERNELS:
            return
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError(f"Cannot find CMake executable: {e}")

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        os.makedirs(os.path.join(self.build_lib, "vllm_ascend"), exist_ok=True)

        def target_name(s: str) -> str:
            return s.removeprefix("vllm_ascend.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]
        try:
            subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
        except OSError as e:
            raise RuntimeError(f"Build library failed: {e}")
        # Install the libraries
        install_args = [
            "cmake",
            "--install",
            ".",
        ]
        try:
            subprocess.check_call(install_args, cwd=self.build_temp)
        except OSError as e:
            raise RuntimeError(f"Install library failed: {e}")

        # copy back to build folder for editable build
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            import shutil
            for root, _, files in os.walk(self.build_temp):
                for file in files:
                    if file.endswith(".so"):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(self.build_lib, "vllm_ascend",
                                                file)
                        shutil.copy(src_path, dst_path)
                        print(f"Copy: {src_path} -> {dst_path}")

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


class custom_install(install):

    def run(self):
        self.run_command("build_ext")
        install.run(self)


ROOT_DIR = os.path.dirname(__file__)
try:
    VERSION = get_version(write_to="vllm_ascend/_version.py")
except LookupError:
    # The checkout action in github action CI does not checkout the tag. It
    # only checks out the commit. In this case, we set a dummy version.
    VERSION = "0.0.0"

ext_modules = []
if envs.COMPILE_CUSTOM_KERNELS:
    ext_modules = [CMakeExtension(name="vllm_ascend.vllm_ascend_C")]


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    try:
        requirements = _read_requirements("requirements.txt")
    except ValueError:
        print("Failed to read requirements.txt in vllm_ascend.")
    return requirements


cmdclass = {
    "build_py": custom_build_info,
    "build_ext": cmake_build_ext,
    "install": custom_install
}

setup(
    name="vllm_ascend",
    # Follow:
    # https://packaging.python.org/en/latest/specifications/version-specifiers
    version=VERSION,
    author="vLLM-Ascend team",
    license="Apache 2.0",
    description=("vLLM Ascend backend plugin"),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm-ascend",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm-ascend",
    },
    # TODO: Add 3.12 back when torch-npu support 3.12
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require={},
    entry_points={
        "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
        "vllm.general_plugins":
        ["ascend_enhanced_model = vllm_ascend:register_model"],
    },
)
