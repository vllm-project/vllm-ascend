# setup.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from setuptools import setup, find_packages, Extension
import pybind11
import torch


class PathManagerBase:
    def __init__(self):
        # torch
        torch_root = os.path.dirname(torch.__file__)
        self.torch_inc = os.path.join(torch_root, "include")
        self.torch_csrc_inc = os.path.join(torch_root, "include/torch/csrc/api/include")
        self.torch_lib = os.path.join(torch_root, "lib")

        # ascend
        ascend_root = os.getenv('ASCEND_TOOLKIT_HOME', None)
        if ascend_root is None:
            raise EnvironmentError("Environment variable 'ASCEND_TOOLKIT_HOME' is not set. Please set this environment variable before running the program.")
        self.ascend_inc = os.path.join(ascend_root, "include")
        self.ascend_lib = os.path.join(ascend_root, "lib64")

    def check(self):
        if not os.path.exists(self.torch_inc):
            raise FileNotFoundError(f"PyTorch include path not found: {self.torch_inc}")
        if not os.path.exists(self.torch_lib):
            raise FileNotFoundError(f"PyTorch lib path not found: {self.torch_lib}")

    def get_include_dirs(self):
        include_dirs = [self.header, self.ascend_inc, self.torch_inc]
        if os.path.exists(self.torch_csrc_inc):
            include_dirs.append(self.torch_csrc_inc)
        return include_dirs

    def get_library_dirs(self):
        return [self.torch_lib, self.ascend_lib]

    def get_extra_link_args(self):
        lib_dirs = self.get_library_dirs()
        link_args = [f"-L{x}" for x in lib_dirs]
        link_args.extend([f"-Wl,-rpath={x}" for x in lib_dirs])
        return link_args


class AllocatorPathManager(PathManagerBase):
    def __init__(self):
        super().__init__()
        self.header = "omni/adaptors/vllm/cpp"
        self.sources = [
            "omni/adaptors/vllm/cpp/npu_mem_allocator.cpp"
        ]
        self.check()

    def check(self):
        super().check()

    def get_include_dirs(self):
        return super().get_include_dirs()

    def get_library_dirs(self):
        return super().get_library_dirs()

    def get_extra_link_args(self):
        return super().get_extra_link_args()

class PlacementPathManager(PathManagerBase):
    def __init__(self):
        super().__init__()

        # pybind11
        self.pybind_inc = pybind11.get_include()

        self.header = "omni/accelerators/placement/omni_placement/cpp/include"
        self.sources = [
            "omni/accelerators/placement/omni_placement/cpp/placement_manager.cpp",
            "omni/accelerators/placement/omni_placement/cpp/placement_mapping.cpp",
            "omni/accelerators/placement/omni_placement/cpp/placement_optimizer.cpp",
            "omni/accelerators/placement/omni_placement/cpp/expert_load_balancer.cpp",
            "omni/accelerators/placement/omni_placement/cpp/dynamic_eplb_greedy.cpp",
            "omni/accelerators/placement/omni_placement/cpp/expert_activation.cpp",
            "omni/accelerators/placement/omni_placement/cpp/tensor.cpp",
            "omni/accelerators/placement/omni_placement/cpp/moe_weights.cpp",
            "omni/accelerators/placement/omni_placement/cpp/distribution.cpp",
            "omni/accelerators/placement/omni_placement/cpp/utils.cpp"
        ]
        self.check()

    def check(self):
        super().check()
        if not os.path.exists(self.header):
            raise FileNotFoundError(f"omni_placement include path not found: {self.header}")

    def get_include_dirs(self):
        return [self.pybind_inc] + super().get_include_dirs()

    def get_library_dirs(self):
        return super().get_library_dirs()

    def get_extra_link_args(self):
        return super().get_extra_link_args()

alloc_paths = AllocatorPathManager()
placement_paths = PlacementPathManager()


# 定义扩展模块
ext_modules = [
    Extension(
        "omni.adaptors.vllm.npu_mem_allocator",
        sources=alloc_paths.sources,
        include_dirs=alloc_paths.get_include_dirs(),
        language='c++',
        extra_compile_args=[
            '-std=c++17',
            '-pthread',
        ],
        extra_link_args=[
            '-pthread',
            '-lascendcl',
            '-ltorch',
            '-ltorch_python',
        ] + alloc_paths.get_extra_link_args(),
        library_dirs=alloc_paths.get_library_dirs(),
        libraries=['torch', 'torch_python', 'ascendcl']
    ),

    Extension(
        "omni.accelerators.placement.omni_placement.omni_placement",
        sources=placement_paths.sources,
        include_dirs=placement_paths.get_include_dirs(),
        library_dirs=placement_paths.get_library_dirs(),
        libraries=['hccl', 'torch', 'torch_python', 'ascendcl'],
        extra_compile_args=[
            '-std=c++17',
            '-pthread',
        ],
        extra_link_args=[
            '-pthread',
            '-lascendcl',
            '-ltorch',
            '-ltorch_python',
        ] + placement_paths.get_extra_link_args(),
        language='c++',
    ),
]


setup(
    name='omni_infer',
    version='0.1.0',
    description='Omni Infer',
    packages=find_packages(
        exclude=(
            "build"
        )
    ),
    install_requires=[
        'torch',
        'torch_npu',
        'pybind11',
    ],
    ext_modules=ext_modules,
)
