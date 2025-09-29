# setup.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from setuptools import setup, Extension
import pybind11
import torch


class PathManager:
    def __init__(self):
        # torch
        torch_root = os.path.dirname(torch.__file__)
        self.torch_inc = os.path.join(torch_root, "include")
        self.torch_csrc_inc = os.path.join(torch_root, "include/torch/csrc/api/include")
        self.torch_lib = os.path.join(torch_root, "lib")

        # pybind11
        self.pybind_inc = pybind11.get_include()

        # ascend
        ascend_root = os.getenv('ASCEND_TOOLKIT_HOME', None)
        if ascend_root is None:
            raise EnvironmentError("Environment variable 'ASCEND_TOOLKIT_HOME' is not set. Please set this environment variable before running the program.")
        self.ascend_inc = os.path.join(ascend_root, "include")
        self.ascend_lib = os.path.join(ascend_root, "lib64")

        # omni_placement headers
        self.header = "omni/accelerators/placement/omni_placement/cpp/include"

        # csrc
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
        if not os.path.exists(self.torch_inc):
            raise FileNotFoundError(f"PyTorch include path not found: {self.torch_inc}")
        if not os.path.exists(self.torch_lib):
            raise FileNotFoundError(f"PyTorch lib path not found: {self.torch_lib}")
        if not os.path.exists(self.header):
            raise FileNotFoundError(f"omni_placement include path not found: {self.header}")

    def get_include_dirs(self):
        include_dirs = [self.header, self.pybind_inc, self.ascend_inc, self.torch_inc]
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


paths = PathManager()


# 定义扩展模块
ext_modules = [
    Extension(
        "omni.accelerators.placement.omni_placement.omni_placement",
        sources=paths.sources,
        include_dirs=paths.get_include_dirs(),
        library_dirs=paths.get_library_dirs(),
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
        ] + paths.get_extra_link_args(),
        language='c++',
    ),
]


setup(
    name='omni_infer',
    version='0.1.0',
    description='Omni Infer',
    ext_modules=ext_modules,
)
