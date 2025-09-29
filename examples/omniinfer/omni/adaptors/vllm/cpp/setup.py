# setup.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import torch
from setuptools import setup, find_packages, Extension


class PathManager:
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

        # my own headers
        self.header = "."

        # csrc
        self.sources = [
            "npu_mem_allocator.cpp"
        ]

        self.check()

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


paths = PathManager()

# 定义扩展模块
ext_modules = [
    Extension(
        "omni_vllm.npu_mem_allocator",
        sources=paths.sources,
        include_dirs=paths.get_include_dirs(),
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
        ] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
        libraries=['torch', 'torch_python', 'ascendcl']
    ),
]


setup(
    name='omni_vllm',  # 包的名称
    version='0.9.0+dynamic',  # 包的版本
    description='Package for optimizing vllm',  # 简短描述
    packages=find_packages(
        exclude=(
            "build"
        )
    ),
    install_requires=[
        'torch',
        'torch_npu',
    ],  # 依赖的其他包
    ext_modules=ext_modules,
    include_package_data=True,
)
