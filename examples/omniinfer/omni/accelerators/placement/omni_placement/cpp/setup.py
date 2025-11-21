# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys
import sysconfig

import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# 检查是否为调试模式
debug_mode = '--debug' in sys.argv

os.environ['ASCEND_TOOLKIT_PATH'] = os.getenv('ASCEND_TOOLKIT_HOME', None)

# 动态获取 Python 相关路径
python_include_dir = sysconfig.get_path("include")  # Python 头文件路径
python_lib_dir = sysconfig.get_path("stdlib")  # Python 标准库路径
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"  # 例如 "python3.11"

# 动态 include 和 library 目录
include_dirs = [
    pybind11.get_include(),  # pybind11 头文件
    python_include_dir,  # 当前 Python 的头文件
    "./include",  # 项目本地 include
    # 如果 Ascend 或 Ray 的路径可能变化，可以通过环境变量动态指定
    os.environ.get("ASCEND_TOOLKIT_PATH",
                   "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") +
    "/include",
    os.environ.get(
        "RAY_CPP_PATH", "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/" +
        python_version + "/site-packages/ray/cpp") + "/include",
]

library_dirs = [
    os.environ.get("ASCEND_TOOLKIT_PATH",
                   "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") +
    "/lib64",
    os.environ.get(
        "RAY_CPP_PATH", "/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/" +
        python_version + "/site-packages/ray/cpp") + "/lib",
    "/lib64",  # 系统库路径
]


# 自定义编译类，移除默认优化标志
class OmniBuildExt(build_ext):

    def build_extensions(self):
        if debug_mode:
            new_compiler_cmd = []
            for item in self.compiler.compiler_so:
                if item.strip() == '-DNDEBUG':
                    continue
                if item.startswith('-O'):
                    continue
                if item.startswith('-g0'):
                    continue
                new_compiler_cmd.append(item)
            self.compiler.compiler_so = new_compiler_cmd

        for ext in self.extensions:
            ext.extra_compile_args = [
                arg for arg in ext.extra_compile_args
                if arg not in ('-fvisibility=hidden', '-g0')
            ] + [
                '-fvisibility=default', '-std=c++17', '-pthread', '-fPIC',
                '-D_GLIBCXX_USE_CXX11_ABI=0', '-g' if debug_mode else '-g0'
            ]
        super().build_extensions()


# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "omni_placement",
        sources=[
            "placement_manager.cpp", "expert_activation.cpp",
            "moe_weights.cpp", "tensor.cpp", "placement_mapping.cpp",
            "placement_optimizer.cpp", "placement_optimizer_for_swap.cpp",
            "expert_load_balancer.cpp"
        ],
        include_dirs=include_dirs,
        language='c++',
        extra_link_args=[
            '-pthread',
            f'-L{library_dirs[1]}',  # Ray 的动态路径
            f'-L{library_dirs[0]}',  # Ascend 的动态路径
            '-lascendcl',
            f'-l{python_version}',  # 动态 Python 库名
            '-L/lib64',
            '-lgfortran',
            f'-Wl,-rpath={library_dirs[0]}',  # 动态 rpath
        ],
        library_dirs=library_dirs,
        libraries=['hccl', 'ascendcl', python_version,
                   'gfortran']  # 动态指定 Python 库
    ),
]

# Setup 配置
setup(
    name="omni_placement",
    version="0.1",
    description="MoE weights management with shared memory",
    ext_modules=ext_modules,
    cmdclass={"build_ext": OmniBuildExt},
    install_requires=["pybind11"],
)
