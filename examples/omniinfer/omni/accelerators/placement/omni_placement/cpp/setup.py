# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys
import sysconfig
import os

# 检查是否为调试模式
debug_mode = '--debug' in sys.argv

os.environ['ASCEND_TOOLKIT_PATH'] =  os.getenv('ASCEND_TOOLKIT_HOME', None)

# 动态获取 Python 相关路径
python_include_dir = sysconfig.get_path("include")  # Python 头文件路径
python_lib_dir = sysconfig.get_path("stdlib")       # Python 标准库路径

# 动态 include 和 library 目录
include_dirs = [
    pybind11.get_include(),  # pybind11 头文件
    python_include_dir,      # 当前 Python 的头文件
    "./include",             # 项目本地 include
    os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") + "/include",
]

library_dirs = [
    os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") + "/lib64",
    "/usr/lib64",
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
                '-fvisibility=default',
                '-std=c++17',
                '-pthread',
                '-fPIC',
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-g' if debug_mode else '-g0'
            ]
        super().build_extensions()

# 定义扩展模块
ext_modules = [
    Pybind11Extension(
        "omni_placement",
        sources=["placement_manager.cpp",
                 "expert_activation.cpp",
                 "moe_weights.cpp",
                 "tensor.cpp",
                 "placement_mapping.cpp",
                 "placement_optimizer.cpp",
                 "expert_load_balancer.cpp"],
        include_dirs=include_dirs,
        language='c++',
        extra_link_args=[
            '-pthread',
            f'-L{library_dirs[0]}',  # Ascend 的动态路径
            '-lascendcl',
            f'-lpython3', 
            '-L/lib64',
            '-lgfortran',
            f'-Wl,-rpath={library_dirs[0]}',  # 动态 rpath
        ],
        library_dirs=library_dirs,
        libraries=['hccl','ascendcl', 'python3', 'gfortran']
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