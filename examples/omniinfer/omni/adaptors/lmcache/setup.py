from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys
import sysconfig
import os
from torch.utils.cpp_extension import include_paths, BuildExtension, library_paths

# 检查是否为调试模式
debug_mode = '--debug' in sys.argv

os.environ['ASCEND_TOOLKIT_PATH'] = os.getenv('ASCEND_TOOLKIT_HOME', None)

# 动态获取 Python 相关路径
python_include_dir = sysconfig.get_path("include")  # Python 头文件路径
python_lib_dir = sysconfig.get_path("stdlib")       # Python 标准库路径
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"  # 例如 "python3.11"

# 获取 Pytorch 头文件路径
torch_include_dirs = include_paths()
torch_lib_dirs = library_paths()

# 动态 include 和 library 目录
include_dirs = [
    pybind11.get_include(),  # pybind11 头文件
    os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") + "/include",
]
include_dirs.extend(torch_include_dirs)

library_dirs = [
    os.environ.get("ASCEND_TOOLKIT_PATH", "/usr/local/Ascend/ascend-toolkit/latest/arm64-linux") + "/lib64",
    "/lib64",  # 系统库路径
]
library_dirs.extend(torch_lib_dirs)

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
                '-fPIC',
                '-D_GLIBCXX_USE_CXX11_ABI=0',
                '-g' if debug_mode else '-g0'
            ]
        super().build_extensions()

# 定义拓展模块
ext_modules = [
    Pybind11Extension(
        "ascend_lmcache.c_ops",
        sources=["csrc/npu_pybind.cpp"],
        include_dirs=include_dirs,
        language='c++',
        extra_link_args=[
            f'-L{library_dirs[0]}',  # Ascend 的动态路径
            '-lascendcl',
            '-L/lib64',
            '-lc10',
            '-ltorch',
            '-ltorch_python',
            '-ltorch_cpu',
            f'-L{library_dirs[2]}',
            f'-Wl,-rpath={library_dirs[0]},-rpath={library_dirs[2]}',  # 动态rpath
        ],
        library_dirs=library_dirs,
        libraries=['ascendcl', "c10"]
    ),
]

# Setup 配置
setup(
    name="ascend_lmcache",
    version="0.3.2",
    description=("lmcache on NPU"),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("csrc", "tests")),
    python_requires=">=3.10,<3.13",
    install_requires=["pybind11"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": OmniBuildExt},
    include_package_data=True,
    entry_points={
        "vllm.general_plugins": ["ascend_lmcache = ascend_lmcache:register_ascend_lmcache"],
    }
)