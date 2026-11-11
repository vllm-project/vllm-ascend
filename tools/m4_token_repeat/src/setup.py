import pybind11
from setuptools import Extension, setup

setup(
    name="m4-cpp-detector",
    version="0.1.0",
    description="M4 CPU repetition detector",
    classifiers=["Private :: Do Not Upload"],
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "tokenizers"],
    py_modules=["m4_cpp", "m4_config", "m4_vocab"],
    ext_modules=[
        Extension(
            "_m4_native",
            ["m4_native.cpp"],
            depends=["position_queue.h"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math"],
        )
    ],
)
