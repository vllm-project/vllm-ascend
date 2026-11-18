import pybind11
from setuptools import Extension, setup

setup(
    name="token-repeat-pattern",
    version="0.1.0",
    description="Pattern-based CPU token-repeat detector for runtime_guard",
    classifiers=["Private :: Do Not Upload"],
    python_requires=">=3.10",
    install_requires=["numpy>=1.24", "tokenizers"],
    py_modules=["pattern_cpp", "pattern_config", "pattern_vocab"],
    ext_modules=[
        Extension(
            "_pattern_native",
            ["pattern_native.cpp"],
            depends=["position_queue.h"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17", "-fno-fast-math"],
        )
    ],
)
