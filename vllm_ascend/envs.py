import os
from typing import Any, Callable, Dict

env_variables: Dict[str, Callable[[], Any]] = {
    # max compile thread num
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "COMPILE_CUSTOM_KERNELS":
    lambda: os.getenv("COMPILE_CUSTOM_KERNELS", None),
    # If set, vllm-ascend will print verbose logs during compliation
    "VERBOSE": lambda: bool(int(os.getenv('VERBOSE', '0'))),
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    "LD_LIBRARY_PATH": lambda: os.getenv("LD_LIBRARY_PATH", None),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
