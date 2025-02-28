import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

env_variables: Dict[str, Callable[[], Any]] = {
    "MAX_JOBS":
    lambda: os.getenv("MAX_JOBS", None),

    "CMAKE_BUILD_TYPE":
    lambda: os.getenv("CMAKE_BUILD_TYPE"),

    # If set, vllm will print verbose logs during installation
    "VERBOSE":
    lambda: bool(int(os.getenv('VERBOSE', '0'))),

    "ASCEND_HOME_PATH":
    lambda: os.environ.get("ASCEND_HOME_PATH", None),

    "LD_LIBRARY_PATH":
    lambda: os.environ.get("LD_LIBRARY_PATH", None),

}
