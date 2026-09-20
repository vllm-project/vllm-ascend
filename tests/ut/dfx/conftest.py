# SPDX-License-Identifier: Apache-2.0
"""Skip only package logging bootstrap when running the pure host suite alone."""

import importlib
import importlib.util
import sys
from types import ModuleType
from unittest.mock import patch

if importlib.util.find_spec("vllm") is None:
    # Package __init__ imports the vLLM logging adapter. The recorder/config
    # themselves use only numpy, pydantic and stdlib; no DFX code is mocked.
    with patch.dict(sys.modules, {"vllm_ascend.logger": ModuleType("vllm_ascend.logger")}):
        package = importlib.import_module("vllm_ascend")
    sys.modules["vllm_ascend"] = package
