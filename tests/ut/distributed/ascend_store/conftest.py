# SPDX-License-Identifier: Apache-2.0
"""Initialize the Core registry normally populated during engine startup."""

from unittest.mock import MagicMock

import pytest
from vllm.v1.core import single_type_kv_cache_manager


@pytest.fixture(autouse=True)
def register_core_cache_specs():
    # Older Core versions register built-in specs at import time. Newer ones
    # defer registration to engine startup, which connector unit tests bypass.
    register = getattr(single_type_kv_cache_manager, "register_all_kvcache_specs", None)
    if register is not None:
        register(MagicMock())
