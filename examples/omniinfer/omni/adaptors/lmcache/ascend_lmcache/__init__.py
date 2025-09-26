# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import sys
from types import ModuleType

class MockModule(ModuleType):
    def __getattr__(self, name) -> None:
        return None

sys.modules["vllm.attention.backends.flash_attn"] = MockModule("flash_attn")
sys.modules["vllm.attention.backends.flashmla"] = MockModule("flashmla")
sys.modules["vllm.attention.backends.mla.common"] = MockModule("common")
sys.modules["infinistore"] = MockModule("infinistore")

def register_ascend_lmcache():
    import ascend_lmcache.adapter.adapter