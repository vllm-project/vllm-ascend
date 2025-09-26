# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import importlib
import sys

def replace_module_use_ascend_lmcache(old_module_name: str):
    new_module_name = old_module_name.replace("lmcache", "ascend_lmcache")
    if old_module_name in sys.modules:
        del sys.modules[old_module_name]
    sys.modules[old_module_name] = importlib.import_module(new_module_name)

_default_ops = ("lmcache.c_ops", "lmcache.integration.vllm.vllm_adapter")

for _ops in _default_ops:
    replace_module_use_ascend_lmcache(_ops)