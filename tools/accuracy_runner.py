# SPDX-License-Identifier: Apache-2.0
"""Optional accuracy runner for trusted repository YAML benchmark configurations."""

import importlib
import math


def run_custom_accuracy(model, port, host_ip, config):
    if config.get("case_type") != "accuracy":
        raise ValueError("Custom accuracy runners require case_type=accuracy")
    module, separator, name = config["runner"].rpartition(".")
    if not separator or not module or not name:
        raise ValueError("runner must be a dotted callable path")
    runner = getattr(importlib.import_module(module), name)
    result = runner(model=model, port=port, host_ip=host_ip, config=config)
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError("Accuracy runner must return a finite numeric percentage")
    if not 0 <= result <= 100:
        raise ValueError("Accuracy must be in [0,100]")
    return result
