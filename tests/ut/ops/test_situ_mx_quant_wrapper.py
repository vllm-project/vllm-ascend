# SPDX-License-Identifier: Apache-2.0
"""Test lazy CANN operator lookup without requiring an NPU."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class TestSituMxQuantWrapper(unittest.TestCase):
    def test_reuses_cann_dispatcher_after_first_call(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/situ_mx_quant.py"
        spec = importlib.util.spec_from_file_location("situ_mx_quant_test_wrapper", path)
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)
        lookups = []
        module = types.ModuleType("cann_ops_nn")

        def resolve(name):
            if name != "situ_mx_quant":
                raise AttributeError(name)
            lookups.append(name)
            return lambda x, **kwargs: (x, kwargs)

        module.__getattr__ = resolve
        kwargs = dict(beta=4.0, linear_beta=25.0, activate_left=True, dst_type=36)
        with patch.dict(sys.modules, {"cann_ops_nn": module}):
            first = wrapper.situ_mx_quant("x", **kwargs)
            initial_count = len(lookups)
            second = wrapper.situ_mx_quant("x", **kwargs)
        self.assertEqual(first, second)
        self.assertGreater(initial_count, 0)
        self.assertEqual(len(lookups), initial_count)
