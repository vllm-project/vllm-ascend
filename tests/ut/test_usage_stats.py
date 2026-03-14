import sys
import types
import unittest

from vllm_ascend.usage_stats import maybe_report_v1_usage_stats


class TestMaybeReportV1UsageStats(unittest.TestCase):

    def setUp(self):
        self._saved_modules = dict(sys.modules)

    def tearDown(self):
        sys.modules.clear()
        sys.modules.update(self._saved_modules)

    def test_noop_when_vllm_v1_utils_missing(self):
        sys.modules.pop("vllm", None)
        sys.modules.pop("vllm.v1", None)
        sys.modules.pop("vllm.v1.utils", None)
        maybe_report_v1_usage_stats(object())

    def test_calls_report_usage_stats_when_available(self):
        calls = []

        vllm_mod = types.ModuleType("vllm")
        v1_mod = types.ModuleType("vllm.v1")
        utils_mod = types.ModuleType("vllm.v1.utils")

        def report_usage_stats(vllm_config):
            calls.append(vllm_config)

        utils_mod.report_usage_stats = report_usage_stats

        sys.modules["vllm"] = vllm_mod
        sys.modules["vllm.v1"] = v1_mod
        sys.modules["vllm.v1.utils"] = utils_mod

        cfg = object()
        maybe_report_v1_usage_stats(cfg)
        self.assertEqual(calls, [cfg])

    def test_swallows_report_usage_stats_errors(self):
        vllm_mod = types.ModuleType("vllm")
        v1_mod = types.ModuleType("vllm.v1")
        utils_mod = types.ModuleType("vllm.v1.utils")

        def report_usage_stats(_):
            raise RuntimeError("boom")

        utils_mod.report_usage_stats = report_usage_stats

        sys.modules["vllm"] = vllm_mod
        sys.modules["vllm.v1"] = v1_mod
        sys.modules["vllm.v1.utils"] = utils_mod

        maybe_report_v1_usage_stats(object())

