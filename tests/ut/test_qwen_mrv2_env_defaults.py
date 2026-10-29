# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the public defaults and independent opt-outs without NPU imports."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FLAGS = (
    "VLLM_ASCEND_QWEN4_EXP_FP32_STABLE_PARTITION",
    "VLLM_ASCEND_QWEN_MRV2_EARLY_QUEUE",
    "VLLM_ASCEND_QWEN_MRV2_MM_DECODE_EARLY_QUEUE",
)


class TestQwenMRV2EnvDefaults(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_test_qwen_mrv2_envs", ROOT / "vllm_ascend/envs.py")
        self.envs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.envs)
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for flag in FLAGS:
            os.environ.pop(flag, None)

    def values(self):
        return tuple(getattr(self.envs, flag) for flag in FLAGS)

    def test_unset_flags_default_to_enabled(self):
        self.assertEqual(self.values(), (True, True, True))

    def test_explicit_zero_disables_each_flag(self):
        os.environ.update(dict.fromkeys(FLAGS, "0"))
        self.assertEqual(self.values(), (False, False, False))

    def test_explicit_one_enables_each_flag(self):
        os.environ.update(dict.fromkeys(FLAGS, "1"))
        self.assertEqual(self.values(), (True, True, True))

    def test_opt_outs_are_independent_and_read_lazily(self):
        for disabled in FLAGS:
            with self.subTest(disabled=disabled):
                os.environ[disabled] = "0"
                self.assertEqual(self.values(), tuple(flag != disabled for flag in FLAGS))
                os.environ.pop(disabled)
                self.assertEqual(self.values(), (True, True, True))


if __name__ == "__main__":
    unittest.main()
