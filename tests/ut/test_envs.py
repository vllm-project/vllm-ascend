#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import inspect
import os

import vllm_ascend.envs as envs_ascend
from tests.ut.base import TestBase


class TestEnvVariables(TestBase):
    def setUp(self):
        self.env_vars = list(envs_ascend.env_variables.keys())

    def test_env_vars_behavior(self):
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                original_val = os.environ.get(var_name)
                var_handler = envs_ascend.env_variables[var_name]

                try:
                    if var_name in os.environ:
                        del os.environ[var_name]
                    self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                    handler_source = inspect.getsource(var_handler)
                    if var_name == "VLLM_ASCEND_KVPOOL_RANGE_DEBUG":
                        test_vals = ["0", "1"]
                    elif "int(" in handler_source:
                        test_vals = ["123", "456"]
                    elif "bool(int(" in handler_source:
                        test_vals = ["0", "1"]
                    else:
                        test_vals = [f"test_{var_name}", f"custom_{var_name}"]

                    for test_val in test_vals:
                        os.environ[var_name] = test_val
                        self.assertEqual(getattr(envs_ascend, var_name), var_handler())

                finally:
                    if original_val is None:
                        os.environ.pop(var_name, None)
                    else:
                        os.environ[var_name] = original_val

    def test_dir_and_getattr(self):
        self.assertEqual(sorted(envs_ascend.__dir__()), sorted(self.env_vars))
        for var_name in self.env_vars:
            with self.subTest(var=var_name):
                getattr(envs_ascend, var_name)

    def test_debug_env_vars_are_strict_and_disabled_by_default(self):
        for name in ("VLLM_ASCEND_KVPOOL_RANGE_DEBUG",):
            original_val = os.environ.pop(name, None)
            try:
                self.assertFalse(getattr(envs_ascend, name))
                for value, expected in (("0", False), ("1", True)):
                    with self.subTest(name=name, value=value):
                        os.environ[name] = value
                        self.assertIs(getattr(envs_ascend, name), expected)
                for value in ("", "2", "true", "-1"):
                    with self.subTest(name=name, invalid=value):
                        os.environ[name] = value
                        with self.assertRaises(ValueError):
                            getattr(envs_ascend, name)
            finally:
                if original_val is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original_val

    def test_env_accessors_match_previous_inline_semantics(self):
        # These accessors replace expressions that were previously inlined at
        # the call sites (ascend_config.py / platform.py / patch/platform /
        # pool_worker.py / msa_m3_triton_a5.py). Each case pins the exact
        # boundary behavior so the convergence stays behavior-preserving.
        cases = {
            # name: (value_when_unset, [(raw_value, expected), ...])
            "DYNAMIC_EPLB": (
                False,
                [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False), ("", False)],
            ),
            "EXPERT_MAP_RECORD": (
                "false",
                [("true", "true"), ("TRUE", "TRUE"), ("1", "1"), ("false", "false")],
            ),
            "VLLM_ASCEND_ENABLE_FLASHCOMM1": (
                False,
                [
                    ("1", True),
                    ("true", True),
                    ("TRUE", True),
                    (" true ", True),
                    ("0", False),
                    ("false", False),
                    ("", False),
                ],
            ),
            "VLLM_ASCEND_LW_DEFER_LAST_SAVE": (
                False,
                [("1", True), ("true", False), ("0", False), ("", False)],
            ),
            "VLLM_MINIMAX_M3_DECODE_INDEX_SCORE_KQ": (
                False,
                [("1", True), ("true", False), ("0", False), ("", False)],
            ),
        }
        for name, (unset_value, value_cases) in cases.items():
            with self.subTest(name=name):
                original_val = os.environ.pop(name, None)
                try:
                    self.assertEqual(getattr(envs_ascend, name), unset_value)
                    for raw_value, expected in value_cases:
                        os.environ[name] = raw_value
                        self.assertEqual(getattr(envs_ascend, name), expected)
                finally:
                    if original_val is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = original_val
