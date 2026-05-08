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
                    if "int(" in handler_source:
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

    def _with_env(self, name: str, value: str | None):
        # Helper context that sets/clears one env var and restores the original.
        original = os.environ.get(name)

        class _Ctx:
            def __enter__(_self):
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
                return _self

            def __exit__(_self, exc_type, exc, tb):
                if original is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = original

        return _Ctx()

    def test_dynamic_eplb_truthy_values_are_case_insensitive(self):
        # Both DYNAMIC_EPLB and EXPERT_MAP_RECORD gate the same EPLB code paths
        # (vllm_ascend/ascend_config.py, vllm_ascend/patch/platform/__init__.py).
        # They must accept the same case-insensitive truthy spellings; previously
        # only DYNAMIC_EPLB was case-insensitive while EXPERT_MAP_RECORD demanded
        # an exact lowercase "true", which silently disabled the patch when users
        # wrote "True" or "1".
        truthy_inputs = ["true", "True", "TRUE", "tRuE", "1", " true ", " 1 "]
        falsy_inputs = ["false", "False", "0", "", "yes", "no", "TRUEISH"]
        for var_name in ("DYNAMIC_EPLB", "EXPERT_MAP_RECORD"):
            self.assertIn(var_name, self.env_vars, f"{var_name} must be defined in vllm_ascend.envs")
            for value in truthy_inputs:
                with self._with_env(var_name, value):
                    self.assertIs(
                        getattr(envs_ascend, var_name),
                        True,
                        f"{var_name}={value!r} should be parsed as True",
                    )
            for value in falsy_inputs:
                with self._with_env(var_name, value):
                    self.assertIs(
                        getattr(envs_ascend, var_name),
                        False,
                        f"{var_name}={value!r} should be parsed as False",
                    )
            with self._with_env(var_name, None):
                self.assertIs(
                    getattr(envs_ascend, var_name),
                    False,
                    f"unset {var_name} should default to False",
                )
