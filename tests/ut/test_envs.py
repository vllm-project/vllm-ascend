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
                    if var_name == "VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M":
                        test_vals = ["1", "2"]
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

    def test_split_qkv_rmsnorm_mrope_block_m_is_strict_and_defaults_to_two(self):
        name = "VLLM_ASCEND_SPLIT_QKV_RMSNORM_MROPE_BLOCK_M"
        original_val = os.environ.pop(name, None)
        try:
            self.assertEqual(getattr(envs_ascend, name), 2)
            for value in ("1", "2"):
                with self.subTest(value=value):
                    os.environ[name] = value
                    self.assertEqual(getattr(envs_ascend, name), int(value))
            for value in ("", "0", "3", "4", "-1", "123", "456", "not-an-integer"):
                with self.subTest(invalid=value):
                    os.environ[name] = value
                    with self.assertRaises(ValueError):
                        getattr(envs_ascend, name)
        finally:
            if original_val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original_val

    def test_retired_ub_and_kvpool_overrides_are_not_registered(self):
        for name in ("VLLM_ASCEND_ROPE_UB_SIZE_KB", "VLLM_ASCEND_KVPOOL_RANGE_DEBUG"):
            with self.subTest(name=name):
                original_val = os.environ.get(name)
                try:
                    os.environ[name] = "invalid"
                    self.assertNotIn(name, envs_ascend.env_variables)
                    with self.assertRaises(AttributeError):
                        getattr(envs_ascend, name)
                finally:
                    if original_val is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = original_val
