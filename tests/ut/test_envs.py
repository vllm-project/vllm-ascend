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

import os
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.envs import env_variables


class TestEnvVariables(TestBase):
    _default_values = {
        "MAX_JOBS": None,
        "CMAKE_BUILD_TYPE": None,
        "COMPILE_CUSTOM_KERNELS": True,
        "CXX_COMPILER": None,
        "C_COMPILER": None,
        "SOC_VERSION": "ASCEND910B1",
        "VERBOSE": False,
        "ASCEND_HOME_PATH": None,
        "HCCN_PATH": "/usr/local/Ascend/driver/tools/hccn_tool",
        "HCCL_SO_PATH": None,
        "PROMPT_DEVICE_ID": None,
        "DECODE_DEVICE_ID": None,
        "LLMDATADIST_COMM_PORT": "26000",
        "LLMDATADIST_SYNC_CACHE_WAIT_TIME": "5000",
        "VLLM_VERSION": None,
        "VLLM_ASCEND_TRACE_RECOMPILES": False,
        "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP": False,
        "VLLM_ASCEND_ENABLE_DBO": False,
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE": False,
        "MOE_ALL2ALL_BUFFER": False,
        "USE_OPTIMIZED_MODEL": True,
        "SELECT_GATING_TOPK_SOTFMAX_EXPERTS": False,
        "VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE": 64,
        "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": True,
        "DISAGGREGATED_PREFILL_RANK_TABLE_PATH": None,
        "VLLM_ASCEND_LLMDD_RPC_IP": "0.0.0.0",
        "VLLM_LLMDD_RPC_PORT": 5557,
        "VLLM_ASCEND_MLA_PA": False,
        "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": False
    }

    _test_values = {
        "MAX_JOBS": ("8", "8"),
        "CMAKE_BUILD_TYPE": ("Debug", "Debug"),
        "COMPILE_CUSTOM_KERNELS": ("0", False),
        "CXX_COMPILER": ("/usr/bin/g++", "/usr/bin/g++"),
        "C_COMPILER": ("/usr/bin/gcc", "/usr/bin/gcc"),
        "SOC_VERSION": ("ASCEND910A", "ASCEND910A"),
        "VERBOSE": ("1", True),
        "ASCEND_HOME_PATH": ("/opt/ascend", "/opt/ascend"),
        "HCCN_PATH": ("/custom/hccn_tool", "/custom/hccn_tool"),
        "HCCL_SO_PATH": ("libhccl_custom.so", "libhccl_custom.so"),
        "PROMPT_DEVICE_ID": ("1", "1"),
        "DECODE_DEVICE_ID": ("2", "2"),
        "LLMDATADIST_COMM_PORT": ("27000", "27000"),
        "LLMDATADIST_SYNC_CACHE_WAIT_TIME": ("6000", "6000"),
        "VLLM_VERSION": ("0.9.1", "0.9.1"),
        "VLLM_ASCEND_TRACE_RECOMPILES": ("1", True),
        "VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP": ("1", True),
        "VLLM_ASCEND_ENABLE_DBO": ("1", True),
        "VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE": ("1", True),
        "MOE_ALL2ALL_BUFFER": ("1", True),
        "USE_OPTIMIZED_MODEL": ("0", False),
        "SELECT_GATING_TOPK_SOTFMAX_EXPERTS": ("1", True),
        "VLLM_ASCEND_KV_CACHE_MEGABYTES_FLOATING_TOLERANCE": ("128", 128),
        "VLLM_ASCEND_ENABLE_TOPK_TOPP_OPTIMIZATION": ("0", False),
        "DISAGGREGATED_PREFILL_RANK_TABLE_PATH":
        ("/path/to/rank_table.json", "/path/to/rank_table.json"),
        "VLLM_ASCEND_LLMDD_RPC_IP": ("192.168.1.1", "192.168.1.1"),
        "VLLM_LLMDD_RPC_PORT": ("5558", 5558),
        "VLLM_ASCEND_MLA_PA": ("1", True),
        "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE": ("1", True)
    }

    def setUp(self):
        env_keys = set(env_variables.keys())
        self.assertEqual(env_keys, set(self._default_values.keys()))
        self.assertEqual(env_keys, set(self._test_values.keys()))

    def test_default_values(self):
        for var_name, getter in env_variables.items():
            with self.subTest(var_name=var_name):
                with patch.dict(os.environ, {}, clear=True):
                    actual = getter()
                    expected = self._default_values[var_name]
                    self.assertEqual(actual, expected)

    def test_set_values(self):
        for var_name, getter in env_variables.items():
            with self.subTest(var_name=var_name):
                input_val, expected = self._test_values[var_name]
                with patch.dict(os.environ, {var_name: input_val}):
                    actual = getter()
                    self.assertEqual(actual, expected)