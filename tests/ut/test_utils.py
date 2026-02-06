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
#

import math
import os
from threading import Lock
from unittest import mock

import pytest
import torch
from vllm.config import (CompilationConfig, ModelConfig, ParallelConfig,
                         VllmConfig)

from tests.ut.base import TestBase
from vllm_ascend import utils
from vllm_ascend.utils import REGISTERED_ASCEND_OPS


class TestUtils(TestBase):

    def setUp(self):
        import importlib

        from vllm_ascend import platform
        importlib.reload(platform)

    def test_nd_to_nz_2d(self):
        # can be divided by 16
        input_tensor = torch.randn(32, 64)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 64 // 16)
        self.assertEqual(output.shape[2], 32)
        self.assertEqual(output.shape[3], 16)

        # cannot be divided by 16
        input_tensor = torch.randn(30, 62)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], math.ceil(62 / 16))
        self.assertEqual(output.shape[2], 32)
        self.assertEqual(output.shape[3], 16)

        # pad to 16
        input_tensor = torch.randn(8, 12)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 1)  # 12->16, 16//16=1
        self.assertEqual(output.shape[2], 16)  # 8->16
        self.assertEqual(output.shape[3], 16)

        # check if the output is contiguous
        input_tensor = torch.randn(32, 64)
        output = utils.nd_to_nz_2d(input_tensor)
        self.assertTrue(output.is_contiguous())

        # check if the output values are preserved
        input_tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        output = utils.nd_to_nz_2d(input_tensor)
        expected = torch.tensor(
            [[[[1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]])
        self.assertTrue(torch.allclose(output, expected))

    def test_aligned_16(self):
        # align to 16
        input_tensor = torch.randn(15, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 16)

        # align to 16
        input_tensor = torch.randn(16, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 16)
        self.assertTrue(torch.equal(input_tensor, output_tensor))

        # align to 32
        input_tensor = torch.randn(17, 64)
        output_tensor = utils.aligned_16(input_tensor)
        self.assertEqual(output_tensor.shape[0], 32)

    @pytest.mark.skip(
        "Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
    def test_enable_custom_op(self):
        result = utils.enable_custom_op()
        self.assertTrue(result)

        utils._CUSTOM_OP_ENABLED = None

        with mock.patch('builtins.__import__') as mock_import_module:
            mock_import_module.side_effect = ImportError("import error")
            self.assertFalse(utils.enable_custom_op())

    def test_find_hccl_library(self):
        with mock.patch.dict(os.environ,
                             {"HCCL_SO_PATH": "/path/to/hccl/libhccl.so"}):
            self.assertEqual(utils.find_hccl_library(),
                             "/path/to/hccl/libhccl.so")
        with mock.patch("torch.version.cann", None):
            self.assertRaises(ValueError, utils.find_hccl_library)
        with mock.patch("torch.version.cann", "Ascend910"):
            self.assertEqual(utils.find_hccl_library(), "libhccl.so")

    def test_current_stream(self):
        with mock.patch("torch.npu.current_stream") as mock_current_stream:
            self.assertEqual(utils.current_stream(), mock_current_stream())

    def test_vllm_version_is(self):
        with mock.patch.dict(os.environ, {"VLLM_VERSION": "1.0.0"}):
            with mock.patch("vllm.__version__", "1.0.0"):
                self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
                self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
            with mock.patch("vllm.__version__", "2.0.0"):
                self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
                self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
        with mock.patch("vllm.__version__", "1.0.0"):
            self.assertTrue(utils.vllm_version_is.__wrapped__("1.0.0"))
            self.assertFalse(utils.vllm_version_is.__wrapped__("2.0.0"))
        with mock.patch("vllm.__version__", "2.0.0"):
            self.assertTrue(utils.vllm_version_is.__wrapped__("2.0.0"))
            self.assertFalse(utils.vllm_version_is.__wrapped__("1.0.0"))
        # Test caching takes effect
        utils.vllm_version_is.cache_clear()
        utils.vllm_version_is("1.0.0")
        misses = utils.vllm_version_is.cache_info().misses
        hits = utils.vllm_version_is.cache_info().hits
        self.assertEqual(misses, 1)
        self.assertEqual(hits, 0)
        utils.vllm_version_is("1.0.0")
        hits = utils.vllm_version_is.cache_info().hits
        self.assertEqual(hits, 1)

    def test_get_max_hidden_layers(self):
        from transformers import PretrainedConfig

        class SimpleConfig(PretrainedConfig):

            def __init__(self, num_hidden_layers=12):
                self.num_hidden_layers = num_hidden_layers

            def to_dict(self):
                return {"num_hidden_layers": self.num_hidden_layers}

        self.assertEqual(utils.get_max_hidden_layers(SimpleConfig()), 12)
        self.assertEqual(utils.get_max_hidden_layers(SimpleConfig(24)), 24)

        class NestedConfig(PretrainedConfig):

            def to_dict(self):
                return {
                    "model": {
                        "encoder": {
                            "num_hidden_layers": 8
                        },
                        "decoder": {
                            "num_hidden_layers": 12
                        }
                    },
                    "other_setting": True
                }

        self.assertEqual(utils.get_max_hidden_layers(NestedConfig()), 12)

        class MultiValueConfig(PretrainedConfig):

            def to_dict(self):
                return {
                    "num_hidden_layers": 6,
                    "submodule": {
                        "num_hidden_layers": 18,
                        "subsub": {
                            "num_hidden_layers": 9
                        }
                    }
                }

        self.assertEqual(utils.get_max_hidden_layers(MultiValueConfig()), 18)

        class NoLayerConfig(PretrainedConfig):

            def to_dict(self):
                return {"attention_heads": 8}

        with self.assertRaises(ValueError) as context:
            utils.get_max_hidden_layers(NoLayerConfig())
        self.assertIn("num_hidden_layers", str(context.exception))

    def test_update_aclgraph_sizes(self):
        test_compilation_config = CompilationConfig(
            cudagraph_capture_sizes=[i for i in range(150)])
        model_path = os.path.join(os.path.dirname(__file__), "fake_weight")
        test_model_config = ModelConfig(model=model_path, enforce_eager=True)
        test_parallel_config = ParallelConfig()
        test_vllm_config = VllmConfig(
            model_config=test_model_config,
            compilation_config=test_compilation_config,
            parallel_config=test_parallel_config)
        utils.update_aclgraph_sizes(test_vllm_config)
        os.environ['HCCL_OP_EXPANSION_MODE'] = 'AIV'
        utils.update_aclgraph_sizes(test_vllm_config)
        del os.environ['HCCL_OP_EXPANSION_MODE']

        self.assertEqual(
            0,
            len(test_vllm_config.compilation_config.cudagraph_capture_sizes))

    @mock.patch("vllm.model_executor.custom_op.CustomOp")
    @mock.patch("vllm_ascend.ops.activation.AscendQuickGELU")
    @mock.patch("vllm_ascend.ops.activation.AscendSiluAndMul")
    @mock.patch("vllm_ascend.ops.layernorm.AscendRMSNorm")
    def test_register_ascend_customop(self, mock_ascend_rmsnorm,
                                      mock_ascend_silu_and_mul,
                                      mock_ascend_quick_gelu, mock_customop):
        utils._ASCEND_CUSTOMOP_IS_REIGISTERED = False

        # ascend custom op is not registered
        utils.register_ascend_customop()
        self.assertEqual(mock_customop.register_oot.call_count,
                         len(REGISTERED_ASCEND_OPS))
        self.assertTrue(utils._ASCEND_CUSTOMOP_IS_REIGISTERED)

        # ascend custom op is already registered
        utils.register_ascend_customop()
        self.assertEqual(mock_customop.register_oot.call_count,
                         len(REGISTERED_ASCEND_OPS))
