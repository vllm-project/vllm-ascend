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
import unittest
from threading import Lock
from unittest import mock

import torch
from vllm.config import (CompilationConfig, ModelConfig, ParallelConfig,
                         VllmConfig)

from tests.ut.base import TestBase
from vllm_ascend import utils


class TestUtils(TestBase):

    def test_is_310p(self):
        utils._IS_310P = None
        with mock.patch("vllm_ascend._build_info.__soc_version__",
                        "Ascend310P3"):
            self.assertTrue(utils.is_310p())
        utils._IS_310P = None
        with mock.patch("vllm_ascend._build_info.__soc_version__",
                        "Ascend910P1"):
            self.assertFalse(utils.is_310p())

    def test_sleep_mode_enabled(self):
        utils._SLEEP_MODE_ENABLED = None
        with mock.patch("vllm_ascend._build_info.__sleep_mode_enabled__",
                        True):
            self.assertTrue(utils.sleep_mode_enabled())
        utils._SLEEP_MODE_ENABLED = None
        with mock.patch("vllm_ascend._build_info.__sleep_mode_enabled__",
                        False):
            self.assertFalse(utils.sleep_mode_enabled())

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

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    @mock.patch('vllm_ascend.utils.is_310p')
    @mock.patch('vllm_ascend.utils.get_ascend_config')
    def test_maybe_converting_weight_acl_format(self, mock_get_config,
                                                mock_310p, mock_npu_cast,
                                                mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_310p.return_value = True

        mock_config = mock.MagicMock()
        mock_config.torchair_graph_config.enabled = True
        mock_get_config.return_value = mock_config
        mock_get_format.return_value = 1

        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        utils.maybe_converting_weight_acl_format(model, ACL_FORMAT_FRACTAL_NZ)
        self.assertEqual(fused_moe.w13_weight.data, 1)

    @mock.patch('torch_npu.get_npu_format')
    @mock.patch('torch_npu.npu_format_cast')
    @mock.patch('vllm.model_executor.layers.fused_moe.layer.FusedMoE',
                new=mock.MagicMock)
    @mock.patch('vllm_ascend.utils.is_310p')
    @mock.patch('vllm_ascend.utils.get_ascend_config')
    def test_maybe_converting_weight_acl_format_format_true(
            self, mock_get_config, mock_310p, mock_npu_cast, mock_get_format):
        ACL_FORMAT_FRACTAL_NZ = 29
        mock_310p.return_value = True

        mock_config = mock.MagicMock()
        mock_config.torchair_graph_config.enabled = True
        mock_get_config.return_value = mock_config
        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ

        mock_npu_cast.return_value = 1

        fused_moe = mock.MagicMock()
        fused_moe.w13_weight = mock.MagicMock()
        fused_moe.w2_weight = mock.MagicMock()
        fused_moe.w13_weight.data = torch.randn(128, 256)
        fused_moe.w2_weight.data = torch.randn(256, 128)
        model = mock.MagicMock()
        model.modules.return_value = [fused_moe]

        mock_get_format.return_value = ACL_FORMAT_FRACTAL_NZ

        utils.maybe_converting_weight_acl_format(model, ACL_FORMAT_FRACTAL_NZ)

    @mock.patch('vllm_ascend.utils.get_ascend_config')
    @mock.patch('vllm_ascend.utils.is_310p', return_value=False)
    def test_maybe_converting_weight_acl_format_not_310_not_graph(
            self, mock_310p, mock_get_config):
        mock_config = mock.MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_get_config.return_value = mock_config

        mock_constant = mock.MagicMock()

        mock_model = mock.MagicMock()
        utils.maybe_converting_weight_acl_format(mock_model, mock_constant)

    @mock.patch('importlib.util.find_spec')
    @mock.patch('importlib.import_module')
    def test_try_register_lib(self, mock_import_module, mock_find_spec):
        # import OK
        mock_find_spec.return_value = mock.MagicMock()
        mock_import_module.return_value = mock.MagicMock()
        lib_name = "existing_lib"
        lib_info = "Library found and imported successfully"
        utils.try_register_lib(lib_name, lib_info)

        # Can't find lib
        mock_find_spec.return_value = None
        lib_name = "non_existing_lib"
        utils.try_register_lib(lib_name)

        # import error
        mock_find_spec.return_value = mock.MagicMock()
        mock_import_module.side_effect = ImportError("import error")
        lib_name = "error_lib"
        utils.try_register_lib(lib_name)

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
                self.assertTrue(utils.vllm_version_is("1.0.0"))
                self.assertFalse(utils.vllm_version_is("2.0.0"))
            with mock.patch("vllm.__version__", "2.0.0"):
                self.assertTrue(utils.vllm_version_is("1.0.0"))
                self.assertFalse(utils.vllm_version_is("2.0.0"))
        with mock.patch("vllm.__version__", "1.0.0"):
            self.assertTrue(utils.vllm_version_is("1.0.0"))
            self.assertFalse(utils.vllm_version_is("2.0.0"))
        with mock.patch("vllm.__version__", "2.0.0"):
            self.assertTrue(utils.vllm_version_is("2.0.0"))
            self.assertFalse(utils.vllm_version_is("1.0.0"))

    def test_update_aclgraph_sizes(self):
        # max_num_batch_sizes < len(original_sizes)
        test_compilation_config = CompilationConfig(
            cudagraph_capture_sizes=[i for i in range(150)])
        model_path = os.path.join(os.path.dirname(__file__), "fake_weight")
        test_model_config = ModelConfig(model=model_path, enforce_eager=True)
        test_parallel_config = ParallelConfig()
        test_vllm_config = VllmConfig(
            model_config=test_model_config,
            compilation_config=test_compilation_config,
            parallel_config=test_parallel_config,
        )
        utils.update_aclgraph_sizes(test_vllm_config)
        self.assertEqual(
            147,
            len(test_vllm_config.compilation_config.cudagraph_capture_sizes))
        # max_num_batch_sizes >= len(original_sizes)
        test_compilation_config = CompilationConfig(
            cudagraph_capture_sizes=[1, 2, 3])
        test_vllm_config = VllmConfig(
            model_config=test_model_config,
            compilation_config=test_compilation_config,
            parallel_config=test_parallel_config,
        )
        utils.update_aclgraph_sizes(test_vllm_config)
        self.assertEqual(
            3,
            len(test_vllm_config.compilation_config.cudagraph_capture_sizes))

    def test_get_torchair_current_work_dir(self):
        cache_dir = utils.TORCHAIR_CACHE_DIR
        work_dir = utils.get_torchair_current_work_dir()
        self.assertEqual(cache_dir, work_dir)
        work_dir = utils.get_torchair_current_work_dir("test")
        self.assertEqual(os.path.join(cache_dir, "test"), work_dir)

    def test_torchair_cache_dir(self):
        utils.write_kv_cache_bytes_to_file(0, 100)
        self.assertTrue(utils.check_torchair_cache_exist(),
                        "Create torchair cache dir failed")
        self.assertTrue(utils.check_kv_cache_bytes_cache_exist(),
                        "Create kv cache bytes cache dir failed")
        kv_cache_bytes = utils.read_kv_cache_bytes_from_file(0)
        self.assertEqual(100, kv_cache_bytes)
        utils.delete_torchair_cache_file()
        self.assertFalse(utils.check_torchair_cache_exist(),
                         "Delete torchair cache dir failed")
        self.assertFalse(utils.check_kv_cache_bytes_cache_exist(),
                         "Delete kv cache bytes cache dir failed")


class TestProfileExecuteDuration(unittest.TestCase):

    def setUp(self):
        utils.ProfileExecuteDuration._instance = None
        utils.ProfileExecuteDuration._observations = []
        utils.ProfileExecuteDuration._lock = Lock()

    def test_singleton_creation(self):
        instance1 = utils.ProfileExecuteDuration()
        self.assertIsNotNone(instance1)
        self.assertIs(instance1, utils.ProfileExecuteDuration._instance)

        instance2 = utils.ProfileExecuteDuration()
        self.assertIs(instance1, instance2)

    def test_thread_safety(self):
        from threading import Thread

        instances = []

        def create_instance():
            instances.append(utils.ProfileExecuteDuration())

        threads = [Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        first_instance = instances[0]
        for instance in instances[1:]:
            self.assertIs(first_instance, instance)

    def test_atexit_registration(self):
        with mock.patch('atexit.register') as mock_register:
            instance = utils.ProfileExecuteDuration()
            mock_register.assert_called_once_with(instance.destroy)

    def test_lock_usage(self):
        original_lock = utils.ProfileExecuteDuration._lock

        with mock.patch.object(utils.ProfileExecuteDuration,
                               '_lock',
                               wraps=original_lock) as mock_lock:
            utils.ProfileExecuteDuration()
            mock_lock.__enter__.assert_called()
            mock_lock.__exit__.assert_called()

    def test_observations_initialization(self):
        instance = utils.ProfileExecuteDuration()
        self.assertEqual(instance._observations, [])
