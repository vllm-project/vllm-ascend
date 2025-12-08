import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import PlatformEnum

from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import (ASCEND_QUANTIZATION_METHOD,
                               COMPRESSED_TENSORS_METHOD, AscendDeviceType)


class TestNPUPlatform(TestBase):

    @staticmethod
    def mock_vllm_config():
        mock_vllm_config = MagicMock()
        mock_vllm_config.compilation_config = MagicMock()
        mock_vllm_config.model_config = MagicMock()
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.cache_config = MagicMock()
        mock_vllm_config.scheduler_config = MagicMock()
        mock_vllm_config.speculative_config = None
        mock_vllm_config.compilation_config.pass_config.enable_sp = False
        mock_vllm_config.compilation_config.cudagraph_mode = None
        return mock_vllm_config

    @staticmethod
    def mock_vllm_ascend_config():
        mock_ascend_config = MagicMock()
        mock_ascend_config.torchair_graph_config.enabled = False
        mock_ascend_config.xlite_graph_config.enabled = False
        mock_ascend_config.enable_shared_expert_dp = False
        return mock_ascend_config

    def setUp(self):
        self.platform = NPUPlatform()

    def test_class_variables(self):
        self.assertEqual(NPUPlatform._enum, PlatformEnum.OOT)
        self.assertEqual(NPUPlatform.device_name, "npu")
        self.assertEqual(NPUPlatform.device_type, "npu")
        self.assertEqual(NPUPlatform.simple_compile_backend, "eager")
        self.assertEqual(NPUPlatform.ray_device_key, "NPU")
        self.assertEqual(NPUPlatform.device_control_env_var,
                         "ASCEND_RT_VISIBLE_DEVICES")
        self.assertEqual(NPUPlatform.dispatch_key, "PrivateUse1")
        self.assertEqual(
            NPUPlatform.supported_quantization,
            [ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD])

    def test_is_sleep_mode_available(self):
        self.assertTrue(self.platform.is_sleep_mode_available())

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.quant_config.AscendQuantConfig")
    def test_pre_register_and_update_with_parser(self, mock_quant_config,
                                                 mock_adapt_patch):
        mock_parser = MagicMock()
        mock_action = MagicMock()
        mock_action.choices = ["awq", "gptq"]
        mock_parser._option_string_actions = {"--quantization": mock_action}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

        self.assertTrue(ASCEND_QUANTIZATION_METHOD in mock_action.choices)
        self.assertEqual(len(mock_action.choices), 3)  # original 2 + ascend

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.quant_config.AscendQuantConfig")
    def test_pre_register_and_update_without_parser(self, mock_quant_config,
                                                    mock_adapt_patch):
        self.platform.pre_register_and_update(None)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.quant_config.AscendQuantConfig")
    def test_pre_register_and_update_with_parser_no_quant_action(
            self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_parser._option_string_actions = {}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.quant_config.AscendQuantConfig")
    def test_pre_register_and_update_with_existing_ascend_quant(
            self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_action = MagicMock()
        mock_action.choices = ["awq", ASCEND_QUANTIZATION_METHOD]
        mock_parser._option_string_actions = {"--quantization": mock_action}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)
        self.assertEqual(len(mock_action.choices), 2)

    def test_get_device_capability(self):
        self.assertIsNone(self.platform.get_device_capability(device_id=0))

    @patch("torch.npu.get_device_name")
    def test_get_device_name(self, mock_get_device_name):
        device_id = 0
        device_name = "Ascend910B2"
        mock_get_device_name.return_value = device_name
        self.assertEqual(self.platform.get_device_name(device_id), device_name)
        mock_get_device_name.assert_called_once_with(0)

    @patch("torch.inference_mode")
    def test_inference_mode(self, mock_inference_mode):
        mock_inference_mode.return_value = None
        self.assertIsNone(self.platform.inference_mode())
        mock_inference_mode.assert_called_once()

    @patch("torch.npu.set_device")
    def test_set_device_normal(self, mock_set_device):
        device = torch.device("npu:0")
        self.platform.set_device(device)
        mock_set_device.assert_called_once_with(device)

    @patch("torch.npu.set_device",
           side_effect=RuntimeError("Device not available"))
    def test_set_device_failure(self, mock_set_device):
        device = torch.device("npu:0")
        with self.assertRaises(RuntimeError):
            self.platform.set_device(device)
        mock_set_device.assert_called_once_with(device)

    @patch("torch.npu.empty_cache")
    def test_empty_cache_normal(self, mock_empty_cache):
        self.platform.empty_cache()
        mock_empty_cache.assert_called_once()

    @patch("torch.npu.empty_cache",
           side_effect=RuntimeError("Cache clearing failed"))
    def test_empty_cache_failure(self, mock_empty_cache):
        with self.assertRaises(RuntimeError):
            self.platform.empty_cache()
        mock_empty_cache.assert_called_once()

    @patch("torch.npu.synchronize")
    def test_synchronize_normal(self, mock_synchronize):
        self.platform.synchronize()
        mock_synchronize.assert_called_once()

    @patch("torch.npu.synchronize",
           side_effect=RuntimeError("Synchronization failed"))
    def test_synchronize_failure(self, mock_synchronize):
        with self.assertRaises(RuntimeError):
            self.platform.synchronize()
        mock_synchronize.assert_called_once()

    @patch("torch.npu.mem_get_info")
    def test_mem_get_info_normal(self, mock_mem_get_info):
        free_memory_size = 1024
        total_memory_size = 2048
        memory_info = (free_memory_size, total_memory_size)
        mock_mem_get_info.return_value = memory_info
        result = self.platform.mem_get_info()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result, memory_info)
        mock_mem_get_info.assert_called_once()

    @patch("torch.npu.mem_get_info",
           side_effect=RuntimeError("NPU not available"))
    def test_mem_get_info_failure(self, mock_mem_get_info):
        with self.assertRaises(RuntimeError):
            self.platform.mem_get_info()
        mock_mem_get_info.assert_called_once()

    @patch("gc.collect")
    @patch("torch.npu.empty_cache")
    @patch("torch.npu.reset_peak_memory_stats")
    def test_clear_npu_memory_normal(self, mock_reset_stats, mock_empty_cache,
                                     mock_gc_collect):
        self.platform.clear_npu_memory()

        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_reset_stats.assert_called_once()

    @patch("gc.collect", side_effect=Exception("GC failed"))
    @patch("torch.npu.empty_cache")
    @patch("torch.npu.reset_peak_memory_stats")
    def test_clear_npu_memory_gc_collect_failure(self, mock_reset_stats,
                                                 mock_empty_cache,
                                                 mock_gc_collect):
        with self.assertRaises(Exception):
            self.platform.clear_npu_memory()

        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_not_called()
        mock_reset_stats.assert_not_called()

    @patch("gc.collect")
    @patch("torch.npu.empty_cache",
           side_effect=RuntimeError("Cache clear failed"))
    @patch("torch.npu.reset_peak_memory_stats")
    def test_clear_npu_memory_empty_cache_failure(self, mock_reset_stats,
                                                  mock_empty_cache,
                                                  mock_gc_collect):
        with self.assertRaises(RuntimeError):
            self.platform.clear_npu_memory()

        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_reset_stats.assert_not_called()

    @patch("gc.collect")
    @patch("torch.npu.empty_cache")
    @patch("torch.npu.reset_peak_memory_stats",
           side_effect=RuntimeError("Reset failed"))
    def test_clear_npu_memory_reset_stats_failure(self, mock_reset_stats,
                                                  mock_empty_cache,
                                                  mock_gc_collect):
        with self.assertRaises(RuntimeError):
            self.platform.clear_npu_memory()

        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_reset_stats.assert_called_once()

    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.utils.update_aclgraph_sizes")
    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("os.environ", {})
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_basic_config_update(
            self, mock_init_recompute, mock_soc_version, mock_update_acl,
            mock_init_ascend, mock_check_ascend):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.enable_expert_parallel = False
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        # Use importlib.reload to reload the platform module, ensuring the mocked init_ascend_config method is used.
        # Without this reload, when calling self.platform.check_and_update_config,
        # it would execute the original unmocked init_ascend_config method, causing the unit test to fail.
        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)

        mock_init_ascend.assert_called_once_with(vllm_config)
        mock_check_ascend.assert_called_once()

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_no_model_config_warning(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_soc_version):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config = None
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        with self.assertLogs(logger="vllm", level="WARNING") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Model config is missing" in cm.output[0])

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_enforce_eager_mode(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_soc_version):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = True
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Compilation disabled, using eager mode by default" in
                        cm.output[0])

        self.assertEqual(
            vllm_config.compilation_config.mode,
            CompilationMode.NONE,
        )

        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.utils.update_default_aclgraph_sizes")
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_unsupported_compilation_level(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_update_default, mock_soc_version):
        mock_update_default.return_value = MagicMock()
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        vllm_config.compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE

        with self.assertLogs(logger="vllm", level="WARNING") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
            self.assertTrue("NPU does not support" in cm.output[0])

            self.assertEqual(
                vllm_config.compilation_config.mode,
                CompilationMode.NONE,
            )
            self.assertEqual(
                vllm_config.compilation_config.cudagraph_mode,
                CUDAGraphMode.NONE,
            )

    @pytest.mark.skip(
        "Revert me when vllm support setting cudagraph_mode on oot platform")
    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_unsupported_cudagraph_mode(
            self, mock_init_ascend, mock_check_ascend, mock_soc_version):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
            self.assertTrue(
                "cudagraph_mode is not support on NPU. falling back to NONE" in
                cm.output[0])

            self.assertEqual(
                vllm_config.compilation_config.mode,
                CompilationMode.NONE,
            )
            self.assertEqual(
                vllm_config.compilation_config.cudagraph_mode,
                CUDAGraphMode.NONE,
            )

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.utils.update_default_aclgraph_sizes")
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_torchair_enabled_compilation(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_update_default, mock_soc_version):
        mock_update_default.return_value = MagicMock()
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.torchair_graph_config.enabled = True
        mock_init_ascend.return_value = mock_ascend_config
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Torchair compilation enabled" in cm.output[0])

        self.assertEqual(
            vllm_config.compilation_config.mode,
            CompilationMode.NONE,
        )
        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_cache_config_block_size(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_soc_version):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.cache_config.block_size = None
        vllm_config.cache_config.enable_prefix_caching = True
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)

        self.assertEqual(vllm_config.cache_config.block_size, 128)

    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._910_93)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_v1_worker_class_selection(
            self, mock_init_recompute, mock_init_ascend, mock_check_ascend,
            mock_soc_version):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.worker_cls = "auto"
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform.check_and_update_config(vllm_config)

        self.assertEqual(
            vllm_config.parallel_config.worker_cls,
            "vllm_ascend.worker.worker_v1.NPUWorker",
        )

        test_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        test_ascend_config.torchair_graph_config.enabled = True
        mock_init_ascend.return_value = test_ascend_config
        vllm_config.parallel_config.worker_cls = "auto"
        self.platform.check_and_update_config(vllm_config)
        self.assertEqual(
            vllm_config.parallel_config.worker_cls,
            "vllm_ascend.torchair.torchair_worker.NPUTorchairWorker",
        )

        test_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        test_ascend_config.xlite_graph_config.enabled = True
        mock_init_ascend.return_value = test_ascend_config
        vllm_config.parallel_config.worker_cls = "auto"
        self.platform.check_and_update_config(vllm_config)
        self.assertEqual(
            vllm_config.parallel_config.worker_cls,
            "vllm_ascend.xlite.xlite_worker.XliteWorker",
        )

    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch('vllm_ascend.utils.get_ascend_device_type',
           return_value=AscendDeviceType._310P)
    @patch(
        "vllm_ascend.core.recompute_schedule_config.RecomputeSchedulerConfig.initialize_from_config"
    )
    def test_check_and_update_config_310p_no_custom_ops(
            self, mock_init_recompute, mock_soc_version, mock_init_ascend,
            mock_check_ascend):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.custom_ops = []
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()

        vllm_config.scheduler_config = MagicMock()
        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)
        self.assertEqual(vllm_config.compilation_config.custom_ops, [])

    @patch('vllm_ascend.platform.get_ascend_config')
    def test_get_attn_backend_cls_use_v1_and_mla(self, mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False
        mock_config.enable_shared_expert_dp = False

        mock_get_ascend_config.return_value = mock_config

        result = self.platform.get_attn_backend_cls(
            selected_backend="ascend",
            head_size=64,
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=64,
            #use_sfa=False,
            use_mla=True,
        )
        self.assertEqual(result,
                         "vllm_ascend.attention.mla_v1.AscendMLABackend")

    @patch('vllm_ascend.platform.get_ascend_config')
    def test_get_attn_backend_cls_use_v1_mla_and_torchair(
            self, mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = True

        mock_get_ascend_config.return_value = mock_config

        result = self.platform.get_attn_backend_cls(
            selected_backend="ascend",
            head_size=64,
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=64,
            #use_sfa=False,
            use_mla=True,
        )
        self.assertEqual(
            result,
            "vllm_ascend.torchair.torchair_mla.AscendMLATorchairBackend")

    @patch('vllm_ascend.platform.get_ascend_config')
    def test_get_attn_backend_cls_use_v1_and_torchair(self,
                                                      mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = True

        mock_get_ascend_config.return_value = mock_config

        result = self.platform.get_attn_backend_cls(
            selected_backend="ascend",
            head_size=64,
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=64,
            #use_sfa=False,
            use_mla=False,
        )
        self.assertEqual(
            result,
            "vllm_ascend.torchair.torchair_attention.AscendAttentionTorchairBackend"
        )

    @patch('vllm_ascend.platform.get_ascend_config')
    def test_get_attn_backend_cls_use_v1_only(self, mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        mock_get_ascend_config.return_value = mock_config

        result = self.platform.get_attn_backend_cls(
            selected_backend="ascend",
            head_size=64,
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=64,
            #use_sfa=False,
            use_mla=False,
        )
        self.assertEqual(
            result,
            "vllm_ascend.attention.attention_v1.AscendAttentionBackend")

    def test_get_punica_wrapper(self):
        result = self.platform.get_punica_wrapper()

        self.assertEqual(result,
                         "vllm_ascend.lora.punica_npu.PunicaWrapperNPU")

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_specific_device(
            self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result
        test_device = torch.device("npu:0")
        result = self.platform.get_current_memory_usage(device=test_device)

        mock_reset_stats.assert_called_once_with(test_device)
        mock_max_memory.assert_called_once_with(test_device)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_default_device(
            self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result

        result = self.platform.get_current_memory_usage()

        mock_reset_stats.assert_called_once_with(None)
        mock_max_memory.assert_called_once_with(None)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("torch.npu.reset_peak_memory_stats",
           side_effect=RuntimeError("Device error"))
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_when_reset_stats_fails(
            self, mock_max_memory, mock_reset_stats):
        with self.assertRaises(RuntimeError):
            self.platform.get_current_memory_usage()
        mock_reset_stats.assert_called_once()
        mock_max_memory.assert_not_called()

    @patch("torch.npu.reset_peak_memory_stats")
    @patch(
        "torch.npu.max_memory_allocated",
        side_effect=RuntimeError("Memory query failed"),
    )
    def test_get_current_memory_usage_when_query_fails(self, mock_max_memory,
                                                       mock_reset_stats):
        with self.assertRaises(RuntimeError):
            self.platform.get_current_memory_usage()
        mock_reset_stats.assert_called_once()
        mock_max_memory.assert_called_once()

    def test_get_device_communicator_cls_returns_correct_value(self):
        self.assertEqual(
            self.platform.get_device_communicator_cls(),
            "vllm_ascend.distributed.communicator.NPUCommunicator",
        )

    def test_is_pin_memory_available_returns_true(self):
        self.assertTrue(self.platform.is_pin_memory_available())

    def test_get_static_graph_wrapper_cls_returns_correct_value(self):
        self.assertEqual(
            self.platform.get_static_graph_wrapper_cls(),
            "vllm_ascend.compilation.acl_graph.ACLGraphWrapper",
        )

    def test_aclgraph_enable(self):
        config = EngineArgs()
        VllmConfig = config.create_engine_config()
        self.assertEqual(VllmConfig.compilation_config.cudagraph_mode,
                         CUDAGraphMode.PIECEWISE)

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(VllmConfig)
            target_msg = "PIECEWISE compilation enabled on NPU. use_inductor not supported - using only ACL Graph mode"
            found = any(target_msg in log for log in cm.output)

            self.assertTrue(
                found,
                f"Expected log message not found. Captured logs: {cm.output}")

            self.assertEqual(
                VllmConfig.compilation_config.mode,
                CompilationMode.VLLM_COMPILE,
            )
            self.assertEqual(
                VllmConfig.compilation_config.cudagraph_mode,
                CUDAGraphMode.PIECEWISE,
            )
