import importlib
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import PrefixStore
from vllm.config import CompilationLevel
from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import PlatformEnum

from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD


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
        mock_vllm_config.compilation_config.pass_config.enable_sequence_parallelism = False
        mock_vllm_config.compilation_config.cudagraph_mode = None
        return mock_vllm_config

    @staticmethod
    def mock_vllm_ascend_config():
        mock_ascend_config = MagicMock()
        mock_ascend_config.torchair_graph_config.enabled = False
        mock_ascend_config.ascend_scheduler_config.enabled = False
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
        self.assertEqual(NPUPlatform.supported_quantization,
                         [ASCEND_QUANTIZATION_METHOD])

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

    def test_is_async_output_supported(self):
        self.assertTrue(
            self.platform.is_async_output_supported(enforce_eager=None))
        self.assertTrue(
            self.platform.is_async_output_supported(enforce_eager=True))
        self.assertTrue(
            self.platform.is_async_output_supported(enforce_eager=False))

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
    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("os.environ", {})
    def test_check_and_update_config_basic_config_update(
            self, mock_is_310p, mock_update_acl, mock_init_ascend,
            mock_check_ascend):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.enable_expert_parallel = False

        # Use importlib.reload to reload the platform module, ensuring the mocked init_ascend_config method is used.
        # Without this reload, when calling self.platform.check_and_update_config,
        # it would execute the original unmocked init_ascend_config method, causing the unit test to fail.
        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)

        mock_init_ascend.assert_called_once_with(vllm_config)
        mock_check_ascend.assert_called_once()

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_no_model_config_warning(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config = None

        with self.assertLogs(logger="vllm", level="WARNING") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Model config is missing" in cm.output[0])

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_enforce_eager_mode(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = True

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Compilation disabled, using eager mode by default" in
                        cm.output[0])
        self.assertEqual(
            vllm_config.compilation_config.level,
            CompilationLevel.NO_COMPILATION,
        )
        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_unsupported_compilation_level(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.level = CompilationLevel.DYNAMO_ONCE

        with self.assertLogs(logger="vllm", level="WARNING") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
            self.assertTrue("NPU does not support" in cm.output[0])
            self.assertEqual(
                vllm_config.compilation_config.level,
                CompilationLevel.NO_COMPILATION,
            )
            self.assertEqual(
                vllm_config.compilation_config.cudagraph_mode,
                CUDAGraphMode.NONE,
            )

    @pytest.mark.skip(
        "Revert me when vllm support setting cudagraph_mode on oot platform")
    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_unsupported_cudagraph_mode(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
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
                vllm_config.compilation_config.level,
                CompilationLevel.NO_COMPILATION,
            )
            self.assertEqual(
                vllm_config.compilation_config.cudagraph_mode,
                CUDAGraphMode.NONE,
            )

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_torchair_enabled_compilation(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.torchair_graph_config.enabled = True
        mock_init_ascend.return_value = mock_ascend_config
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.level = CompilationLevel.PIECEWISE

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
        self.assertTrue("Torchair compilation enabled" in cm.output[0])
        self.assertEqual(
            vllm_config.compilation_config.level,
            CompilationLevel.NO_COMPILATION,
        )
        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_cache_config_block_size(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.cache_config.block_size = None
        vllm_config.cache_config.enable_prefix_caching = True

        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)

        self.assertEqual(vllm_config.cache_config.block_size, 128)

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_v1_worker_class_selection(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.worker_cls = "auto"

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

    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.utils.is_310p", return_value=True)
    def test_check_and_update_config_310p_no_custom_ops(
            self, mock_is_310p, mock_init_ascend, mock_check_ascend):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config(
        )
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.custom_ops = []

        from vllm_ascend import platform

        importlib.reload(platform)

        self.platform.check_and_update_config(vllm_config)
        self.assertEqual(vllm_config.compilation_config.custom_ops, [])

    @patch("vllm_ascend.utils.is_310p", return_value=False)
    @patch("vllm_ascend.ascend_config.check_ascend_config")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_ascend_scheduler_config(
            self, mock_init_ascend, mock_check_ascend, mock_is_310p):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.ascend_scheduler_config.enabled = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()

        with patch("vllm_ascend.core.schedule_config.AscendSchedulerConfig"
                   ) as mock_scheduler:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
            mock_scheduler.initialize_from_config.assert_called_once()

    @patch('vllm_ascend.platform.get_ascend_config')
    def test_get_attn_backend_cls_use_v1_and_mla(self, mock_get_ascend_config):
        mock_config = MagicMock()
        mock_config.torchair_graph_config.enabled = False

        mock_get_ascend_config.return_value = mock_config

        result = self.platform.get_attn_backend_cls(
            selected_backend="ascend",
            head_size=64,
            dtype="float16",
            kv_cache_dtype="float16",
            block_size=64,
            use_v1=True,
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
            use_v1=True,
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
            use_v1=True,
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
            use_v1=True,
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

    def test_supports_v1(self):
        from vllm.config import ModelConfig

        mock_config = MagicMock(spec=ModelConfig)
        self.assertTrue(self.platform.supports_v1(mock_config))

    def test_get_static_graph_wrapper_cls_returns_correct_value(self):
        self.assertEqual(
            self.platform.get_static_graph_wrapper_cls(),
            "vllm_ascend.compilation.acl_graph.ACLGraphWrapper",
        )

    @patch("torch.distributed.is_hccl_available", return_value=True)
    @patch("torch_npu._C._distributed_c10d.ProcessGroupHCCL")
    @patch("torch.distributed.ProcessGroup")
    def test_successful_initialization(self, mock_pg, mock_pg_hccl, _):
        mock_prefix = MagicMock(spec=PrefixStore)
        mock_backend = MagicMock()
        mock_pg_hccl.return_value = mock_backend
        group_rank = 0
        group_size = 4

        mock_pg_instance = MagicMock(spec=ProcessGroup)
        mock_pg.return_value = mock_pg_instance

        # Use importlib.reload() to force-reload the platform module and ensure the mocked ProcessGroup is used.
        # Without this reload, when executing self.platform.stateless_init_device_torch_dist_pg(),
        # it would invoke the original unmocked ProcessGroup implementation instead of our test mock,
        # which would cause the unit test to fail.
        from vllm_ascend import platform

        importlib.reload(platform)

        result = self.platform.stateless_init_device_torch_dist_pg(
            backend="hccl",
            prefix_store=mock_prefix,
            group_rank=group_rank,
            group_size=group_size,
            timeout=timedelta(seconds=30),
        )

        mock_pg.assert_called_once_with(mock_prefix, group_rank, group_size)
        mock_pg_hccl.assert_called_once_with(mock_prefix, group_rank,
                                             group_size, unittest.mock.ANY)
        mock_backend._set_sequence_number_for_group.assert_called_once()
        mock_pg_instance._register_backend.assert_called_once_with(
            torch.device("npu"), unittest.mock.ANY, mock_backend)
        self.assertEqual(result, mock_pg_instance)

    @patch("torch.distributed.is_hccl_available", return_value=False)
    def test_hccl_unavailable(self, _):
        with self.assertRaises(AssertionError):
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.stateless_init_device_torch_dist_pg(
                backend="hccl",
                prefix_store=MagicMock(),
                group_rank=0,
                group_size=4,
                timeout=timedelta(seconds=30),
            )
