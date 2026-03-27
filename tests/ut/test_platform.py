import importlib
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import PlatformEnum
from vllm.v1.attention.selector import AttentionSelectorConfig  # type: ignore

from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.utils import (
    ASCEND_QUANTIZATION_METHOD,
    COMPRESSED_TENSORS_METHOD,
    AscendDeviceType,
)


class TestNPUPlatform(TestBase):
    @staticmethod
    def mock_vllm_config():
        mock_vllm_config = MagicMock()
        mock_vllm_config.compilation_config = MagicMock()
        mock_vllm_config.model_config = MagicMock()
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.cache_config = MagicMock()
        mock_vllm_config.scheduler_config = MagicMock()
        mock_vllm_config.scheduler_config.max_num_seqs = None
        mock_vllm_config.speculative_config = None
        mock_vllm_config.additional_config = {}
        mock_vllm_config.compilation_config.pass_config.enable_sp = False
        mock_vllm_config.compilation_config.cudagraph_mode = None
        return mock_vllm_config

    @staticmethod
    def mock_vllm_ascend_config():
        mock_ascend_config = MagicMock()
        mock_ascend_config.xlite_graph_config.enabled = False
        mock_ascend_config.xlite_graph_config.full_mode = False
        mock_ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        mock_ascend_config.ascend_fusion_config = None
        mock_ascend_config.recompute_scheduler_enable = False
        mock_ascend_config.SLO_limits_for_dynamic_batch = -1
        mock_ascend_config.enable_shared_expert_dp = False
        mock_ascend_config.update_compile_ranges_split_points = MagicMock()
        return mock_ascend_config

    def setUp(self):
        self.platform = NPUPlatform()
        self.platform.supported_quantization[:] = ["ascend", "compressed-tensors"]

    def test_class_variables(self):
        self.assertEqual(NPUPlatform._enum, PlatformEnum.OOT)
        self.assertEqual(NPUPlatform.device_name, "npu")
        self.assertEqual(NPUPlatform.device_type, "npu")
        self.assertEqual(NPUPlatform.simple_compile_backend, "eager")
        self.assertEqual(NPUPlatform.ray_device_key, "NPU")
        self.assertEqual(NPUPlatform.device_control_env_var, "ASCEND_RT_VISIBLE_DEVICES")
        self.assertEqual(NPUPlatform.dispatch_key, "PrivateUse1")
        self.assertEqual(NPUPlatform.supported_quantization, [ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD])

    def test_is_sleep_mode_available(self):
        self.assertTrue(self.platform.is_sleep_mode_available())

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.modelslim_config.AscendModelSlimConfig")
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
    @patch("vllm_ascend.quantization.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_without_parser(self, mock_quant_config,
                                                    mock_adapt_patch):
        self.platform.pre_register_and_update(None)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_with_parser_no_quant_action(
            self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_parser._option_string_actions = {}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_with_existing_ascend_quant(
            self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_action = MagicMock()
        mock_action.choices = ["awq", ASCEND_QUANTIZATION_METHOD]
        mock_parser._option_string_actions = {"--quantization": mock_action}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)
        self.assertEqual(len(mock_action.choices), 2)

    def test_apply_config_platform_defaults_sets_ascend_default_max(self):
        test_cases = [
            (40, 3, 160),
            (200, 3, 512),
        ]

        for max_num_seqs, num_speculative_tokens, expected_max in test_cases:
            with self.subTest(
                max_num_seqs=max_num_seqs,
                num_speculative_tokens=num_speculative_tokens,
                expected_max=expected_max,
            ):
                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.scheduler_config.max_num_seqs = max_num_seqs
                vllm_config.speculative_config = MagicMock(
                    num_speculative_tokens=num_speculative_tokens
                )
                vllm_config.compilation_config.max_cudagraph_capture_size = None
                vllm_config.compilation_config.cudagraph_capture_sizes = None

                self.platform.apply_config_platform_defaults(vllm_config)

                self.assertEqual(
                    vllm_config.compilation_config.max_cudagraph_capture_size,
                    expected_max,
                )

    def test_apply_config_platform_defaults_respects_explicit_max(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.max_cudagraph_capture_size = 456
        vllm_config.compilation_config.cudagraph_capture_sizes = None

        self.platform.apply_config_platform_defaults(vllm_config)

        self.assertEqual(vllm_config.compilation_config.max_cudagraph_capture_size, 456)

    def test_apply_config_platform_defaults_respects_explicit_sizes(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.max_cudagraph_capture_size = None
        vllm_config.compilation_config.cudagraph_capture_sizes = [1, 2, 4]

        self.platform.apply_config_platform_defaults(vllm_config)

        self.assertIsNone(vllm_config.compilation_config.max_cudagraph_capture_size)
        self.assertEqual(vllm_config.compilation_config.cudagraph_capture_sizes, [1, 2, 4])

    def test_apply_config_platform_defaults_skips_when_scheduler_max_num_seqs_is_missing(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.max_cudagraph_capture_size = None
        vllm_config.compilation_config.cudagraph_capture_sizes = None

        self.platform.apply_config_platform_defaults(vllm_config)

        self.assertIsNone(vllm_config.compilation_config.max_cudagraph_capture_size)

    @patch("vllm_ascend.platform.refresh_block_size")
    @patch("vllm_ascend.platform.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.platform.enable_sp", return_value=False)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    def test_check_and_update_config_preserves_platform_default_max_input(
        self,
        mock_auto_detect,
        mock_init_ascend,
        _mock_enable_sp,
        _mock_device_type,
        _mock_refresh_block_size,
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.scheduler_config.max_num_seqs = 77
        vllm_config.compilation_config.max_cudagraph_capture_size = None
        vllm_config.compilation_config.cudagraph_capture_sizes = None
        vllm_config.compilation_config.mode = CompilationMode.DYNAMO_TRACE_ONCE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        vllm_config.compilation_config.custom_ops = []
        vllm_config.model_config.enforce_eager = False
        vllm_config.model_config.enable_sleep_mode = True
        vllm_config.model_config.is_encoder_decoder = False
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.worker_cls = "manual"
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 1

        self.platform.apply_config_platform_defaults(vllm_config)

        observed_inputs: list[int | None] = []
        vllm_config._set_cudagraph_sizes = MagicMock(
            side_effect=lambda: observed_inputs.append(
                vllm_config.compilation_config.max_cudagraph_capture_size
            )
        )

        self.platform.check_and_update_config(vllm_config)

        self.assertEqual(observed_inputs, [77])

    def test_get_device_capability(self):
        self.assertIsNone(self.platform.get_device_capability(device_id=0))

    @patch("torch.npu.get_device_name")
    def test_get_device_name(self, mock_get_device_name):
        device_id = 0
        device_name = "Ascend910B2"
        mock_get_device_name.return_value = device_name
        self.assertEqual(self.platform.get_device_name(device_id), device_name)
        mock_get_device_name.assert_called_once_with(0)

    @patch("torch.npu.get_device_properties")
    def test_get_device_uuid(self, mock_get_device_properties):
        device_id = 0
        device_properties = MagicMock()
        device_properties.uuid = "01020304-0000-0000-0000-01020304"
        mock_get_device_properties.return_value = device_properties
        self.assertEqual(self.platform.get_device_uuid(device_id), device_properties.uuid)
        mock_get_device_properties.assert_called_once_with(0)        

    @patch("torch.inference_mode")
    def test_inference_mode(self, mock_inference_mode):
        mock_inference_mode.return_value = None
        self.assertIsNone(self.platform.inference_mode())
        mock_inference_mode.assert_called_once()

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.utils.update_aclgraph_sizes")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("os.environ", {})
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_basic_config_update(
        self, mock_init_recompute, mock_soc_version, mock_update_acl, mock_init_ascend, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_no_model_config_warning(
        self, mock_init_recompute, mock_init_ascend, mock_soc_version, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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
            self.platform = platform.NPUPlatform()

            with patch.object(platform.NPUPlatform, "_fix_incompatible_config"):
                self.platform.check_and_update_config(vllm_config)

        self.assertTrue("Model config is missing" in cm.output[0])

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_enforce_eager_mode(self, mock_init_recompute, mock_init_ascend, mock_soc_version, mock_auto_detect):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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
            self.platform = platform.NPUPlatform()

            with patch.object(platform.NPUPlatform, "_fix_incompatible_config"):
                self.platform.check_and_update_config(vllm_config)

        self.assertTrue("Compilation disabled, using eager mode by default" in cm.output[0])

        self.assertEqual(
            vllm_config.compilation_config.mode,
            CompilationMode.NONE,
        )

        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_unsupported_compilation_level(
        self, mock_init_recompute, mock_init_ascend, mock_soc_version, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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
            self.platform = platform.NPUPlatform()

            with patch.object(platform.NPUPlatform, "_fix_incompatible_config"):
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

    @pytest.mark.skip("Revert me when vllm support setting cudagraph_mode on oot platform")
    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_unsupported_cudagraph_mode(self, mock_init_ascend, mock_soc_version, mock_auto_detect):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = False
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL

        with self.assertLogs(logger="vllm", level="INFO") as cm:
            from vllm_ascend import platform

            importlib.reload(platform)
            self.platform.check_and_update_config(vllm_config)
            self.assertTrue("cudagraph_mode is not support on NPU. falling back to NONE" in cm.output[0])

            self.assertEqual(
                vllm_config.compilation_config.mode,
                CompilationMode.NONE,
            )
            self.assertEqual(
                vllm_config.compilation_config.cudagraph_mode,
                CUDAGraphMode.NONE,
            )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_cache_config_block_size(
        self, mock_init_recompute, mock_init_ascend, mock_soc_version, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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

    def test_update_block_size_for_backend_preserves_hybrid_block_size(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.is_hybrid = True
        vllm_config.cache_config.block_size = 1024
        vllm_config.cache_config.user_specified_block_size = False

        self.platform.update_block_size_for_backend(vllm_config)

        self.assertEqual(vllm_config.cache_config.block_size, 1024)

    def test_update_block_size_for_backend_preserves_user_block_size(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.is_hybrid = False
        vllm_config.cache_config.block_size = 512
        vllm_config.cache_config.user_specified_block_size = True

        self.platform.update_block_size_for_backend(vllm_config)

        self.assertEqual(vllm_config.cache_config.block_size, 512)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_v1_worker_class_selection(
        self, mock_init_recompute, mock_init_ascend, mock_soc_version, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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
            "vllm_ascend.worker.worker.NPUWorker",
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

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType._310P)
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_310p_no_custom_ops(self, mock_init_recompute, mock_soc_version, mock_init_ascend, mock_auto_detect):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
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

    def test_get_attn_backend_cls_use_v1_and_mla(self):
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=True,
            use_sparse=False,
        )
        result = self.platform.get_attn_backend_cls("ascend", attn_selector_config)
        self.assertEqual(result, "vllm_ascend.attention.mla_v1.AscendMLABackend")

    def test_get_attn_backend_cls_use_v1_only(self):
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=False,
            use_sparse=False,
        )
        result = self.platform.get_attn_backend_cls("ascend", attn_selector_config)
        self.assertEqual(result, "vllm_ascend.attention.attention_v1.AscendAttentionBackend")

    def test_get_punica_wrapper(self):
        result = self.platform.get_punica_wrapper()

        self.assertEqual(result, "vllm_ascend.lora.punica_npu.PunicaWrapperNPU")

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_specific_device(self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result
        test_device = torch.device("npu:0")
        result = self.platform.get_current_memory_usage(device=test_device)

        mock_reset_stats.assert_called_once_with(test_device)
        mock_max_memory.assert_called_once_with(test_device)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("torch.npu.reset_peak_memory_stats")
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_default_device(self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result

        result = self.platform.get_current_memory_usage()

        mock_reset_stats.assert_called_once_with(None)
        mock_max_memory.assert_called_once_with(None)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("torch.npu.reset_peak_memory_stats", side_effect=RuntimeError("Device error"))
    @patch("torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_when_reset_stats_fails(self, mock_max_memory, mock_reset_stats):
        with self.assertRaises(RuntimeError):
            self.platform.get_current_memory_usage()
        mock_reset_stats.assert_called_once()
        mock_max_memory.assert_not_called()

    @patch("torch.npu.reset_peak_memory_stats")
    @patch(
        "torch.npu.max_memory_allocated",
        side_effect=RuntimeError("Memory query failed"),
    )
    def test_get_current_memory_usage_when_query_fails(self, mock_max_memory, mock_reset_stats):
        with self.assertRaises(RuntimeError):
            self.platform.get_current_memory_usage()
        mock_reset_stats.assert_called_once()
        mock_max_memory.assert_called_once()

    def test_get_device_communicator_cls_returns_correct_value(self):
        self.assertEqual(
            self.platform.get_device_communicator_cls(),
            "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator",
        )

    def test_is_pin_memory_available_returns_true(self):
        self.assertTrue(self.platform.is_pin_memory_available())

    def test_get_static_graph_wrapper_cls_returns_correct_value(self):
        self.assertEqual(
            self.platform.get_static_graph_wrapper_cls(),
            "vllm_ascend.compilation.acl_graph.ACLGraphWrapper",
        )
