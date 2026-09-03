import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.platforms import PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig  # type: ignore

from tests.ut.base import TestBase
from vllm_ascend.ascend_forward_context import MoECommType, override_mrv2_in_profile_run
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import (
    NPUPlatform,
    _setup_compile_backend,
    _validate_eplb_config,
    _validate_sfa_dcp_kv_sp,
)
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD


class TestNPUPlatform(TestBase):
    @staticmethod
    def mock_vllm_config():
        mock_vllm_config = MagicMock()
        mock_vllm_config.compilation_config = MagicMock()
        mock_vllm_config.model_config = MagicMock()
        mock_vllm_config.model_config.is_hybrid = False
        mock_vllm_config.model_config.is_encoder_decoder = False
        mock_vllm_config.device_config = MagicMock()
        mock_vllm_config.device_config.device_type = "npu"
        mock_vllm_config.parallel_config = MagicMock()
        mock_vllm_config.parallel_config.data_parallel_size = 1
        mock_vllm_config.parallel_config.prefill_context_parallel_size = 1
        mock_vllm_config.parallel_config.tensor_parallel_size = 1
        mock_vllm_config.parallel_config.pipeline_parallel_size = 1
        mock_vllm_config.parallel_config.context_parallel_size = 1
        mock_vllm_config.parallel_config.decode_context_parallel_size = 1
        mock_vllm_config.parallel_config.nnodes_within_dp = 1
        mock_vllm_config.use_v2_model_runner = False
        mock_vllm_config.parallel_config.enable_eplb = False
        mock_vllm_config.parallel_config.enable_elastic_ep = False
        mock_vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=True,
            communicator=None,
        )
        mock_vllm_config.cache_config = MagicMock()
        mock_vllm_config.scheduler_config = MagicMock()
        mock_vllm_config.scheduler_config.max_num_seqs = None
        mock_vllm_config.scheduler_config.policy = "fcfs"
        mock_vllm_config.scheduler_config.async_scheduling = False
        mock_vllm_config.scheduler_config.scheduler_cls = None
        mock_vllm_config.speculative_config = None
        mock_vllm_config.kv_transfer_config = None
        mock_vllm_config.additional_config = {}
        mock_vllm_config.compilation_config.pass_config.enable_sp = False
        mock_vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        return mock_vllm_config

    @staticmethod
    def mock_vllm_ascend_config():
        mock_ascend_config = MagicMock()
        mock_ascend_config.xlite_graph_config.enabled = False
        mock_ascend_config.xlite_graph_config.full_mode = False
        mock_ascend_config.ascend_compilation_config.enable_npugraph_ex = False
        mock_ascend_config.ascend_fusion_config = None
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = False
        mock_ascend_config.scheduler_config.enable_balance_scheduling = False
        mock_ascend_config.scheduler_config.batch_job_sched_config.enabled = False
        mock_ascend_config.mc2_comm_alg = ""
        mock_ascend_config.enable_fused_mc2 = False
        mock_ascend_config.enable_shared_expert_dp = False
        mock_ascend_config.scheduler_config.short_request_first_config.enabled = False
        mock_ascend_config.scheduler_config.profiling_chunk_config.enabled = False
        mock_ascend_config.scheduler_config.dyntra_lb_config.enabled = False
        mock_ascend_config.update_compile_ranges_split_points = MagicMock()
        return mock_ascend_config

    def setUp(self):
        self._enable_sp_patch = patch("vllm_ascend.utils.enable_sp", return_value=False)
        self._enable_sp_patch.start()
        self.addCleanup(self._enable_sp_patch.stop)
        self.platform = NPUPlatform()
        self.platform.supported_quantization[:] = ["ascend", "compressed-tensors"]

    def test_class_variables(self):
        self.assertEqual(NPUPlatform._enum, PlatformEnum.OOT)
        self.assertEqual(NPUPlatform.device_name, "npu")
        self.assertEqual(NPUPlatform.device_type, "npu")
        self.assertEqual(NPUPlatform.simple_compile_backend, "eager")
        self.assertEqual(NPUPlatform.ray_device_key, "NPU")
        self.assertEqual(NPUPlatform.device_control_env_var, "ASCEND_RT_VISIBLE_DEVICES")

    @patch("vllm_ascend.platform.enable_sp", return_value=False)
    @patch("vllm_ascend.platform.enable_sfa_dcp_replicated_indexer", return_value=True)
    @patch("vllm_ascend.platform.model_uses_sfa_sparse", return_value=True)
    def test_sfa_dcp_replicated_indexer_aligns_interleave_to_block_size(
        self,
        _mock_sparse,
        _mock_replicated_indexer,
        _mock_enable_sp,
    ):
        vllm_config = self.mock_vllm_config()
        vllm_config.parallel_config.decode_context_parallel_size = 8
        vllm_config.parallel_config.tensor_parallel_size = 8
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 128

        with patch("vllm_ascend.platform.logger.warning_once") as mock_warning:
            _validate_sfa_dcp_kv_sp(vllm_config)

        self.assertEqual(
            vllm_config.parallel_config.cp_kv_cache_interleave_size,
            vllm_config.cache_config.block_size,
        )
        mock_warning.assert_called_once()
        self.assertIn(
            "Override cp_kv_cache_interleave_size to 128",
            mock_warning.call_args.args[0],
        )

    def test_validate_eplb_config_allows_v2_load_collection_phase(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.additional_config = {"eplb_config": {"load_collection_phase": "prefill"}}

        with patch.dict("os.environ", {}, clear=True):
            _validate_eplb_config(vllm_config)

        self.assertIsNone(vllm_config.parallel_config.eplb_config.communicator)

    def test_validate_eplb_config_warns_and_forces_async_mode(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=False,
            communicator="torch_gloo",
        )

        with patch.dict("os.environ", {}, clear=True), patch("vllm_ascend.platform.logger.warning") as warning:
            _validate_eplb_config(vllm_config)

        self.assertTrue(vllm_config.parallel_config.eplb_config.use_async)
        self.assertEqual(vllm_config.parallel_config.eplb_config.communicator, "torch_gloo")
        warning.assert_called_once()

    def test_validate_eplb_config_rejects_nccl_before_sync_normalization(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=False,
            communicator="torch_nccl",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "torch_gloo"),
        ):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_async_keeps_gloo_auto_select(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.enable_elastic_ep = False
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=True,
            communicator=None,
        )

        with patch.dict("os.environ", {}, clear=True):
            _validate_eplb_config(vllm_config)

        self.assertIsNone(
            vllm_config.parallel_config.eplb_config.communicator,
        )

    def test_validate_eplb_config_async_rejects_elastic_ep(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.enable_elastic_ep = True
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=True,
            communicator=None,
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "elastic EP"),
        ):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_async_rejects_nccl_communicator(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.enable_elastic_ep = False
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=True,
            communicator="torch_nccl",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "torch_gloo"),
        ):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_async_rejects_nixl_communicator(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.enable_elastic_ep = False
        vllm_config.parallel_config.eplb_config = MagicMock(
            use_async=True,
            communicator="nixl",
        )

        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "got 'nixl'"),
        ):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_allows_load_collection_phase_with_dbo_and_spec_decode(
        self,
    ):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.enable_eplb = True
        vllm_config.parallel_config.enable_dbo = True
        vllm_config.speculative_config = object()
        vllm_config.additional_config = {"eplb_config": {"load_collection_phase": "decode"}}

        with patch.dict("os.environ", {}, clear=True):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_rejects_v2_legacy_fields(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.additional_config = {"eplb_config": {"dynamic_eplb": True}}

        with self.assertRaisesRegex(ValueError, "legacy fields are not supported: dynamic_eplb"):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_rejects_v1_load_collection_phase(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = False
        vllm_config.additional_config = {"eplb_config": {"load_collection_phase": "decode"}}

        with self.assertRaisesRegex(ValueError, "only supported by Model Runner V2"):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_rejects_upstream_eplb_on_v1(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = False
        vllm_config.parallel_config.enable_eplb = True

        with self.assertRaisesRegex(ValueError, "only supported by Model Runner V2"):
            _validate_eplb_config(vllm_config)

    def test_validate_eplb_config_rejects_v2_legacy_environment(self):
        vllm_config = self.mock_vllm_config()
        vllm_config.use_v2_model_runner = True

        with (
            patch.dict("os.environ", {"DYNAMIC_EPLB": "1"}, clear=True),
            self.assertRaisesRegex(ValueError, "Model Runner V1 controls"),
        ):
            _validate_eplb_config(vllm_config)
        self.assertEqual(NPUPlatform.dispatch_key, "PrivateUse1")
        self.assertEqual(NPUPlatform.supported_quantization, [ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD])

    def test_get_recompute_scheduler_cls(self):
        from vllm_ascend import platform

        cases = (
            (
                False,
                False,
                "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
            ),
            (
                True,
                False,
                "vllm_ascend.core.recompute_scheduler.AsyncRecomputeScheduler",
            ),
            (
                False,
                True,
                "vllm_ascend.core.recompute_scheduler.DyntraLBRecomputeScheduler",
            ),
            (
                True,
                True,
                "vllm_ascend.core.recompute_scheduler.AsyncDyntraLBRecomputeScheduler",
            ),
        )
        for async_scheduling, dyntra_lb_enabled, expected_scheduler_cls in cases:
            with self.subTest(
                async_scheduling=async_scheduling,
                dyntra_lb_enabled=dyntra_lb_enabled,
            ):
                self.assertEqual(
                    platform._get_recompute_scheduler_cls(
                        async_scheduling=async_scheduling,
                        dyntra_lb_enabled=dyntra_lb_enabled,
                    ),
                    expected_scheduler_cls,
                )

    def test_is_sleep_mode_available(self):
        self.assertTrue(self.platform.is_sleep_mode_available())

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.configs.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_with_parser(self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_action = MagicMock()
        mock_action.choices = ["awq", "gptq"]
        mock_parser._option_string_actions = {"--quantization": mock_action}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

        self.assertTrue(ASCEND_QUANTIZATION_METHOD in mock_action.choices)
        self.assertEqual(len(mock_action.choices), 3)  # original 2 + ascend

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.configs.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_without_parser(self, mock_quant_config, mock_adapt_patch):
        self.platform.pre_register_and_update(None)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.configs.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_with_parser_no_quant_action(self, mock_quant_config, mock_adapt_patch):
        mock_parser = MagicMock()
        mock_parser._option_string_actions = {}

        self.platform.pre_register_and_update(mock_parser)

        mock_adapt_patch.assert_called_once_with(is_global_patch=True)

    @patch("vllm_ascend.utils.adapt_patch")
    @patch("vllm_ascend.quantization.configs.modelslim_config.AscendModelSlimConfig")
    def test_pre_register_and_update_with_existing_ascend_quant(self, mock_quant_config, mock_adapt_patch):
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
                vllm_config.speculative_config = MagicMock(num_speculative_tokens=num_speculative_tokens)
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

    def test_validate_indexer_pp_config_rejects_indexshare_partition(self):
        indexer_types = ["full", "full", "full", "shared", "shared", "shared"]
        indexer_types.extend(["full", "shared", "shared", "shared"] * 18)
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=78,
            indexer_types=indexer_types,
        )

        with (
            patch("vllm.envs.VLLM_PP_LAYER_PARTITION", None),
            pytest.raises(ValueError, match="layer 39 uses a shared Indexer"),
        ):
            self.platform._validate_indexer_pp_config(vllm_config)

    def test_validate_indexer_pp_config_accepts_aligned_partition(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.pipeline_parallel_size = 2
        vllm_config.model_config.hf_text_config = SimpleNamespace(
            num_hidden_layers=8,
            indexer_types=["full", "shared", "shared", "shared"] * 2,
        )

        with patch("vllm.envs.VLLM_PP_LAYER_PARTITION", None):
            self.platform._validate_indexer_pp_config(vllm_config)

    def test_validate_indexer_pp_config_rejects_index_cache_partition(self):
        test_cases = (
            (
                2,
                SimpleNamespace(
                    num_hidden_layers=6,
                    use_index_cache=True,
                    index_topk_freq=4,
                    index_skip_topk_offset=3,
                ),
                "layer 3 skips Top-K computation",
            ),
            (
                3,
                SimpleNamespace(
                    num_hidden_layers=6,
                    use_index_cache=True,
                    index_topk_pattern="FFSFFS",
                ),
                "layer 2 skips Top-K computation",
            ),
        )

        for pp_size, hf_text_config, error_match in test_cases:
            with self.subTest(pp_size=pp_size, error_match=error_match):
                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.parallel_config.pipeline_parallel_size = pp_size
                vllm_config.model_config.hf_text_config = hf_text_config

                with (
                    patch("vllm.envs.VLLM_PP_LAYER_PARTITION", None),
                    pytest.raises(ValueError, match=error_match),
                ):
                    self.platform._validate_indexer_pp_config(vllm_config)

    @patch.object(
        NPUPlatform,
        "_validate_indexer_pp_config",
        side_effect=ValueError("invalid Indexer PP partition"),
    )
    def test_check_and_update_config_validates_indexer_before_worker_start(
        self,
        mock_validate_indexer,
    ):
        vllm_config = TestNPUPlatform.mock_vllm_config()

        with pytest.raises(ValueError, match="invalid Indexer PP partition"):
            self.platform.check_and_update_config(vllm_config)

        mock_validate_indexer.assert_called_once_with(vllm_config)

    def test_apply_config_platform_defaults_skips_when_scheduler_max_num_seqs_is_missing(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.max_cudagraph_capture_size = None
        vllm_config.compilation_config.cudagraph_capture_sizes = None

        self.platform.apply_config_platform_defaults(vllm_config)

        self.assertIsNone(vllm_config.compilation_config.max_cudagraph_capture_size)

    @patch("vllm_ascend.platform.refresh_block_size")
    @patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3))
    @patch("vllm_ascend.platform.enable_sp", return_value=False)
    @patch("vllm_ascend.platform.init_ascend_config")
    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    def test_check_and_update_config_preserves_platform_default_max_input(
        self,
        mock_auto_detect,
        mock_init_ascend,
        _mock_enable_sp,
        _mock_device_type,
        _mock_refresh_block_size,
    ):
        ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        ascend_config.enable_dsa_cp = False
        mock_init_ascend.return_value = ascend_config
        vllm_config = TestNPUPlatform.mock_vllm_config()
        # A raw string is truthy in Python, but the platform must pass the
        # validated False value from AscendConfig to the compile backend.
        vllm_config.additional_config = {"enable_dsa_cp": "false"}
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
            side_effect=lambda: observed_inputs.append(vllm_config.compilation_config.max_cudagraph_capture_size)
        )

        with patch("vllm_ascend.platform._setup_compile_backend", wraps=_setup_compile_backend) as mock_setup:
            self.platform.check_and_update_config(vllm_config)

        self.assertEqual(observed_inputs, [77])
        self.assertIs(mock_setup.call_args.kwargs["enable_dsa_cp"], False)

    @patch("vllm_ascend.platform.refresh_block_size")
    @patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A3))
    @patch("vllm_ascend.platform.enable_sp", return_value=True)
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    def test_check_and_update_config_skips_sp_capture_sizes_without_cudagraph(
        self,
        mock_auto_detect,
        mock_init_ascend,
        _mock_enable_sp,
        _mock_device_type,
        _mock_refresh_block_size,
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.compilation_config.mode = CompilationMode.NONE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.compilation_config.cudagraph_capture_sizes = []
        vllm_config.compilation_config.custom_ops = []
        vllm_config.compilation_config.splitting_ops = []
        vllm_config.model_config.enforce_eager = False
        vllm_config.model_config.enable_sleep_mode = True
        vllm_config.model_config.is_encoder_decoder = False
        vllm_config.parallel_config.tensor_parallel_size = 16
        vllm_config.parallel_config.worker_cls = "manual"
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 1
        vllm_config._set_cudagraph_sizes = MagicMock()
        vllm_config.update_sizes_for_sequence_parallelism = MagicMock(return_value=[])

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        self.platform.check_and_update_config(vllm_config)

        vllm_config.update_sizes_for_sequence_parallelism.assert_not_called()

    def test_setup_compile_backend_aligns_tp_token_layout_capture_sizes(self):
        cases = [
            (True, False, False, True),
            (False, True, False, True),
            (False, False, True, True),
            (False, False, False, False),
        ]
        for upstream_sp, enable_shared_expert_dp, enable_dsa_cp, should_align in cases:
            with self.subTest(
                upstream_sp=upstream_sp,
                enable_shared_expert_dp=enable_shared_expert_dp,
                enable_dsa_cp=enable_dsa_cp,
            ):
                vllm_config = TestNPUPlatform.mock_vllm_config()
                compilation_config = vllm_config.compilation_config
                compilation_config.mode = CompilationMode.VLLM_COMPILE
                compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
                compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8]
                compilation_config.max_cudagraph_capture_size = 8
                compilation_config.splitting_ops = []
                vllm_config.additional_config = {
                    "ascend_compilation_config": {
                        "enable_npugraph_ex": True,
                        "enable_static_kernel": True,
                    }
                }
                vllm_config.model_config.enforce_eager = False
                vllm_config.parallel_config.tensor_parallel_size = 8
                vllm_config._set_cudagraph_sizes = MagicMock()
                vllm_config.update_sizes_for_sequence_parallelism = MagicMock(return_value=[8])

                with patch("vllm_ascend.platform.enable_sp", return_value=upstream_sp):
                    _setup_compile_backend(
                        vllm_config,
                        compile_backend="test_backend",
                        enable_shared_expert_dp=enable_shared_expert_dp,
                        enable_dsa_cp=enable_dsa_cp,
                    )

                if should_align:
                    vllm_config.update_sizes_for_sequence_parallelism.assert_called_once_with([1, 2, 4, 8])
                    self.assertEqual(compilation_config.cudagraph_capture_sizes, [8])
                    self.assertEqual(compilation_config.max_cudagraph_capture_size, 8)
                else:
                    vllm_config.update_sizes_for_sequence_parallelism.assert_not_called()
                    self.assertEqual(compilation_config.cudagraph_capture_sizes, [1, 2, 4, 8])

    def test_setup_compile_backend_honors_cudagraph_capture_size_limit(self):
        cases = (
            ([1, 2, 4, 8, 16, 32], [1, 4, 8, 32], True),
            ([1, 2, 4, 8], [1, 2, 4, 8], False),
        )
        profile = SimpleNamespace(cudagraph_capture_size_limit=4)

        for original_sizes, expected_sizes, should_prune in cases:
            with self.subTest(original_sizes=original_sizes):
                vllm_config = TestNPUPlatform.mock_vllm_config()
                compilation_config = vllm_config.compilation_config
                compilation_config.mode = CompilationMode.VLLM_COMPILE
                compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
                compilation_config.cudagraph_capture_sizes = list(original_sizes)
                compilation_config.max_cudagraph_capture_size = original_sizes[-1]
                compilation_config.splitting_ops = []
                compilation_config.post_init_cudagraph_sizes = MagicMock()
                vllm_config.additional_config = {
                    "ascend_compilation_config": {
                        "enable_npugraph_ex": True,
                        "enable_static_kernel": True,
                    }
                }
                vllm_config.model_config.enforce_eager = False
                vllm_config.parallel_config.tensor_parallel_size = 1
                vllm_config._set_cudagraph_sizes = MagicMock()

                with (
                    patch.dict("os.environ", {}, clear=True),
                    patch("vllm_ascend.platform.enable_sp", return_value=False),
                    patch(
                        "vllm_ascend.platform.get_current_hardware_profile",
                        return_value=profile,
                    ),
                ):
                    _setup_compile_backend(vllm_config, compile_backend="test_backend")

                self.assertEqual(compilation_config.cudagraph_capture_sizes, expected_sizes)
                if should_prune:
                    compilation_config.post_init_cudagraph_sizes.assert_called_once()
                else:
                    compilation_config.post_init_cudagraph_sizes.assert_not_called()

    def test_get_device_capability(self):
        self.assertIsNone(self.platform.get_device_capability(device_id=0))

    @patch("vllm_ascend.platform.torch.npu.get_device_name")
    def test_get_device_name(self, mock_get_device_name):
        device_id = 0
        device_name = "Ascend910B2"
        mock_get_device_name.return_value = device_name
        self.assertEqual(self.platform.get_device_name(device_id), device_name)
        mock_get_device_name.assert_called_once_with(0)

    @patch("vllm_ascend.platform.torch.npu.get_device_properties")
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

    def test_set_additional_forward_context_v2_includes_required_moe_fields(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.prefill_context_parallel_size = 2
        dummy_comm_method = object()

        with (
            patch("vllm_ascend.platform.envs_vllm.VLLM_USE_V2_MODEL_RUNNER", True, create=True),
            patch("vllm_ascend.platform.is_moe_model", return_value=True),
            patch("vllm_ascend.platform.enable_sp", return_value=False),
            patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=4),
            patch("vllm.distributed.get_dp_group", return_value=MagicMock(world_size=1)),
            patch("vllm_ascend.ascend_forward_context.select_moe_comm_method", return_value=MoECommType.ALLGATHER),
            patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=dummy_comm_method),
        ):
            kwargs = self.platform.set_additional_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                dp_metadata=None,
                num_tokens=5,
            )

        self.assertFalse(kwargs["in_profile_run"])
        self.assertEqual(kwargs["padded_num_tokens"], 8)
        self.assertEqual(kwargs["max_tokens_across_pcp"], 5)
        self.assertIs(kwargs["moe_comm_method"], dummy_comm_method)
        self.assertEqual(kwargs["dynamic_mx_quant_scale_alg"], 0)

    def test_set_additional_forward_context_v1_includes_dynamic_mx_scale_alg(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.use_v2_model_runner = False

        with patch(
            "vllm_ascend.quantization.utils.get_dynamic_mx_quant_scale_alg",
            return_value=1,
        ):
            kwargs = self.platform.set_additional_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                dp_metadata=None,
                num_tokens=5,
            )

        self.assertEqual(kwargs, {"dynamic_mx_quant_scale_alg": 1})

    def test_set_additional_forward_context_reads_v2_profile_override(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.use_v2_model_runner = True

        with (
            patch("vllm_ascend.platform.envs_vllm.VLLM_USE_V2_MODEL_RUNNER", True, create=True),
            patch("vllm_ascend.platform.is_moe_model", return_value=True),
            patch("vllm_ascend.platform.enable_sp", return_value=False),
            patch("vllm.distributed.get_tensor_model_parallel_world_size", return_value=2),
            patch("vllm.distributed.get_dp_group", return_value=MagicMock(world_size=1)),
            patch("vllm_ascend.ascend_forward_context.select_moe_comm_method", return_value=MoECommType.ALLGATHER),
            patch("vllm_ascend.ascend_forward_context.get_mc2_mask", return_value=None),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_moe_comm_method", return_value=object()),
            override_mrv2_in_profile_run(True),
        ):
            kwargs = self.platform.set_additional_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                dp_metadata=None,
                num_tokens=4,
            )

        self.assertTrue(kwargs["in_profile_run"])

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("os.environ", {})
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_basic_config_update(
        self, mock_init_recompute, _mock_hardware_profile, mock_init_ascend, mock_auto_detect
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
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_no_model_config_warning(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config = None
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            self.assertLogs(logger="vllm", level="INFO") as cm,
            patch.object(platform, "_fix_incompatible_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

        self.assertTrue("Model config is missing" in cm.output[0])

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_enforce_eager_mode(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_init_ascend.return_value = TestNPUPlatform.mock_vllm_ascend_config()
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.enforce_eager = True
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        mock_init_recompute.return_value = MagicMock()
        vllm_config.scheduler_config = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            self.assertLogs(logger="vllm", level="INFO") as cm,
            patch.object(platform, "_fix_incompatible_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

        self.assertTrue(
            any("Compilation disabled, using eager mode by default" in log for log in cm.output),
            cm.output,
        )

        self.assertEqual(
            vllm_config.compilation_config.mode,
            CompilationMode.NONE,
        )

        self.assertEqual(
            vllm_config.compilation_config.cudagraph_mode,
            CUDAGraphMode.NONE,
        )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_unsupported_compilation_level(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
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

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            self.assertLogs(logger="vllm", level="WARNING") as cm,
            patch.object(platform, "_fix_incompatible_config"),
        ):
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
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_unsupported_cudagraph_mode(
        self, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
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
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_cache_config_block_size(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
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

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_recompute_scheduler_rejects_pd_mixed_no_kv_transfer(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = None
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.scheduler_config = MagicMock()
        mock_init_recompute.return_value = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            pytest.raises(ValueError, match=r"recompute_scheduler_enable.*PD-disaggregated.*PD-mixed"),
            patch.object(platform, "_fix_incompatible_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

        mock_init_recompute.assert_not_called()

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_recompute_scheduler_rejects_pd_mixed_kv_both(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_both", engine_id="engine0")
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.scheduler_config = MagicMock()
        mock_init_recompute.return_value = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            pytest.raises(ValueError, match=r"recompute_scheduler_enable.*PD-disaggregated.*PD-mixed"),
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

        mock_init_recompute.assert_not_called()

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_recompute_scheduler_warns_and_disables_kv_producer(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = True
        mock_ascend_config.scheduler_config.profiling_chunk_config.enabled = False
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_producer", engine_id="engine0")
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        scheduler_config = MagicMock()
        vllm_config.scheduler_config = scheduler_config
        mock_init_recompute.return_value = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
            patch.object(platform.logger, "warning") as mock_warning,
        ):
            self.platform.check_and_update_config(vllm_config)

        mock_init_recompute.assert_not_called()
        self.assertFalse(vllm_config.additional_config["scheduler_config"]["recompute_scheduler_enable"])
        self.assertIs(vllm_config.scheduler_config, scheduler_config)
        mock_warning.assert_called_once()
        self.assertIn(
            "recompute_scheduler_enable is ignored on PD-disaggregated P nodes",
            mock_warning.call_args.args[0],
        )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_recompute_scheduler_accepts_kv_consumer(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = True
        mock_ascend_config.scheduler_config.profiling_chunk_config.enabled = False
        mock_init_ascend.return_value = mock_ascend_config

        recompute_scheduler_config = MagicMock()
        mock_init_recompute.return_value = recompute_scheduler_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_consumer", engine_id="engine0")
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 1
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.async_scheduling = False

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

        mock_init_recompute.assert_called_once_with(vllm_config)
        self.assertIs(vllm_config.scheduler_config, recompute_scheduler_config)
        self.assertEqual(
            recompute_scheduler_config.scheduler_cls,
            "vllm_ascend.core.recompute_scheduler.RecomputeScheduler",
        )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_selects_dyntra_lb_recompute_scheduler(
        self,
        mock_init_recompute,
        mock_init_ascend,
        mock_auto_detect,
    ):
        cases = (
            (
                False,
                "vllm_ascend.core.recompute_scheduler.DyntraLBRecomputeScheduler",
            ),
            (
                True,
                "vllm_ascend.core.recompute_scheduler.AsyncDyntraLBRecomputeScheduler",
            ),
        )
        for async_scheduling, expected_scheduler_cls in cases:
            with self.subTest(async_scheduling=async_scheduling):
                mock_init_recompute.reset_mock()
                mock_init_ascend.reset_mock()

                mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
                mock_ascend_config.scheduler_config.recompute_scheduler_enable = True
                mock_ascend_config.scheduler_config.dyntra_lb_config.enabled = True
                mock_init_ascend.return_value = mock_ascend_config

                recompute_scheduler_config = MagicMock()
                mock_init_recompute.return_value = recompute_scheduler_config

                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.scheduler_config.async_scheduling = async_scheduling
                vllm_config.kv_transfer_config = MagicMock(kv_role="kv_consumer", engine_id="engine0")
                vllm_config.parallel_config.data_parallel_size = 2
                vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
                vllm_config.cache_config.block_size = 1

                from vllm_ascend import platform

                importlib.reload(platform)
                self.platform = platform.NPUPlatform()

                with (
                    patch.object(platform, "_fix_incompatible_config"),
                    patch.object(platform, "check_kv_extra_config"),
                ):
                    self.platform.check_and_update_config(vllm_config)

                mock_init_recompute.assert_called_once_with(vllm_config)
                self.assertIs(vllm_config.scheduler_config, recompute_scheduler_config)
                self.assertEqual(recompute_scheduler_config.scheduler_cls, expected_scheduler_cls)
                self.assertTrue(mock_ascend_config.scheduler_config.recompute_scheduler_enable)
                self.assertTrue(mock_ascend_config.scheduler_config.dyntra_lb_config.enabled)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_short_request_first_rejects_unsupported_combinations(
        self, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        cases = [
            (
                "priority",
                lambda config, ascend: setattr(config.scheduler_config, "policy", "priority"),
                "policy='fcfs'",
            ),
            (
                "batch_job",
                lambda config, ascend: setattr(ascend.scheduler_config.batch_job_sched_config, "enabled", True),
                "batch_job_sched_config",
            ),
            (
                "profiling_chunk",
                lambda config, ascend: setattr(ascend.scheduler_config.profiling_chunk_config, "enabled", True),
                "profiling_chunk_config",
            ),
            (
                "kv_consumer",
                lambda config, ascend: setattr(
                    config, "kv_transfer_config", MagicMock(kv_role="kv_consumer", engine_id="engine0")
                ),
                "PD-disaggregated D nodes",
            ),
        ]
        for name, configure, message in cases:
            with self.subTest(name=name):
                ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
                ascend_config.scheduler_config.short_request_first_config.enabled = True
                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.kv_transfer_config = None
                configure(vllm_config, ascend_config)
                mock_init_ascend.return_value = ascend_config

                with (
                    pytest.raises(ValueError, match=message),
                    patch.object(platform, "_fix_incompatible_config"),
                    patch.object(platform, "check_kv_extra_config"),
                ):
                    self.platform.check_and_update_config(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_short_request_first_accepts_standard_prefill_paths(
        self, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        for kv_role in (None, "kv_producer", "kv_both"):
            with self.subTest(kv_role=kv_role):
                ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
                ascend_config.scheduler_config.short_request_first_config.enabled = True
                mock_init_ascend.return_value = ascend_config
                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.kv_transfer_config = (
                    None if kv_role is None else MagicMock(kv_role=kv_role, engine_id="engine0")
                )

                with (
                    patch.object(platform, "_fix_incompatible_config"),
                    patch.object(platform, "check_kv_extra_config"),
                ):
                    self.platform.check_and_update_config(vllm_config)

                self.assertIsNone(vllm_config.scheduler_config.scheduler_cls)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_short_request_first_selects_async_scheduler(
        self, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        for kv_role in (None, "kv_producer", "kv_both"):
            with self.subTest(kv_role=kv_role):
                ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
                ascend_config.scheduler_config.short_request_first_config.enabled = True
                mock_init_ascend.return_value = ascend_config
                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.scheduler_config.async_scheduling = True
                vllm_config.kv_transfer_config = (
                    None if kv_role is None else MagicMock(kv_role=kv_role, engine_id="engine0")
                )

                with (
                    patch.object(platform, "_fix_incompatible_config"),
                    patch.object(platform, "check_kv_extra_config"),
                ):
                    self.platform.check_and_update_config(vllm_config)

                self.assertEqual(
                    vllm_config.scheduler_config.scheduler_cls,
                    "vllm_ascend.core.short_request_first_scheduler.ShortRequestFirstAsyncScheduler",
                )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_short_request_first_survives_p_recompute_disable(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        ascend_config.scheduler_config.short_request_first_config.enabled = True
        ascend_config.scheduler_config.recompute_scheduler_enable = True
        mock_init_ascend.return_value = ascend_config
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_producer", engine_id="engine0")

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()
        with (
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
            patch.object(platform.logger, "warning") as mock_warning,
        ):
            self.platform.check_and_update_config(vllm_config)

        mock_init_recompute.assert_not_called()
        self.assertFalse(vllm_config.additional_config["scheduler_config"]["recompute_scheduler_enable"])
        self.assertTrue(ascend_config.scheduler_config.short_request_first_config.enabled)
        self.assertIsNone(vllm_config.scheduler_config.scheduler_cls)
        self.assertIn("recompute_scheduler_enable is ignored", mock_warning.call_args.args[0])

    def test_validate_kv_load_failure_policy_rejects_hybrid_recompute(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.model_config.is_hybrid = True
        vllm_config.kv_transfer_config = MagicMock(kv_load_failure_policy="recompute")

        with pytest.raises(AssertionError, match="Hybrid models do not support recompute mode kv load failure policy"):
            from vllm_ascend import platform

            platform._validate_kv_load_failure_policy(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_selects_dyntra_lb_scheduler(
        self, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.dyntra_lb_config.enabled = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_consumer", engine_id="engine0")
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 1
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.async_scheduling = False

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
            patch("vllm_ascend.logger.configure_ascend_file_logging"),
        ):
            self.platform.check_and_update_config(vllm_config)

        self.assertEqual(
            vllm_config.scheduler_config.scheduler_cls,
            "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler",
        )

    def test_get_dyntra_lb_scheduler_cls(self):
        from vllm_ascend import platform

        cases = (
            (
                False,
                "vllm_ascend.core.dyntra_lb_scheduler.DyntraLBScheduler",
            ),
            (
                True,
                "vllm_ascend.core.dyntra_lb_scheduler.AsyncDyntraLBScheduler",
            ),
        )
        for async_scheduling, expected_scheduler_cls in cases:
            with self.subTest(async_scheduling=async_scheduling):
                self.assertEqual(
                    platform._get_dyntra_lb_scheduler_cls(async_scheduling=async_scheduling),
                    expected_scheduler_cls,
                )

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_rejects_dyntra_lb_outside_disaggregated_d_node(
        self, mock_init_ascend, mock_auto_detect
    ):
        for kv_role in (None, "kv_both", "kv_producer"):
            with self.subTest(kv_role=kv_role):
                mock_init_ascend.reset_mock()

                mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
                mock_ascend_config.scheduler_config.dyntra_lb_config.enabled = True
                mock_init_ascend.return_value = mock_ascend_config

                vllm_config = TestNPUPlatform.mock_vllm_config()
                vllm_config.kv_transfer_config = (
                    None if kv_role is None else MagicMock(kv_role=kv_role, engine_id="engine0")
                )
                vllm_config.parallel_config.data_parallel_size = 2
                vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
                vllm_config.cache_config.block_size = 1
                vllm_config.scheduler_config = MagicMock()

                from vllm_ascend import platform

                importlib.reload(platform)
                self.platform = platform.NPUPlatform()

                with (
                    pytest.raises(
                        ValueError,
                        match=r"DyntraLB is only supported on PD-disaggregated D nodes",
                    ),
                    patch.object(platform, "_fix_incompatible_config"),
                    patch.object(platform, "check_kv_extra_config"),
                ):
                    self.platform.check_and_update_config(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    def test_check_and_update_config_rejects_multinode_dyntra_lb_decoder(self, mock_init_ascend, mock_auto_detect):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.dyntra_lb_config.enabled = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_consumer", engine_id="engine0")
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.parallel_config.nnodes_within_dp = 2
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
        vllm_config.cache_config.block_size = 1
        vllm_config.scheduler_config = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            pytest.raises(
                ValueError,
                match=r"DyntraLB only supports decoder instances.*single node",
            ),
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_balance_scheduler_rejects_pd_disaggregated_kv_producer(
        self, mock_init_recompute, mock_init_ascend, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = False
        mock_ascend_config.scheduler_config.enable_balance_scheduling = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_producer", engine_id="engine0")
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.scheduler_config = MagicMock()
        mock_init_recompute.return_value = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            pytest.raises(ValueError, match=r"enable_balance_scheduling.*PD-mixed.*PD-disaggregated"),
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_balance_scheduler_rejects_pd_disaggregated_kv_consumer(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
    ):
        mock_ascend_config = TestNPUPlatform.mock_vllm_ascend_config()
        mock_ascend_config.scheduler_config.recompute_scheduler_enable = False
        mock_ascend_config.scheduler_config.enable_balance_scheduling = True
        mock_init_ascend.return_value = mock_ascend_config

        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.kv_transfer_config = MagicMock(kv_role="kv_consumer", engine_id="engine0")
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.scheduler_config = MagicMock()
        mock_init_recompute.return_value = MagicMock()

        from vllm_ascend import platform

        importlib.reload(platform)
        self.platform = platform.NPUPlatform()

        with (
            pytest.raises(ValueError, match=r"enable_balance_scheduling.*PD-mixed.*PD-disaggregated"),
            patch.object(platform, "_fix_incompatible_config"),
            patch.object(platform, "check_kv_extra_config"),
        ):
            self.platform.check_and_update_config(vllm_config)

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

    def test_validate_parallel_config_rejects_pcp_for_model_runner_v1(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.prefill_context_parallel_size = 2

        with pytest.raises(ValueError, match="Prefill Context Parallel"):
            from vllm_ascend import platform

            platform._validate_parallel_config(vllm_config)

    def test_validate_parallel_config_defers_pcp_validation_to_model_runner_v2(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.use_v2_model_runner = True
        vllm_config.parallel_config.prefill_context_parallel_size = 2

        from vllm_ascend import platform

        platform._validate_parallel_config(vllm_config)

    def test_validate_parallel_config_accepts_dp_only(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.data_parallel_size = 2
        vllm_config.parallel_config.prefill_context_parallel_size = 1

        from vllm_ascend import platform

        platform._validate_parallel_config(vllm_config)

    def test_validate_parallel_config_accepts_neither(self):
        vllm_config = TestNPUPlatform.mock_vllm_config()
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1

        from vllm_ascend import platform

        platform._validate_parallel_config(vllm_config)

    @patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization")
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A3),
    )
    @patch("vllm_ascend.ascend_config.init_ascend_config")
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_v1_worker_class_selection(
        self, mock_init_recompute, mock_init_ascend, _mock_hardware_profile, mock_auto_detect
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
    @patch(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType._310P),
    )
    @patch("vllm_ascend.core.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config")
    def test_check_and_update_config_310p_no_custom_ops(
        self, mock_init_recompute, _mock_hardware_profile, mock_init_ascend, mock_auto_detect
    ):
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

    @patch("vllm_ascend.platform.get_ascend_config")
    def test_get_attn_backend_cls_use_v1_and_mla(self, mock_get_ascend_config):
        mock_get_ascend_config.return_value.rl_config.enabled = False
        mock_get_ascend_config.return_value.rl_config.enable_training_consistency = False
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

    @patch("vllm_ascend.platform.get_ascend_config")
    def test_get_attn_backend_cls_use_v1_only(self, mock_get_ascend_config):
        mock_get_ascend_config.return_value.rl_config.enabled = False
        mock_get_ascend_config.return_value.rl_config.enable_training_consistency = False
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

    @patch("vllm_ascend.platform.get_ascend_config")
    def test_get_attn_backend_cls_selects_cp_backend(self, mock_get_ascend_config):
        mock_get_ascend_config.return_value.rl_config.enabled = False
        mock_get_ascend_config.return_value.rl_config.enable_training_consistency = False
        cases = (
            (
                True,
                True,
                False,
                "vllm_ascend.attention.mla_v1.AscendMLABackend",
            ),
            (
                False,
                True,
                False,
                "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            ),
            (
                True,
                False,
                True,
                "vllm_ascend.attention.mla_v1.AscendMLABackend",
            ),
            (
                False,
                False,
                True,
                "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            ),
        )
        for use_mla, use_pcp, use_dcp, expected_backend in cases:
            with self.subTest(use_mla=use_mla, use_pcp=use_pcp, use_dcp=use_dcp):
                attn_selector_config = AttentionSelectorConfig(
                    dtype=torch.float16,
                    head_size=0,
                    kv_cache_dtype=None,
                    block_size=128,
                    use_mla=use_mla,
                    use_sparse=False,
                    use_pcp=use_pcp,
                    use_dcp=use_dcp,
                )
                result = self.platform.get_attn_backend_cls("ascend", attn_selector_config)
                self.assertEqual(result, expected_backend)

    def test_get_attn_backend_cls_rejects_pcp_and_dcp(self):
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_pcp=True,
            use_dcp=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "does not support PCP and DCP simultaneously"):
            self.platform.get_attn_backend_cls("ascend", attn_selector_config)

    def test_get_attn_backend_cls_selects_sfa_pcp_backend(self):
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=True,
            use_sparse=True,
            use_pcp=True,
        )
        result = self.platform.get_attn_backend_cls("ascend", attn_selector_config)
        self.assertEqual(result, "vllm_ascend.attention.sfa_v1.AscendSFABackend")

    def test_get_attn_backend_cls_rejects_unsupported_pcp_backend(self):
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=False,
            use_sparse=True,
            use_pcp=True,
        )
        with self.assertRaisesRegex(NotImplementedError, "PCP does not support attention backend"):
            self.platform.get_attn_backend_cls("ascend", attn_selector_config)

    @patch("vllm_ascend.platform.import_module")
    @patch("vllm_ascend.platform.util.find_spec", return_value=object())
    @patch("vllm_ascend.platform.get_ascend_config")
    def test_get_attn_backend_cls_fa3_uses_training_consistency_without_selected_backend(
        self, mock_get_ascend_config, mock_find_spec, mock_import_module
    ):
        mock_get_ascend_config.return_value.rl_config.enabled = True
        mock_get_ascend_config.return_value.rl_config.enable_training_consistency = True
        mock_import_module.return_value.flash_attn_with_kvcache = MagicMock()
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=False,
            use_sparse=False,
        )

        result = self.platform.get_attn_backend_cls(None, attn_selector_config)

        self.assertEqual(result, "vllm_ascend.attention.fa3_v1.AscendFABackend")
        mock_find_spec.assert_called_once_with("flash_attn_npu_v3")

    @patch("vllm_ascend.platform.get_ascend_config")
    def test_get_attn_backend_cls_fa3_disabled_without_training_consistency(self, mock_get_ascend_config):
        mock_get_ascend_config.return_value.rl_config.enabled = True
        mock_get_ascend_config.return_value.rl_config.enable_training_consistency = False
        attn_selector_config = AttentionSelectorConfig(
            dtype=torch.float16,
            head_size=0,
            kv_cache_dtype=None,
            block_size=128,
            use_mla=False,
            use_sparse=False,
            use_batch_invariant=True,
        )

        result = self.platform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN, attn_selector_config)

        self.assertEqual(result, "vllm_ascend.attention.attention_v1.AscendAttentionBackend")

    def test_get_punica_wrapper(self):
        result = self.platform.get_punica_wrapper()

        self.assertEqual(result, "vllm_ascend.lora.punica_npu.PunicaWrapperNPU")

    @patch("vllm_ascend.platform.torch.npu.reset_peak_memory_stats")
    @patch("vllm_ascend.platform.torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_specific_device(self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result
        # Avoid constructing torch.device("npu:0") on CPU-only runners.
        test_device = MagicMock(name="npu:0")
        result = self.platform.get_current_memory_usage(device=test_device)

        mock_reset_stats.assert_called_once_with(test_device)
        mock_max_memory.assert_called_once_with(test_device)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("vllm_ascend.platform.torch.npu.reset_peak_memory_stats")
    @patch("vllm_ascend.platform.torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_with_default_device(self, mock_max_memory, mock_reset_stats):
        max_memory_allocated_result = 1024.0
        mock_max_memory.return_value = max_memory_allocated_result

        result = self.platform.get_current_memory_usage()

        mock_reset_stats.assert_called_once_with(None)
        mock_max_memory.assert_called_once_with(None)
        self.assertEqual(result, max_memory_allocated_result)

    @patch("vllm_ascend.platform.torch.npu.reset_peak_memory_stats", side_effect=RuntimeError("Device error"))
    @patch("vllm_ascend.platform.torch.npu.max_memory_allocated")
    def test_get_current_memory_usage_when_reset_stats_fails(self, mock_max_memory, mock_reset_stats):
        with self.assertRaises(RuntimeError):
            self.platform.get_current_memory_usage()
        mock_reset_stats.assert_called_once()
        mock_max_memory.assert_not_called()

    @patch("vllm_ascend.platform.torch.npu.reset_peak_memory_stats")
    @patch(
        "vllm_ascend.platform.torch.npu.max_memory_allocated",
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
