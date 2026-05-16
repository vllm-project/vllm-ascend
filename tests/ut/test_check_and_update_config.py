import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
from vllm.config.compilation import CompilationMode, CUDAGraphMode

from tests.ut.base import PytestBase


class TestCheckAndUpdateConfigPartial(PytestBase):
    """Tests for check_and_update_config method (lines 285-345)"""

    @staticmethod
    def create_mock_configs():
        """Create mock configs for testing"""
        vllm_config = MagicMock()
        vllm_config.compilation_config = MagicMock()
        vllm_config.model_config = MagicMock()
        vllm_config.parallel_config = MagicMock()
        vllm_config.parallel_config.all2all_backend = "none"
        vllm_config.parallel_config.data_parallel_size = 1
        vllm_config.parallel_config.decode_context_parallel_size = 1
        vllm_config.parallel_config.prefill_context_parallel_size = 1
        vllm_config.parallel_config.tensor_parallel_size = 1
        vllm_config.parallel_config.worker_cls = "manual"
        vllm_config.parallel_config.cp_kv_cache_interleave_size = 16
        vllm_config.cache_config = MagicMock()
        vllm_config.cache_config.block_size = 16
        vllm_config.kv_transfer_config = None
        vllm_config.speculative_config = None
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.max_num_seqs = None
        vllm_config.additional_config = {}
        vllm_config._set_cudagraph_sizes = MagicMock()

        ascend_config = MagicMock()
        ascend_config.xlite_graph_config = MagicMock()
        ascend_config.xlite_graph_config.enabled = False
        ascend_config.xlite_graph_config.full_mode = False
        ascend_config.ascend_compilation_config = MagicMock()
        ascend_config.ascend_compilation_config.enable_npugraph_ex = True
        ascend_config.ascend_fusion_config = None
        ascend_config.recompute_scheduler_enable = False
        ascend_config.SLO_limits_for_dynamic_batch = -1
        ascend_config.enable_mc2_hierarchy_comm = False
        ascend_config.update_compile_ranges_split_points = MagicMock()

        return vllm_config, ascend_config

    def _get_patches(self):
        """Get common patches for tests"""
        return [
            patch("vllm_ascend.platform.init_ascend_config"),
            patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization"),
            patch("vllm_ascend.platform.refresh_block_size"),
            patch("vllm_ascend.platform.get_ascend_device_type"),
            patch("vllm_ascend.platform.enable_sp", return_value=False),
            patch("vllm_ascend.platform.is_moe_model", return_value=False),
        ]

    def _setup_basic_config(self, vllm_config, ascend_config, mock_device_type):
        """Setup basic configuration for tests"""
        from vllm_ascend.utils import AscendDeviceType

        mock_device_type.return_value = AscendDeviceType.A3
        vllm_config.model_config.enforce_eager = False
        vllm_config.model_config.enable_sleep_mode = True
        vllm_config.model_config.is_encoder_decoder = False
        vllm_config.compilation_config.mode = CompilationMode.VLLM_COMPILE
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        vllm_config.compilation_config.splitting_ops = []
        vllm_config.compilation_config.pass_config = MagicMock()
        vllm_config.compilation_config.pass_config.enable_sp = False
        return ascend_config

    def _enter_patches(self, stack, patches):
        """Enter all patches and return mock objects"""
        mocks = [stack.enter_context(p) for p in patches]
        return mocks[0], mocks

    # ==================== Xlite Graph Config Tests ====================

    @pytest.mark.parametrize("xlite_enabled,xlite_full_mode,splitting_ops,expected_cudagraph_mode,expected_splitting_ops,expected_enforce_eager,log_message", [
        (True, True, None, CUDAGraphMode.NONE, [], True, "ACLGraph is disabled under xlite full mode"),
        (True, True, ["existing_op"], CUDAGraphMode.NONE, ["existing_op"], True, "ACLGraph is disabled under xlite full mode"),
        (True, False, [], CUDAGraphMode.FULL_DECODE_ONLY, [], False, "Falling back to FULL_DECODE_ONLY"),
        (False, False, [], CUDAGraphMode.FULL_DECODE_ONLY, [], False, None),
    ])
    def test_xlite_graph_config(
        self, xlite_enabled, xlite_full_mode, splitting_ops,
        expected_cudagraph_mode, expected_splitting_ops, expected_enforce_eager, log_message
    ):
        """Test xlite graph config scenarios"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes"):
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    ascend_config.xlite_graph_config.enabled = xlite_enabled
                    ascend_config.xlite_graph_config.full_mode = xlite_full_mode
                    vllm_config.compilation_config.splitting_ops = splitting_ops
                    if expected_cudagraph_mode != CUDAGraphMode.NONE:
                        vllm_config.compilation_config.cudagraph_mode = expected_cudagraph_mode
                    mock_init.return_value = ascend_config

                    if log_message:
                        with patch("vllm_ascend.platform.logger") as mock_logger:
                            NPUPlatform.check_and_update_config(vllm_config)
                            assert any(log_message in str(call) for call in mock_logger.info.call_args_list)
                    else:
                        NPUPlatform.check_and_update_config(vllm_config)

                    assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode
                    assert vllm_config.compilation_config.splitting_ops == expected_splitting_ops
                    assert vllm_config.model_config.enforce_eager == expected_enforce_eager

    # ==================== Compilation Mode Validation Tests ====================

    @pytest.mark.parametrize("compilation_mode,expected_cudagraph_mode,should_warn", [
        (CompilationMode.DYNAMO_TRACE_ONCE, CUDAGraphMode.NONE, True),
        (CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.NONE, True),
        (CompilationMode.NONE, CUDAGraphMode.NONE, False),
        (CompilationMode.VLLM_COMPILE, CUDAGraphMode.NONE, False),
    ])
    def test_compilation_mode_validation(self, compilation_mode, expected_cudagraph_mode, should_warn):
        """Test compilation mode validation"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes"):
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    vllm_config.compilation_config.mode = compilation_mode
                    mock_init.return_value = ascend_config

                    if should_warn:
                        with patch("vllm_ascend.platform.logger") as mock_logger:
                            NPUPlatform.check_and_update_config(vllm_config)
                            assert any("NPU does not support" in str(call) for call in mock_logger.warning.call_args_list)
                    else:
                        NPUPlatform.check_and_update_config(vllm_config)

                    assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode

    # ==================== Encoder-Decoder Model Tests ====================

    @pytest.mark.parametrize("is_encoder_decoder,cudagraph_mode,expected_cudagraph_mode,should_warn", [
        (True, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.PIECEWISE, True),
        (True, CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE, False),
        (True, CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, False),
        (True, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE, False),
        (False, CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_DECODE_ONLY, False),
    ])
    def test_encoder_decoder_model(self, is_encoder_decoder, cudagraph_mode, expected_cudagraph_mode, should_warn):
        """Test encoder-decoder model scenarios"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes"):
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    vllm_config.model_config.is_encoder_decoder = is_encoder_decoder
                    vllm_config.compilation_config.cudagraph_mode = cudagraph_mode
                    if cudagraph_mode == CUDAGraphMode.PIECEWISE:
                        vllm_config.compilation_config.splitting_ops = []
                        vllm_config.compilation_config.set_splitting_ops_for_v1 = MagicMock()
                    else:
                        vllm_config.compilation_config.splitting_ops = []
                    mock_init.return_value = ascend_config

                    if should_warn:
                        with patch("vllm_ascend.platform.logger") as mock_logger:
                            NPUPlatform.check_and_update_config(vllm_config)
                            assert any("encoder-decoder model doesn't support FULL_DECODE_ONLY" in str(call)
                                       for call in mock_logger.warning.call_args_list)
                    else:
                        NPUPlatform.check_and_update_config(vllm_config)

                    assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode

    # ==================== CUDAGraph Mode Tests ====================

    @pytest.mark.parametrize("cudagraph_mode,compilation_mode,splitting_ops,expected_mode,expected_splitting_ops,expected_npugraph_ex,should_raise,log_message", [
        (CUDAGraphMode.NONE, CompilationMode.VLLM_COMPILE, [], CompilationMode.NONE, [], False, False, None),
        (CUDAGraphMode.PIECEWISE, CompilationMode.VLLM_COMPILE, [], CompilationMode.VLLM_COMPILE, ["vllm::mla_forward"], False, False, "PIECEWISE compilation enabled"),
        (CUDAGraphMode.PIECEWISE, CompilationMode.VLLM_COMPILE, ["existing_op"], CompilationMode.VLLM_COMPILE, ["existing_op", "vllm::mla_forward"], False, False, "PIECEWISE compilation enabled"),
        (CUDAGraphMode.PIECEWISE, CompilationMode.NONE, [], None, None, None, True, None),
        (CUDAGraphMode.FULL_DECODE_ONLY, CompilationMode.VLLM_COMPILE, ["existing_op"], CompilationMode.VLLM_COMPILE, [], None, False, "FULL_DECODE_ONLY compilation enabled"),
        (CUDAGraphMode.FULL, CompilationMode.VLLM_COMPILE, ["existing_op"], CompilationMode.VLLM_COMPILE, [], None, False, "FULL_DECODE_ONLY compilation enabled"),
    ])
    def test_cudagraph_mode(
        self, cudagraph_mode, compilation_mode, splitting_ops,
        expected_mode, expected_splitting_ops, expected_npugraph_ex,
        should_raise, log_message
    ):
        """Test CUDAGraph mode scenarios"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes"):
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    vllm_config.compilation_config.cudagraph_mode = cudagraph_mode
                    vllm_config.compilation_config.mode = compilation_mode
                    vllm_config.compilation_config.splitting_ops = splitting_ops.copy() if splitting_ops else []
                    if cudagraph_mode == CUDAGraphMode.PIECEWISE:
                        vllm_config.compilation_config.set_splitting_ops_for_v1 = MagicMock()
                    mock_init.return_value = ascend_config

                    if should_raise:
                        with pytest.raises(AssertionError) as exc_info:
                            NPUPlatform.check_and_update_config(vllm_config)
                        assert "When enabling VLLM_COMPILE aclgraph" in str(exc_info.value)
                    else:
                        if log_message:
                            with patch("vllm_ascend.platform.logger") as mock_logger:
                                NPUPlatform.check_and_update_config(vllm_config)
                                assert any(log_message in str(call) for call in mock_logger.info.call_args_list)
                        else:
                            NPUPlatform.check_and_update_config(vllm_config)

                        if expected_mode is not None:
                            assert vllm_config.compilation_config.mode == expected_mode
                        if expected_splitting_ops is not None:
                            assert vllm_config.compilation_config.splitting_ops == expected_splitting_ops
                        if expected_npugraph_ex is not None:
                            assert ascend_config.ascend_compilation_config.enable_npugraph_ex == expected_npugraph_ex

    @pytest.mark.parametrize("cudagraph_mode,should_call_update", [
        (CUDAGraphMode.PIECEWISE, True),
        (CUDAGraphMode.FULL_DECODE_ONLY, False),
        (CUDAGraphMode.FULL, False),
        (CUDAGraphMode.NONE, False),
    ])
    def test_update_aclgraph_sizes_call(self, cudagraph_mode, should_call_update):
        """Test update_aclgraph_sizes is called only for PIECEWISE mode"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes") as mock_update:
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    vllm_config.compilation_config.cudagraph_mode = cudagraph_mode
                    vllm_config.compilation_config.splitting_ops = []
                    if cudagraph_mode == CUDAGraphMode.PIECEWISE:
                        vllm_config.compilation_config.set_splitting_ops_for_v1 = MagicMock()
                    mock_init.return_value = ascend_config

                    NPUPlatform.check_and_update_config(vllm_config)

                    if should_call_update:
                        mock_update.assert_called_once_with(vllm_config)
                    else:
                        mock_update.assert_not_called()

    # ==================== Integration Tests ====================

    @pytest.mark.parametrize("scenario,xlite_enabled,xlite_full_mode,compilation_mode,cudagraph_mode,is_encoder_decoder,expected_cudagraph_mode,expected_enforce_eager,expected_compilation_mode", [
        ("xlite_full_mode", True, True, CompilationMode.VLLM_COMPILE, CUDAGraphMode.NONE, False, CUDAGraphMode.NONE, True, CompilationMode.NONE),
        ("piecewise_mode", False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.PIECEWISE, False, CUDAGraphMode.PIECEWISE, False, CompilationMode.VLLM_COMPILE),
        ("encoder_decoder", False, False, CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY, True, CUDAGraphMode.PIECEWISE, False, CompilationMode.VLLM_COMPILE),
        ("unsupported_mode_with_xlite", True, False, CompilationMode.STOCK_TORCH_COMPILE, CUDAGraphMode.NONE, False, CUDAGraphMode.NONE, False, CompilationMode.NONE),
    ])
    def test_full_workflow(
        self, scenario, xlite_enabled, xlite_full_mode, compilation_mode,
        cudagraph_mode, is_encoder_decoder, expected_cudagraph_mode,
        expected_enforce_eager, expected_compilation_mode
    ):
        """Test full workflow scenarios"""
        from vllm_ascend.platform import NPUPlatform

        with patch.dict(os.environ, {}, clear=True):
            with patch("vllm_ascend.platform.update_aclgraph_sizes"):
                with ExitStack() as stack:
                    patches = self._get_patches()
                    mock_init, mocks = self._enter_patches(stack, patches)
                    mock_device_type = mocks[3]
                    vllm_config, ascend_config = self.create_mock_configs()
                    self._setup_basic_config(vllm_config, ascend_config, mock_device_type)
                    ascend_config.xlite_graph_config.enabled = xlite_enabled
                    ascend_config.xlite_graph_config.full_mode = xlite_full_mode
                    vllm_config.compilation_config.mode = compilation_mode
                    vllm_config.compilation_config.cudagraph_mode = cudagraph_mode
                    vllm_config.model_config.is_encoder_decoder = is_encoder_decoder
                    vllm_config.compilation_config.splitting_ops = []
                    if cudagraph_mode == CUDAGraphMode.PIECEWISE:
                        vllm_config.compilation_config.set_splitting_ops_for_v1 = MagicMock()
                    if scenario == "xlite_full_mode":
                        vllm_config.compilation_config.splitting_ops = None
                    mock_init.return_value = ascend_config

                    NPUPlatform.check_and_update_config(vllm_config)

                    assert vllm_config.compilation_config.cudagraph_mode == expected_cudagraph_mode
                    assert vllm_config.model_config.enforce_eager == expected_enforce_eager
                    if expected_compilation_mode is not None:
                        assert vllm_config.compilation_config.mode == expected_compilation_mode
