import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (
    AscendMergedColumnParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
    AscendUnquantizedLinearMethod,
)


class BaseLinearTest(unittest.TestCase):
    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config", return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group", return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype
        mock_is_meta = mock.PropertyMock(return_value=False)
        type(self.layer.weight.data).is_meta = mock_is_meta
        self.layer.precast_fp32_weight = False

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 0
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 1
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 2
        mock_get_config.return_value = mock_config
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.enable_sp", return_value=False)
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_mlp_optimize(self, mock_enable_sp, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.enable_sp", return_value=False)
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_enable_sp, mock_get_current_vllm_config):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):
    def test_merged_mlp_tp_init(self):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.scheduler_config.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestAscendReplicatedLinear(BaseLinearTest):
    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


class TestColumnParallelOpDispatch(unittest.TestCase):
    """Tests for _get_column_parallel_op factory — share_expert, g_proj."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _get_column_op(self, prefix: str):
        from vllm_ascend.ops.linear_op import _get_column_parallel_op

        return _get_column_parallel_op(prefix, self.mock_layer)

    def test_share_expert_disabled_with_sp_column(self):
        """share_expert / shared_expert prefix → None when SP enabled."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.share_expert.gate_up_proj"))
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.shared_expert.gate_up_proj"))

    def test_g_proj_matches_sp_column_path(self):
        """g_proj (Step3p5 attention gate) is included in SP column prefixes."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNotNone(self._get_column_op("model.layers.0.self_attn.g_proj"))

    def test_multimodal_encoder_prefix_skips_sp_column(self):
        """Multimodal encoder variants should not enter the SP column path."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNone(self._get_column_op("model.vision_model_proj.indexer_proj"))
        self.assertIsNone(self._get_column_op("model.vision_tower_encoder.qkv_proj"))


class TestRowParallelOpDispatch(unittest.TestCase):
    """Tests for _get_row_parallel_op — mtp_block, share_expert."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _op(self, prefix: str):
        from vllm_ascend.ops.linear_op import _get_row_parallel_op

        return _get_row_parallel_op(prefix, self.mock_layer)

    def test_share_expert_disabled_with_sp_row(self):
        """share_expert / shared_expert prefix → None when SP enabled."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNone(self._op("model.layers.0.mlp.share_expert.down_proj"))
        self.assertIsNone(self._op("model.layers.0.mlp.shared_expert.down_proj"))

    def test_multimodal_encoder_prefix_skips_sp_row(self):
        """Multimodal encoder variants should not enter the SP row path."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNone(self._op("model.multi_modal_projector.down_proj"))
        self.assertIsNone(self._op("model.patch_merge_mlp.out_proj"))


class TestSequenceRowParallelOp(unittest.TestCase):
    def setUp(self):
        self.mock_group = MagicMock(world_size=4, rank_in_group=0)
        self.group_patch = patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group)
        self.group_patch.start()

    def tearDown(self):
        self.group_patch.stop()

    def _make_op(self, input_is_parallel=True):
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        layer = MagicMock()
        op = SequenceRowParallelOp(layer)
        op.input_is_parallel = input_is_parallel
        op.reduce_results = True
        op.skip_bias_add = False
        op.bias = None
        op.quant_method = MagicMock()
        op.unique_prefix = "model.layers.0.mlp.down_proj"
        return op

    def test_prequantized_tuple_uses_with_scale_custom_op(self):
        op = self._make_op()
        quantized_input = torch.empty(8, 16, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(8, 1, 2, dtype=torch.uint8)
        expected = torch.empty(2, 8, dtype=torch.bfloat16)

        with patch.object(
            torch.ops.vllm, "matmul_and_reduce_with_scale", return_value=expected, create=True
        ) as mock_with_scale:
            output, output_bias = op.apply_impl((quantized_input, input_scale))

        self.assertIs(output, expected)
        self.assertIsNone(output_bias)
        mock_with_scale.assert_called_once_with(quantized_input, input_scale, op.unique_prefix)

    def test_prequantized_tuple_must_be_prepartitioned(self):
        op = self._make_op(input_is_parallel=False)
        quantized_input = torch.empty(8, 16, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(8, 1, 2, dtype=torch.uint8)

        with self.assertRaisesRegex(ValueError, "must already be partitioned"):
            op.apply_impl((quantized_input, input_scale))

    def test_with_scale_custom_op_reconstructs_quantized_input_tuple(self):
        from vllm_ascend.ops.register_custom_ops import _matmul_and_reduce_with_scale_impl

        quantized_input = torch.empty(8, 16, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty(8, 1, 2, dtype=torch.uint8)
        expected = torch.empty(2, 8, dtype=torch.bfloat16)
        layer = MagicMock()
        layer.tp_rank = 0
        layer.skip_bias_add = False
        layer.bias = None
        layer.custom_op.matmul_and_reduce.return_value = expected
        forward_context = MagicMock()
        forward_context.no_compile_layers = {"model.layers.0.mlp.down_proj": layer}

        with patch(
            "vllm_ascend.ops.register_custom_ops.get_forward_context",
            return_value=forward_context,
        ):
            output = _matmul_and_reduce_with_scale_impl(quantized_input, input_scale, "model.layers.0.mlp.down_proj")

        self.assertIs(output, expected)
        layer.custom_op.matmul_and_reduce.assert_called_once_with((quantized_input, input_scale), None)

    def test_with_scale_fake_uses_layer_params_dtype(self):
        from vllm_ascend.ops.register_custom_ops import _matmul_and_reduce_with_scale_impl_fake

        quantized_input = torch.empty(8, 16, dtype=torch.int8)
        input_scale = torch.empty(8, 1, dtype=torch.float32)
        layer = MagicMock()
        layer.output_size_per_partition = 4
        layer.params_dtype = torch.float16
        forward_context = MagicMock()
        forward_context.no_compile_layers = {"model.layers.0.mlp.down_proj": layer}

        with (
            patch(
                "vllm_ascend.ops.register_custom_ops.get_forward_context",
                return_value=forward_context,
            ),
            patch("vllm_ascend.ops.register_custom_ops._EXTRA_CTX") as extra_ctx,
        ):
            extra_ctx.flash_comm_v1_enabled = False
            output = _matmul_and_reduce_with_scale_impl_fake(
                quantized_input, input_scale, "model.layers.0.mlp.down_proj"
            )

        self.assertEqual(output.shape, (8, 4))
        self.assertEqual(output.dtype, torch.float16)


class TestGetParallelOpShareExpert(unittest.TestCase):
    """Tests for get_parallel_op — share_expert/shared_expert disables TP."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self.mock_group = MagicMock()
        self.mock_group.rank_in_group = 1
        self.mock_group.world_size = 2
        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.enable_sp_by_pass", return_value=False),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _call(self, prefix: str):
        from vllm_ascend.ops.linear_op import get_parallel_op

        return get_parallel_op(False, prefix, self.mock_layer, False)

    def test_share_expert_disables_tp(self):
        """share_expert / shared_expert / shared_experts → (None, 0, 1)."""
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=True):
            for prefix in (
                "model.layers.0.mlp.share_expert.gate_up_proj",
                "model.layers.0.mlp.shared_expert.gate_up_proj",
                "model.layers.0.mlp.shared_experts.gate_up_proj",
            ):
                custom_op, tp_rank, tp_size = self._call(prefix)
                self.assertIsNone(custom_op)
                self.assertEqual(tp_rank, 0)
                self.assertEqual(tp_size, 1)

    def test_share_expert_keeps_tp_when_only_flashcomm_is_enabled(self):
        with (
            patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True),
        ):
            custom_op, tp_rank, tp_size = self._call("model.layers.0.mlp.shared_experts.gate_up_proj")

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)


if __name__ == "__main__":
    unittest.main()
