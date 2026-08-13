import unittest
from types import SimpleNamespace
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

    def test_sequence_column_op_calls_tensor_all_gather_matmul_in_eager(self):
        from vllm_ascend.ops.linear_op import SequenceColumnParallelOp

        layer = MagicMock()
        layer.prefix = "model.layers.1.mlp.gate_up_proj"
        layer.gather_output = False
        layer.skip_bias_add = False
        layer.return_bias = True
        layer.quant_method = AscendUnquantizedLinearMethod()
        layer.weight = torch.nn.Parameter(torch.empty(16, 8, dtype=torch.float16), requires_grad=False)
        layer.bias = None
        output = torch.empty(4, 16, dtype=torch.float16)

        op = SequenceColumnParallelOp(layer)
        op.update_attrs()
        input_ = torch.empty(2, 8, dtype=torch.float16)
        with (
            patch("vllm_ascend.ops.linear_op.is_vl_model", return_value=False),
            patch.object(
                torch.ops.vllm, "all_gather_unquantized_matmul", return_value=output, create=True
            ) as mock_fused,
        ):
            result, output_bias = op.apply_impl(input_)

        self.assertIs(result, output)
        self.assertIsNone(output_bias)
        mock_fused.assert_called_once_with(input_, layer.weight, None)

    def test_dynamic_all_gather_matmul_skips_a5(self):
        from vllm_ascend.device.hardware import AscendDeviceType
        from vllm_ascend.ops.linear_op import _apply_tensor_all_gather_matmul_if_supported
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
        from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

        layer = MagicMock()
        layer.quant_method = AscendLinearMethod(AscendW8A8DynamicLinearMethod())
        layer.weight = torch.empty(16, 8, dtype=torch.int8)
        layer.weight_scale = torch.empty(16, dtype=torch.float32)
        input_ = torch.empty(2, 8, dtype=torch.bfloat16)

        with patch("vllm_ascend.ops.linear_op.get_ascend_device_type", return_value=AscendDeviceType.A5):
            result = _apply_tensor_all_gather_matmul_if_supported(layer, input_, None, need_all_gather=True)

        self.assertIsNone(result)


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


class TestSequenceRowParallelMatmulAndReduce(unittest.TestCase):
    def test_dynamic_w8a8_uses_mm_reduce_scatter_fusion(self):
        from vllm_ascend.device.hardware import AscendDeviceType
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp
        from vllm_ascend.quantization.method_adapters import AscendLinearMethod
        from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

        quant_method = AscendW8A8DynamicLinearMethod()
        layer = MagicMock()
        layer.quant_method = AscendLinearMethod(quant_method)
        layer.weight = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
        layer.weight_scale = torch.randn(16, dtype=torch.bfloat16)
        layer.weight_scale_fp32 = torch.randn(16, dtype=torch.float32)
        layer.tp_size = 2
        layer.tp_rank = 0
        layer.prefix = "model.language_model.model.layers.0.self_attn.o_proj"

        op = SequenceRowParallelOp(layer)
        op.quant_method = layer.quant_method
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        x_quant = torch.randint(-8, 8, (4, 8), dtype=torch.int8)
        pertoken_scale = torch.randn(4, dtype=torch.float32)
        fused_output = torch.randn(2, 16, dtype=torch.bfloat16)

        tp_group = MagicMock()
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_comm"

        with (
            patch(
                "vllm_ascend.ops.linear_op._EXTRA_CTX",
                SimpleNamespace(flash_comm_v1_enabled=True, mmrs_fusion=True, pad_size=0),
            ),
            patch("vllm_ascend.ops.linear_op.get_ascend_device_type", return_value=AscendDeviceType.A2),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp_group),
            patch(
                "vllm_ascend.ops.linear_op.DeviceOperator.npu_dynamic_quant",
                return_value=(x_quant, pertoken_scale),
            ) as mock_dynamic_quant,
            patch(
                "vllm_ascend.ops.linear_op.DeviceOperator.npu_mm_reduce_scatter_base",
                return_value=fused_output,
            ) as mock_mmrs,
        ):
            output = op.matmul_and_reduce(x, bias_=None)

        self.assertIs(output, fused_output)
        mock_dynamic_quant.assert_called_once_with(x, act_quant_type=quant_method.act_quant_type)
        mock_mmrs.assert_called_once()
        _, kwargs = mock_mmrs.call_args
        self.assertEqual(kwargs["x1_scale"].shape, (4, 1))
        self.assertTrue(torch.equal(kwargs["x1_scale"].squeeze(dim=1), pertoken_scale))
        self.assertEqual(kwargs["x2_scale"].shape, (1, 16))
        self.assertTrue(torch.equal(kwargs["x2_scale"].squeeze(dim=0), layer.weight_scale_fp32))
        self.assertEqual(kwargs["output_dtype"], x.dtype)
        self.assertEqual(kwargs["comm_mode"], "aiv")

    def test_dynamic_w8a8_mmrs_fusion_disabled_on_a5(self):
        from vllm_ascend.device.hardware import AscendDeviceType
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        layer = MagicMock()
        layer.quant_method = MagicMock()
        layer.quant_method.apply.return_value = torch.randn(4, 16, dtype=torch.bfloat16)
        layer.weight = torch.randint(-8, 8, (8, 16), dtype=torch.int8)
        layer.weight_scale = torch.randn(16, dtype=torch.bfloat16)
        layer.tp_size = 2
        layer.tp_rank = 0
        layer.prefix = "model.language_model.model.layers.0.self_attn.o_proj"

        op = SequenceRowParallelOp(layer)
        op.quant_method = layer.quant_method
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        reduced_output = torch.randn(2, 16, dtype=torch.bfloat16)

        tp_group = MagicMock()
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_comm"

        with (
            patch(
                "vllm_ascend.ops.linear_op._EXTRA_CTX",
                SimpleNamespace(flash_comm_v1_enabled=True, mmrs_fusion=True, pad_size=0),
            ),
            patch("vllm_ascend.ops.linear_op.get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp_group),
            patch("vllm_ascend.ops.linear_op.DeviceOperator.npu_mm_reduce_scatter_base") as mock_mmrs,
            patch(
                "vllm_ascend.ops.linear_op.tensor_model_parallel_reduce_scatter",
                return_value=reduced_output,
            ) as mock_reduce_scatter,
        ):
            output = op.matmul_and_reduce(x, bias_=None)

        self.assertIs(output, reduced_output)
        layer.quant_method.apply.assert_called_once_with(layer, x, bias=None)
        mock_reduce_scatter.assert_called_once()
        mock_mmrs.assert_not_called()

    def test_unquantized_mmrs_fusion_allowed_on_a5(self):
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

        from vllm_ascend.device.hardware import AscendDeviceType
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        layer = MagicMock()
        layer.quant_method = UnquantizedLinearMethod()
        layer.weight = torch.randn(16, 8, dtype=torch.bfloat16)
        layer.tp_size = 2
        layer.tp_rank = 0
        layer.prefix = "model.language_model.model.layers.0.self_attn.o_proj"

        op = SequenceRowParallelOp(layer)
        op.quant_method = layer.quant_method
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        fused_output = torch.randn(2, 16, dtype=torch.bfloat16)

        tp_group = MagicMock()
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_comm"

        with (
            patch(
                "vllm_ascend.ops.linear_op._EXTRA_CTX",
                SimpleNamespace(flash_comm_v1_enabled=True, mmrs_fusion=True, pad_size=0),
            ),
            patch("vllm_ascend.ops.linear_op.get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp_group),
            patch(
                "vllm_ascend.ops.linear_op.DeviceOperator.npu_mm_reduce_scatter_base",
                return_value=fused_output,
            ) as mock_mmrs,
        ):
            output = op.matmul_and_reduce(x, bias_=None)

        self.assertIs(output, fused_output)
        mock_mmrs.assert_called_once()

    def test_dynamic_w8a8_mmrs_fusion_only_for_self_attn_o_proj(self):
        from vllm_ascend.device.hardware import AscendDeviceType
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        layer = MagicMock()
        layer.quant_method.apply.return_value = torch.randn(4, 16, dtype=torch.bfloat16)
        layer.tp_size = 2
        layer.tp_rank = 0
        layer.prefix = "model.language_model.model.layers.0.mlp.down_proj"

        op = SequenceRowParallelOp(layer)
        op.quant_method = layer.quant_method
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        reduced_output = torch.randn(2, 16, dtype=torch.bfloat16)

        tp_group = MagicMock()
        tp_group.device_group._get_backend.return_value.get_hccl_comm_name.return_value = "hccl_comm"

        with (
            patch(
                "vllm_ascend.ops.linear_op._EXTRA_CTX",
                SimpleNamespace(flash_comm_v1_enabled=True, mmrs_fusion=True, pad_size=0),
            ),
            patch("vllm_ascend.ops.linear_op.get_ascend_device_type", return_value=AscendDeviceType.A2),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp_group),
            patch("vllm_ascend.ops.linear_op.DeviceOperator.npu_mm_reduce_scatter_base") as mock_mmrs,
            patch(
                "vllm_ascend.ops.linear_op.tensor_model_parallel_reduce_scatter",
                return_value=reduced_output,
            ) as mock_reduce_scatter,
        ):
            output = op.matmul_and_reduce(x, bias_=None)

        self.assertIs(output, reduced_output)
        layer.quant_method.apply.assert_called_once_with(layer, x, bias=None)
        mock_reduce_scatter.assert_called_once()
        mock_mmrs.assert_not_called()


class TestGetParallelOpShareExpert(unittest.TestCase):
    """Tests for get_parallel_op — share_expert/shared_expert disables TP."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self.mock_group = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=True),
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
        for prefix in (
            "model.layers.0.mlp.share_expert.gate_up_proj",
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            "model.layers.0.mlp.shared_experts.gate_up_proj",
        ):
            custom_op, tp_rank, tp_size = self._call(prefix)
            self.assertIsNone(custom_op)
            self.assertEqual(tp_rank, 0)
            self.assertEqual(tp_size, 1)


if __name__ == "__main__":
    unittest.main()
