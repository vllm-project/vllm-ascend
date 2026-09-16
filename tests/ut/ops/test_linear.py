import os
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
from vllm_ascend.ops.linear_op import MatmulCommRowParallelOp, _get_row_parallel_op
from vllm_ascend.utils import AscendDeviceType, enable_mm_comm_fuse


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
        self.layer.skip_weight_nz_conversion = False

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

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_skips_nz_for_marked_layer(self, mock_format_cast, mock_get_config):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 2
        mock_get_config.return_value = mock_config
        self.layer.skip_weight_nz_conversion = True
        self.layer.precast_fp32_weight = True

        self.method.process_weights_after_loading(self.layer)

        mock_format_cast.assert_not_called()


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

    def _get_column_op(self, prefix: str, output_size: int | None = None):
        from vllm_ascend.ops.linear_op import _get_column_parallel_op

        return _get_column_parallel_op(prefix, self.mock_layer, output_size)

    def test_share_expert_disabled_with_sp_column(self):
        """share_expert / shared_expert prefix → None when SP enabled."""
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.share_expert.gate_up_proj"))
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.shared_expert.gate_up_proj"))

    def test_head_wise_g_proj_matches_sp_column_path(self):
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        model_config = MagicMock()
        model_config.hf_text_config.num_attention_heads = 64
        with patch(
            "vllm_ascend.ops.linear_op.get_current_vllm_config",
            return_value=MagicMock(model_config=model_config),
        ):
            self.assertIsNotNone(self._get_column_op("model.layers.0.self_attn.g_proj", output_size=64))

    def test_full_projection_g_proj_does_not_match_sp_column_path(self):
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        model_config = MagicMock()
        model_config.hf_text_config.num_attention_heads = 96
        with patch(
            "vllm_ascend.ops.linear_op.get_current_vllm_config",
            return_value=MagicMock(model_config=model_config),
        ):
            self.assertIsNone(
                self._get_column_op(
                    "model.layers.0.self_attn.g_proj",
                    output_size=96 * 128,
                )
            )

    def test_g_proj_without_explicit_output_size_does_not_match_sp_column_path(self):
        self._patches.append(patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True))
        self._patches[-1].start()
        model_config = MagicMock()
        model_config.hf_text_config.num_attention_heads = 64
        with patch(
            "vllm_ascend.ops.linear_op.get_current_vllm_config",
            return_value=MagicMock(model_config=model_config),
        ):
            self.assertIsNone(self._get_column_op("model.layers.0.self_attn.g_proj"))

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


class TestGetParallelOpShareExpert(unittest.TestCase):
    """Tests for independent shared-expert weight placement."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self.mock_group = MagicMock()
        self.mock_group.rank_in_group = 1
        self.mock_group.world_size = 4
        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _call(self, prefix: str, *, shared_expert_dp: bool):
        from vllm_ascend.ops.linear_op import get_parallel_op

        with patch(
            "vllm_ascend.ops.linear_op.shared_expert_dp_enabled",
            return_value=shared_expert_dp,
        ):
            return get_parallel_op(False, prefix, self.mock_layer, False)

    def test_share_expert_disables_tp(self):
        """share_expert / shared_expert / shared_experts → (None, 0, 1)."""
        for prefix in (
            "model.layers.0.mlp.share_expert.gate_up_proj",
            "model.layers.0.mlp.shared_expert.gate_up_proj",
            "model.layers.0.mlp.shared_experts.gate_up_proj",
        ):
            custom_op, tp_rank, tp_size = self._call(prefix, shared_expert_dp=True)
            self.assertIsNone(custom_op)
            self.assertEqual(tp_rank, 0)
            self.assertEqual(tp_size, 1)

    def test_flashcomm_without_shared_expert_dp_keeps_tp_shards(self):
        custom_op, tp_rank, tp_size = self._call(
            "model.layers.0.mlp.shared_experts.gate_up_proj",
            shared_expert_dp=False,
        )
        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 4)


class TestEnableMatmulCommFuse(unittest.TestCase):
    """Tests for enable_mm_comm_fuse env parsing and the A2 device guard."""

    def tearDown(self):
        os.environ.pop("VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE", None)

    def test_disabled_by_default(self):
        self.assertEqual(enable_mm_comm_fuse(), 0)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2)
    def test_mode_1_on_a2(self, _mock_device):
        os.environ["VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE"] = "1"
        self.assertEqual(enable_mm_comm_fuse(), 1)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2)
    def test_mode_2_on_a2(self, _mock_device):
        os.environ["VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE"] = "2"
        self.assertEqual(enable_mm_comm_fuse(), 2)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
    def test_falls_back_to_zero_on_non_a2(self, _mock_device):
        os.environ["VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE"] = "1"
        self.assertEqual(enable_mm_comm_fuse(), 0)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2)
    def test_invalid_string_falls_back_to_zero(self, _mock_device):
        os.environ["VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE"] = "abc"
        self.assertEqual(enable_mm_comm_fuse(), 0)

    @patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A2)
    def test_out_of_range_value_falls_back_to_zero(self, _mock_device):
        os.environ["VLLM_ASCEND_ENABLE_MATMUL_COMM_FUSE"] = "3"
        self.assertEqual(enable_mm_comm_fuse(), 0)


class TestMatmulCommRowParallelOpDispatch(unittest.TestCase):
    """Tests for _get_row_parallel_op selection of the fused op."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self._patches = [
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=False),
            patch("vllm_ascend.ops.linear_op.is_moe_layer", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _op(self, prefix: str, quant_config=None):
        return _get_row_parallel_op(prefix, self.mock_layer, quant_config)

    def test_fused_op_selected_when_enabled(self):
        with patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=1):
            op = self._op("model.layers.0.down_proj")
        self.assertIsInstance(op, MatmulCommRowParallelOp)

    def test_fused_op_not_selected_when_disabled(self):
        self.assertIsNone(self._op("model.layers.0.down_proj"))

    def test_fused_op_not_selected_for_quantized_layers(self):
        with patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=1):
            op = self._op("model.layers.0.down_proj", quant_config=MagicMock())
        self.assertIsNone(op)

    def test_fused_op_not_selected_beyond_tp_limit(self):
        tp16 = SimpleNamespace(world_size=16, rank_in_group=0, device_group="fake_group")
        with (
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=1),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp16),
        ):
            op = self._op("model.layers.0.down_proj")
        self.assertIsNone(op)

    def test_fused_op_not_selected_for_tp1(self):
        tp1 = SimpleNamespace(world_size=1, rank_in_group=0, device_group="fake_group")
        with (
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=1),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=tp1),
        ):
            op = self._op("model.layers.0.down_proj")
        self.assertIsNone(op)

    def test_oproj_specialization_keeps_priority_over_fused(self):
        from vllm_ascend.ops.linear_op import OProjRowParallelOp

        with (
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=2),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=True),
        ):
            op = self._op("model.layers.0.self_attn.o_proj")
        self.assertIsInstance(op, OProjRowParallelOp)

    def test_sp_row_keeps_priority_over_fused(self):
        from vllm_ascend.ops.linear_op import SequenceRowParallelOp

        with (
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=2),
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True),
        ):
            op = self._op("model.layers.0.mlp.down_proj")
        self.assertIsInstance(op, SequenceRowParallelOp)


class TestMatmulCommRowParallelOp(unittest.TestCase):
    """Tests for MatmulCommRowParallelOp init/attrs and apply_impl paths."""

    def setUp(self):
        self.mock_group = SimpleNamespace(world_size=2, rank_in_group=0, device_group="fake_group")

        self.mock_layer = MagicMock()
        self.mock_layer.weight_t = None
        self.mock_layer.bias = None
        self.mock_layer.skip_bias_add = False
        self.mock_layer.return_bias = True
        self.mock_layer.input_is_parallel = True
        self.mock_layer.reduce_results = True
        self.mock_layer.quant_method = MagicMock()

        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.enable_mm_comm_fuse", return_value=1),
            patch(
                "vllm_ascend.ops.linear_op.MatmulCommRowParallelOp.get_hcomm_info",
                return_value="fake_hcomm",
            ),
        ]
        for p in self._patches:
            p.start()

        MatmulCommRowParallelOp._HCOMM_INFO_MAP.clear()
        self.op = MatmulCommRowParallelOp(self.mock_layer)
        self.op.update_attrs()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        MatmulCommRowParallelOp._HCOMM_INFO_MAP.clear()

    def test_update_attrs_flags_apply_weight_t(self):
        """reduce_results + tp>1 flagged in update_attrs, not in __init__."""
        self.assertTrue(self.mock_layer.apply_weight_t)
        self.assertEqual(self.op.hcomm_info, "fake_hcomm")

    def test_update_attrs_no_flag_for_tp1(self):
        self.mock_group.world_size = 1
        layer = MagicMock()
        layer.apply_weight_t = False
        op = MatmulCommRowParallelOp(layer)
        op.update_attrs()
        self.assertFalse(layer.apply_weight_t)
        self.assertIsNone(op.hcomm_info)

    def test_apply_mode_1_all_reduce(self):
        """Mode 1: fused all-reduce op called with stashed weight and bias."""
        weight_t = torch.randn(8, 16)
        self.mock_layer.weight_t = weight_t
        input_ = torch.randn(4, 8)
        bias_ = torch.randn(16)
        self.op.bias = bias_
        with patch("torch_npu.npu_mm_all_reduce_base", return_value=torch.randn(4, 16)) as mock_fuse:
            output, output_bias = self.op.apply_impl(input_)
        self.assertEqual(mock_fuse.call_count, 1)
        self.assertIs(mock_fuse.call_args.args[0], input_)
        self.assertIs(mock_fuse.call_args.args[1], weight_t)
        self.assertEqual(mock_fuse.call_args.args[2], "fake_hcomm")
        self.assertIs(mock_fuse.call_args.kwargs["bias"], bias_)
        self.assertEqual(output.shape, (4, 16))
        self.assertIsNone(output_bias)

    def test_apply_mode_1_skips_bias_on_nonzero_rank(self):
        self.mock_group.rank_in_group = 1
        op = MatmulCommRowParallelOp(self.mock_layer)
        op.update_attrs()
        self.mock_layer.weight_t = torch.randn(8, 16)
        with patch("torch_npu.npu_mm_all_reduce_base", return_value=torch.randn(4, 16)) as mock_fuse:
            op.apply_impl(torch.randn(4, 8))
        self.assertIsNone(mock_fuse.call_args.kwargs["bias"])

    def test_apply_mode_2_reduce_scatter_with_pad(self):
        """Mode 2: pad → reduce_scatter → all_gather → slice → add bias."""
        self.op.mm_comm_fuse_mode = 2
        # 3 tokens on tp=2 → padded to 4 before the fused reduce-scatter
        input_ = torch.randn(3, 8)
        weight_t = torch.randn(8, 16)
        self.mock_layer.weight_t = weight_t
        self.op.bias = torch.randn(16)

        rs_output = torch.randn(2, 16)  # [padded_tokens // world_size, out]
        gathered = torch.randn(4, 16)  # full padded output

        with (
            patch("torch_npu.npu_mm_reduce_scatter_base", return_value=rs_output) as mock_rs,
            patch(
                "torch.distributed.all_gather_into_tensor", side_effect=lambda out, *_a, **_k: out.copy_(gathered)
            ) as mock_ag,
        ):
            output, _ = self.op.apply_impl(input_)

        rs_args = mock_rs.call_args
        self.assertEqual(rs_args.args[0].shape, (4, 8))
        self.assertIs(rs_args.args[1], weight_t)
        self.assertEqual(rs_args.args[2], "fake_hcomm")
        self.assertEqual(rs_args.args[3], 2)
        # all_gather wrote into a [4, out] buffer on the device group
        self.assertEqual(mock_ag.call_args.args[0].shape, (4, 16))
        self.assertEqual(mock_ag.call_args.kwargs["group"], "fake_group")
        # output sliced back to the original token count
        self.assertEqual(output.shape, (3, 16))

    def test_apply_mode_2_no_pad_when_divisible(self):
        self.op.mm_comm_fuse_mode = 2
        self.mock_layer.weight_t = torch.randn(8, 16)
        rs_output = torch.randn(2, 16)
        with (
            patch("torch_npu.npu_mm_reduce_scatter_base", return_value=rs_output) as mock_rs,
            patch("torch.distributed.all_gather_into_tensor"),
        ):
            self.op.apply_impl(torch.randn(4, 8))
        self.assertEqual(mock_rs.call_args.args[0].shape, (4, 8))

    def test_apply_fallback_tp1_uses_quant_method(self):
        """tp=1 (or reduce_results=False) falls back to quant_method.apply."""
        self.mock_group.world_size = 1
        op = MatmulCommRowParallelOp(self.mock_layer)
        op.update_attrs()
        op.apply_impl(torch.randn(4, 8))
        self.mock_layer.quant_method.apply.assert_called_once()

    def test_apply_missing_stash_transposes_on_the_fly(self):
        """Defensive: unquantized layer without the stashed copy builds [K, O] on the fly."""
        weight = torch.randn(8, 16)
        self.mock_layer.weight = weight
        self.mock_layer.weight_t = None
        with patch("torch_npu.npu_mm_all_reduce_base", return_value=torch.randn(4, 16)) as mock_fuse:
            self.op.apply_impl(torch.randn(4, 8))
        weight_arg = mock_fuse.call_args.args[1]
        self.assertEqual(weight_arg.shape, (16, 8))
        self.assertTrue(weight_arg.is_contiguous())
        self.assertTrue(torch.equal(weight_arg, weight.t()))

    def test_apply_mode_1_flattens_3d_input(self):
        """Mode 1 flattens >2D input for the fused op and restores leading dims."""
        self.mock_layer.weight_t = torch.randn(8, 16)
        input_ = torch.randn(2, 3, 8)
        with patch("torch_npu.npu_mm_all_reduce_base", return_value=torch.randn(6, 16)) as mock_fuse:
            output, _ = self.op.apply_impl(input_)
        self.assertEqual(mock_fuse.call_args.args[0].shape, (6, 8))
        self.assertEqual(output.shape, (2, 3, 16))

    def test_get_hcomm_info_cached_per_group(self):
        g1, g2 = MagicMock(), MagicMock()
        backend1 = g1._get_backend.return_value
        backend1.get_hccl_comm_name.return_value = "handle_g1"
        backend2 = g2._get_backend.return_value
        backend2.get_hccl_comm_name.return_value = "handle_g2"
        with (
            patch("vllm_ascend.ops.linear_op.torch.__version__", "2.1"),
            patch("vllm_ascend.ops.linear_op.dist.get_rank", return_value=0),
            patch("vllm_ascend.ops.linear_op.dist.get_global_rank", return_value=0),
        ):
            h1a = MatmulCommRowParallelOp.get_hcomm_info(g1)
            h1b = MatmulCommRowParallelOp.get_hcomm_info(g1)
            h2 = MatmulCommRowParallelOp.get_hcomm_info(g2)
        self.assertEqual(h1a, "handle_g1")
        self.assertEqual(h1b, "handle_g1")
        self.assertEqual(h2, "handle_g2")
        # resolved once per group: the second g1 call hits the cache
        self.assertEqual(backend1.get_hccl_comm_name.call_count, 1)


class TestWeightTStash(unittest.TestCase):
    """Tests for the weight_t stash in process_weights_after_loading."""

    def _process(self, layer: MagicMock) -> None:
        method = AscendUnquantizedLinearMethod()
        with (
            patch("vllm_ascend.ops.linear._should_keep_nd_for_310p_weight", return_value=False),
            patch("vllm_ascend.ops.linear.maybe_trans_nz", side_effect=lambda w: w),
        ):
            method.process_weights_after_loading(layer)

    def test_stash_before_nz_keeps_nd_transposed_copy(self):
        """The stash runs before NZ conversion, so it is the plain ND
        transposed copy of the original weight."""
        layer = MagicMock()
        layer.prefix = "model.layers.0.down_proj"
        layer.weight = MagicMock()
        layer.weight.data = torch.randn(8, 16)
        layer.apply_weight_t = True
        layer.precast_fp32_weight = False
        layer.skip_weight_nz_conversion = False

        self._process(layer)

        self.assertTrue(torch.equal(layer.weight_t, layer.weight.data.t()))
        self.assertEqual(layer.weight_t.shape, (16, 8))
        self.assertTrue(layer.weight_t.is_contiguous())

    def test_no_stash_without_flag(self):
        layer = MagicMock()
        layer.prefix = "model.layers.0.down_proj"
        layer.weight = MagicMock()
        layer.weight.data = torch.randn(8, 16)
        layer.precast_fp32_weight = False
        layer.skip_weight_nz_conversion = False

        self._process(layer)

        self.assertIsNone(getattr(layer, "weight_t", None))


if __name__ == "__main__":
    unittest.main()
