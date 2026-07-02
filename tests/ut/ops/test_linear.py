import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops import linear_op
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
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp_with_layer_shard", return_value=False),
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
    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.enable_sp", return_value=False)
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_mlp_optimize(self, mock_enable_sp, mock_get_current_vllm_config, mock_get_weight_prefetch_method):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
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

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch("vllm_ascend.ops.linear.enable_sp", return_value=False)
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_enable_sp, mock_get_current_vllm_config, mock_get_weight_prefetch_method):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
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
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
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


class TestVisionModelParallelOpExclusion(unittest.TestCase):
    """FC1/FC2 (SequenceParallel / FlashComm2) must be skipped for ViT
    prefixes: vision-language models keep standard TP for the vision tower,
    only the LLM body runs the communication-fused ops.

    Exercises the ``"vision_model" not in prefix`` guards in
    ``_get_column_parallel_op`` and ``_get_row_parallel_op``.
    """

    def setUp(self):
        # Route every prefix to the SP / FlashComm2 branches by disabling the
        # other specialized paths (MLP/OProj/DSA/matmul/oshard).
        self.patches = [
            patch("vllm_ascend.ops.linear_op.enable_sp", return_value=True),
            patch("vllm_ascend.ops.linear_op.flashcomm2_enable", return_value=True),
            patch("vllm_ascend.ops.linear_op.mlp_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.oproj_tp_enable", return_value=False),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch(
                "vllm_ascend.ops.linear_op.enable_dsa_cp_with_layer_shard",
                return_value=False,
            ),
            patch(
                "vllm_ascend.ops.linear_op.matmul_allreduce_enable",
                return_value=False,
            ),
            patch(
                "vllm_ascend.ops.linear_op.flashcomm2_oshard_manager.flashcomm2_oshard_enable",
                return_value=False,
            ),
            # FlashComm2 op construction reaches into tp/ascend groups; stub them
            # so the control case can be built on a CPU runner.
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=MagicMock(world_size=2)),
            patch(
                "vllm_ascend.ops.linear_op.get_flashcomm2_odp_group",
                return_value=MagicMock(world_size=2),
            ),
            patch(
                "vllm_ascend.ops.linear_op.get_ascend_config",
                return_value=MagicMock(flashcomm2_oproj_tensor_parallel_size=2),
            ),
            patch(
                "vllm_ascend.ops.linear_op.get_flashcomm2_reorgnized_batch_ids",
                return_value=[[0]],
            ),
        ]
        for p in self.patches:
            p.start()
        self.addCleanup(self._stop_patches)

    def _stop_patches(self):
        for p in self.patches:
            p.stop()

    def _column_op(self, prefix):
        return linear_op._get_column_parallel_op(prefix, layer=MagicMock())

    def _row_op(self, prefix):
        return linear_op._get_row_parallel_op(prefix, layer=MagicMock())

    # ---- column-parallel (FC1) -------------------------------------------
    def test_column_sp_for_llm_qkv(self):
        op = self._column_op("model.layers.0.self_attn.qkv_proj")
        self.assertIsInstance(op, linear_op.SequenceColumnParallelOp)

    def test_column_sp_excluded_for_vision_model(self):
        op = self._column_op("vision_model.blocks.0.attn.qkv_proj")
        self.assertIsNone(op)

    # ---- row-parallel (FC2 FlashComm2 o_proj) ----------------------------
    def test_row_flashcomm2_for_llm_oproj(self):
        op = self._row_op("model.layers.0.self_attn.o_proj")
        self.assertIsInstance(op, linear_op.Flashcomm2OProjRowParallelOp)

    def test_row_flashcomm2_excluded_for_vision_model(self):
        op = self._row_op("vision_model.blocks.0.attn.o_proj")
        # Excluded from FC2 and from the SP fallback -> falls through to None.
        self.assertIsNone(op)

    # ---- row-parallel (FC1 SequenceParallel down_proj) -------------------
    def test_row_sp_for_llm_down_proj(self):
        op = self._row_op("model.layers.0.mlp.down_proj")
        self.assertIsInstance(op, linear_op.SequenceRowParallelOp)

    def test_row_sp_excluded_for_vision_model(self):
        op = self._row_op("vision_model.blocks.0.mlp.down_proj")
        self.assertIsNone(op)


if __name__ == "__main__":
    unittest.main()
