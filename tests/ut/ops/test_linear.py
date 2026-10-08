import unittest
from contextlib import contextmanager
from itertools import product
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_processing,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops import linear
from vllm_ascend.ops.linear import (
    AscendDCPGroupColumnParallelLinear,
    AscendMergedColumnParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
    AscendUnquantizedLinearMethod,
)
from vllm_ascend.ops.linear_op import DCPGroupColumnParallelOp
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLinearMethod


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

        # maybe_trans_nz reads ndim/shape to screen out k=1/n=1 weights, so the
        # mock must expose a realistic 2D (non-singleton) shape, not a MagicMock.
        type(self.layer.weight.data).ndim = mock.PropertyMock(return_value=2)
        type(self.layer.weight.data).shape = mock.PropertyMock(return_value=torch.Size([64, 32]))
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
        # Real tensor so precast can materialize weight_fp32 alongside the NZ skip.
        weight = torch.randn(8, 4, dtype=torch.float16)
        self.layer.weight.data = weight
        self.layer.prefix = "model.layers.0.mlp.gate"

        self.method.process_weights_after_loading(self.layer)

        mock_format_cast.assert_not_called()
        self.assertEqual(self.layer.weight_fp32.dtype, torch.float32)
        torch.testing.assert_close(self.layer.weight_fp32, weight.to(torch.float32))

    @patch("vllm_ascend.utils.get_ascend_config")
    @mock.patch("vllm_ascend.ops.linear.maybe_trans_nz", side_effect=lambda x: x)
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_precasts_fp32_weight(
        self, mock_format_cast, mock_maybe_trans_nz, mock_get_config
    ):
        mock_config = MagicMock()
        mock_config.weight_nz_mode = 0
        mock_get_config.return_value = mock_config

        weight = torch.randn(8, 4, dtype=torch.float16)
        layer = mock.MagicMock()
        layer.weight.data = weight
        layer.prefix = "model.layers.0.mlp.gate"
        layer.precast_fp32_weight = True
        layer.skip_weight_nz_conversion = True

        self.method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight_fp32.dtype, torch.float32)
        torch.testing.assert_close(layer.weight_fp32, weight.to(torch.float32))
        mock_format_cast.assert_not_called()


class TestShouldReshapeWoATo3d(unittest.TestCase):
    """Tests for _should_reshape_wo_a_to_3d — DSV4 wo_a 3D layout decision."""

    @staticmethod
    def _profile_supporting(mx_quant_fusion: bool):
        profile = MagicMock()
        profile.supports.return_value = mx_quant_fusion
        return profile

    def _reshape_decision(self, prefix, dtype, mx_quant_fusion):
        with patch(
            "vllm_ascend.ops.linear.get_current_hardware_profile",
            return_value=self._profile_supporting(mx_quant_fusion),
        ):
            from vllm_ascend.ops.linear import _should_reshape_wo_a_to_3d

            return _should_reshape_wo_a_to_3d(prefix, dtype)

    def test_bf16_wo_a_reshapes_on_a5(self):
        """Regression: ModelSlim keeps unquantized bf16 wo_a while the model
        carries a global quant config; it still needs the 3D reshape in
        weight_loader (previously gated on quant_config is None)."""
        self.assertTrue(self._reshape_decision("model.layers.0.self_attn.wo_a", torch.bfloat16, mx_quant_fusion=True))

    def test_bf16_wo_a_reshapes_without_mx_quant_fusion(self):
        self.assertTrue(self._reshape_decision("model.layers.0.self_attn.wo_a", torch.bfloat16, mx_quant_fusion=False))

    def test_quantized_wo_a_not_reshaped_by_weight_loader(self):
        """fp8 wo_a is left to the quantization path (process_weights_after_loading)."""
        self.assertFalse(
            self._reshape_decision("model.layers.0.self_attn.wo_a", torch.float8_e4m3fn, mx_quant_fusion=True)
        )

    def test_non_wo_a_layer_not_reshaped(self):
        self.assertFalse(
            self._reshape_decision("model.layers.0.self_attn.o_proj", torch.bfloat16, mx_quant_fusion=True)
        )


class TestAscendRowParallelLinear(BaseLinearTest):
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_mlp_optimize(self, mock_get_current_vllm_config):
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
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_get_current_vllm_config):
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
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.share_expert.gate_up_proj"))
        self.assertIsNone(self._get_column_op("model.layers.0.mlp.shared_expert.gate_up_proj"))

    def test_g_proj_does_not_use_removed_sp_column_path(self):
        """g_proj (Step3p5 attention gate) is included in SP column prefixes."""
        self.assertIsNone(self._get_column_op("model.layers.0.self_attn.g_proj"))

    def test_multimodal_encoder_prefix_skips_sp_column(self):
        """Multimodal encoder variants should not enter the SP column path."""
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
        self.assertIsNone(self._op("model.layers.0.mlp.share_expert.down_proj"))
        self.assertIsNone(self._op("model.layers.0.mlp.shared_expert.down_proj"))

    def test_multimodal_encoder_prefix_skips_sp_row(self):
        """Multimodal encoder variants should not enter the SP row path."""
        self.assertIsNone(self._op("model.multi_modal_projector.down_proj"))
        self.assertIsNone(self._op("model.patch_merge_mlp.out_proj"))


class TestGetParallelOpShareExpert(unittest.TestCase):
    """Tests for get_parallel_op — share_expert/shared_expert disables TP."""

    def setUp(self):
        self.mock_layer = MagicMock()
        self.mock_group = MagicMock()
        self.mock_group.rank_in_group = 1
        self.mock_group.world_size = 2
        self._patches = [
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
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

    def test_sequence_parallel_does_not_replicate_shared_expert_weights(self):
        """After decoupling, SP alone keeps TP (weights not replicated)."""
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = self._call("model.layers.0.mlp.shared_experts.gate_up_proj")

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

    def test_shared_expert_keeps_tp_without_dp_or_sequence_parallel(self):
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = self._call("model.layers.0.mlp.shared_experts.gate_up_proj")

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

    def test_shared_expert_ignores_disable_tp_from_model(self):
        """Models pass disable_tp=is_sequence_parallel for shared experts; after
        decoupling, only the shared-expert DP switch replicates weights."""
        from vllm_ascend.ops.linear_op import get_parallel_op

        prefix = "model.layers.0.mlp.shared_experts.gate_up_proj"
        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=False):
            custom_op, tp_rank, tp_size = get_parallel_op(True, prefix, self.mock_layer, False)

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 1)
        self.assertEqual(tp_size, 2)

        with patch("vllm_ascend.ops.linear_op.shared_expert_dp_enabled", return_value=True):
            custom_op, tp_rank, tp_size = get_parallel_op(True, prefix, self.mock_layer, False)

        self.assertIsNone(custom_op)
        self.assertEqual(tp_rank, 0)
        self.assertEqual(tp_size, 1)


class TestAscendDCPGroupColumnParallelLinear(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _parallel_context(rank=1, dcp=2):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=dcp, prefill_context_parallel_size=1, tensor_parallel_size=4
            ),
            model_config=SimpleNamespace(enforce_eager=True),
        )
        group = SimpleNamespace(world_size=4, rank_in_group=rank)
        with (
            patch.object(linear, "get_current_vllm_config", return_value=config),
            patch.object(linear, "get_tensor_model_parallel_rank", return_value=rank),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=group),
            patch("vllm.distributed.parallel_state.get_tp_group", return_value=group),
            patch("vllm_ascend.ops.linear_op.enable_dsa_cp", return_value=False),
            patch.object(linear, "_should_reshape_wo_a_to_3d", return_value=False),
        ):
            yield

    @staticmethod
    @contextmanager
    def _unquantized_gemm():
        with (
            patch.object(linear.UnquantizedLinearMethod, "process_weights_after_loading"),
            patch.object(linear, "_should_keep_nd_for_compatibility_weight", return_value=True),
            patch("torch.ops.vllm.unquantized_gemm", side_effect=torch.nn.functional.linear, create=True) as gemm,
        ):
            yield gemm

    @staticmethod
    @contextmanager
    def _quantization():
        config = SimpleNamespace(
            get_quant_method=lambda *args, **kwargs: AscendLinearMethod(AscendW8A8DynamicLinearMethod())
        )
        with (
            patch("vllm_ascend.quantization.methods.w8a8.w8a8_dynamic.maybe_trans_nz", side_effect=lambda w: w),
            patch("torch_npu.npu_dynamic_quant", side_effect=lambda x, **kwargs: (x, torch.ones(x.shape[0]))),
            patch(
                "torch_npu.npu_quant_matmul", side_effect=lambda x, w, scale, **kwargs: (x @ w.float()) * scale
            ) as matmul,
        ):
            yield config, matmul

    def test_group_projection_loads_actual_shards_and_uses_ascend_gemm(self):
        for rank, dcp in product(range(4), [2, 4]):
            with self.subTest(rank=rank, dcp=dcp), self._unquantized_gemm() as unquantized_gemm:
                full_weight = torch.arange(24 * 5, dtype=torch.float32).view(24, 5) / 100
                x = torch.arange(15, dtype=torch.float32).view(3, 5)
                with self._parallel_context(rank, dcp):
                    layer = AscendDCPGroupColumnParallelLinear(5, 24, prefix="model.layers.0.self_attn.q_b_proj")
                    self.assertIsInstance(layer.quant_method, AscendUnquantizedLinearMethod)
                    self.assertIsInstance(layer.custom_op, DCPGroupColumnParallelOp)
                    self.assertEqual((layer.tp_rank, layer.tp_size), (rank // dcp, 4 // dcp))
                    layer.weight.weight_loader(layer.weight, full_weight)
                    start = (rank // dcp) * (6 * dcp)
                    torch.testing.assert_close(layer.weight, full_weight[start : start + 6 * dcp])
                    output = layer(x)[0].view(3, 2 * dcp, 3)
                    expected = torch.nn.functional.linear(x, full_weight).view(3, 8, 3)
                    torch.testing.assert_close(
                        output, expected[:, (rank // dcp) * (2 * dcp) : (rank // dcp + 1) * (2 * dcp)]
                    )
                    unquantized_gemm.assert_called_once()

    def test_prefill_projection_uses_local_weight_width_and_refreshes(self):
        for bias, rank, dcp in product([False, True], range(4), [2, 4]):
            with self.subTest(bias=bias, rank=rank, dcp=dcp), self._unquantized_gemm() as unquantized_gemm:
                model_config = SimpleNamespace(
                    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
                )
                full_weight = torch.arange(24 * 5, dtype=torch.float32).view(24, 5) / 100
                x = torch.arange(15, dtype=torch.float32).view(3, 5)
                with self._parallel_context(rank, dcp):
                    layer = AscendDCPGroupColumnParallelLinear(
                        5, 24, bias=bias, prefix="model.layers.0.self_attn.q_proj"
                    )
                    captured_weights = None
                    for offset in (0, 100):
                        layer.weight.weight_loader(layer.weight, full_weight + offset)
                        if bias:
                            layer.bias.weight_loader(layer.bias, torch.arange(24, dtype=torch.float32) + offset)
                        process_weights_after_loading(layer, model_config, torch.device("cpu"))
                        if captured_weights is None:
                            captured_weights = dict(layer.local_proj.named_parameters())
                        for name, tensor in layer.local_proj.named_parameters():
                            self.assertEqual(tensor.data_ptr(), captured_weights[name].data_ptr())
                        self.assertIsInstance(layer.local_proj, linear.AscendColumnParallelLinear)
                        self.assertTrue(layer.local_proj.is_weights_processed)
                        self.assertEqual((layer.local_proj.tp_rank, layer.local_proj.tp_size), (rank, 4))
                        layer.update_param_tp_status()
                        self.assertEqual((layer.local_proj.weight.tp_rank, layer.local_proj.weight.tp_size), (rank, 4))
                        unquantized_gemm.reset_mock()
                        actual, output_bias = layer.forward_local(x)
                        local_weight = (full_weight + offset)[rank * 6 : (rank + 1) * 6]
                        local_bias = (
                            (torch.arange(24, dtype=torch.float32) + offset)[rank * 6 : (rank + 1) * 6]
                            if bias
                            else None
                        )
                        torch.testing.assert_close(actual, torch.nn.functional.linear(x, local_weight, local_bias))
                        self.assertIs(output_bias, None)
                        self.assertEqual(unquantized_gemm.call_count, 1)
                        self.assertEqual(unquantized_gemm.call_args.args[1].shape, (6, 5))
                        torch.testing.assert_close(unquantized_gemm.call_args.args[1], local_weight)

    def test_quantized_local_projection_is_derived_before_transpose_and_processed_once(self):
        for layerwise, rank, dcp in product([False, True], range(4), [2, 4]):
            with self.subTest(layerwise=layerwise, rank=rank, dcp=dcp):
                self._check_quantized_local_projection_is_derived_before_transpose_and_processed_once(
                    layerwise, rank, dcp
                )

    def _check_quantized_local_projection_is_derived_before_transpose_and_processed_once(self, layerwise, rank, dcp):
        with self._quantization() as quantization:
            model_config = SimpleNamespace(
                word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
            )
            quant, matmul = quantization
            full_weight = torch.arange(24 * 5, dtype=torch.int8).view(24, 5)
            scales = torch.arange(1, 25, dtype=torch.float32).view(24, 1)
            offsets = torch.zeros_like(scales)
            x = torch.arange(15, dtype=torch.float32).view(3, 5)
            processing_order = []
            original_process = AscendW8A8DynamicLinearMethod.process_weights_after_loading

            def record_process(method, layer):
                processing_order.append(layer)
                if isinstance(layer, linear._DCPDerivedColumnParallelLinear):
                    self.assertFalse(layer.is_weights_processed)
                # The scheme must see raw [output, input] tensors on both invocations.
                self.assertEqual(layer.weight.shape, (layer.output_size_per_partition, 5))
                original_process(method, layer)

            with (
                self._parallel_context(rank, dcp),
                patch.object(AscendW8A8DynamicLinearMethod, "process_weights_after_loading", record_process),
            ):
                layer = AscendDCPGroupColumnParallelLinear(
                    5, 24, quant_config=quant, prefix="model.layers.0.self_attn.q_proj"
                )
                initial_local_proj = layer.local_proj
                self.assertIs(initial_local_proj, None)
                self.assertFalse(any((name.startswith("local_proj.") for name, _ in layer.named_parameters())))
                for name, tensor in (
                    ("weight", full_weight),
                    ("weight_scale", scales),
                    ("weight_offset", offsets),
                ):
                    param = getattr(layer, name)
                    param.weight_loader(param, tensor)
                if layerwise:
                    layer.quant_method.process_weights_after_loading(layer)
                else:
                    process_weights_after_loading(layer, model_config, torch.device("cpu"))
                assert layer.local_proj is not None
                self.assertTrue(layer.local_proj.is_weights_processed)
                layer.local_proj.quant_method.process_weights_after_loading(layer.local_proj)
                self.assertEqual(processing_order, [layer.local_proj, layer])
                self.assertEqual(layer.weight.shape, (5, 6 * dcp))
                self.assertEqual(layer.local_proj.weight.shape, (5, 6))
                local_slice = slice(rank * 6, (rank + 1) * 6)
                torch.testing.assert_close(layer.local_proj.weight_scale_fp32, scales[local_slice].flatten())
                local_result = layer.forward_local(x)[0]
                expected = (
                    torch.nn.functional.linear(x, full_weight[local_slice].float()) * scales[local_slice].flatten()
                )
                torch.testing.assert_close(local_result, expected)
                self.assertEqual(matmul.call_args.args[1].shape, (5, 6))
                group_start = (rank // dcp) * (6 * dcp)
                group_slice = slice(group_start, group_start + 6 * dcp)
                group_result = layer(x)[0]
                expected_group = (
                    torch.nn.functional.linear(x, full_weight[group_slice].float()) * scales[group_slice].flatten()
                )
                torch.testing.assert_close(group_result, expected_group)
                self.assertEqual(matmul.call_args.args[1].shape, (5, 6 * dcp))

    def test_invalid_local_projection_fails_explicitly(self):
        for case in ["output", "quant_method", "block_alignment", "not_prepared"]:
            with self.subTest(case=case):
                self._check_invalid_local_projection_fails_explicitly(case)

    def _check_invalid_local_projection_fails_explicitly(self, case):
        quant = SimpleNamespace(get_quant_method=lambda *args, **kwargs: AscendUnquantizedLinearMethod())
        if case == "quant_method":
            unsupported = AscendUnquantizedLinearMethod()
            unsupported.supports_weight_preprocessing = False
            quant.get_quant_method = lambda *args, **kwargs: unsupported
        with self._parallel_context():
            if case in ("output", "quant_method"):
                with self.assertRaisesRegex(ValueError, "DCP Q"):
                    AscendDCPGroupColumnParallelLinear(5, 25 if case == "output" else 24, quant_config=quant)
            else:
                layer = AscendDCPGroupColumnParallelLinear(5, 24, quant_config=quant)
                if case == "block_alignment":
                    layer.weight_block_size = (4, 4)
                    with self.assertRaisesRegex(ValueError, "quantization blocks"):
                        layer.prepare_weights_for_processing()
                else:
                    with self.assertRaisesRegex(RuntimeError, "must be prepared"):
                        layer.forward_local(torch.zeros(1, 5))

    def test_local_projection_preserves_captured_tensors_across_layerwise_reload(self):
        for quantized in [False, True]:
            with self.subTest(quantized=quantized), self._unquantized_gemm(), self._quantization() as quantization:
                model_config = SimpleNamespace(
                    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.float32
                )
                quant = quantization[0] if quantized else None
                dtype = torch.int8 if quantized else torch.float32
                weight = torch.arange(120, dtype=dtype).view(24, 5)
                x = torch.arange(10, dtype=torch.float32).view(2, 5)
                with self._parallel_context():
                    layer = AscendDCPGroupColumnParallelLinear(
                        5, 24, quant_config=quant, prefix="model.layers.0.self_attn.q_proj"
                    )
                    record_metadata_for_reloading(layer)
                    captured = None
                    for offset in (0, 1, 2):
                        if offset:
                            initialize_layerwise_reload(layer)
                        tensors = {"weight": weight + offset}
                        if quantized:
                            tensors.update(
                                weight_scale=torch.full((24, 1), 1.0 + offset), weight_offset=torch.zeros(24, 1)
                            )
                        for name, tensor in tensors.items():
                            parameter = getattr(layer, name)
                            parameter.weight_loader(parameter, tensor)
                        if offset:
                            finalize_layerwise_processing(layer, model_config)
                        else:
                            process_weights_after_loading(layer, model_config, torch.device("cpu"))
                            captured = dict(layer.local_proj._processed_weights)
                        assert captured is not None
                        for name, tensor in captured.items():
                            self.assertEqual(getattr(layer.local_proj, name).data_ptr(), tensor.data_ptr())
                        expected_weight = (weight + offset)[6:12].float()
                        expected = torch.nn.functional.linear(x, expected_weight)
                        if quantized:
                            expected *= 1 + offset
                            torch.testing.assert_close(captured["weight_scale_fp32"], torch.full((6,), 1.0 + offset))
                        torch.testing.assert_close(layer.forward_local(x)[0], expected)
                        torch.testing.assert_close(
                            captured["weight"].float(), expected_weight.T if quantized else expected_weight
                        )


if __name__ == "__main__":
    unittest.main()
