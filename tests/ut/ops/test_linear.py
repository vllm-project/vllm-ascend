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
from vllm_ascend.ops.linear_op import OProjRowParallelOp


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

    @patch("vllm_ascend.ops.linear_op.get_potential_max_tokens", return_value=16)
    @patch("vllm_ascend.ops.linear.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_get_current_vllm_config, _mock_capacity):
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

        input_tensor = torch.randn(16, 16)
        with (
            patch(
                "vllm_ascend.ops.linear_op.dist.all_to_all_single",
                side_effect=lambda recv, send, group: recv.copy_(send),
            ),
            patch(
                "vllm_ascend.ops.linear_op.dist.reduce_scatter_tensor",
                side_effect=lambda output, input_, group: output.copy_(input_[: output.shape[0]]),
            ),
        ):
            output, _ = linear(input_tensor)
        self.assertEqual(output.shape, (16, 8))


class TestOProjRowParallelOp(unittest.TestCase):
    def test_uneven_dp_batches_preserve_projection_and_collective_buffers(self):
        capacity, input_size, output_size, tp_size = 4, 6, 3, 2
        chunk_size = input_size // tp_size
        weight = torch.arange(output_size * input_size, dtype=torch.float32).reshape(output_size, input_size)
        bias = torch.arange(output_size, dtype=torch.float32)
        group = SimpleNamespace(world_size=tp_size, rank_in_group=0, device_group=object())
        layers = [
            SimpleNamespace(
                weight=shard.contiguous(),
                bias=bias,
                skip_bias_add=False,
                return_bias=True,
                reduce_results=True,
                input_is_parallel=True,
                input_size_per_partition=chunk_size,
                output_size=output_size,
                prefix="o_proj",
                quant_method=SimpleNamespace(
                    apply=lambda layer, x, bias: torch.nn.functional.linear(x, layer.weight, bias)
                ),
            )
            for shard in weight.chunk(tp_size, dim=1)
        ]
        ops = [OProjRowParallelOp(layer) for layer in layers]
        for op in ops:
            op.update_attrs()
        buffer_addresses = {}

        with (
            patch("vllm_ascend.ops.linear_op.get_otp_group", return_value=group),
            patch("vllm_ascend.ops.linear_op.get_potential_max_tokens", return_value=capacity),
        ):
            # Shrinking batches must clear old padding; an idle rank still
            # contributes its weight shard to the active rank's projection.
            for batch_sizes in ((4, 1), (1, 3), (0, 2)):
                inputs = [torch.randn(size, input_size) for size in batch_sizes]
                padded = torch.stack([torch.nn.functional.pad(x, (0, 0, 0, capacity - x.shape[0])) for x in inputs])
                for rank, op in enumerate(ops):
                    group.rank_in_group = rank
                    received = padded[:, :, rank * chunk_size : (rank + 1) * chunk_size].reshape(-1, chunk_size)
                    expected = torch.nn.functional.linear(inputs[rank], weight, bias)
                    reduced = torch.nn.functional.linear(padded[rank], weight, bias)
                    with (
                        patch(
                            "vllm_ascend.ops.linear_op.dist.all_to_all_single",
                            side_effect=lambda recv, send, group, received=received: recv.copy_(received.reshape(-1)),
                        ) as exchange,
                        patch(
                            "vllm_ascend.ops.linear_op.dist.reduce_scatter_tensor",
                            side_effect=lambda result, partial, group, reduced=reduced: result.copy_(reduced),
                        ) as reduce,
                    ):
                        actual, output_bias = op.apply(inputs[rank])
                    torch.testing.assert_close(actual, expected)
                    self.assertIsNone(output_bias)
                    exchange.assert_called_once()
                    reduce.assert_called_once()
                    recv, send = exchange.call_args.args
                    result, partial = reduce.call_args.args
                    self.assertIs(exchange.call_args.kwargs["group"], group.device_group)
                    self.assertIs(reduce.call_args.kwargs["group"], group.device_group)
                    torch.testing.assert_close(
                        send.view(tp_size, capacity, chunk_size),
                        padded[rank].reshape(capacity, tp_size, chunk_size).transpose(0, 1),
                    )
                    torch.testing.assert_close(
                        partial, torch.nn.functional.linear(received, layers[rank].weight, bias if rank == 0 else None)
                    )
                    addresses = [buf.data_ptr() for buf in (send, recv, partial, result)]
                    self.assertEqual(addresses, buffer_addresses.setdefault(rank, addresses))

            group.rank_in_group = 0
            with (
                patch("vllm_ascend.ops.linear_op.is_forward_context_available", return_value=True),
                patch(
                    "vllm_ascend.ops.linear_op.get_forward_context",
                    return_value=SimpleNamespace(attn_metadata=None),
                ) as forward_context,
                patch(
                    "vllm_ascend.ops.linear_op.dist.all_to_all_single",
                    side_effect=lambda recv, send, group: recv.zero_(),
                ) as exchange,
                patch(
                    "vllm_ascend.ops.linear_op.dist.reduce_scatter_tensor",
                    side_effect=lambda result, partial, group: result.zero_(),
                ) as reduce,
            ):
                # KDA keeps the full profiling batch before its output
                # projection, even though OTP only needs decode capacity.
                profile_size = capacity * 8
                actual, _ = ops[0].apply(torch.randn(profile_size, input_size))
                torch.testing.assert_close(actual, torch.zeros(profile_size, output_size))
                exchange.assert_called_once()
                reduce.assert_called_once()
                recv, send = exchange.call_args.args
                result, partial = reduce.call_args.args
                torch.testing.assert_close(send, torch.zeros(tp_size * capacity * chunk_size))
                self.assertEqual([buf.data_ptr() for buf in (send, recv, partial, result)], buffer_addresses[0])

                forward_context.return_value.attn_metadata = {}
                exchange.reset_mock()
                with self.assertRaisesRegex(ValueError, "capacity must cover local tokens"):
                    ops[0].apply(torch.empty(capacity + 1, input_size))
                exchange.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
