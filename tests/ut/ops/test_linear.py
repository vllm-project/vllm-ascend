import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from tests.ut.base import TestBase
from vllm_ascend import ascend_config, ascend_forward_context
from vllm_ascend.ascend_forward_context import override_mrv2_in_profile_run
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops import linear_op
from vllm_ascend.ops.linear import (
    AscendMergedColumnParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
    AscendUnquantizedLinearMethod,
)
from vllm_ascend.ops.linear_op import (
    MLPColumnParallelOp,
    MLPRowParallelOp,
    OProjRowParallelOp,
    _mlp_tp_local_tokens,
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

        # The list mixes _patch instantiations of different generics, so give
        # it a uniform type for the start()/stop() loops below.
        self.patches: list[Any] = [
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
            # The fine-grained TP ops self-pad into static-capacity buffers and
            # call raw torch.distributed collectives; stub them shape-agnostically.
            patch("vllm_ascend.ops.linear_op.get_potential_max_tokens", return_value=32),
            patch("torch.distributed.all_to_all_single", lambda recv, send, group=None: recv.copy_(send)),
            patch("torch.distributed.all_gather_into_tensor", lambda out, inp, group=None: out.zero_()),
            patch("torch.distributed.reduce_scatter_tensor", lambda out, inp, group=None: out.zero_()),
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
            prefix="mlp.down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        # The down op trims with the token count the gate_up op recorded.
        input_tensor = torch.randn(16, 8)
        _mlp_tp_local_tokens["mlp"] = 16
        self.addCleanup(_mlp_tp_local_tokens.clear)
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

        # Feature dim must equal input_size (tp_size x input_size_per_partition):
        # the op reshapes [num_tokens, tp, chunk] for the exchange.
        input_tensor = torch.randn(16, 16)
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


# --- Self-padded static-capacity exchange (OProjRowParallelOp / MLP TP ops) ---
# Collectives are faked with a deterministic peer rank; real HCCL behavior is
# hardware-tested (the oversize fail-fast surfaces as a dynamo ``Unsupported``
# under torch.compile, so it is asserted on the eager path only).
CAPACITY, TP_SIZE, CHUNK = 8, 2, 3
# The peer rank's all_to_all send buffer, all_gather input and reduce_scatter contribution.
_PEER = SimpleNamespace(all_to_all_send=torch.zeros(0), ag_in=torch.zeros(0), rs_chunk=torch.zeros(0))


def _layer(prefix, weight, output_size, input_size_per_partition=None):
    """Minimal layer surface consumed by the ops' update_attrs/apply_impl."""
    return SimpleNamespace(
        prefix=prefix,
        weight=weight,
        output_size=output_size,
        bias=None,
        quant_method=SimpleNamespace(apply=lambda layer, x, bias=None: F.linear(x, layer.weight, bias)),
        skip_bias_add=False,
        return_bias=False,
        gather_output=False,
        input_is_parallel=True,
        reduce_results=True,
        input_size_per_partition=weight.shape[1] if input_size_per_partition is None else input_size_per_partition,
    )


def _pad_rows(x, capacity=CAPACITY):
    out = x.new_zeros((capacity, x.shape[1]))
    out[: x.shape[0]] = x
    return out


class TestSelfPaddedExchanges(unittest.TestCase):
    """Padding choreography and oversize behavior of the o_proj / MLP TP ops."""

    def setUp(self):
        group = SimpleNamespace(world_size=TP_SIZE, rank_in_group=0, device_group=object())
        # The list mixes _patch instantiations of different generics.
        patches: list[Any] = [
            patch("vllm_ascend.ops.linear_op.get_otp_group", return_value=group),
            patch("vllm_ascend.ops.linear_op.get_mlp_tp_group", return_value=group),
            patch("vllm_ascend.ops.linear_op.get_potential_max_tokens", return_value=CAPACITY),
            patch(
                "torch.distributed.all_to_all_single",
                lambda recv, send, group=None: recv.view(TP_SIZE, CAPACITY, CHUNK).copy_(
                    torch.stack([send.view(TP_SIZE, CAPACITY, CHUNK)[0], _PEER.all_to_all_send[0]])
                ),
            ),
            patch(
                "torch.distributed.all_gather_into_tensor",
                lambda out, inp, group=None: out.copy_(torch.cat([inp, _PEER.ag_in])),
            ),
            patch(
                "torch.distributed.reduce_scatter_tensor",
                lambda out, inp, group=None: out.copy_(inp[:CAPACITY] + _PEER.rs_chunk),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        linear_op._mlp_tp_local_tokens.clear()
        self.addCleanup(linear_op._mlp_tp_local_tokens.clear)
        # Oversize tests toggle the real mirror via override_mrv2_in_profile_run;
        # a patched getter here would shadow it. Reset the mirror instead.
        ascend_forward_context._IN_PROFILE_RUN = False
        self.addCleanup(setattr, ascend_forward_context, "_IN_PROFILE_RUN", False)

    def test_oproj_exchange_pads_and_trims(self):
        w = torch.randn(4, 2 * CHUNK)  # full [out, attn_dim]; rank 0 owns chunk 0
        op = OProjRowParallelOp(_layer("model.layers.0.self_attn.o_proj", w[:, :CHUNK], 4, CHUNK))
        op.update_attrs()
        x, x_peer = torch.randn(5, 2 * CHUNK), torch.randn(6, 2 * CHUNK)  # per-rank counts differ
        _PEER.all_to_all_send = torch.zeros(TP_SIZE, CAPACITY, CHUNK)
        _PEER.all_to_all_send[0, :6], _PEER.all_to_all_send[1, :6] = x_peer[:, :CHUNK], x_peer[:, CHUNK:]
        _PEER.rs_chunk = _pad_rows(x)[:, CHUNK:] @ w[:, CHUNK:].T
        out = op.apply(x)
        torch.testing.assert_close(out, x @ w.T)
        self.assertTrue(torch.all(op._send_buf[:, 5:] == 0))  # padding tail stays zero

    def test_oversize_raises_outside_profile_zero_fills_inside(self):
        op = OProjRowParallelOp(_layer("model.layers.0.self_attn.o_proj", torch.randn(4, CHUNK), 4, CHUNK))
        op.update_attrs()
        _PEER.all_to_all_send = torch.zeros(TP_SIZE, CAPACITY, CHUNK)
        _PEER.rs_chunk = torch.zeros(CAPACITY, 4)
        with self.assertRaisesRegex(ValueError, "static exchange capacity"):
            op.apply(torch.randn(CAPACITY + 1, 2 * CHUNK))
        # Real runner API (not a patch): also covers the profile-run mirror.
        with override_mrv2_in_profile_run(True):
            out = op.apply(torch.randn(CAPACITY + 1, 2 * CHUNK))
        self.assertEqual(out.shape, (CAPACITY + 1, 4))
        self.assertTrue(torch.all(out == 0))

    def test_mlp_gate_up_down_chain(self):
        hidden, inter = 4, 6
        w_gate_up, w_down = torch.randn(2 * inter, hidden), torch.randn(hidden, inter)
        gate_up = MLPColumnParallelOp(_layer("model.layers.0.mlp.gate_up_proj", w_gate_up, 2 * inter))
        gate_up.update_attrs()
        down = MLPRowParallelOp(_layer("model.layers.0.mlp.down_proj", w_down, hidden))
        down.update_attrs()

        # Regular chain: gate_up gathers each rank's padded tokens and records
        # the local count; down trims its reduce_scatter with that count.
        x, x_peer = torch.randn(5, hidden), torch.randn(6, hidden)
        _PEER.ag_in = _pad_rows(x_peer)
        gu_out = gate_up.apply(x)
        self.assertEqual(gu_out.shape, (TP_SIZE * CAPACITY, 2 * inter))
        self.assertEqual(linear_op._mlp_tp_local_tokens["model.layers.0.mlp"], 5)
        torch.testing.assert_close(gu_out, torch.cat([_pad_rows(x), _pad_rows(x_peer)]) @ w_gate_up.T)
        act = gu_out[:, :inter]  # stand-in for the activation between the two ops
        _PEER.rs_chunk = _pad_rows(x_peer) @ w_gate_up.T[:, :inter] @ w_down.T  # peer's partial sum
        out = down.apply(act)
        torch.testing.assert_close(out, (_pad_rows(x) @ w_gate_up.T[:, :inter] @ w_down.T)[:5] + _PEER.rs_chunk[:5])

        # Oversized profile batch: gate_up zero-fills, down skips the exchange.
        with override_mrv2_in_profile_run(True):
            gu_out = gate_up.apply(torch.randn(CAPACITY + 1, hidden))
            self.assertEqual(gu_out.shape, (CAPACITY + 1, 2 * inter))
            self.assertTrue(torch.all(gu_out == 0))
            out = down.apply(torch.randn(CAPACITY + 1, inter))
        self.assertEqual(out.shape, (CAPACITY + 1, hidden))
        self.assertTrue(torch.all(out == 0))

        # A down op whose gate_up never registered (non-fused up/down naming) fails loudly.
        linear_op._mlp_tp_local_tokens.clear()
        with self.assertRaisesRegex(ValueError, "gate_up-recorded"):
            down.apply(torch.randn(TP_SIZE * CAPACITY, inter))


if __name__ == "__main__":
    unittest.main()
