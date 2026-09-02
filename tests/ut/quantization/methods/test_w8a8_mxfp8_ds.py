from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.fp8 import AscendW8A8MXFP8DSDynamicLinearMethod


def _mock_vllm_config(o_groups: int, o_lora_rank: int):
    """Minimal vllm_config mock for AscendW8A8MXFP8DSDynamicLinearMethod."""
    mock = Mock()
    mock.model_config = Mock(hf_config=Mock(o_groups=o_groups, o_lora_rank=o_lora_rank))
    # parent __init__ reads quant_config.quant_description.get("group_size", 32)
    mock.quant_config = Mock(quant_description={"group_size": 32})
    return mock


# Concrete dims chosen so block_size (128) divides the per-rank output/input:
#   n_groups=8, o_lora_rank=128, otp_size=2 -> groups_per_rank=4
#   output_per_rank = 4 * 128 = 512 ; input(group_hidden_dim) = 256
N_GROUPS = 8
O_LORA_RANK = 128
BLOCK_SIZE = 128
INPUT_SIZE = 256


def _make_wo_a_layer(output_per_rank: int, tp_size: int):
    """Build a wo_a layer in the post-weight_loader (per-rank-sharded) state.

    ``tp_size`` is the layer's active TP-style size (standard TP, or the OTP
    group size when oproj_tp is on) -- the value the production
    AscendColumnParallelLinear caches via get_parallel_op and that
    process_weights_after_loading divides n_groups by.
    """
    layer = nn.Module()
    layer.prefix = "model.layers.0.self_attn.o_proj.wo_a"
    layer.tp_size = tp_size
    layer.weight = nn.Parameter(
        torch.randint(-8, 8, (output_per_rank, INPUT_SIZE), dtype=torch.int8).to(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = nn.Parameter(
        torch.ones((output_per_rank // BLOCK_SIZE, INPUT_SIZE // BLOCK_SIZE), dtype=torch.float32),
        requires_grad=False,
    )
    return layer


class TestAscendW8A8MXFP8DSOProjTP(TestBase):
    """AscendW8A8MXFP8DSDynamicLinearMethod reshapes wo_a using the layer's own
    TP-style size (``layer.tp_size``), which already reflects oproj_tp when it
    is enabled. So the wo_a reshape's leading group dim is n_groups // layer.tp_size
    -- per-rank groups under OTP, full groups under standard TP."""

    def _make_scheme(self):
        # fp8.__init__ calls get_current_vllm_config in the fp8 namespace; its
        # parent w8a8_mxfp8.__init__ calls get_current_vllm_config /
        # ensure_mxfp8_linear_available in the w8a8_mxfp8 namespace. Both
        # get_current_vllm_config sites must return the same mock.
        mock_vllm = _mock_vllm_config(N_GROUPS, O_LORA_RANK)
        with (
            patch("vllm_ascend.quantization.methods.fp8.get_current_vllm_config", return_value=mock_vllm),
            patch("vllm_ascend.quantization.methods.w8a8_mxfp8.get_current_vllm_config", return_value=mock_vllm),
            patch("vllm_ascend.quantization.methods.w8a8_mxfp8.ensure_mxfp8_linear_available"),
        ):
            scheme = AscendW8A8MXFP8DSDynamicLinearMethod({"weight_block_size": [BLOCK_SIZE, BLOCK_SIZE]})
        return scheme

    def test_n_groups_read_from_config(self):
        scheme = self._make_scheme()
        self.assertEqual(scheme.n_groups, N_GROUPS)
        self.assertEqual(scheme.o_lora_rank, O_LORA_RANK)

    def test_wo_a_reshape_leading_dim_matches_groups_per_rank(self):
        # oproj_tp on: layer.tp_size == otp_size == 2, so each rank holds
        # groups_per_rank = n_groups // 2 groups (groups_per_rank * o_lora_rank
        # rows). process_weights_after_loading must reshape wo_a so the leading
        # group dim equals groups_per_rank -- otherwise the runtime batch matmul
        # (batch=groups_per_rank) desyncs.
        scheme = self._make_scheme()
        groups_per_rank = N_GROUPS // 2
        output_per_rank = groups_per_rank * O_LORA_RANK
        layer = _make_wo_a_layer(output_per_rank, tp_size=2)
        scheme.process_weights_after_loading(layer)
        # weight final shape: (groups_per_rank, input, o_lora_rank)
        self.assertEqual(layer.weight.shape[0], groups_per_rank)
        self.assertEqual(layer.weight.shape[2], O_LORA_RANK)
        # weight_scale leading dim tracks groups_per_rank too.
        self.assertEqual(layer.weight_scale.shape[0], groups_per_rank)

    def test_wo_a_reshape_standard_tp(self):
        # Regression guard: standard TP (layer.tp_size == 1), full weight
        # (n_groups groups per rank).
        scheme = self._make_scheme()
        output_full = N_GROUPS * O_LORA_RANK
        layer = _make_wo_a_layer(output_full, tp_size=1)
        scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape[0], N_GROUPS)

    def test_wo_a_reshape_tp4(self):
        # Standard TP with tp_size=4: per-rank groups = n_groups // 4.
        scheme = self._make_scheme()
        groups_per_rank = N_GROUPS // 4
        output_per_rank = groups_per_rank * O_LORA_RANK
        layer = _make_wo_a_layer(output_per_rank, tp_size=4)
        scheme.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape[0], groups_per_rank)
