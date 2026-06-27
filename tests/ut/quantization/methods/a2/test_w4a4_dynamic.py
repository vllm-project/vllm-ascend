from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_moe_layer,
)
from vllm_ascend.quantization.methods import registry
from vllm_ascend.quantization.methods.w4a4_dynamic import AscendW4A4DynamicFusedMoEMethod


class TestAscendW4A4DynamicFusedMoEMethod(TestBase):
    # Per-rank shapes the mega kernel is compiled for (process_weights fails fast
    # on anything else). gate_up transpose -> K=H (%16==0), N=2I (%64==0);
    # down -> K=I (%16==0), N=H (%64==0).
    num_experts = 2
    hidden_size = 2048
    intermediate_size = 128

    # get_mc2_group() needs a distributed env; raise AttributeError so the
    # parent __init__'s own try/except falls back to an empty comm-group name.
    @patch("vllm_ascend.quantization.methods.w4a4_dynamic.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.get_mc2_group", side_effect=AttributeError)
    @patch("vllm_ascend.quantization.methods.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.get_current_vllm_config")
    def setUp(self, mock_vllm, mock_ascend, mock_mc2, mock_vllm_w4a4):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        # The scheme's __init__ requires the Hadamard converter's marker in the quant
        # description; supply it so construction succeeds (see test_rejects_* for the
        # missing-marker path).
        mock_vllm_w4a4.return_value.quant_config.quant_description = {"hadamard_block_size": 64}
        self.scheme = AscendW4A4DynamicFusedMoEMethod()

    @patch("vllm_ascend.quantization.methods.w4a4_dynamic.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w4a8.get_mc2_group", side_effect=AttributeError)
    @patch("vllm_ascend.quantization.methods.w4a8.get_ascend_config")
    @patch("vllm_ascend.quantization.methods.w4a8.get_current_vllm_config")
    def test_rejects_non_hadamard_checkpoint(self, mock_vllm, mock_ascend, mock_mc2, mock_vllm_w4a4):
        # A plain W4A4_DYNAMIC checkpoint (no hadamard_block_size marker) must fail fast,
        # since the kernel unconditionally applies its in-kernel block-diagonal Hadamard.
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        mock_vllm_w4a4.return_value.quant_config.quant_description = {}
        with self.assertRaises(NotImplementedError):
            AscendW4A4DynamicFusedMoEMethod()

    def test_registered_under_w4a4_dynamic_moe(self):
        self.assertIs(
            registry._SCHEME_REGISTRY[("W4A4_DYNAMIC", "moe")],
            AscendW4A4DynamicFusedMoEMethod,
        )

    def test_init_forces_per_channel(self):
        # W4A4 checkpoints are per-output-channel (group_size == 0).
        self.assertEqual(self.scheme.group_size, 0)
        self.assertTrue(self.scheme.is_per_channel_weight)

    def test_process_weights_after_loading(self):
        E, H, i_dim = self.num_experts, self.hidden_size, self.intermediate_size
        layer = create_moe_layer(
            num_experts=E,
            hidden_size=H,
            intermediate_size=i_dim,
            params_dtype=torch.float32,
        )
        self.scheme.process_weights_after_loading(layer)

        # FRACTAL_NZ int4 weights: flat int8, half the element count (2 int4/byte).
        self.assertEqual(layer.w13_nz.dtype, torch.int8)
        self.assertEqual(layer.w13_nz.numel(), E * H * (2 * i_dim) // 2)
        self.assertEqual(layer.w2_nz.numel(), E * i_dim * H // 2)
        # per-channel fp32 scales, squeezed.
        self.assertEqual(tuple(layer.w13_scale_mega.shape), (E, 2 * i_dim))
        self.assertEqual(tuple(layer.w2_scale_mega.shape), (E, H))
        self.assertEqual(layer.w13_scale_mega.dtype, torch.float32)
        # dims handed to the kernel: (E, H, I, 2I).
        self.assertEqual(layer.mega_dims, (E, H, i_dim, 2 * i_dim))
        # vendor int8 copies freed down to tiny placeholders.
        self.assertEqual(tuple(layer.w13_weight.data.shape), (E, 1, 1))
        self.assertEqual(tuple(layer.w2_weight.data.shape), (E, 1, 1))
