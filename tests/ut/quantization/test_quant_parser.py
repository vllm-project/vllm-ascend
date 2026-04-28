from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_parser import (
    get_rollback_quant_type,
    parse_mxfp_quant_params,
    parse_quant_moe_down_proj_params,
)


class TestGetRollbackQuantType(TestBase):
    def test_returns_down_proj_quant_type(self):
        config = {
            "model.layers.0.mlp.gate_proj": "W8A8_MXFP8",
            "model.layers.0.mlp.down_proj": "W4A4_MXFP4",
        }
        result = get_rollback_quant_type(config)
        self.assertEqual(result, "W4A4_MXFP4")

    def test_returns_default_when_no_down_proj(self):
        config = {"model.layers.0.mlp.gate_proj": "W4A8_MXFP"}
        result = get_rollback_quant_type(config)
        self.assertEqual(result, "W8A8_MXFP8")


class TestParseMxfpQuantParams(TestBase):
    def test_default_values(self):
        act, weight, scale, per_token, round_mode = parse_mxfp_quant_params()
        self.assertIsNotNone(act)
        self.assertIsNotNone(weight)
        self.assertIsNotNone(round_mode)


class TestParseQuantMoeDownProjParams(TestBase):
    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp8_scale_dtype_available")
    def test_w8a8_mxfp8_uses_rint_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params("W8A8_MXFP8", "round")
        self.assertEqual(round_mode, "rint")

    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp4_dtype_available")
    def test_w4a4_mxfp4_respects_parsed_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params("W4A4_MXFP4", "round")
        self.assertEqual(round_mode, "round")
