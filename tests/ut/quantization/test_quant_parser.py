from unittest.mock import patch
import torch
from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_parser import (
    QuantTypeMapping,
    get_rollback_quant_type,
    parse_mxfp_quant_params,
    parse_quant_moe_down_proj_params,
)


class TestQuantTypeMapping(TestBase):

    def test_get_quant_settings_returns_dict(self):
        settings = QuantTypeMapping.get_quant_settings()
        self.assertIsInstance(settings, dict)

    def test_get_quant_settings_contains_expected_keys(self):
        settings = QuantTypeMapping.get_quant_settings()
        self.assertIn("W8A8_MXFP8", settings)
        self.assertIn("W4A4_MXFP4", settings)
        self.assertIn("W4A8_MXFP", settings)

    def test_w8a8_mxfp8_settings(self):
        settings = QuantTypeMapping.get_quant_settings()["W8A8_MXFP8"]
        self.assertEqual(settings["act_quant_type"], torch.float8_e4m3fn)
        self.assertIsNone(settings["weight_quant_type"])

    def test_w4a4_mxfp4_settings(self):
        settings = QuantTypeMapping.get_quant_settings()["W4A4_MXFP4"]
        self.assertIn("act_quant_type", settings)
        self.assertIn("weight_quant_type", settings)
        self.assertIn("scale_dtype", settings)
        self.assertIn("per_token_scale_dtype", settings)


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

    def test_returns_down_proj_type_with_multiple_entries(self):
        config = {
            "model.layers.0.mlp.gate_proj": "W8A8_MXFP8",
            "model.layers.0.mlp.up_proj": "W8A8_MXFP8",
            "model.layers.0.mlp.down_proj": "W4A8_MXFP",
        }
        result = get_rollback_quant_type(config)
        self.assertEqual(result, "W4A8_MXFP")


class TestParseMxfpQuantParams(TestBase):

    def test_default_values(self):
        act, weight, scale, per_token, round_mode = parse_mxfp_quant_params()
        self.assertEqual(act, torch.float8_e4m3fn)
        self.assertEqual(weight, torch.float8_e4m3fn)
        self.assertIsNone(scale)
        self.assertIsNone(per_token)
        self.assertEqual(round_mode, "rint")

    def test_custom_values(self):
        act, weight, scale, per_token, round_mode = parse_mxfp_quant_params(
            act_quant_type=torch.float16,
            weight_quant_type=torch.float8_e4m3fn,
            round_mode="round",
        )
        self.assertEqual(act, torch.float16)
        self.assertEqual(weight, torch.float8_e4m3fn)
        self.assertEqual(round_mode, "round")

    def test_scale_type_overrides(self):
        act, weight, scale, per_token, round_mode = parse_mxfp_quant_params(
            scale_type="float8_e8m0fnu",
            per_token_scale_type="float8_e8m0fnu",
        )
        self.assertEqual(scale, "float8_e8m0fnu")
        self.assertEqual(per_token, "float8_e8m0fnu")


class TestParseQuantMoeDownProjParams(TestBase):

    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp8_scale_dtype_available")
    def test_w8a8_mxfp8_uses_rint_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params(
            "W8A8_MXFP8", "round"
        )
        self.assertEqual(round_mode, "rint")

    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp4_dtype_available")
    def test_w4a4_mxfp4_respects_parsed_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params(
            "W4A4_MXFP4", "round"
        )
        self.assertEqual(round_mode, "round")

    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp4_dtype_available")
    def test_w4a4_mxfp4_rint_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params(
            "W4A4_MXFP4", "rint"
        )
        self.assertEqual(round_mode, "rint")

    @patch("vllm_ascend.quantization.quant_parser.ensure_mxfp8_scale_dtype_available")
    def test_w4a8_mxfp_uses_rint_round_mode(self, mock_ensure):
        mock_ensure.return_value = None
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params(
            "W4A8_MXFP", "round"
        )
        self.assertEqual(round_mode, "rint")
