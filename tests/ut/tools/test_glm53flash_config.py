# SPDX-License-Identifier: Apache-2.0
import copy
import unittest

from tools.ci.glm53flash_config import cut_config, cut_quant


class TestFlashCutConfig(unittest.TestCase):
    def setUp(self):
        self.source = {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "vision_config": {"depth": 24},
            "text_config": {
                "num_hidden_layers": 45,
                "num_nextn_predict_layers": 1,
                "layer_types": ["linear_attention" if i % 4 != 3 else "deepseek_sparse_attention" for i in range(45)],
                "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
                "indexer_types": ["full"] * 45,
                "linear_attn_config": {
                    "kda_layers": [i for i in range(45) if i % 4 != 3],
                    "full_attn_layers": list(range(3, 45, 4)),
                },
                "hidden_size": 4096,
                "n_routed_experts": 288,
                "num_experts_per_tok": 8,
            },
        }

    def test_nine_layers(self):
        original = copy.deepcopy(self.source)
        reduced = cut_config(self.source)
        self.assertEqual(self.source, original)
        self.assertEqual(reduced["num_hidden_layers"], 9)
        self.assertEqual(reduced["linear_attn_config"]["full_attn_layers"], [3, 7])
        self.assertEqual(reduced["linear_attn_config"]["kda_layers"], [0, 1, 2, 4, 5, 6, 8])
        self.assertEqual(len(reduced["mlp_layer_types"]), 9)
        self.assertEqual(reduced["num_nextn_predict_layers"], 0)
        self.assertEqual(reduced["n_routed_experts"], 288)

    def test_multimodal_preserves_vision(self):
        reduced = cut_config(self.source, mtp=True, multimodal=True)
        self.assertEqual(reduced["vision_config"], self.source["vision_config"])
        self.assertEqual(reduced["text_config"]["num_nextn_predict_layers"], 1)

    def test_five_layers(self):
        self.assertEqual(cut_config(self.source, layers=5)["linear_attn_config"]["full_attn_layers"], [3])

    def test_unreviewed_layer_field(self):
        self.source["text_config"]["future_layer_field"] = list(range(45))
        with self.assertRaises(ValueError):
            cut_config(self.source)

    def test_fp8_not_silently_reused(self):
        self.source["quantization_config"] = {"quant_method": "fp8"}
        with self.assertRaises(ValueError):
            cut_config(self.source)

    def test_quant_mtp_mapping(self):
        source = {
            "model_quant_type": "W8A8_DYNAMIC",
            "model.language_model.layers.3.mlp.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.3.self_attn.weight": "FLOAT",
            "model.language_model.layers.9.mlp.weight": "W8A8_DYNAMIC",
            "model.language_model.layers.45.mlp.weight": "W8A8_DYNAMIC",
        }
        result = cut_quant(source, 45, mtp=True)
        self.assertEqual(result["model.layers.3.self_attn.weight"], "FLOAT")
        self.assertEqual(result["model.layers.9.mlp.weight"], "W8A8_DYNAMIC")
        self.assertEqual(len(result), 4)
        self.assertNotIn("model.layers.9.mlp.weight", cut_quant(source, 45))

    def test_quant_fp8_rejected(self):
        with self.assertRaises(ValueError):
            cut_quant({"model_quant_type": "FP8"}, 45)


if __name__ == "__main__":
    unittest.main()
