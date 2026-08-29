"""Tests for GLM-5.3-Flash (glm5_next) ModelSlim quantization config handling.

Covers the three tolerances added for the multimodal glm5_next wrapper and
its MTP draft predictor:
- the inner text model registers with model_type "glm5_next_text"
- layers absent from the quant description stay unquantized (FLOAT)
- the draft module queries "model.layers.N.*" while descriptions are keyed
  under "language_model.model.layers.N.*"
"""

from tests.ut.base import TestBase
from vllm_ascend.quantization.modelslim_config import (
    AscendModelSlimConfig,
    get_linear_quant_type,
    get_packed_modules_mapping,
)


class TestGlm5NextModelSlimTolerance(TestBase):
    def setUp(self):
        self.desc = {
            "group_size": 0,
            "metadata": {},
            # Fused (gate_up_proj) shards are keyed by their checkpoint names.
            "language_model.model.layers.0.mlp.gate_proj.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.0.mlp.up_proj.weight": "W8A8_DYNAMIC",
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            # An explicit FLOAT layer.
            "language_model.model.layers.0.self_attn.b_proj.weight": "FLOAT",
        }
        self.cfg = AscendModelSlimConfig(dict(self.desc))

    def test_glm5_next_text_has_packed_modules_mapping(self):
        """The inner text model registers with the *_text model_type."""
        for model_type in ("glm5_next", "glm5_next_text"):
            mapping = get_packed_modules_mapping(model_type)
            self.assertIn("gate_up_proj", mapping)
            self.assertEqual(mapping["gate_up_proj"], ["gate_proj", "up_proj"])
            self.assertIn("fused_qkvbfg_a_proj", mapping)

    def test_missing_layer_falls_back_to_float(self):
        """Layers absent from the description stay unquantized.

        The GLM-5.3 MTP SharedHead placeholder reuses the main lm_head and has
        no checkpoint entry; the quant lookup must not raise for it.
        """
        packed = get_packed_modules_mapping("glm5_next_text")
        quant_type = get_linear_quant_type(self.desc, "model.layers.45.head", packed)
        self.assertEqual(quant_type, "FLOAT")

    def test_is_layer_skipped_for_absent_and_float_layers(self):
        packed = get_packed_modules_mapping("glm5_next_text")
        # Absent from the description entirely -> skipped (unquantized).
        self.assertTrue(self.cfg.is_layer_skipped_ascend("model.layers.45.head", packed))
        # Explicit FLOAT layer -> skipped.
        prefix = "language_model.model.layers.0.self_attn.b_proj"
        self.assertTrue(self.cfg.is_layer_skipped_ascend(prefix, packed))
        # Quantized layer -> not skipped.
        prefix = "language_model.model.layers.0.mlp.experts.0.gate_proj"
        self.assertFalse(self.cfg.is_layer_skipped_ascend(prefix, packed))

    def test_draft_prefix_falls_back_to_wrapper_alias(self):
        """Draft 'model.layers.N.*' resolves to the wrapper's
        'language_model.model.layers.N.*' description keys."""
        mapped = self.cfg.quant_prefix_mapper("glm5_next", "model.layers.0.mlp.gate_up_proj")
        self.assertEqual(mapped, "language_model.model.layers.0.mlp.gate_up_proj")

    def test_draft_prefix_without_alias_stays_unmapped(self):
        """A prefix whose wrapper alias also misses keeps its original form."""
        mapped = self.cfg.quant_prefix_mapper("glm5_next", "model.layers.45.head")
        self.assertEqual(mapped, "model.layers.45.head")
