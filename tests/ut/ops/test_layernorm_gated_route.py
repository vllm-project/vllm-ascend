"""Source-level checks for the narrow PR1 wrapper boundary."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAYER_NORM = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated.py"
DISPATCH = ROOT / "vllm_ascend" / "ops" / "triton" / "layernorm_gated_dispatch.py"


class RouteSourceTests(unittest.TestCase):
    def test_sources_parse_and_exclude_wide_path_symbols(self):
        layer_source = LAYER_NORM.read_text()
        dispatch_source = DISPATCH.read_text()
        ast.parse(layer_source)
        ast.parse(dispatch_source)
        self.assertNotIn("c2_", layer_source.lower())
        self.assertNotIn("c2_", dispatch_source.lower())
        self.assertNotIn("compile_target", layer_source.lower())
        self.assertNotIn("try_get_compile_target", layer_source)

    def test_exact_device_targets_and_dtype_gate_are_present(self):
        source = LAYER_NORM.read_text()
        self.assertIn('"Ascend910B3"', source)
        self.assertIn('"Ascend910_9382"', source)
        self.assertIn("torch.float16", source)
        self.assertIn("torch.bfloat16", source)
        self.assertIn("try_get_vectorcore_num", source)
        self.assertIn("qualified=qualified", source)


if __name__ == "__main__":
    unittest.main()
