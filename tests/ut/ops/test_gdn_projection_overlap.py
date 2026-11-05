# SPDX-License-Identifier: Apache-2.0
"""Test stream dependencies without importing the NPU runtime."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class GDNProjectionTest(unittest.TestCase):
    def setUp(self):
        self.trace = []
        trace = self.trace

        class Event:
            def record(self, stream):
                trace.append(("record", stream.name))

        class Stream:
            name = "side"

            def wait_event(self, event):
                trace.append(("wait", self.name))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class Tensor:
            shape = (4, 8)

            def record_stream(self, stream):
                trace.append(("lifetime", stream.name))

        self.tensor = Tensor()
        self.main = Stream()
        self.main.name = "main"
        self.capture = True
        self.config = SimpleNamespace(gdn_projection_overlap_max_tokens=8)
        self.method = type("UnquantizedLinearMethod", (), {})
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            compiler=SimpleNamespace(is_compiling=lambda: False),
            npu=SimpleNamespace(
                Stream=Stream,
                Event=Event,
                stream=lambda stream: stream,
                current_stream=lambda: self.main,
                is_current_stream_capturing=lambda: self.capture,
            ),
        )
        modules = {}
        for name, attrs in {
            "torch": vars(self.torch),
            "vllm.model_executor.layers.linear": {"UnquantizedLinearMethod": self.method},
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": self.method},
            "vllm_ascend.ascend_config": {"get_ascend_config": lambda: self.config},
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn_projection.py"
        spec = importlib.util.spec_from_file_location("gdn_projection_under_test", path)
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(self.module)

        def projection(name):
            def run(hidden):
                trace.append(("gemm", name))
                return Tensor(), None

            return SimpleNamespace(quant_method=self.method(), run=run)

        class Projection:
            def __init__(inner, name):
                inner.impl = projection(name)
                inner.quant_method = inner.impl.quant_method

            def __call__(inner, hidden):
                return inner.impl.run(hidden)

        self.layer = SimpleNamespace(
            in_proj_qkvz=Projection("qkvz"),
            in_proj_ba=Projection("ba"),
            vllm_config=SimpleNamespace(
                lora_config=None,
                model_config=SimpleNamespace(dtype="bf16"),
                parallel_config=SimpleNamespace(use_ubatching=False),
            ),
        )

    def test_capture_forks_after_input_and_joins_before_consumer(self):
        self.module.initialize_gdn_projection_overlap(self.layer)
        self.module.project_gdn_inputs(self.layer, self.tensor)
        self.assertEqual(
            self.trace,
            [
                ("record", "main"),
                ("lifetime", "side"),
                ("wait", "side"),
                ("gemm", "ba"),
                ("record", "side"),
                ("gemm", "qkvz"),
                ("wait", "main"),
                ("lifetime", "main"),
            ],
        )

    def test_eager_and_large_batches_keep_serial_path(self):
        self.module.initialize_gdn_projection_overlap(self.layer)
        for capturing, shape in [(False, (4, 8)), (True, (9, 8))]:
            self.capture = capturing
            self.tensor.shape = shape
            self.trace.clear()
            self.module.project_gdn_inputs(self.layer, self.tensor)
            self.assertEqual(self.trace, [("gemm", "qkvz"), ("gemm", "ba")])

    def test_disabled_quantized_and_lora_paths_do_not_create_stream(self):
        for limit, lora, method in [(0, None, self.method()), (8, object(), self.method()), (8, None, object())]:
            self.config.gdn_projection_overlap_max_tokens = limit
            self.layer.vllm_config.lora_config = lora
            self.layer.in_proj_ba.quant_method = method
            self.module.initialize_gdn_projection_overlap(self.layer)
            self.assertIsNone(self.layer._ascend_ba_stream)

    def test_negative_limit_fails_before_creating_stream(self):
        self.config.gdn_projection_overlap_max_tokens = -1
        with self.assertRaises(ValueError):
            self.module.initialize_gdn_projection_overlap(self.layer)

    def test_dynamo_tracing_does_not_capture_stream_side_effects(self):
        self.module.initialize_gdn_projection_overlap(self.layer)
        self.torch.compiler.is_compiling = lambda: True
        self.module.project_gdn_inputs(self.layer, self.tensor)
        self.assertEqual(self.trace, [("gemm", "qkvz"), ("gemm", "ba")])


if __name__ == "__main__":
    unittest.main()
