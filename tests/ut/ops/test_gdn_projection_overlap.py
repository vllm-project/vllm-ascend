# SPDX-License-Identifier: Apache-2.0
"""Test stream dependencies without importing the NPU runtime."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class GDNProjectionTest(unittest.TestCase):
    def setUp(self):
        self.trace: list[tuple[str, str]] = []
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
            dtype = "bf16"

            def record_stream(self, stream):
                trace.append(("lifetime", stream.name))

        self.tensor = Tensor()
        self.main = Stream()
        self.main.name = "main"
        self.capture = True
        self.config = SimpleNamespace(gdn_projection_overlap_max_tokens=8)
        self.compilation_mode = SimpleNamespace(NONE=0, VLLM_COMPILE=3)
        self.graph_mode = SimpleNamespace(NONE=0, PIECEWISE=1, FULL=2, FULL_DECODE_ONLY=3)
        self.method = type("UnquantizedLinearMethod", (), {})
        self.torch = SimpleNamespace(
            bfloat16="bf16",
            compiler=SimpleNamespace(is_compiling=lambda: False),
            npu=SimpleNamespace(
                Stream=Stream,
                current_device=lambda: 0,
                Event=Event,
                stream=lambda stream: stream,
                current_stream=lambda: self.main,
                is_current_stream_capturing=lambda: self.capture,
            ),
        )
        modules = {}
        for name, attrs in {
            "torch": vars(self.torch),
            "vllm.config": {
                "CompilationMode": self.compilation_mode,
                "CUDAGraphMode": self.graph_mode,
                "VllmConfig": SimpleNamespace,
            },
            "vllm.model_executor.layers.linear": {"UnquantizedLinearMethod": self.method},
            "vllm_ascend.ops.linear": {"AscendUnquantizedLinearMethod": self.method},
            "vllm_ascend.ascend_config": {"get_ascend_config": lambda: self.config},
        }.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/gdn_projection.py"
        spec = importlib.util.spec_from_file_location("gdn_projection_under_test", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the GDN projection module from {path}")
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
        )
        self.vllm_config = SimpleNamespace(
            lora_config=None,
            model_config=SimpleNamespace(dtype="bf16"),
            parallel_config=SimpleNamespace(use_ubatching=False, prefill_context_parallel_size=1),
            compilation_config=SimpleNamespace(
                mode=self.compilation_mode.NONE,
                cudagraph_mode=self.graph_mode.FULL_DECODE_ONLY,
                static_forward_context={"first": self.layer},
            ),
        )

    def test_capture_forks_after_input_and_joins_before_consumer(self):
        self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
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

    def test_eager_ineligible_shape_and_dtype_keep_serial_path(self):
        self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
        for capturing, rows, dtype in [(False, 4, "bf16"), (True, 9, "bf16"), (True, 0, "bf16"), (True, 4, "fp16")]:
            with self.subTest(capturing=capturing, rows=rows, dtype=dtype):
                self.capture = capturing
                self.tensor.shape = (rows, 8)
                self.tensor.dtype = dtype
                self.trace.clear()
                self.module.project_gdn_inputs(self.layer, self.tensor)
                self.assertEqual(self.trace, [("gemm", "qkvz"), ("gemm", "ba")])

    def test_disabled_quantized_and_lora_paths_do_not_create_stream(self):
        for limit, lora, method in [(0, None, self.method()), (8, object(), self.method()), (8, None, object())]:
            self.config.gdn_projection_overlap_max_tokens = limit
            self.vllm_config.lora_config = lora
            self.layer.in_proj_ba.quant_method = method
            self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
            self.assertFalse(hasattr(self.layer, "_ascend_ba_stream"))

    def test_layers_share_device_stream_but_keep_separate_events(self):
        self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
        second = copy.copy(self.layer)
        self.module.initialize_gdn_projection_overlap(second, self.vllm_config)
        self.assertIs(second._ascend_ba_stream, self.layer._ascend_ba_stream)
        for first_event, second_event in zip(self.layer._ascend_ba_events, second._ascend_ba_events):
            self.assertIsNot(first_event, second_event)
        self.torch.npu.current_device = lambda: 1
        third = copy.copy(self.layer)
        self.module.initialize_gdn_projection_overlap(third, self.vllm_config)
        self.assertIsNot(third._ascend_ba_stream, self.layer._ascend_ba_stream)

    def test_eager_piecewise_and_compiled_config_do_not_create_stream(self):
        for graph, compile_mode in [
            (self.graph_mode.NONE, self.compilation_mode.NONE),
            (self.graph_mode.PIECEWISE, self.compilation_mode.NONE),
            (self.graph_mode.FULL, self.compilation_mode.VLLM_COMPILE),
        ]:
            self.vllm_config.compilation_config.cudagraph_mode = graph
            self.vllm_config.compilation_config.mode = compile_mode
            self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
            self.assertFalse(hasattr(self.layer, "_ascend_ba_stream"))

    def test_custom_collective_and_context_parallel_do_not_create_stream(self):
        for pcp, gather, custom_op in [(2, False, None), (1, True, None), (1, False, object())]:
            self.vllm_config.parallel_config.prefill_context_parallel_size = pcp
            self.layer.in_proj_ba.gather_output = gather
            self.layer.in_proj_qkvz.custom_op = custom_op
            self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
            self.assertFalse(hasattr(self.layer, "_ascend_ba_stream"))

    def test_dynamo_tracing_does_not_capture_stream_side_effects(self):
        self.module.initialize_gdn_projection_overlap(self.layer, self.vllm_config)
        self.torch.compiler.is_compiling = lambda: True
        self.module.project_gdn_inputs(self.layer, self.tensor)
        self.assertEqual(self.trace, [("gemm", "qkvz"), ("gemm", "ba")])


if __name__ == "__main__":
    unittest.main()
