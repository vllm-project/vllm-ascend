# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for config-time grid completion before worker dispatchers exist.

The platform functions are compiled unchanged from their production AST so the
tests do not load the NPU plugin or construct models. Native graph capture and
memory cost still require hardware validation.
"""

import ast
import copy
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
PLATFORM = ROOT / "vllm_ascend/platform.py"


class GraphMode(Enum):
    NONE = 0
    FULL_DECODE_ONLY = 1
    PIECEWISE = 2

    def requires_piecewise_compilation(self):
        return self == GraphMode.PIECEWISE

    def has_full_cudagraphs(self):
        return self == GraphMode.FULL_DECODE_ONLY


def load_platform_functions():
    names = {"_complete_qwen_mrv2_capture_sizes", "_setup_compile_backend"}
    source = ast.parse(PLATFORM.read_text(encoding="utf-8"))
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]
    tree = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    namespace = {"logger": Mock(), "enable_sp": lambda config: False, "update_cudagraph_capture_sizes": Mock()}
    exec(compile(ast.fix_missing_locations(tree), str(PLATFORM), "exec"), namespace)
    return namespace


class TestQsaCaptureGrid(unittest.TestCase):
    def setUp(self):
        config_module = ModuleType("vllm.config")
        config_module.CompilationMode = SimpleNamespace(NONE="NONE", VLLM_COMPILE="VLLM_COMPILE")
        compilation_module = ModuleType("vllm.config.compilation")
        compilation_module.CUDAGraphMode = GraphMode
        modules = patch.dict(sys.modules, {"vllm.config": config_module, "vllm.config.compilation": compilation_module})
        modules.start()
        self.addCleanup(modules.stop)
        self.functions = load_platform_functions()

    def config(self, sizes=None, maximum=128, query_len=4, max_reqs=32):
        config = SimpleNamespace(
            use_v2_model_runner=True,
            model_config=SimpleNamespace(architecture="Qwen4ExpForCausalLM", enforce_eager=False),
            compilation_config=SimpleNamespace(
                cudagraph_mode=GraphMode.FULL_DECODE_ONLY,
                cudagraph_capture_sizes=sizes if sizes is not None else [1, 2, 4, 8, *range(16, 129, 8)],
                max_cudagraph_capture_size=maximum,
            ),
            speculative_config=SimpleNamespace(
                method="mtp",
                num_speculative_tokens=query_len - 1,
                uses_dynamic_speculative_decoding=lambda: False,
                enable_adaptive_verification=False,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_reqs, max_num_batched_tokens=4096),
            parallel_config=SimpleNamespace(tensor_parallel_size=2, data_parallel_size=1, pipeline_parallel_size=1),
            additional_config={"ascend_compilation_config": {}},
        )
        config._set_cudagraph_sizes = Mock()
        return config

    def complete(self, config):
        self.functions["_complete_qwen_mrv2_capture_sizes"](config)

    def test_default_sparse_grid_completed_without_larger_maximum(self):
        config = self.config()
        self.complete(config)
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, [1, 2, *range(4, 129, 4)])
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 128)

    def test_off_stride_explicit_maximum_preserved(self):
        config = self.config([1, 7, 17, 130], maximum=130, max_reqs=40)
        self.complete(config)
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, sorted({1, 7, 17, 130, *range(4, 129, 4)}))
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 130)

    def test_request_capacity_does_not_expand_requested_token_sizes(self):
        config = self.config([1, 2, 64], maximum=64, max_reqs=3)
        self.complete(config)
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, [1, 2, 4, 8, 12, 64])
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 64)

    def test_rounded_raw_bucket_uses_configured_maximum(self):
        config = self.config([1, 13], maximum=16)
        self.complete(config)
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, [1, 4, 8, 12, 13, 16])
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 16)

    def test_configured_maximum_smaller_than_raw_last_size_is_respected(self):
        config = self.config([1, 2, 64], maximum=16)
        self.complete(config)
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, [1, 2, 4, 8, 12, 16, 64])
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 16)

    def test_arbitrary_fixed_mtp_width_and_parallel_config(self):
        for query_len in (2, 3, 5, 9):
            with self.subTest(query_len=query_len):
                config = self.config([1, 2, 37], maximum=37, query_len=query_len, max_reqs=6)
                config.parallel_config = SimpleNamespace(
                    tensor_parallel_size=16, data_parallel_size=4, pipeline_parallel_size=3
                )
                self.complete(config)
                self.assertEqual(
                    config.compilation_config.cudagraph_capture_sizes,
                    sorted({1, 2, 37, *range(query_len, min(37, 6 * query_len) + 1, query_len)}),
                )

    def test_existing_complete_grid_is_idempotent(self):
        config = self.config([1, 2, 4, 8, 12, 16], maximum=16)
        original = config.compilation_config.cudagraph_capture_sizes
        self.complete(config)
        self.complete(config)
        self.assertIs(config.compilation_config.cudagraph_capture_sizes, original)
        self.functions["logger"].info.assert_not_called()

    def test_non_target_modes_and_models_keep_their_grid(self):
        mutations = (
            lambda c: setattr(c, "use_v2_model_runner", False),
            lambda c: setattr(c.model_config, "architecture", "OtherForCausalLM"),
            lambda c: setattr(c.model_config, "enforce_eager", True),
            lambda c: setattr(c.compilation_config, "cudagraph_mode", GraphMode.NONE),
            lambda c: setattr(c.compilation_config, "cudagraph_mode", GraphMode.PIECEWISE),
            lambda c: setattr(c, "speculative_config", None),
            lambda c: setattr(c.speculative_config, "method", "eagle"),
            lambda c: setattr(c.speculative_config, "num_speculative_tokens", 0),
            lambda c: setattr(c.speculative_config, "uses_dynamic_speculative_decoding", lambda: True),
            lambda c: setattr(c.speculative_config, "enable_adaptive_verification", True),
        )
        for mutate in mutations:
            config = self.config()
            original = config.compilation_config.cudagraph_capture_sizes
            mutate(config)
            self.complete(config)
            self.assertIs(config.compilation_config.cudagraph_capture_sizes, original)

    def test_multimodal_architecture_uses_same_grid(self):
        config = self.config()
        config.model_config.architecture = "Qwen4ExpForConditionalGeneration"
        self.complete(config)
        self.assertIn(12, config.compilation_config.cudagraph_capture_sizes)

    def test_platform_completes_after_upstream_recompute_before_worker_copy(self):
        config = self.config([1], maximum=128)

        def recompute():
            config.compilation_config.cudagraph_capture_sizes = [1, 2, 4, 8, *range(16, 129, 8)]

        config._set_cudagraph_sizes.side_effect = recompute
        self.functions["_setup_compile_backend"](config, compile_backend="test")
        config._set_cudagraph_sizes.assert_called_once_with()
        # Worker/speculator constructors receive this same normalized config.
        worker_config = copy.deepcopy(config)
        self.assertEqual(worker_config.compilation_config.cudagraph_capture_sizes, [1, 2, *range(4, 129, 4)])
        self.assertEqual(worker_config.compilation_config.max_cudagraph_capture_size, 128)
        self.assertEqual(worker_config.compilation_config.oot_compiler, "test")

    def test_existing_sp_alignment_remains_after_completion(self):
        config = self.config([1, 2, 4, 8, 16, 24, 32], maximum=32)
        config.parallel_config.tensor_parallel_size = 8
        self.functions["enable_sp"] = lambda c: True
        self.functions["_setup_compile_backend"](config, compile_backend="test")
        self.assertEqual(config.compilation_config.cudagraph_capture_sizes, [8, 16, 24, 32])
        self.assertEqual(config.compilation_config.max_cudagraph_capture_size, 32)
        self.functions["update_cudagraph_capture_sizes"].assert_called_once_with(config, [8, 16, 24, 32])


if __name__ == "__main__":
    unittest.main()
