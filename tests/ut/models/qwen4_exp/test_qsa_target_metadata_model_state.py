# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU integration checks for removing fusion without changing target routing."""

import importlib.util
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / "vllm_ascend"
MODELS = SOURCE / "models/qwen4_exp"
PACKAGE = "qsa_no_fusion_integration"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NoFusionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.calls = []
        self.raise_in_builder = False
        owner = self

        class UpstreamState:
            pass

        class HybridState:
            def prepare_attn(self, input_batch, mode, *args, **kwargs):
                scope = owner.helper._SCOPE.get()
                owner.calls.append((self, input_batch, mode, args, kwargs, scope))
                if owner.raise_in_builder:
                    raise ValueError("upstream builder failed")
                return owner.result

        upstream = ModuleType("vllm.models.qwen4_exp.amd.model_state")
        upstream.Qwen4ExpModelState = UpstreamState
        hybrid = ModuleType("vllm_ascend.worker.v2.model_states.mamba_hybrid")
        hybrid.AscendMambaHybridModelState = HybridState
        torch = ModuleType("torch")
        torch.Tensor = object
        cache = ModuleType("vllm.models.qwen4_exp.common.qsa_cache")
        cache._build_qsa_metadata_torch = lambda *args, **kwargs: None
        common = ModuleType("vllm.models.qwen4_exp.common")
        common.qsa_cache = cache
        compilation = ModuleType("vllm.config.compilation")
        compilation.CUDAGraphMode = SimpleNamespace(FULL="FULL", FULL_DECODE_ONLY="FULL_DECODE_ONLY")
        package = ModuleType(PACKAGE)
        package.__path__ = [str(MODELS)]
        self.stack.enter_context(
            patch.dict(
                sys.modules,
                {
                    PACKAGE: package,
                    "torch": torch,
                    "vllm.models.qwen4_exp.amd.model_state": upstream,
                    "vllm.models.qwen4_exp.common": common,
                    "vllm.config.compilation": compilation,
                    "vllm_ascend.worker.v2.model_states.mamba_hybrid": hybrid,
                },
            )
        )
        self.helper = load_module(PACKAGE + ".qsa_target_metadata", MODELS / "qsa_target_metadata.py")
        self.addCleanup(sys.modules.pop, PACKAGE + ".qsa_target_metadata", None)
        self.module = load_module(PACKAGE + ".model_state", MODELS / "model_state.py")
        self.addCleanup(sys.modules.pop, PACKAGE + ".model_state", None)
        self.state = self.module.AscendQwen4ExpModelState()
        self.state.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=8,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            speculative_config=SimpleNamespace(num_speculative_tokens=3),
            model_config=SimpleNamespace(enforce_eager=False),
        )
        self.batch = SimpleNamespace(num_tokens_after_padding=8, num_reqs_after_padding=2)
        self.result = object()

    def test_supported_full_enters_original_producer_scope_and_forwards_arguments(self):
        result = self.state.prepare_attn(self.batch, "FULL", "extra", for_capture=False, marker=17)
        self.assertIs(result, self.result)
        self.assertEqual(len(self.calls), 1)
        state, batch, mode, args, kwargs, scope = self.calls[0]
        self.assertIs(state, self.state)
        self.assertIs(batch, self.batch)
        self.assertEqual((mode, args, kwargs), ("FULL", ("extra",), {"for_capture": False, "marker": 17}))
        self.assertTrue(scope.defer)
        self.assertFalse(scope.for_capture)
        self.assertEqual(scope.key, (8, 2))
        self.assertIsNone(self.helper._SCOPE.get())

    def test_each_unsupported_configuration_keeps_upstream_path(self):
        config = self.state.vllm_config
        changes = [
            (config.parallel_config, "tensor_parallel_size", 4),
            (config.parallel_config, "data_parallel_size", 2),
            (config.parallel_config, "pipeline_parallel_size", 2),
            (config.parallel_config, "decode_context_parallel_size", 2),
            (config.parallel_config, "prefill_context_parallel_size", 2),
            (config.compilation_config, "cudagraph_mode", "NONE"),
            (config.model_config, "enforce_eager", True),
            (config.speculative_config, "num_speculative_tokens", 2),
            (config, "speculative_config", None),
        ]
        for obj, field, value in changes:
            with self.subTest(field=field, value=value), patch.object(obj, field, value):
                self.assertFalse(self.state._qsa_target_metadata_enabled)
                self.assertIs(self.state.prepare_attn(self.batch, "FULL", marker=5), self.result)
                self.assertIsNone(self.calls[-1][-1])
                self.assertEqual(self.calls[-1][4], {"marker": 5})
                self.assertFalse(hasattr(self.state, "_qsa_target_metadata"))

    def test_non_full_batch_preserves_eager_scope_with_supported_configuration(self):
        self.state.prepare_attn(self.batch, "NONE")
        self.assertFalse(self.calls[-1][-1].defer)
        self.assertIsNone(self.state._qsa_target_metadata.pending)
        self.assertIsNone(self.helper._SCOPE.get())

    def test_capture_flag_reaches_real_preparation_only_inside_manager_capture(self):
        with self.helper.target_metadata_capture(self.state):
            self.state.prepare_attn(self.batch, "NONE", for_capture=True)
            scope = self.calls[-1][-1]
            self.assertTrue(scope.for_capture)
            self.assertTrue(scope.defer)
            self.assertEqual(self.calls[-1][4], {"for_capture": True})
        self.assertIsNone(self.helper._SCOPE.get())
        self.assertEqual(self.state._qsa_target_metadata.capture_depth, 0)

    def test_upstream_exception_propagates_and_cleans_target_scope(self):
        self.raise_in_builder = True
        with self.assertRaisesRegex(ValueError, "upstream builder failed"):
            self.state.prepare_attn(self.batch, "FULL")
        self.assertIsNone(self.helper._SCOPE.get())
        self.assertIsNone(self.state._qsa_target_metadata.pending)
        self.assertEqual(self.state._qsa_target_metadata.stats["preparation_failures"], 1)

    def test_stale_fusion_environment_cannot_switch_supported_path(self):
        for value in ("0", "1", "not-an-integer"):
            with self.subTest(value=value), patch.dict(os.environ, {"VLLM_ASCEND_QWEN4_EXP_FUSED_METADATA": value}):
                self.assertTrue(self.state._qsa_target_metadata_enabled)
                self.state.prepare_attn(self.batch, "FULL")
                self.assertTrue(self.calls[-1][-1].defer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
