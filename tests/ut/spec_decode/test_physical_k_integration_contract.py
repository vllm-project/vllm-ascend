# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only integration contracts for the physical-K overlay on PR #15098.

Run directly with Python when vLLM/torch_npu are not installed. These tests
exercise the real controller and sampler adapter, but do not replace NPU
graph-replay or end-to-end correctness tests.
"""

from __future__ import annotations

import ast
import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
SPECULATOR = ROOT / "vllm_ascend/worker/v2/spec_decode/dspark/speculator.py"


def load_standalone(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def method_node(path, class_name, method_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def method_calls(node):
    return {
        getattr(call.func, "attr", getattr(call.func, "id", "")): call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


controller_module = load_standalone(
    "vllm_ascend/spec_decode/dynamic/draft_k_controller.py", "physical_k_contract_controller"
)
config_module = load_standalone("vllm_ascend/dynamic_spec_config.py", "physical_k_contract_config")


class TestHybridController(unittest.TestCase):
    def controller(self, **kwargs):
        return controller_module.AdaptiveDraftKController(max_k=5, min_k=3, slack=0, hybrid_enabled=True, **kwargs)

    def test_low_acceptance_requires_hysteresis(self):
        controller = self.controller()
        self.assertEqual(controller.cap(5), 5)
        for _ in range(3):
            controller.observe([5] * 16, [[1, 2]] * 16)
            self.assertEqual(controller.current_k, 5)
        controller.observe([5] * 16, [[1, 2]] * 16)
        self.assertEqual(controller.current_k, 3)

    def test_small_batch_restores_full_width(self):
        controller = self.controller()
        controller.update([1] * 16)
        controller.observe([3] * 4, [[1, 2]] * 4)
        self.assertEqual(controller.current_k, 5)

    def test_high_acceptance_restores_full_width(self):
        controller = self.controller()
        controller.update([1] * 16)
        controller.observe([3] * 16, [[1, 2, 3, 4]] * 16)
        self.assertEqual(controller.current_k, 3)
        controller.observe([3] * 16, [[1, 2, 3, 4]] * 16)
        self.assertEqual(controller.current_k, 5)

    def test_periodic_probe_restores_full_width(self):
        controller = self.controller(hybrid_probe_interval=5)
        controller.cap(5)
        for _ in range(4):
            controller.observe([5] * 16, [[1, 2]] * 16)
        self.assertEqual(controller.current_k, 3)
        controller.observe([3] * 16, [[1, 2]] * 16)
        self.assertEqual(controller.current_k, 5)
        self.assertEqual(controller.last_reason, "periodic_full_k_probe")

    def test_explicit_zero_does_not_erase_recommendation(self):
        controller = self.controller()
        controller.update([1] * 16)
        self.assertEqual(controller.cap(0), 0)
        self.assertEqual(controller.current_k, 3)

    def test_config_normalization_is_idempotent(self):
        compact = {
            "method": "dspark",
            "policy": "hardware_aware",
            "physical_k": {"min_k": 3, "capture_k": [3, 5]},
        }
        resolved = config_module.resolve_method_params(compact)
        expanded = {"method_params": resolved}
        self.assertEqual(resolved, config_module.resolve_method_params(expanded))
        self.assertTrue(config_module.v2_physical_k_enabled(compact))
        self.assertTrue(config_module.v2_physical_k_enabled(expanded))

    def test_random_feedback_respects_bounds(self):
        controller = self.controller()
        rng = random.Random(15098)
        for _ in range(1000):
            k = controller.cap(5)
            batch = rng.choice([1, 4, 8, 16])
            controller.observe([k] * batch, [list(range(rng.randrange(k + 1) + 1)) for _ in range(batch)])
            self.assertGreaterEqual(controller.current_k, 3)
            self.assertLessEqual(controller.current_k, 5)


class TestRunnerWiring(unittest.TestCase):
    def test_piecewise_wrapper_is_entered(self):
        calls = method_calls(method_node(RUNNER, "NPUModelRunner", "initialize_kv_cache"))
        self.assertIn("adaptive_verification_gate_wrapper", calls)

    def test_allocation_precedes_attention_classification(self):
        calls = method_calls(method_node(RUNNER, "NPUModelRunner", "prepare_inputs"))
        self.assertLess(calls["compact_batch"], calls["reallocate_drafts"])
        self.assertLess(calls["reallocate_drafts"], calls["build_attn_state"])
        self.assertIn("copy_", calls)

    def test_next_width_is_published(self):
        node = method_node(RUNNER, "NPUModelRunner", "prepare_inputs")
        assignments = [
            item
            for item in ast.walk(node)
            if isinstance(item, ast.Assign)
            and any(getattr(target, "attr", None) == "_vllm_ascend_physical_draft_k" for target in item.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIn("num_spec_tokens_to_schedule", ast.unparse(assignments[0].value))

    def test_dsa_metadata_module_compiles(self):
        path = ROOT / "vllm_ascend/attention/dsa_v1.py"
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_dp_sync_signature_preserved(self):
        for relative in ("dspark", "dflash"):
            path = ROOT / f"vllm_ascend/worker/v2/spec_decode/{relative}/speculator.py"
            cls = "AscendDSparkSpeculator" if relative == "dspark" else "AscendDFlashSpeculator"
            args = [arg.arg for arg in method_node(path, cls, "propose").args.args]
            self.assertIn("num_tokens_across_dp", args)
            self.assertEqual(args.count("dp_sync"), 1)


class TestSamplerDelegation(unittest.TestCase):
    def sampler(self, active_k, fail=False):
        node = method_node(SPECULATOR, "AscendDSparkSpeculator", "_sample_sequential")

        class UpstreamSampler:
            def _sample_sequential(self, num_reqs, head_hidden):
                self.calls.append((self.draft_tokens, self.draft_token_confidence_probs))
                if fail:
                    raise RuntimeError("upstream failure")

        module = ast.parse("from __future__ import annotations\nclass Sampler(UpstreamSampler):\n    pass\n")
        module.body[1].body = [node]
        namespace = {"UpstreamSampler": UpstreamSampler}
        exec(compile(ast.fix_missing_locations(module), str(SPECULATOR), "exec"), namespace)
        sampler = namespace["Sampler"]()
        sampler.num_speculative_steps = active_k
        sampler._vllm_ascend_max_speculative_steps = 5
        sampler.draft_tokens = object()
        sampler.draft_token_confidence_probs = SimpleNamespace(ndim=2)
        sampler._physical_token_buffer = object()
        sampler._physical_confidence_buffers = {3: object()}
        sampler.calls = []
        return sampler

    def test_full_width_delegates_original_buffers(self):
        sampler = self.sampler(5)
        original = (sampler.draft_tokens, sampler.draft_token_confidence_probs)
        sampler._sample_sequential(2, None)
        self.assertEqual(sampler.calls, [original])

    def test_short_width_uses_adapters_then_restores(self):
        sampler = self.sampler(3)
        original = (sampler.draft_tokens, sampler.draft_token_confidence_probs)
        sampler._sample_sequential(2, None)
        self.assertEqual(sampler.calls, [(sampler._physical_token_buffer, sampler._physical_confidence_buffers[3])])
        self.assertEqual((sampler.draft_tokens, sampler.draft_token_confidence_probs), original)

    def test_short_width_restores_after_exception(self):
        sampler = self.sampler(3, fail=True)
        original = (sampler.draft_tokens, sampler.draft_token_confidence_probs)
        with self.assertRaisesRegex(RuntimeError, "upstream failure"):
            sampler._sample_sequential(2, None)
        self.assertEqual((sampler.draft_tokens, sampler.draft_token_confidence_probs), original)


if __name__ == "__main__":
    unittest.main()
