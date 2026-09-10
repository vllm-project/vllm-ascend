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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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


controller_module = load_standalone("vllm_ascend/spec_decode/dynamic/policy.py", "physical_k_contract_controller")
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

    def test_speculators_explicitly_inherit_physical_k_adapters(self):
        for name, cls, mixin in (
            ("dspark", "AscendDSparkSpeculator", "PhysicalKDSparkMixin"),
            ("dflash", "AscendDFlashSpeculator", "PhysicalKDFlashMixin"),
        ):
            path = ROOT / f"vllm_ascend/worker/v2/spec_decode/{name}/speculator.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
            self.assertEqual(ast.unparse(node.bases[0]), mixin)


class TestSamplerDelegation(unittest.TestCase):
    def sampler(self, active_k, fail=False):
        node = method_node(
            ROOT / "vllm_ascend/worker/v2/spec_decode/physical_k.py", "PhysicalKDSparkMixin", "_sample_sequential"
        )

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


class TestRuntimeAdapters(unittest.TestCase):
    """Execute complete adapter modules against CPU-only upstream interfaces.

    Stub only the unavailable vLLM/torch imports. Policy, scheduler integration,
    profiling configuration and descriptor/capture adapters are the real code.
    Tensor kernels and real AsyncScheduler integration have separate UTs.
    """

    @classmethod
    def setUpClass(cls):
        # Load native NumPy extensions before patch.dict snapshots sys.modules;
        # removing and reimporting them for each test is unsafe.
        importlib.import_module("numpy")

    def setUp(self):
        modules = {}

        def module(name, **attrs):
            if name not in modules:
                modules[name] = ModuleType(name)
                modules[name].__path__ = []
                if "." in name:
                    parent, child = name.rsplit(".", 1)
                    setattr(module(parent), child, modules[name])
            vars(modules[name]).update(attrs)
            return modules[name]

        class GraphMode(Enum):
            NONE = 0
            FULL = 1

            def decode_mode(self):
                return self

        @dataclass(frozen=True)
        class Descriptor:
            cg_mode: GraphMode
            num_tokens: int
            num_reqs: int
            uniform_token_count: int
            num_active_loras: int

        module("torch")
        logger = Mock()
        logger.isEnabledFor.return_value = False
        module("vllm.logger", logger=logger)
        module("vllm.config.compilation", CUDAGraphMode=GraphMode)
        module("vllm.envs", VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=128)
        module("vllm.v1.kv_cache_interface", KVCacheConfig=object)
        module("vllm.v1.worker.gpu.block_table", BlockTables=object)
        module("vllm.v1.worker.gpu.cudagraph_utils", BatchExecutionDescriptor=Descriptor)
        module("vllm.v1.worker.gpu.input_batch", InputBuffers=object)
        module("vllm.v1.worker.utils", AttentionGroup=object)
        self.adaptive = module(
            "vllm.v1.worker.gpu.spec_decode.adaptive_verification",
            AdaptiveVerificationManager=Mock(),
            _assign_draft_token_budget=object(),
            _assign_draft_token_budget_compiled=object(),
        )
        self.capture = module("vllm.v1.worker.gpu.spec_decode.dflash.cudagraph")
        self.sched_module = module("vllm.v1.core.sched.scheduler")
        self.outputs = module("vllm.v1.outputs")
        module(
            "vllm_ascend.dynamic_spec_config",
            **{name: getattr(config_module, name) for name in ("resolve_method_params", "v2_physical_k_enabled")},
        )
        module(
            "vllm_ascend.spec_decode.dynamic.policy",
            AdaptiveDraftKController=controller_module.AdaptiveDraftKController,
            ProposalGate=controller_module.ProposalGate,
        )
        module("vllm_ascend.worker.v2.spec_decode")
        stub_imports = patch.dict(sys.modules, modules)
        stub_imports.start()
        self.addCleanup(stub_imports.stop)
        self.scheduler = load_standalone("vllm_ascend/core/dynamic_spec_scheduler.py", "contract_scheduler")
        self.physical = load_standalone(
            "vllm_ascend/worker/v2/spec_decode/physical_k.py", "vllm_ascend.worker.v2.spec_decode.physical_k"
        )
        self.verification = load_standalone(
            "vllm_ascend/worker/v2/spec_decode/verification.py", "contract_verification"
        )
        self.graph = load_standalone("vllm_ascend/worker/v2/spec_decode/physical_k_graph.py", "contract_graph")
        self.mode = GraphMode
        self.descriptor = Descriptor

    def config(self):
        return SimpleNamespace(
            use_v2_model_runner=True,
            additional_config={
                "dynamic_spec_config": {
                    "method": "dspark",
                    "policy": "hardware_aware",
                    "physical_k": {"min_k": 3, "capture_k": [3, 5]},
                }
            },
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
        )

    def install_scheduler(self):
        class Scheduler:
            def __init__(self, vllm_config):
                self.vllm_config = vllm_config

            def _update_after_schedule(self, output):
                self.placeholder_width = output.num_spec_tokens_to_schedule
                return "upstream-result"

        self.sched_module.Scheduler = Scheduler
        self.scheduler.install_scheduler_policy()
        return Scheduler

    def test_scheduler_install_is_idempotent(self):
        scheduler_cls = self.install_scheduler()
        original = (scheduler_cls.__init__, scheduler_cls._update_after_schedule)
        self.scheduler.install_scheduler_policy()
        self.assertEqual(original, (scheduler_cls.__init__, scheduler_cls._update_after_schedule))

    def test_scheduler_caps_before_placeholder_creation(self):
        scheduler = self.install_scheduler()(self.config())
        scheduler._ascend_physical_k_controller.update([1] * 16)
        for configured, expected in ((5, 3), (3, 3), (0, 0)):
            output = SimpleNamespace(
                num_spec_tokens_to_schedule=configured, scheduled_spec_decode_tokens={"a": [0] * 5}
            )
            self.assertEqual(scheduler._update_after_schedule(output), "upstream-result")
            self.assertEqual(scheduler.placeholder_width, expected)
            self.assertEqual(len(output.scheduled_spec_decode_tokens["a"]), 5)

    def test_disabled_scheduler_delegates_without_changing_width(self):
        config = self.config()
        config.additional_config = {}
        scheduler = self.install_scheduler()(config)
        output = SimpleNamespace(num_spec_tokens_to_schedule=5)
        self.assertIsNone(scheduler._ascend_physical_k_controller)
        self.assertEqual(scheduler._update_after_schedule(output), "upstream-result")
        self.assertEqual(output.num_spec_tokens_to_schedule, 5)

    def test_compact_and_legacy_scheduler_initialization_match(self):
        scheduler_cls = self.install_scheduler()
        compact = self.config()
        legacy = self.config()
        dynamic = legacy.additional_config["dynamic_spec_config"]
        dynamic["method_params"] = config_module.resolve_method_params(dynamic)
        del dynamic["physical_k"]
        a = scheduler_cls(compact)._ascend_physical_k_controller
        b = scheduler_cls(legacy)._ascend_physical_k_controller
        rng = random.Random(93924)
        for _ in range(1000):
            k = a.cap(5)
            b.cap(5)
            width = rng.choice([1, 4, 8, 16])
            tokens = [list(range(rng.randrange(k + 1) + 1)) for _ in range(width)]
            a.observe([k] * width, tokens)
            b.observe([k] * width, tokens)
            self.assertEqual(vars(a), vars(b))

    def test_feedback_preserves_native_and_legacy_bookkeeping(self):
        for native in (True, False):
            with self.subTest(native=native):
                controller = controller_module.AdaptiveDraftKController(max_k=5, min_k=3, slack=0)
                request = SimpleNamespace(spec_token_ids=[99], is_finished=lambda: False, is_prefill_chunk=False)
                scheduler = SimpleNamespace(_ascend_physical_k_controller=controller, requests={"a": request})
                output = SimpleNamespace(req_ids=["a"], proposal_lengths=[2])
                self.scheduler.update_dynamic_feedback(scheduler, None, output, native_proposal_lengths=native)
                self.assertEqual(controller.current_k, 3)
                self.assertEqual(request.spec_token_ids, [99] if native else [-1, -1])

    def test_feedback_uses_actual_scheduled_widths(self):
        for native in (True, False):
            controller = controller_module.AdaptiveDraftKController(max_k=5, min_k=3, slack=0)
            scheduler = SimpleNamespace(_ascend_physical_k_controller=controller)
            scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"a": [0] * 3})
            output = SimpleNamespace(req_ids=["prefill", "a"], sampled_token_ids=[[9], [1, 2, 3]])
            self.scheduler.update_dynamic_feedback(scheduler, scheduled, output, native_proposal_lengths=native)
            self.assertEqual(controller.last_scheduled_widths, [3])
            self.assertEqual(controller.last_accepted_lengths, [2])

    def test_malformed_legacy_lengths_do_not_modify_requests(self):
        scheduler = SimpleNamespace(requests={}, _latest_proposal_lengths={"old": 2})
        self.scheduler.update_dynamic_feedback(
            scheduler, None, SimpleNamespace(req_ids=["a"], proposal_lengths=[]), native_proposal_lengths=False
        )
        self.assertEqual(scheduler._latest_proposal_lengths, {"old": 2})

    def test_output_fields_keep_native_constructors(self):
        @dataclass
        class Output:
            spec_token_ids: object = None
            proposal_lengths: object = None

        self.outputs.ModelRunnerOutput = Output
        self.outputs.DraftTokenIds = Output
        self.outputs.EMPTY_MODEL_RUNNER_OUTPUT = Output()
        original = Output.__init__
        self.scheduler.install_output_fields()
        self.assertIs(Output.__init__, original)

    def test_output_field_backport_is_idempotent(self):
        @dataclass
        class Output:
            value: int = 7

        self.outputs.ModelRunnerOutput = Output
        self.outputs.DraftTokenIds = Output
        self.outputs.EMPTY_MODEL_RUNNER_OUTPUT = Output()
        self.scheduler.install_output_fields()
        original = Output.__init__
        self.scheduler.install_output_fields()
        self.assertIs(Output.__init__, original)
        output = Output(9, proposal_lengths=[3], spec_token_ids=[[1]])
        self.assertEqual((output.value, output.proposal_lengths, output.spec_token_ids), (9, [3], [[1]]))

    def test_request_reordering_and_clipping_share_one_helper(self):
        ids = ["b", "missing", "a", "negative"]
        tokens = [[1, 2, 3], [4, 5], [6, 7], [8]]
        trimmed, lengths = self.scheduler.trim_proposal_tokens(ids, tokens, {"a": 9, "b": 1, "negative": -2})
        self.assertEqual(trimmed, [[1], [4, 5], [6, 7], []])
        self.assertEqual(lengths, [1, 2, 9, -2])
        self.assertEqual(tokens, [[1, 2, 3], [4, 5], [6, 7], [8]])
        self.assertEqual(self.scheduler.trim_proposal_tokens([], [], None), ([], []))

    def manager(self):
        return SimpleNamespace(
            req_states=SimpleNamespace(max_num_batched_tokens=24),
            num_speculative_steps=5,
            set_cost_curves=Mock(),
        )

    def test_piecewise_profile_grid_and_median_curves(self):
        manager = self.manager()
        self.assertIs(self.verification.configure_piecewise_manager(manager), manager)
        self.assertEqual(
            list(manager.batches_to_profile([100])),
            [{"num_tokens": n, "context_len": 128} for n in (6, 12, 24) for _ in range(3)],
        )
        self.assertEqual(manager._cudagraph_limit, 0)
        samples = [
            SimpleNamespace(num_reqs=b, num_target_tokens=t, drafter_ms=d, forward_ms=f)
            for b, t, d, f in ((2, 6, 10, 30), (2, 6, 2, 4), (2, 6, 4, 8), (4, 12, 6, 9))
        ]
        manager.set_initial_cost_curves(samples)
        manager.set_cost_curves.assert_called_once_with([(2, 4.0), (4, 6.0)], [(6, 8.0), (12, 9.0)])

    def factory_args(self, enabled=True):
        return dict(
            enable_adaptive_verification=enabled,
            attn_groups=[],
            attn_cg_support=None,
            req_states=object(),
            query_start_loc=object(),
            num_bonus_tokens=1,
            max_total_logits=100,
        )

    def test_factory_reuses_manager_and_restores_upstream_entry(self):
        manager = self.manager()
        factory = Mock(return_value=manager)
        runner_module = SimpleNamespace(maybe_create_adaptive_verification_manager=factory)
        with self.verification.adaptive_verification_gate_wrapper(runner_module):
            self.assertIs(runner_module.maybe_create_adaptive_verification_manager(**self.factory_args()), manager)
        self.assertIs(runner_module.maybe_create_adaptive_verification_manager, factory)
        self.adaptive.AdaptiveVerificationManager.assert_not_called()
        self.assertIs(self.adaptive._assign_draft_token_budget_compiled, self.adaptive._assign_draft_token_budget)

    def test_factory_only_relaxes_always_gate(self):
        for message, permitted in (("requires AttentionCGSupport.ALWAYS", True), ("CPU query lengths mismatch", False)):
            with self.subTest(message=message):
                factory = Mock(side_effect=ValueError(message))
                runner_module = SimpleNamespace(maybe_create_adaptive_verification_manager=factory)
                manager = self.manager()
                self.adaptive.AdaptiveVerificationManager.reset_mock()
                self.adaptive.AdaptiveVerificationManager.return_value = manager
                with self.verification.adaptive_verification_gate_wrapper(runner_module):
                    if permitted:
                        self.assertIs(
                            runner_module.maybe_create_adaptive_verification_manager(**self.factory_args()), manager
                        )
                        self.adaptive.AdaptiveVerificationManager.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError, message):
                            runner_module.maybe_create_adaptive_verification_manager(**self.factory_args())
                        self.adaptive.AdaptiveVerificationManager.assert_not_called()
                self.assertIs(runner_module.maybe_create_adaptive_verification_manager, factory)

    def test_factory_forwards_upstream_validation_inputs(self):
        for enabled in (False, True):
            for error in (None, "requires AttentionCGSupport.ALWAYS", "CPU query lengths mismatch"):
                with self.subTest(enabled=enabled, error=error):
                    manager = self.manager()
                    factory = Mock(return_value=manager, side_effect=ValueError(error) if error else None)
                    runner_module = SimpleNamespace(maybe_create_adaptive_verification_manager=factory)
                    args = self.factory_args(enabled)
                    args.update(
                        vllm_config=object(),
                        target_layer_names={"target.attn"},
                        additional_attn_cg_support=(object(), "draft.attn"),
                    )
                    self.adaptive.AdaptiveVerificationManager.reset_mock()
                    self.adaptive.AdaptiveVerificationManager.return_value = manager
                    with self.verification.adaptive_verification_gate_wrapper(runner_module):
                        if error and (not enabled or "ALWAYS" not in error):
                            with self.assertRaisesRegex(ValueError, error):
                                runner_module.maybe_create_adaptive_verification_manager(**args)
                            self.adaptive.AdaptiveVerificationManager.assert_not_called()
                        else:
                            self.assertIs(runner_module.maybe_create_adaptive_verification_manager(**args), manager)
                    factory.assert_called_once_with(**args)
                    self.assertIs(runner_module.maybe_create_adaptive_verification_manager, factory)

    def test_disabled_factory_delegates_unmodified(self):
        marker = object()
        factory = Mock(return_value=marker)
        runner_module = SimpleNamespace(maybe_create_adaptive_verification_manager=factory)
        args = self.factory_args(False)
        with self.verification.adaptive_verification_gate_wrapper(runner_module):
            self.assertIs(runner_module.maybe_create_adaptive_verification_manager(**args), marker)
        factory.assert_called_once_with(**args)
        self.adaptive.AdaptiveVerificationManager.assert_not_called()

    def graph_manager(self, sample_from_anchor=True):
        config = self.config()
        config.speculative_config.use_dspark = lambda: sample_from_anchor
        config.speculative_config.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(sample_from_anchor=True)
        )
        speculator = SimpleNamespace(
            vllm_config=config,
            num_speculative_steps=5,
            num_query_per_req=5 if sample_from_anchor else 6,
            sample_from_anchor=sample_from_anchor,
        )
        return SimpleNamespace(
            vllm_config=config,
            speculator=speculator,
            _v2_varlen_physical_k=True,
            cudagraph_mode=self.mode.FULL,
            _capture_descs={},
            _candidates={},
            max_num_reqs=16,
            decode_query_len=speculator.num_query_per_req,
            lora_capture_cases=[0],
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 15, 24], max_cudagraph_capture_size=80),
        )

    def test_descriptors_keep_widest_width_per_token_bucket(self):
        manager = self.graph_manager()
        self.graph.extend_capture_descriptors(manager)
        descs = manager._capture_descs[self.mode.FULL]
        self.assertEqual(len(descs), len({d.num_tokens for d in descs}))
        self.assertEqual(next(d.uniform_token_count for d in descs if d.num_tokens == 15), 5)
        self.assertTrue(all(d.num_tokens == d.num_reqs * d.uniform_token_count for d in descs))
        self.assertTrue(all(d.uniform_token_count in (3, 5) for d in descs))
        self.assertTrue(all(d in descs for candidates in manager._candidates.values() for d in candidates))
        self.assertTrue(all(len(candidates) == len(set(candidates)) for candidates in manager._candidates.values()))

    def test_eager_mode_does_not_add_descriptors(self):
        manager = self.graph_manager()
        manager.cudagraph_mode = self.mode.NONE
        self.graph.extend_capture_descriptors(manager)
        self.assertEqual(manager._capture_descs, {})

    def test_capture_metadata_and_forward_share_width_and_restore(self):
        for anchor in (True, False):
            with self.subTest(anchor=anchor):
                manager = self.graph_manager(anchor)
                seen = []
                metadata = SimpleNamespace()

                def prepare(*args, seen=seen, manager=manager, metadata=metadata):
                    seen.append(manager.speculator.num_speculative_steps)
                    return SimpleNamespace(attn_metadata={"layer": metadata})

                self.capture._prepare_dflash_inputs_to_capture = prepare
                width = 3 if anchor else 4

                def record_forward(*args, seen=seen, manager=manager):
                    seen.append(manager.speculator.num_speculative_steps)

                with self.graph.physical_k_capture_scope(manager, record_forward) as forward:
                    self.capture._prepare_dflash_inputs_to_capture(
                        2, width * 2, None, None, [], None, 256, False, False
                    )
                    forward(2, width * 2, None, None, None, self.mode.FULL)
                self.assertEqual(seen, [3, 3])
                self.assertEqual(metadata.actual_seq_lengths_q, [width, width * 2])
                self.assertEqual(manager.speculator.num_speculative_steps, 5)
                self.assertIs(self.capture._prepare_dflash_inputs_to_capture, prepare)

    def test_capture_restores_global_and_width_after_failure(self):
        manager = self.graph_manager()
        original = Mock(side_effect=RuntimeError("capture failure"))
        self.capture._prepare_dflash_inputs_to_capture = original
        with (
            self.assertRaisesRegex(RuntimeError, "capture failure"),
            self.graph.physical_k_capture_scope(manager, lambda *args: None),
        ):
            self.capture._prepare_dflash_inputs_to_capture(2, 6, None, None, [], None, 256, False, False)
        self.assertIs(self.capture._prepare_dflash_inputs_to_capture, original)
        self.assertEqual(manager.speculator.num_speculative_steps, 5)


if __name__ == "__main__":
    unittest.main()
