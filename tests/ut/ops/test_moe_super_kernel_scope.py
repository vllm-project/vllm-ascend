# SPDX-License-Identifier: Apache-2.0
"""CPU-only scope contract tests. Actual graph fusion needs hardware profiling."""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace, cls=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class ScopeTest(unittest.TestCase):
    def setUp(self):
        self.stream = "main"
        self.stacks = {"main": [], "shared": []}
        self.calls = []

        def begin(name):
            self.calls.append(("begin", self.stream, name))
            self.stacks[self.stream].append(name)

        def end(name):
            self.calls.append(("end", self.stream, name))
            self.assertEqual(self.stacks[self.stream].pop(), name)

        self.ns = {
            "contextmanager": contextmanager,
            "torch": NS(npu=NS(super_kernel_scope_begin=begin, super_kernel_scope_end=end)),
        }
        load_function("vllm_ascend/utils.py", "super_kernel_scope", self.ns)

        @contextmanager
        def switch(stream, enabled):
            old = self.stream
            if enabled:
                self.stream = stream
            try:
                yield
            finally:
                self.stream = old

        self.ns.update(
            npu_stream_switch=switch,
            shared_experts_calculation_stream=lambda: "shared",
            SharedExpertMLPPath=NS(A8_INT_FUSED=1, A8_MXFP_FUSED=2),
            SubmittedSharedExpertGateUp=NS,
        )

    def assert_work_scope(self, enabled, stream, name):
        self.assertEqual(self.stream, stream)
        self.assertEqual(self.stacks[stream], [name] if enabled else [])

    def test_none_scope_is_forwarded_and_outer_scope_restored(self):
        # None is the torch_npu exclusion sentinel, not a disabled context.
        # This tests wrapper forwarding/cleanup; device fusion needs hardware.
        for enabled in (False, True):
            for fail in (False, True):
                with self.subTest(enabled=enabled, fail=fail):
                    self.calls.clear()
                    with self.ns["super_kernel_scope"]("moe_shared", True):
                        try:
                            with self.ns["super_kernel_scope"](None, enabled):
                                self.assertEqual(
                                    self.stacks["main"], ["moe_shared", None] if enabled else ["moe_shared"]
                                )
                                if fail:
                                    raise ValueError("quant failed")
                        except ValueError:
                            self.assertTrue(fail)
                        self.assertEqual(self.stacks["main"], ["moe_shared"])
                    expected = [("begin", "main", "moe_shared")]
                    if enabled:
                        expected += [("begin", "main", None), ("end", "main", None)]
                    expected += [("end", "main", "moe_shared")]
                    self.assertEqual(self.calls, expected)
                    self.assertEqual(self.stacks, {"main": [], "shared": []})

    def test_routed_scope_and_cleanup(self):
        fn = load_function("vllm_ascend/ops/fused_moe/fused_moe.py", "_forward_impl", self.ns, "AscendMoERunner")
        for enabled in (False, True):
            for fail in (False, True):
                with self.subTest(enabled=enabled, fail=fail):
                    self.calls.clear()

                    def routed(*, enabled=enabled, fail=fail, **kwargs):
                        self.assert_work_scope(enabled, "main", "moe_routed")
                        if fail:
                            raise ValueError("operator failure")
                        return "routed-result"

                    runner = NS(
                        _enable_moe_super_kernel_scopes=enabled,
                        _sequence_parallel_context=nullcontext,
                        ascend_shared_experts=None,
                        is_internal_router=False,
                        routed_experts=NS(forward_impl=routed),
                    )
                    if fail:
                        with self.assertRaisesRegex(ValueError, "operator failure"):
                            fn(runner, "hidden", "logits", None)
                    else:
                        self.assertEqual(fn(runner, "hidden", "logits", None), "routed-result")
                    self.assertEqual(self.stacks, {"main": [], "shared": []})
                    self.assertEqual(len(self.calls), 2 if enabled else 0)

    def test_shared_scope_after_stream_switch_and_cleanup(self):
        fn = load_function(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "enqueue_gate_up_after_router",
            self.ns,
            "AscendSharedExperts",
        )
        for enabled in (False, True):
            for overlap in (False, True):
                for fail in (False, True):
                    with self.subTest(enabled=enabled, overlap=overlap, fail=fail):
                        self.calls.clear()

                        def part1(value, *, enabled=enabled, fail=fail, overlap=overlap):
                            self.assert_work_scope(enabled, "shared" if overlap else "main", "moe_shared")
                            self.assertEqual(value, "prepared")
                            if fail:
                                raise ValueError("operator failure")
                            return "fused-gate-up-result"

                        executor = NS(
                            _enable_moe_super_kernel_scopes=enabled,
                            multistream_overlap=overlap,
                            _select_mlp_path=lambda: 3,
                            _wait_for_milestone=lambda *args: None,
                            part1=part1,
                        )
                        prepared = NS(gate_up_input="prepared", hidden_states="hidden", local_dp_metadata=None)
                        if fail:
                            with self.assertRaisesRegex(ValueError, "operator failure"):
                                fn(executor, prepared, None)
                        else:
                            self.assertEqual(fn(executor, prepared, None).gate_up, "fused-gate-up-result")
                        self.assertEqual(self.stream, "main")
                        self.assertEqual(self.stacks, {"main": [], "shared": []})
                        self.assertEqual(len(self.calls), 2 if enabled else 0)

    def test_shared_capture_join_precedes_scope_begin(self):
        self.ns.update(PreparedSharedExpertExecution=NS, RoutedMoEMilestones=NS)
        fn = load_function(
            "vllm_ascend/ops/fused_moe/shared_experts.py", "prepare_for_router_overlap", self.ns, "AscendSharedExperts"
        )

        def wait(event, name):
            self.assertEqual(self.stream, "shared")
            self.assertEqual(self.stacks["shared"], [], "capture join must precede scope begin")
            self.calls.append(("wait", event, name))

        def prepare(*args):
            self.assert_work_scope(True, "shared", "moe_shared")
            return "hidden", None

        executor = NS(
            _enable_moe_super_kernel_scopes=True,
            multistream_overlap=True,
            parallel_mode=lambda: 0,
            _wait_for_milestone=wait,
            _prepare_execution_input=prepare,
            _prepare_gate_up_input=lambda x: x,
        )
        fn(executor, NS(hidden_states="hidden"), "input-event")
        self.assertEqual(self.calls[0], ("wait", "input-event", "shared_input_ready"))
        self.assertEqual(self.stacks, {"main": [], "shared": []})

    def test_router_scope_closes_before_output_milestone(self):
        fn = load_function(
            "vllm_ascend/ops/fused_moe/fused_moe.py", "_prepare_router_and_milestones", self.ns, "AscendMoERunner"
        )
        for enabled in (False, True):
            for fail in (False, True):
                order = []

                def event(*, order=order):
                    self.assertEqual(self.stacks["main"], ["moe_routed"])
                    order.append("event")
                    return len(order)

                def router(value, *, enabled=enabled, fail=fail, order=order):
                    self.assertEqual(self.stacks["main"], ["moe_routed", "moe_router"] if enabled else ["moe_routed"])
                    order.append("router")
                    if fail:
                        raise ValueError("router failure")
                    return "logits"

                self.ns["maybe_record_event"] = event
                obj = NS(
                    _enable_moe_super_kernel_scopes=enabled,
                    is_internal_router=True,
                    _prepare_router_input=lambda *a: "input",
                    _prepare_shared_for_router_overlap=lambda *a: "prepared",
                    _apply_router_linear=router,
                    ascend_shared_experts=NS(enqueue_gate_up_after_router=lambda *a: "gateup"),
                )
                with self.ns["super_kernel_scope"]("moe_routed", True):
                    if fail:
                        with self.assertRaisesRegex(ValueError, "router failure"):
                            fn(obj, None, "hidden", "old-logits")
                    else:
                        self.assertEqual(fn(obj, None, "hidden", "old-logits"), ("logits", 1, 3, "gateup"))
                        self.assertEqual(order, ["event", "router", "event"])
                self.assertEqual(self.stacks, {"main": [], "shared": []})

    def test_shared_mxfp_input_quantization_excludes_only_target_scope(self):
        quant = NS(W8A8MXFP=1, OTHER=2)
        self.ns["QuantType"] = quant
        fn = load_function(
            "vllm_ascend/ops/fused_moe/shared_experts.py", "_prepare_gate_up_input", self.ns, "AscendSharedExperts"
        )
        token = object()
        for enabled in (False, True):
            for overlap in (False, True):
                for qtype in (quant.W8A8MXFP, quant.OTHER):
                    for fail in (False, True):
                        target = enabled and overlap and qtype == quant.W8A8MXFP

                        def prepare(layer, value, *, fail=fail, target=target):
                            self.assertIs(value, token)
                            self.assertEqual(
                                self.stacks[self.stream], ["moe_shared", None] if target else ["moe_shared"]
                            )
                            if fail:
                                raise ValueError("quant failed")
                            return token

                        executor = NS(
                            _enable_moe_super_kernel_scopes=enabled,
                            multistream_overlap=overlap,
                            quant_type=qtype,
                            _select_mlp_path=lambda: 3,
                            layer=NS(gate_up_proj=object()),
                            _prepare_linear_input_for_overlap=prepare,
                        )
                        with self.ns["super_kernel_scope"]("moe_shared", True):
                            if fail:
                                with self.assertRaisesRegex(ValueError, "quant failed"):
                                    fn(executor, token)
                            else:
                                self.assertIs(fn(executor, token), token)
                        self.assertEqual(self.stacks, {"main": [], "shared": []})

    def test_nested_branches_keep_distinct_scope_sets(self):
        fn = load_function(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "enqueue_gate_up_after_router",
            self.ns,
            "AscendSharedExperts",
        )
        self.stacks = {"main": ["full_model"], "shared": ["full_model"]}

        def part1(value):
            self.assertEqual(self.stacks["main"], ["full_model", "moe_routed"])
            self.assertEqual(self.stacks["shared"], ["full_model", "moe_shared"])
            return "shared-result"

        executor = NS(
            _enable_moe_super_kernel_scopes=True,
            multistream_overlap=True,
            _select_mlp_path=lambda: 3,
            _wait_for_milestone=lambda *args: None,
            part1=part1,
        )
        prepared = NS(gate_up_input="prepared", hidden_states="hidden", local_dp_metadata=None)
        with self.ns["super_kernel_scope"]("moe_routed", True):
            self.assertEqual(fn(executor, prepared, None).gate_up, "shared-result")
            self.assertEqual(self.stacks["main"], ["full_model", "moe_routed"])
            self.assertEqual(self.stacks["shared"], ["full_model"])
        self.assertEqual(self.stacks, {"main": ["full_model"], "shared": ["full_model"]})

    def test_shared_collectives_on_main_keep_shared_scope(self):
        modes = NS(SEQUENCE_PARALLEL_ONLY=1, SHARED_EXPERT_DATA_PARALLEL_ONLY=2)
        self.ns.update(SharedExpertParallelMode=modes, PreparedSharedExpertInput=NS)

        def collective(value, *args, **kwargs):
            self.assertEqual(self.stream, "main")
            self.assertEqual(self.stacks["main"], ["moe_routed", "moe_shared"])
            return value

        self.ns["tensor_model_parallel_all_gather"] = collective
        prepare = load_function(
            "vllm_ascend/ops/fused_moe/shared_experts.py", "prepare_input_before_routed", self.ns, "AscendSharedExperts"
        )
        finish = load_function("vllm_ascend/ops/fused_moe/shared_experts.py", "forward", self.ns, "AscendSharedExperts")
        executor = NS(
            _enable_moe_super_kernel_scopes=True,
            multistream_overlap=True,
            parallel_mode=lambda: modes.SEQUENCE_PARALLEL_ONLY,
        )
        with self.ns["super_kernel_scope"]("moe_routed", True):
            self.assertEqual(prepare(executor, "input").hidden_states, "input")
        for mode, overlap in [(modes.SHARED_EXPERT_DATA_PARALLEL_ONLY, True), (modes.SEQUENCE_PARALLEL_ONLY, False)]:
            executor = NS(
                _enable_moe_super_kernel_scopes=True,
                multistream_overlap=overlap,
                parallel_mode=lambda mode=mode: mode,
                _finish_shared_mlp=lambda *args: "output",
                wait_for_output=lambda: None,
                _finalize_local_dp_output=collective,
                _pad_and_reduce_scatter=collective,
            )
            submitted = NS(local_dp_metadata=(1, 0))
            with self.ns["super_kernel_scope"]("moe_routed", True):
                self.assertEqual(finish(executor, None, None, submitted_gate_up=submitted), "output")
        self.assertEqual(self.stacks, {"main": [], "shared": []})


if __name__ == "__main__":
    unittest.main()
