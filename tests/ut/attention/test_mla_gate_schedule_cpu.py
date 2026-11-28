# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check production Gate scheduling without importing a device runtime.

These stream/event doubles verify submission dependencies, not NPU concurrency
or ACL graph capture. Run directly with Python or through pytest.
"""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def load_gate_methods(scope):
    path = Path(__file__).resolve().parents[3] / "vllm_ascend/attention/mla_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendMLAImpl")
    names = {
        "_merge_kimi_c8_dcp_output",
        "_stage_gate",
        "_late_fork_gate",
        "_resolve_gate_core_limit",
        "_fork_gate",
        "_join_gate",
    }
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in methods} != names:
        raise AssertionError("A production scheduling method was removed")
    cls = ast.ClassDef(name="GateRuntime", bases=[], keywords=[], body=methods, decorator_list=[])
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(path), "exec"), scope)
    return scope["GateRuntime"]


class FakeTensor:
    def __init__(self, runtime, name, shape=(3, 16, 512)):
        self.runtime, self.name, self.shape = runtime, name, shape

    def contiguous(self):
        self.runtime.log.append(("contiguous", self.runtime.current.name, self.runtime.quota))
        return self

    def record_stream(self, stream):
        self.runtime.log.append(("record_stream", self.name, stream.name))


class FakeStream:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name

    def wait_event(self, event):
        self.runtime.log.append(("wait", self.name, event.name))


class FakeEvent:
    def __init__(self, runtime, name):
        self.runtime, self.name = runtime, name

    def record(self, stream):
        self.runtime.log.append(("record", self.name, stream.name))

    def query(self):
        raise AssertionError("Scheduling must not query a device event on the CPU")

    def synchronize(self):
        raise AssertionError("Scheduling must not synchronize the CPU")


class MXScheme:
    pass


class GateHarness:
    def __init__(self, *, compiling=False, graph_ex=True, mx=False):
        self.log, self.quota = [], None
        self.main, self.gate = FakeStream(self, "main"), FakeStream(self, "gate")
        self.current = self.main
        self.hidden = FakeTensor(self, "hidden")
        self.output = FakeTensor(self, "attention")
        self.lse = FakeTensor(self, "lse")
        self.packed = FakeTensor(self, "packed", (8, 2, 3, 516))
        self.combined = FakeTensor(self, "combined", (3, 2, 512))
        self.gate_output = FakeTensor(self, "gate_output")
        self.expected_group = object()
        scope = {
            "torch": SimpleNamespace(
                compiler=SimpleNamespace(is_compiling=lambda: compiling),
                npu=SimpleNamespace(
                    is_available=lambda: True,
                    current_stream=lambda: self.current,
                    Event=lambda: self.fail("Events must be precreated"),
                ),
            ),
            "torch_npu": SimpleNamespace(
                npu=SimpleNamespace(npugraph_ex=SimpleNamespace(scope=SimpleNamespace(limit_core_num=self.limit)))
            ),
            "get_ascend_config": lambda: SimpleNamespace(
                ascend_compilation_config=SimpleNamespace(enable_npugraph_ex=graph_ex)
            ),
            "npu_stream_switch": self.switch,
            "mla_gate_stream": lambda: self.fail("Stream must be precreated"),
            "pack_dcp_output_lse": self.pack,
            "exchange_and_combine_dcp_output": self.exchange_combine,
            "get_vectorcore_num": lambda: 56,
            "is_mx_quant_type": lambda scheme: isinstance(scheme, MXScheme),
            "MLA_GATE_CORE_LIMIT_BF16": (32, 1),
            "MLA_GATE_CORE_LIMIT_MXFP": (16, 8),
        }
        self.impl = load_gate_methods(scope)()
        self.impl.use_output_gate, self.impl.dcp_size = True, 8
        self.impl.dcp_device_group = self.expected_group
        self.impl._gate_stream = self.gate
        self.impl._gate_in_event = FakeEvent(self, "pack_done")
        self.impl._gate_done_event = FakeEvent(self, "gate_done")
        self.impl._gate_pending = False
        self.impl.g_proj = self.project
        if mx:
            self.impl.g_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=MXScheme()))
            self.impl._gate_core_limit = self.impl._resolve_gate_core_limit()
            self.impl.g_proj = self.project

    @staticmethod
    def fail(message):
        raise AssertionError(message)

    @contextmanager
    def switch(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous

    @contextmanager
    def limit(self, *quota):
        if self.current is not self.gate:
            self.fail("Gate core scope must be entered on the gate stream")
        self.quota = quota
        self.log.append(("limit", quota))
        try:
            yield
        finally:
            self.quota = None

    def project(self, hidden):
        if hidden is not self.hidden:
            self.fail("Gate must consume the staged hidden states")
        self.log.append(("gate", self.current.name, self.quota))
        return self.gate_output, None

    def pack(self, output, lse, world, *, scatter_dim):
        if output is not self.output or lse is not self.lse or world != 8 or scatter_dim != 1:
            self.fail("Pack geometry changed")
        self.log.append(("pack", self.current.name))
        return self.packed

    def exchange_combine(self, packed, head_dim, *, scatter_dim, scatter_group, vector_core_budget):
        if packed is not self.packed or scatter_group is not self.expected_group or (head_dim, scatter_dim) != (512, 1):
            self.fail("Exchange must consume the packed output on the original DCP group")
        if ("wait", "main", "gate_done") in self.log:
            self.fail("Communication and Combine must not wait for Gate")
        self.log.append(("alltoall", self.current.name))
        self.log.append(("combine", self.current.name, vector_core_budget))
        return self.combined


class TestMLAGateScheduleCPU(unittest.TestCase):
    def test_pack_releases_fixed_quota_gate_and_combine_does_not_join(self):
        for compiling in (False, True):
            for mx, quota, budget in ((False, (32, 1), 55), (True, (16, 8), 48)):
                with self.subTest(compiling=compiling, mx=mx):
                    harness = GateHarness(compiling=compiling, mx=mx)
                    impl = harness.impl
                    impl._stage_gate(harness.hidden, defer=True)
                    self.assertEqual(harness.log, [])
                    self.assertIs(impl._merge_kimi_c8_dcp_output(harness.output, harness.lse), harness.combined)
                    self.assertEqual(
                        harness.log,
                        [
                            ("pack", "main"),
                            ("record_stream", "hidden", "gate"),
                            ("record", "pack_done", "main"),
                            ("wait", "gate", "pack_done"),
                            ("limit", quota),
                            ("contiguous", "gate", quota),
                            ("gate", "gate", quota),
                            ("record", "gate_done", "gate"),
                            ("record_stream", "gate_output", "main"),
                            ("alltoall", "main"),
                            ("combine", "main", budget),
                        ],
                    )
                    self.assertTrue(impl._gate_pending)
                    impl._late_fork_gate()  # The forward fallback must not launch a second gate.
                    self.assertEqual(sum(entry[0] == "gate" for entry in harness.log), 1)
                    harness.log.append(("v_up", "main"))
                    impl._join_gate()
                    self.assertEqual(harness.log[-2:], [("v_up", "main"), ("wait", "main", "gate_done")])
                    before = len(harness.log)
                    impl._join_gate()
                    self.assertEqual(len(harness.log), before)

    def test_repeated_invocations_reuse_stream_and_event_objects(self):
        harness = GateHarness(compiling=True)
        impl = harness.impl
        objects = impl._gate_stream, impl._gate_in_event, impl._gate_done_event
        for _ in range(2):
            harness.log.clear()
            impl._stage_gate(harness.hidden, defer=True)
            impl._merge_kimi_c8_dcp_output(harness.output, harness.lse)
            impl._join_gate()
            self.assertEqual(objects, (impl._gate_stream, impl._gate_in_event, impl._gate_done_event))
            self.assertEqual(sum(entry[0] == "gate" for entry in harness.log), 1)

    def test_serial_compile_fallback_preserves_full_combine_budget(self):
        harness = GateHarness(compiling=True, graph_ex=False)
        impl = harness.impl
        impl._stage_gate(harness.hidden, defer=True)
        impl._merge_kimi_c8_dcp_output(harness.output, harness.lse)
        self.assertEqual(
            harness.log,
            [
                ("pack", "main"),
                ("contiguous", "main", None),
                ("gate", "main", None),
                ("alltoall", "main"),
                ("combine", "main", None),
            ],
        )
        impl._join_gate()
        self.assertFalse(impl._gate_pending)

    def test_quota_uses_loaded_mx_scheme_instead_of_scale_attribute(self):
        harness = GateHarness()
        impl = harness.impl
        for method in (MXScheme(), SimpleNamespace(quant_method=MXScheme())):
            impl.g_proj = SimpleNamespace(quant_method=method)
            self.assertEqual(impl._resolve_gate_core_limit(), (16, 8))
        impl.g_proj = SimpleNamespace(weight_scale=object(), quant_method=object())
        self.assertEqual(impl._resolve_gate_core_limit(), (32, 1))


if __name__ == "__main__":
    unittest.main()
