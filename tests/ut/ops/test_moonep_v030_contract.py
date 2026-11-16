# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free contract checks; never import vllm, torch or NPU libraries."""

from __future__ import annotations
import __future__

import ast
import copy
import dataclasses
import unittest
from enum import Enum
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
MOE = ROOT / "vllm_ascend/ops/fused_moe"


def nodes(path):
    return ast.parse(path.read_text(), filename=str(path)).body


def execute(body, namespace):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(body), type_ignores=[]))
    exec(compile(module, "<source-contract>", "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def method(path, cls, name):
    owner = next(n for n in nodes(path) if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


class SlotWeightContractTest(unittest.TestCase):
    def setUp(self):
        selected = [
            n for n in nodes(MOE / "moonep_mlp.py") if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign))
        ]
        self.calls = []

        def gmm1(mci):
            self.calls.append(mci)
            return mci.layer.w13_weight

        self.quant = SimpleNamespace(
            use_expert_weight_list=False,
            supports_fused_activation=lambda activation: False,
            apply_gmm1=gmm1,
            apply_act_quant=lambda mci, hidden: (hidden, None),
            apply_gmm2=lambda mci, hidden, scale: hidden,
        )

        self.ns = execute(
            selected,
            {
                "__name__": __name__,
                "dataclass": dataclasses.dataclass,
                "replace": dataclasses.replace,
                "MappingProxyType": MappingProxyType,
                "MoEActivation": SimpleNamespace(SWIGLUSTEP="step"),
                "_unified_apply_activation": lambda mci, hidden, quant: hidden,
            },
        )
        self.original = SimpleNamespace(w13_weight="original-w13", w2_weight="original-w2", activation="silu")
        self.weights = SimpleNamespace(
            w1="slot-w13",
            w2="slot-w2",
            w1_scale="slot-scale13",
            w2_scale="slot-scale2",
            w1_bias=None,
            w2_bias=None,
            w1_scale_bias=None,
            w2_scale_bias=None,
            w1_offset=None,
            w2_offset=None,
        )
        self.view = self.ns["SlotMLPLayerView"](self.original, self.weights)

    def test_bf16_native_getter_consumes_slot_aliases(self):
        fn = method(MOE / "routed_experts.py", "AscendUnquantizedFusedMoEMethod", "get_mlp_weights")
        ns = execute([fn], {})
        self.assertEqual(ns["get_mlp_weights"](object(), self.view), ("slot-w13", "slot-w2"))
        self.assertEqual(self.original.w13_weight, "original-w13")

    def test_mxfp_native_getter_consumes_both_weights_and_scales(self):
        path = ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py"
        owner = next(
            n
            for n in nodes(path)
            if isinstance(n, ast.ClassDef)
            and any(isinstance(f, ast.FunctionDef) and f.name == "_get_mlp_weights" for f in n.body)
        )
        fn = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_get_mlp_weights")
        ns = execute([fn], {})
        result = ns["_get_mlp_weights"](SimpleNamespace(use_expert_weight_list=False), self.view)
        self.assertEqual(result, (["slot-w13"], ["slot-scale13"], ["slot-w2"], ["slot-scale2"]))

    def test_native_mlp_receives_read_only_view_without_changing_input(self):
        @dataclasses.dataclass(frozen=True)
        class Input:
            layer: object
            weights: object
            activation: str = "silu"
            lora_context: object = None

        mci = Input(self.original, self.weights)
        result = self.ns["apply_slot_mlp"](mci, self.quant)
        self.assertEqual(result, ("slot-w13", None))
        self.assertIs(mci.layer, self.original)
        self.assertIsNot(self.calls[0].layer, self.original)
        self.assertEqual(self.view.activation, "silu")
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            self.view.original = object()

    def test_split_layout_fails_before_native_mlp(self):
        mci = SimpleNamespace(layer=self.original)
        with self.assertRaises(NotImplementedError):
            self.ns["apply_slot_mlp"](mci, SimpleNamespace(use_expert_weight_list=True))
        self.assertEqual(self.calls, [])

    def test_missing_quant_method_fails_before_native_mlp(self):
        with self.assertRaises(RuntimeError):
            self.ns["apply_slot_mlp"](SimpleNamespace(layer=self.original), None)
        self.assertEqual(self.calls, [])


class MoonEPIntegrationContractTest(unittest.TestCase):
    def selector(self):
        path = ROOT / "vllm_ascend/ascend_forward_context.py"
        tree = nodes(path)
        selected = [n for n in tree if isinstance(n, ast.ClassDef) and n.name == "MoECommType"]
        selected += [n for n in tree if isinstance(n, ast.FunctionDef) and n.name == "select_moe_comm_method"]
        state = SimpleNamespace(enabled=False, keep=False, capacity=4096, world=8, native=None)
        policy = object()
        ns = execute(
            selected,
            {
                "Enum": Enum,
                "QuantType": SimpleNamespace(NONE=object()),
                "is_moe_model": lambda config: True,
                "get_ascend_config": lambda: SimpleNamespace(enable_moonep=state.enabled),
                "get_ep_group": lambda: SimpleNamespace(world_size=state.world),
                "get_mc2_tokens_capacity": lambda: 512,
                "get_moonep_tokens_capacity": lambda: state.capacity,
                "get_current_hardware_profile": lambda: SimpleNamespace(moe_comm_policy=policy),
                "MoECommPolicy": SimpleNamespace(ALLGATHER=object()),
                "_MOE_COMM_SELECTORS": {policy: lambda *args: state.native},
                "envs": SimpleNamespace(MOONEP_KEEP_MC2=False),
                "logger": SimpleNamespace(debug=lambda *args: None),
            },
        )
        config = SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True), lora_config=None)
        return ns, state, config

    def test_off_preserves_native_policy_and_does_not_require_shmem_capacity(self):
        ns, state, config = self.selector()
        state.capacity = None
        for native in ns["MoECommType"]:
            state.native = native
            self.assertIs(ns["select_moe_comm_method"](8192, config), native)

    def test_keep_mc2_only_preserves_eligible_native_steps(self):
        ns, state, config = self.selector()
        state.enabled = True
        kind = ns["MoECommType"]
        for native in (kind.MC2, kind.FUSED_MC2, kind.ALLTOALL):
            state.native = native
            for keep in (False, True):
                ns["envs"].MOONEP_KEEP_MC2 = keep
                expected = native if keep and native in (kind.MC2, kind.FUSED_MC2) else kind.SHMEM
                self.assertIs(ns["select_moe_comm_method"](16, config), expected)

    def test_unsupported_configuration_and_overflow_fail_without_native_fallback(self):
        ns, state, config = self.selector()
        state.enabled = True
        state.native = ns["MoECommType"].ALLTOALL
        with self.assertRaises(RuntimeError):
            ns["select_moe_comm_method"](4097, config)
        state.world = 1
        with self.assertRaises(RuntimeError):
            ns["select_moe_comm_method"](1, config)
        state.world = 8
        config.lora_config = object()
        with self.assertRaises(NotImplementedError):
            ns["select_moe_comm_method"](1, config)

    def test_fused_result_has_only_v030_fields(self):
        fn = method(MOE / "moonep_comm.py", "MoonEPCommImpl", "fused_experts")
        result_cls = next(
            n
            for n in nodes(MOE / "moe_comm_method.py")
            if isinstance(n, ast.ClassDef) and n.name == "FusedExpertsResult"
        )
        fields = {n.target.id for n in result_cls.body if isinstance(n, ast.AnnAssign)}
        call = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "FusedExpertsResult"
        )
        supplied = {kw.arg for kw in call.keywords}
        self.assertFalse(supplied - fields)
        self.assertTrue(
            {"before_dispatch_evt", "before_gmm2_evt", "before_combine_evt", "shared_activation_overlap_start_evt"}
            <= supplied
        )

    def check_stage_events(self, shared, multistream):
        fn = copy.deepcopy(method(MOE / "moonep_comm.py", "MoonEPCommImpl", "fused_experts"))

        class StripImports(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None

        fn = StripImports().visit(fn)
        events = []

        def record():
            event = object()
            events.append(event)
            return event

        layer = SimpleNamespace(w13_weight=object(), return_with_event=shared)
        state = object()
        tls = SimpleNamespace(state=None)
        meta = SimpleNamespace(state=state, buffer=object(), plan=object())
        dispatch = SimpleNamespace(combine_metadata=meta, group_list_type=1, group_list="counts")
        selected = []
        needed = shared and multistream

        def slot(**kwargs):
            self.assertIs(kwargs["state"], state)
            self.assertIs(kwargs["quant_method"], quant)
            self.assertEqual(kwargs["record_before_gmm2"], needed)
            return "rows", "real-gmm2-event" if needed else None

        ns = execute(
            [fn],
            {
                "torch": SimpleNamespace(
                    npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(record_event=record))
                ),
                "get_ascend_config": lambda: SimpleNamespace(multistream_overlap_shared_expert=multistream),
                "_MOONEP_RUNTIME": SimpleNamespace(_state_by_w13={id(layer.w13_weight): state}),
                "_SLOT_TLS": tls,
                "select_prefetch_layer": selected.append,
                "build_token_dispatch_input": lambda **kw: kw["fused_experts_input"],
                "run_slot_forward": slot,
                "FusedExpertsResult": lambda **kw: SimpleNamespace(**kw),
            },
        )
        dispatcher = SimpleNamespace(
            token_dispatch=lambda arg: dispatch,
            token_combine=lambda rows, metadata: (rows, metadata),
        )
        owner = SimpleNamespace(token_dispatcher=dispatcher, moe_config=object())
        inp = SimpleNamespace(layer=layer, lora_context=None)
        quant = object()
        result = ns["fused_experts"](owner, inp, quant)
        if needed:
            self.assertEqual(len(events), 2)
            self.assertIs(result.before_dispatch_evt, events[0])
            self.assertIs(result.before_combine_evt, events[1])
            self.assertEqual(result.before_gmm2_evt, "real-gmm2-event")
            self.assertEqual(result.shared_activation_overlap_start_evt, "real-gmm2-event")
        else:
            self.assertEqual(events, [])
            self.assertIsNone(result.before_dispatch_evt)
            self.assertIsNone(result.before_combine_evt)
            self.assertIsNone(result.before_gmm2_evt)
            self.assertIsNone(result.shared_activation_overlap_start_evt)
        self.assertIsNone(tls.state)
        self.assertEqual(selected, [layer])

        def failure(**kwargs):
            raise ValueError("injected")

        ns["run_slot_forward"] = failure
        with self.assertRaises(ValueError):
            ns["fused_experts"](owner, inp, quant)
        self.assertIsNone(tls.state)

    def test_actual_fused_entry_propagates_events_and_clears_layer_state(self):
        self.check_stage_events(shared=True, multistream=True)

    def test_no_shared_experts_omit_events_even_when_overlap_is_enabled(self):
        self.check_stage_events(shared=False, multistream=True)

    def test_serial_shared_experts_omit_unused_events(self):
        self.check_stage_events(shared=True, multistream=False)

    def test_no_shared_experts_and_no_overlap_omit_events(self):
        self.check_stage_events(shared=False, multistream=False)

    def test_both_runners_initialize_runtime_before_prefetch_hooks(self):
        for name in ("worker/model_runner_v1.py", "worker/v2/model_runner.py"):
            path = ROOT / "vllm_ascend" / name
            fn = method(path, "NPUModelRunner", "load_model")
            calls = [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertLess(calls.index("ensure_runtime_initialized"), calls.index("maybe_install_pf_stream"))
            self.assertIn("set_moonep_tokens_capacity", path.read_text())


if __name__ == "__main__":
    unittest.main()
