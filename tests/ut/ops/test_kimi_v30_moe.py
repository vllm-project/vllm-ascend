# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the 0.30 SiTU hooks and Decode dependency graph.

NPU kernels/streams are simulated; numerical SiTU reference runs on CPU.
"""

import ast
import unittest
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(path, names, ns, class_name=None):
    tree = ast.parse((ROOT / path).read_text())
    if class_name:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        cls.bases = []
        cls.decorator_list = []
        cls.body = [n for n in cls.body if getattr(n, "name", None) in names]
        nodes = [cls]
    else:
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", None) in names
            or isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        ]
    nodes.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec"), ns)


class Activation(Enum):
    SITU = "situ"


class Quant(Enum):
    NONE = 0
    W4A8MXFP = 1


class TestSituHooks(unittest.TestCase):
    def test_unquant_sparse_counts_and_native_fallback(self):
        def reference(x, beta, linear_beta):
            gate, up = x.float().chunk(2, -1)
            return (beta * torch.tanh(gate / beta) * gate.sigmoid() * linear_beta * torch.tanh(up / linear_beta)).to(
                x.dtype
            )

        sparse = Mock(side_effect=lambda x, counts, beta, alpha, high_precision: reference(x, beta, alpha))
        ns = dict(
            torch=NS(
                float16=torch.float16,
                bfloat16=torch.bfloat16,
                int64=torch.int64,
                ops=NS(custom=NS(npu_situ_and_mul_sparse=sparse)),
            ),
            MoEActivation=Activation,
            QuantType=Quant,
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            cumsum_group_list=lambda counts, src, dst: counts
            if src == 1
            else torch.diff(counts, prepend=counts.new_zeros(1)),
            _apply_situ=reference,
        )
        load("vllm_ascend/ops/fused_moe/moe_mlp.py", {"_unified_apply_activation"}, ns)
        x = torch.randn(3, 8).bfloat16()
        for typ, counts in [(1, [0, 2, 1]), (0, [0, 2, 3])]:
            inp = NS(
                activation=Activation.SITU,
                activation_situ_beta=4.0,
                activation_situ_linear_beta=25.0,
                quant=NS(quant_type=Quant.NONE),
                group_list_type=typ,
                group_list=torch.tensor(counts),
            )
            actual = ns["_unified_apply_activation"](inp, x, None)
            torch.testing.assert_close(actual, reference(x, 4.0, 25.0))
            self.assertTrue(sparse.call_args.kwargs["high_precision"])
            self.assertEqual(sparse.call_args.args[1].tolist(), [0, 2, 1])
        ns["get_current_hardware_profile"] = lambda: NS(supports=lambda cap: False)
        sparse.reset_mock()
        torch.testing.assert_close(ns["_unified_apply_activation"](inp, x, None), reference(x, 4.0, 25.0))
        sparse.assert_not_called()

    def test_mxfp_hook_splits_only_when_supported_or_scheduled(self):
        gate_up = torch.randn(3, 8).bfloat16()
        activation = gate_up[:, :4]
        scale = torch.ones(3, 1)
        calls = []

        def mark(name, value):
            def call(*a, **kw):
                calls.append(name)
                return value

            return Mock(side_effect=call)

        grouped = mark("grouped_situ", (activation, scale))
        fused = mark("fused", (activation, scale, None))
        legacy = mark("legacy_situ", (activation, scale))
        custom = NS(grouped_situ_mx_quant=grouped)
        fake_torch = NS(
            bfloat16=torch.bfloat16,
            float8_e4m3fn=torch.float8_e4m3fn,
            int64=torch.int64,
            ops=NS(custom=custom, _C_ascend=NS(situ_mx_quant=legacy)),
        )
        ns = dict(
            torch=fake_torch,
            torch_npu=NS(float8_e8m0fnu="e8m0", float4_e2m1fn_x2="fp4", npu_grouped_matmul=mark("gmm1", [gate_up])),
            DeviceOperator=NS(npu_grouped_matmul_situ_quant=fused),
            MoEActivation=Activation,
            get_current_hardware_profile=lambda: NS(supports=lambda cap: True),
            HardwareCapability=NS(CANN_MEGAMOE_MXFP=1),
            SITU_MX_DST_TYPE_E4M3FN=36,
            dispose_tensor=lambda x: None,
            maybe_normalize_mxfp_scale_layout=lambda x: x,
            cumsum_group_list=lambda counts, src, dst: counts,
        )
        clsname = "AscendW4A8MXFPDynamicFusedMoEMethod"
        load(
            "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
            {"_use_grouped_situ", "apply_gmm1_act_quant"},
            ns,
            clsname,
        )
        method = ns[clsname]()
        method.group_size = 32
        method.use_expert_weight_list = False
        method.quant_type = Quant.W4A8MXFP
        method._quant_hidden_states = lambda x, s: (x, s)
        method._get_mlp_weights = lambda layer: ([layer.w13_weight], [layer.w13_weight_scale], [], [])
        inp = NS(
            hidden_states=torch.zeros(3, 4),
            dynamic_scale=scale,
            layer=NS(w13_weight=object(), w13_weight_scale=object()),
            activation=Activation.SITU,
            group_list_type=1,
            group_list=torch.tensor([0, 2, 1]),
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            lora_context=None,
            shared_expert_schedule=None,
        )
        method.apply_gmm1_act_quant(inp)
        self.assertEqual(calls, ["gmm1", "grouped_situ"])
        self.assertEqual(grouped.call_args.args[1].tolist(), [0, 2, 1])
        self.assertFalse(grouped.call_args.kwargs["high_precision"])
        del custom.grouped_situ_mx_quant
        calls.clear()
        method.apply_gmm1_act_quant(inp)
        self.assertEqual(calls, ["fused"])
        # After MegaMoE loading the dense weights were released. Even without
        # the custom grouped SiTU op or a shared schedule, use live GMM views.
        inp.layer.cann_mega_moe_w13_weight_list = []
        calls.clear()
        method.apply_gmm1_act_quant(inp)
        self.assertEqual(calls, ["gmm1", "legacy_situ"])
        del inp.layer.cann_mega_moe_w13_weight_list
        inp.shared_expert_schedule = NS(after_gmm1=lambda: calls.append("shared_down"))
        calls.clear()
        method.apply_gmm1_act_quant(inp)
        self.assertEqual(calls, ["gmm1", "shared_down", "legacy_situ"])


class TestDecodeDependencies(unittest.TestCase):
    def test_bf16_and_mxfp_share_vector_barriers(self):
        self.check_dependency_graph(False)
        self.check_dependency_graph(True)

    def check_dependency_graph(self, quantized):
        nodes = []
        last = {"main": None, "shared": None}
        current = ["main"]

        def op(name, deps=()):
            idx = len(nodes)
            nodes.append((current[0], name, [d for d in (last[current[0]], *deps) if d is not None]))
            last[current[0]] = idx
            return idx

        class Stream:
            def __init__(self, name):
                self.name = name

            def record_event(self):
                return last[self.name]

            def wait_event(self, event):
                op("wait", (event,))

        streams = {name: Stream(name) for name in last}

        @contextmanager
        def switch(stream):
            old = current[0]
            current[0] = stream.name
            try:
                yield
            finally:
                current[0] = old

        limits = []
        npu = NS(current_stream=lambda: streams[current[0]])
        ns = dict(
            torch=NS(npu=npu),
            torch_npu=NS(
                npu=NS(npugraph_ex=NS(scope=NS(limit_core_num=lambda *limit: (limits.append(limit) or nullcontext()))))
            ),
            npu_stream_switch=switch,
            shared_experts_calculation_stream=lambda: streams["shared"],
        )
        load(
            "vllm_ascend/ops/fused_moe/kimi_decode_schedule.py",
            {"KimiDecodeSchedule", "ROUTER_CORE_LIMIT", "BF16_SHARED_CORE_LIMIT", "MXFP_SHARED_CORE_LIMIT"},
            ns,
        )
        x = torch.tensor([[1.0, 2.0]])

        def gate_up(prepared, ready):
            with switch(streams["shared"]):
                streams["shared"].wait_event(ready)
                op("gate_up")
            return NS(gate_up=x, hidden_states=x)

        def activate(x):
            op("shared_situ")
            return x * 2

        def down(x, y):
            op("shared_down")
            return y + 3

        def scatter(x):
            op("shared_rs")
            return x

        gate = NS(weight_scale=1) if quantized else NS()
        shared = NS(
            layer=NS(gate_up_proj=gate, down_proj=object()),
            prepare_for_router_overlap=lambda p, r: p,
            enqueue_gate_up_after_router=gate_up,
            apply_activation=activate,
            _prepare_linear_input_for_overlap=lambda layer, x: x,
            part2=down,
            _pad_and_reduce_scatter=scatter,
        )
        op("ready")
        schedule = ns["KimiDecodeSchedule"](shared, x, last["main"])
        op("dispatch")
        schedule.after_dispatch()
        op("gmm1")
        schedule.after_gmm1()
        op("routed_situ")
        op("gmm2")
        op("combine")
        with patch.object(torch.Tensor, "record_stream") as record_stream:
            actual = schedule.finalize()
        record_stream.assert_called_once_with(streams["main"])
        npu.current_stream().wait_event(last["shared"])
        op("add")
        torch.testing.assert_close(actual, x * 2 + 3)

        def ancestors(i):
            return set(nodes[i][2]).union(*(ancestors(d) for d in nodes[i][2]))

        ids = {name: i for i, (_, name, _) in enumerate(nodes) if name != "wait"}
        for before, after in [
            ("dispatch", "shared_situ"),
            ("shared_situ", "gmm1"),
            ("gmm1", "shared_down"),
            ("combine", "shared_rs"),
            ("shared_rs", "add"),
        ]:
            self.assertIn(ids[before], ancestors(ids[after]))
        self.assertNotIn(ids["shared_down"], ancestors(ids["routed_situ"]))
        self.assertEqual(limits, [(16, 32) if quantized else (32, 1)])


class TestRuntimeMoEBoundary(unittest.TestCase):
    def test_latent_transform_once_and_layer_cursor_once(self):
        ns = dict(torch=torch, _unpack=lambda value: value, _resolve_layer_name=lambda name: name)
        path = "vllm_ascend/ops/fused_moe/fused_moe.py"
        load(path, {"_forward_standard"}, ns, "AscendMoERunner")
        load(path, {"_ascend_kimi_moe"}, ns)
        runner = ns["AscendMoERunner"]()
        runner._kimi_gate = object()
        runner._fused_output_is_reduced = True
        runner.apply_routed_input_transform = Mock(side_effect=lambda x: (x * 2, x))
        runner._maybe_pad_hidden_states = lambda original, projected: (projected, None, None)
        runner._forward_impl = Mock(side_effect=lambda projected, logits, original, ids: (original * 3, projected * 2))
        runner._forward_entry = Mock(side_effect=AssertionError("nested MoE boundary advances cursor twice"))
        runner._maybe_reduce_routed_output_before_transform = lambda x, reduced: (x, reduced)
        runner._maybe_reduce_shared_expert_output = lambda x, reduced: x
        runner._maybe_apply_routed_scale_to_output = lambda a, b: (a, b)
        runner.apply_routed_output_transform = lambda x: x * 5
        runner._maybe_reduce_final_output = lambda x, trunc, reduced: x
        runner._maybe_add_zero_expert_output = lambda x: x
        resolve = Mock(return_value=runner)
        ns["get_layer_from_name"] = resolve
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        result = ns["_ascend_kimi_moe"](x, None, "from_forward_context")
        torch.testing.assert_close(result, x * 23)
        resolve.assert_called_once()
        runner.apply_routed_input_transform.assert_called_once()
        runner._forward_entry.assert_not_called()
        self.assertIsNone(runner._kimi_router_logits)

    def test_030_router_keeps_fp32_weights_for_bf16_activations(self):
        ns = dict(
            torch=torch,
            get_moe_num_logical_experts=lambda *a, **kw: 2,
            get_ascend_config=lambda: NS(enable_force_eplb=False),
        )
        load("vllm_ascend/ops/fused_moe/routed_experts.py", {"_select_experts"}, ns, "AscendRoutedExperts")
        layer = ns["AscendRoutedExperts"]()
        weights = torch.tensor([[0.123456, 0.876544]], dtype=torch.float32)
        layer.router = NS(_select_experts=lambda **kw: (weights, torch.tensor([[0, 1]])))
        layer.log2phy = None
        layer.n_shared_experts = None
        layer.global_redundant_expert_num = 0
        layer.moe_config = NS(num_experts=2)
        result, _ = layer._select_experts(torch.ones(1, 4, dtype=torch.bfloat16), torch.zeros(1, 2), False)
        self.assertIs(result, weights)


if __name__ == "__main__":
    unittest.main()
