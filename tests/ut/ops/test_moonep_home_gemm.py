"""CPU scheduling regression tests. No NPU initialization or device operators."""

import ast
import types
import unittest
from pathlib import Path

import torch


class HomeGmmScheduleTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.quant_calls = 0
        self.activation_calls = 0
        self.gmm2_calls = 0
        self.prefetch_ready = False
        self.outputs = []
        self.quant_inputs = []
        self.disposed = []
        outer = self
        self.activation = types.SimpleNamespace(
            SILU="silu",
            SWIGLUOAI="oai",
            SWIGLUSTEP="step",
            GELU="gelu",
            GELU_TANH="gelu_tanh",
        )

        class Stream:
            def record_event(self):
                outer.events.append("event")
                return "gmm2-event"

            def wait_event(self, event):
                outer.events.append("wait")
                outer.prefetch_ready = True

        def gmm(x, weight, group_list, **kwargs):
            outer.events.append("gemm2")
            outer.gmm2_calls += 1
            return [self.group_mm(x[0], weight[0], group_list)]

        fake_npu = types.SimpleNamespace(
            npu=types.SimpleNamespace(current_stream=lambda: Stream()),
            npu_grouped_matmul=gmm,
            npu_swiglu=lambda x: torch.nn.functional.silu(x[:, :2]) * x[:, 2:],
        )

        def quant(hidden_states, **kwargs):
            outer.quant_calls += 1
            outer.events.append("quant")
            return hidden_states.clone(), torch.ones(hidden_states.shape[0], 1)

        def quant_gmm2(hidden_states, weight, group_list, **kwargs):
            outer.gmm2_calls += 1
            outer.events.append("gemm2")
            return self.group_mm(hidden_states, weight, group_list)

        source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_home_gemm.py"
        tree = ast.parse(source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        tree = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *functions,
            ],
            type_ignores=[],
        )
        self.ns = {
            "torch": torch,
            "torch_npu": fake_npu,
            "MoEActivation": self.activation,
            "QuantType": types.SimpleNamespace(W4A8MXFP="mx"),
            "DeviceOperator": types.SimpleNamespace(
                npu_dynamic_quant=quant,
                maybe_normalize_mxfp_scale_layout=lambda x: x,
                npu_grouped_matmul_gmm2=quant_gmm2,
            ),
            "dispose_tensor": lambda x: self.disposed.append(id(x)),
            "AscendSwigluOAIAndMul": None,
            "AscendSwigluStepAndMul": None,
        }
        exec(compile(ast.fix_missing_locations(tree), str(source), "exec"), self.ns)

        class Range:
            def run(self, x, w, counts, out, begin, end):
                outer.events.append(("range", begin, end))
                if begin >= 2 and not outer.prefetch_ready:
                    raise AssertionError("Slot weights consumed before prefetch join")
                if id(x) in outer.disposed:
                    raise AssertionError("Shared quantized input disposed before Slot GEMM1")
                outer.outputs.append(id(out))
                outer.quant_inputs.append(id(x))
                row = 0
                for i, count in enumerate(counts.tolist()):
                    if begin <= i < end:
                        out[row : row + count].copy_((x[row : row + count].float() @ w[i].float()).bfloat16())
                    row += count

            def bf16(self, x, w, counts, out, begin, end):
                self.run(x, w, counts, out, begin, end)

            def mxfp4(self, x, w, ws, xs, counts, out, begin, end):
                self.run(x, w, counts, out, begin, end)

        self.range = Range()

    @staticmethod
    def group_mm(x, w, counts):
        out = []
        start = 0
        for i, count in enumerate(counts.tolist()):
            out.append((x[start : start + count].float() @ w[i].float()).bfloat16())
            start += count
        return torch.cat(out)

    def inputs(self, counts, quantized=False):
        torch.manual_seed(42)
        counts = torch.tensor(counts, dtype=torch.int64)
        x = torch.randn(int(counts.sum()), 4).bfloat16()
        w1 = torch.randn(4, 4, 4).bfloat16()
        w2 = torch.randn(4, 2, 4).bfloat16()
        weights = types.SimpleNamespace(
            w1=w1,
            w2=w2,
            w1_scale=torch.ones(4),
            w2_scale=torch.ones(4),
            w1_bias=None,
            w2_bias=None,
            w1_scale_bias=None,
            w2_scale_bias=None,
            w1_offset=None,
            w2_offset=None,
        )
        return types.SimpleNamespace(
            hidden_states=x,
            group_list=counts,
            group_list_type=1,
            lora_context=None,
            weights=weights,
            quant=types.SimpleNamespace(
                is_quant=quantized,
                quant_type="mx",
                mxfp=types.SimpleNamespace(
                    weight_quant_type="fp4", scale_dtype=None, per_token_scale_dtype=None, use_bf16=True
                ),
            ),
            activation="silu",
            dynamic_scale=None,
            swiglu_limit=0,
            need_trans=False,
            topk_scales=None,
        )

    def run_case(self, counts, event=True, quantized=False, record_gmm2=False):
        mci = self.inputs(counts, quantized)
        expected_up = self.group_mm(mci.hidden_states, mci.weights.w1, mci.group_list)
        expected_hidden = torch.nn.functional.silu(expected_up[:, :2]) * expected_up[:, 2:]
        expected = self.group_mm(expected_hidden, mci.weights.w2, mci.group_list)
        if not event:
            self.prefetch_ready = True

        def activation(x, **kwargs):
            self.activation_calls += 1
            self.events.append("activation")
            return torch.nn.functional.silu(x[:, :2]) * x[:, 2:], torch.ones(x.shape[0], 1), None

        ops = torch.ops._C_ascend
        old = getattr(ops, "npu_swiglu_group_quant", None)
        ops.npu_swiglu_group_quant = activation
        try:
            actual, unused = self.ns["home_gemm_apply_mlp"](
                mlp_compute_input=mci,
                home_experts=2,
                intermediate_size=2,
                range_kernel=self.range,
                prefetch_ready=object() if event else None,
                record_before_gmm2=record_gmm2,
            )
        finally:
            if old is None:
                del ops.npu_swiglu_group_quant
            else:
                ops.npu_swiglu_group_quant = old
        torch.testing.assert_close(actual, expected)
        if record_gmm2:
            self.assertEqual(unused, "gmm2-event")
            self.assertLess(self.events.index("event"), self.events.index("gemm2"))
        else:
            self.assertIsNone(unused)
            self.assertNotIn("event", self.events)
        self.assertEqual(len(set(self.outputs)), 1)
        self.assertEqual(self.gmm2_calls, 1)
        self.assertEqual([e for e in self.events if isinstance(e, tuple)], [("range", 0, 2), ("range", 2, 4)])
        if event:
            self.assertLess(self.events.index(("range", 0, 2)), self.events.index("wait"))
            self.assertLess(self.events.index("wait"), self.events.index(("range", 2, 4)))
        else:
            self.assertNotIn("wait", self.events)
        if quantized:
            self.assertEqual(self.quant_calls, 1)
            self.assertEqual(self.activation_calls, 1)
            self.assertEqual(len(set(self.quant_inputs)), 1)
            self.assertIn(self.quant_inputs[-1], self.disposed)

    def test_bf16_mixed_unequal_rows(self):
        self.run_case([1, 3, 2, 1])

    def test_empty_home_groups(self):
        self.run_case([0, 0, 2, 3])

    def test_empty_slot_groups(self):
        self.run_case([2, 3, 0, 0])

    def test_empty_middle_groups(self):
        self.run_case([2, 0, 0, 3])

    def test_first_layer_without_side_event(self):
        self.run_case([1, 1, 2, 2], event=False)

    def test_quantize_once_shared_input_and_output(self):
        self.run_case([1, 3, 2, 1], quantized=True)

    def test_bf16_shared_consumer_records_gmm2_event(self):
        self.run_case([1, 3, 2, 1], record_gmm2=True)

    def test_quantized_shared_consumer_records_gmm2_event(self):
        self.run_case([1, 3, 2, 1], quantized=True, record_gmm2=True)

    def test_swiglustep_custom_limit_and_default_fallback(self):
        calls = []

        def step(x, limit):
            calls.append(limit)
            gate, up = x.chunk(2, dim=-1)
            return torch.nn.functional.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)

        self.ns["AscendSwigluStepAndMul"] = types.SimpleNamespace(swiglustep_forward=step)
        x = torch.tensor([[20.0, -2.0, 10.0, -10.0]])
        for limit in (None, 0.0, 2.5):
            with self.subTest(limit=limit):
                actual = self.ns["_bf16_activation"](x, self.activation.SWIGLUSTEP, limit)
                effective = limit or 7.0
                expected = torch.nn.functional.silu(x[:, :2]).clamp(max=effective) * x[:, 2:].clamp(
                    -effective, effective
                )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(calls[-1], effective)

    def test_quantized_non_silu_activations_fail_before_work(self):
        for activation in (
            self.activation.SWIGLUOAI,
            self.activation.SWIGLUSTEP,
            self.activation.GELU,
            self.activation.GELU_TANH,
        ):
            with self.subTest(activation=activation):
                mci = self.inputs([1, 1, 1, 1], True)
                mci.activation = activation
                with self.assertRaisesRegex(NotImplementedError, "SiLU"):
                    self.ns["home_gemm_apply_mlp"](
                        mlp_compute_input=mci, home_experts=2, intermediate_size=2, range_kernel=self.range
                    )
                self.assertEqual(self.events, [])

    def test_reject_unadapted_backend_before_work(self):
        mci = self.inputs([1, 1, 1, 1], True)
        mci.quant.quant_type = "unadapted"
        with self.assertRaises(NotImplementedError):
            self.ns["home_gemm_apply_mlp"](
                mlp_compute_input=mci,
                home_experts=2,
                intermediate_size=2,
                range_kernel=self.range,
                prefetch_ready=object(),
            )
        self.assertEqual(self.events, [])

    def test_reject_lora_before_work(self):
        mci = self.inputs([1, 1, 1, 1])
        mci.lora_context = object()
        with self.assertRaises(NotImplementedError):
            self.ns["home_gemm_apply_mlp"](
                mlp_compute_input=mci,
                home_experts=2,
                intermediate_size=2,
                range_kernel=self.range,
            )
        self.assertEqual(self.events, [])


class HomeSlotIntegrationTests(unittest.TestCase):
    def setUp(self):
        from test_moonep_slot_cache_stream import CacheStreamTests

        self.case = CacheStreamTests(methodName="runTest")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.ns["_PREFETCH_OVERLAP"] = "dispatch|home_gemm1"
        self.case.ns["_CACHED_EXPERT_COUNT"] = 0
        self.case.state.caches = {}
        self.case.state.epn = self.case.state.B = 2
        self.case.state.F = 2
        runner = types.SimpleNamespace(activation_logged=True)
        self.case.state.home_gmm_runner = lambda: runner
        self.received_event = None

        def home(**kwargs):
            self.case.events.append("home-gemm1")
            self.received_event = kwargs["prefetch_ready"]
            if self.received_event is not None:
                self.case.fake_npu.npu.current_stream().wait_event(self.received_event)
            self.case.events.append("slot-gemm1")
            return self.case.state.slots["w13"].clone(), None

        self.case.ns["home_gemm_apply_mlp"] = home

    def test_forward_defers_join_until_after_home(self):
        from test_moonep_slot_cache_stream import Plan

        plan = Plan(torch.tensor([[2, 4]], dtype=torch.int32))
        self.case.hook(plan)
        start = len(self.case.events)
        self.case.events.append("dispatch")
        out = self.case.forward(plan)
        events = self.case.events[start:]
        self.assertEqual(events, ["dispatch", "home-gemm1", "wait", "slot-gemm1"])
        self.assertEqual(out.flatten().tolist(), [22.0, 44.0])
        self.assertIsNotNone(self.received_event)
        self.assertNotIn("slot_prefetch_ready", self.case.ctx)

    def test_first_use_prefetches_on_main_then_computes(self):
        from test_moonep_slot_cache_stream import Plan

        self.case.forward(Plan(torch.tensor([[2, 4]], dtype=torch.int32)))
        self.assertEqual(self.case.events, ["prefetch", "home-gemm1", "slot-gemm1"])
        self.assertIsNone(self.received_event)

    def test_mode3_rejects_unadapted_cache_combination(self):
        from test_moonep_slot_cache_stream import Plan

        self.case.ns["_CACHED_EXPERT_COUNT"] = 2
        with self.assertRaisesRegex(RuntimeError, "cache disabled"):
            self.case.forward(Plan(torch.tensor([[2, 4]], dtype=torch.int32)))
        self.assertEqual(self.case.events, [])

    def test_mode3_udma_still_requires_queue_isolation(self):
        from test_moonep_slot_cache_stream import Plan

        self.case.ctx["_prefetch_use_udma"] = True
        self.case.ctx["udma_prefetch_overlap_ready"] = False
        with self.assertRaisesRegex(RuntimeError, "dedicated QP initialization"):
            self.case.hook(Plan(torch.tensor([[2, 4]], dtype=torch.int32)))
        self.assertEqual(self.case.events, [])


if __name__ == "__main__":
    unittest.main()
