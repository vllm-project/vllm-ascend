# SPDX-License-Identifier: Apache-2.0
"""CPU event-DAG regressions for the PR45/48/51 routed front and MC2 handoff."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from test_kimi_v30_review import load_class


class EventGraph:
    def __init__(self):
        self.nodes = []
        self.last = dict.fromkeys(("main", "router", "shared", "comm"))
        self.current = "main"
        self.streams = {name: self.Stream(self, name) for name in self.last}

    def op(self, name, dependencies=()):
        index = len(self.nodes)
        deps = [d for d in (self.last[self.current], *dependencies) if d is not None]
        self.nodes.append((self.current, name, deps))
        self.last[self.current] = index
        return index

    class Stream:
        def __init__(self, graph, name):
            self.graph, self.name = graph, name

        def record_event(self):
            return self.graph.last[self.name]

        def wait_event(self, event):
            self.graph.op("wait", (event,))

    @contextmanager
    def switch(self, stream, enabled=True):
        old = self.current
        if enabled:
            self.current = stream.name
        try:
            yield
        finally:
            self.current = old

    def index(self, name):
        return next(i for i, (_, label, _) in enumerate(self.nodes) if label == name)

    def ancestors(self, index):
        deps = self.nodes[index][2]
        return set(deps).union(*(self.ancestors(d) for d in deps))


class TestKimiFrontOverlap(unittest.TestCase):
    def test_final_join_waits_for_this_invocations_comm_event_after_latent_up(self):
        order = []
        done = object()

        class Base:
            def apply_routed_output_transform(self, x):
                order.append("latent_up")
                return x * 2

        def wait(event):
            self.assertIs(event, done)
            order.append("join_rs")

        ns = dict(Base=Base, torch=NS(npu=NS(current_stream=lambda: NS(wait_event=wait))))
        cls = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            {"apply_routed_output_transform"},
            ns,
            "Base",
        )
        runner = cls()
        runner.routed_output_transform = object()
        runner._can_overlap_sp_shared_with = lambda transform: True
        runner._kimi_shared_output_done = done
        runner.ascend_shared_experts = NS(wait_for_output=Mock(side_effect=AssertionError("wrong stream join")))
        actual = runner.apply_routed_output_transform(torch.ones(1))
        torch.testing.assert_close(actual, torch.full((1,), 2.0))
        self.assertEqual(order, ["latent_up", "join_rs"])

    def test_router_quant_and_topk_projection_remain_independent(self):
        for preprojected in (False, True):
            for superkernel in (False, True):
                with self.subTest(preprojected=preprojected, superkernel=superkernel):
                    self.check_front(preprojected, superkernel=superkernel)

    def test_non_decode_retains_main_stream_gather_wait(self):
        self.check_front(True, decode=False)

    def check_front(self, preprojected, *, decode=True, superkernel=False):
        graph = EventGraph()
        inputs = torch.arange(24).reshape(3, 8).to(torch.bfloat16)
        projected = inputs[:, :4] * 2
        logits = torch.arange(18).reshape(3, 6).float()
        padded_logits = torch.nn.functional.pad(logits, (0, 0, 0, 2))
        records, limits = [], []
        npu = NS(current_stream=lambda: graph.streams[graph.current])

        class TorchProxy:
            def __getattr__(self, key):
                return npu if key == "npu" else getattr(torch, key)

        class Base:
            def apply_routed_input_transform(self, x):
                raise AssertionError("latent projection repeated or quantization not separated")

        def gather(x, dim):
            graph.op("all_gather")
            return x.repeat(2, 1)

        ns = dict(
            torch=TorchProxy(),
            F=torch.nn.functional,
            nullcontext=nullcontext,
            npu_stream_switch=graph.switch,
            PreparedSharedExpertInput=NS,
            shared_experts_calculation_stream=lambda: graph.streams["shared"],
            tensor_model_parallel_all_gather=gather,
            SharedExpertParallelMode=NS(SEQUENCE_PARALLEL_ONLY="sp", SHARED_EXPERT_DATA_PARALLEL_ONLY="dp"),
            RoutedMoEMilestones=NS,
            PreparedSharedExpertExecution=NS,
            _EXTRA_CTX=NS(num_tokens=6),
        )
        shared_cls = load_class(
            "vllm_ascend/ops/fused_moe/shared_experts.py",
            "AscendSharedExperts",
            {"prepare_input_async", "prepare_for_router_overlap", "_prepare_execution_input", "_wait_for_milestone"},
            ns,
        )
        shared = shared_cls()
        shared.multistream_overlap = True
        shared.parallel_mode = lambda: "sp"

        @contextmanager
        def core(*limit):
            limits.append((graph.current, limit))
            yield

        def select(metadata, actual_logits, force_balance, ids):
            self.assertEqual(metadata.shape, (5, 4))
            self.assertIsNone(ids)
            self.assertFalse(force_balance)
            torch.testing.assert_close(actual_logits, padded_logits)
            graph.op("topk")
            return torch.topk(actual_logits, 2)

        def quantize(layer, x):
            self.assertIs(x, inputs)
            graph.op("latent_quant")
            return (x, "prepared scale")

        class Projection:
            quant_method = NS(quant_method=NS(prepare_input_for_overlap=Mock(side_effect=quantize)))

            def __call__(self, x):
                self_input, scale = x
                assert self_input is inputs and scale == "prepared scale"
                graph.op("latent_projection")
                return projected, None

        ns.update(
            Base=Base,
            torch_npu=NS(npu=NS(npugraph_ex=NS(scope=NS(limit_core_num=core)))),
            ROUTER_CORE_LIMIT=(32, 1),
            MoECommType=NS(MC2=1),
            get_moe_comm_method=lambda *args: NS(prepare_finalize=NS(tp_size=2)),
            _EXTRA_CTX=NS(padded_num_tokens=10, in_profile_run=False, num_tokens=6),
            PreparedSharedExpertInput=lambda hidden_states, **kw: NS(
                hidden_states=hidden_states, ready_event=kw.pop("ready_event", None), **kw
            ),
        )
        cls = load_class(
            "vllm_ascend/ops/fused_moe/fused_moe.py",
            "AscendMoERunner",
            {"apply_routed_input_transform", "_prepare_kimi_decode_topk", "_forward_impl"},
            ns,
            "Base",
        )
        runner = cls()
        runner.ascend_shared_experts = shared
        runner._kimi_gate = object()
        runner._kimi_router_stream = graph.streams["router"]
        runner._kimi_shared_comm_stream = graph.streams["comm"]
        runner._can_overlap_sp_shared_with = lambda transform: True
        runner._can_run_kimi_decode = lambda x: decode
        runner._kimi_superkernel_active = superkernel
        # A new invocation must overwrite an earlier call's readiness.
        runner._kimi_shared_input_ready = object()
        runner._compute_router_logits = lambda *args: (graph.op("router_gemm") and logits)
        runner.routed_input_transform = Projection()
        runner.routed_experts = NS(_select_experts=select)
        runner.moe_config = NS(hidden_dim=4)
        graph.op("input_ready")
        with patch.object(torch.Tensor, "record_stream", lambda tensor, stream: records.append(stream.name)):
            actual, shared_input = runner.apply_routed_input_transform(
                inputs, routed_input=projected if preprojected else None
            )
        self.assertIs(actual, projected)
        torch.testing.assert_close(shared_input, inputs.repeat(2, 1))
        self.assertEqual(limits, [("router", (32, 1))] if decode else [])
        self.assertIn("comm", records)
        self.assertEqual(graph.nodes[graph.index("all_gather")][0], "comm")
        self.assertNotIn(graph.index("all_gather"), graph.ancestors(graph.index("router_gemm")))
        if preprojected:
            runner.routed_input_transform.quant_method.quant_method.prepare_input_for_overlap.assert_not_called()
            self.assertNotIn("latent_projection", [n[1] for n in graph.nodes])
        else:
            quant, gate, topk, projection = [
                graph.index(name) for name in ("latent_quant", "router_gemm", "topk", "latent_projection")
            ]
            self.assertNotIn(gate, graph.ancestors(quant))
            self.assertNotIn(quant, graph.ancestors(gate))
            self.assertIn(gate, graph.ancestors(projection))
            self.assertIn(quant, graph.ancestors(projection))
            self.assertNotIn(topk, graph.ancestors(projection))
            self.assertNotIn(projection, graph.ancestors(topk))
            self.assertNotIn(graph.index("all_gather"), graph.ancestors(projection))

        if not decode:
            dispatch = graph.op("fallback_dispatch")
            self.assertIn(graph.index("all_gather"), graph.ancestors(dispatch))
            self.assertIsNone(runner._kimi_shared_input_ready)
            return

        # Exercise the real runner handoff and Shared input preparation:
        # Dispatch must not inherit AG, but Shared quantization/Gate-Up must.
        def prepare_shared(x):
            torch.testing.assert_close(x, inputs.repeat(2, 1))
            graph.op("shared_quant")
            return x

        def enqueue_shared(prepared, ready):
            graph.streams["shared"].wait_event(ready)
            graph.op("shared_gate_up")
            return NS(gate_up=prepared.gate_up_input)

        shared._prepare_gate_up_input = prepare_shared
        shared.enqueue_gate_up_after_router = enqueue_shared
        shared.apply_activation = lambda x: (graph.op("shared_situ"), x)[1]
        shared.layer = NS(gate_up_proj=NS())
        ns.update(BF16_SHARED_CORE_LIMIT=(32, 1), MXFP_SHARED_CORE_LIMIT=(16, 32))
        schedule_cls = load_class(
            "vllm_ascend/ops/fused_moe/kimi_decode_schedule.py",
            "KimiDecodeSchedule",
            {"__init__", "before_dispatch", "after_dispatch"},
            ns,
        )
        schedule_cls.finalize = lambda schedule: shared_input
        ns["KimiDecodeSchedule"] = schedule_cls
        runner._sequence_parallel_context = nullcontext
        runner.routed_output_transform = object()

        def routed_forward(**kwargs):
            schedule = kwargs["shared_expert_schedule"]
            schedule.before_dispatch(npu.current_stream().record_event())
            graph.op("dispatch")
            schedule.after_dispatch()
            graph.op("gmm1")
            return projected, None

        runner.routed_experts.forward_impl = routed_forward
        with patch.object(torch.Tensor, "record_stream", lambda *args: None):
            runner._forward_impl(projected, logits, shared_input)
        ag = graph.index("all_gather")
        self.assertNotIn(ag, graph.ancestors(graph.index("dispatch")))
        for name in ("shared_quant", "shared_gate_up"):
            self.assertEqual(graph.nodes[graph.index(name)][0], "shared")
            self.assertIn(ag, graph.ancestors(graph.index(name)))
        self.assertNotIn(graph.index("dispatch"), graph.ancestors(graph.index("shared_gate_up")))
        self.assertIn(graph.index("dispatch"), graph.ancestors(graph.index("shared_situ")))
        self.assertIn(graph.index("shared_situ"), graph.ancestors(graph.index("gmm1")))

    def test_precomputed_topk_preserves_mc2_prepare_masks_and_finalization(self):
        states = torch.ones(3, 4, dtype=torch.bfloat16)
        weights = torch.ones(5, 2)
        ids = torch.zeros(5, 2, dtype=torch.int32)
        mask = torch.tensor([True, True, True, False, False])
        main = Mock()

        class TorchProxy:
            npu = NS(current_stream=lambda: main)

            def __getattr__(self, key):
                return getattr(torch, key)

        prepare = Mock(
            return_value=NS(
                hidden_states=torch.ones(5, 4),
                router_logits=torch.ones(5, 6),
                mc2_mask=mask,
                padded_hidden_states_shape=(5, 4),
                pertoken_scale=None,
            )
        )
        comm = NS(prepare=prepare, finalize=Mock(side_effect=lambda **kw: kw["hidden_states"][:3]))
        ctx = NS(in_profile_run=False, moe_comm_type=1)
        ns = dict(
            torch=TorchProxy(),
            get_forward_context=lambda: NS(),
            _EXTRA_CTX=ctx,
            get_moe_comm_method=lambda *args: comm,
            AllGatherCommImpl=type("AllGather", (), {}),
            _record_v2_eplb_load=Mock(),
        )
        cls = load_class("vllm_ascend/ops/fused_moe/routed_experts.py", "AscendRoutedExperts", {"forward_impl"}, ns)
        layer = cls()
        layer.enable_npugraph_ex_static_kernel = False
        layer.moe_config = NS(is_sequence_parallel=True)
        layer.quant_type = "mxfp4"
        layer.router = object()
        layer._use_v2_model_runner = True
        layer.dynamic_eplb = False
        layer.return_with_event = False
        layer._select_experts_with_stream = Mock(side_effect=AssertionError("TopK recomputed"))

        def apply(**kw):
            self.assertIs(kw["topk_weights"], weights)
            self.assertIs(kw["topk_ids"], ids)
            self.assertIs(layer.ascend_mc2_mask, mask)
            self.assertIs(layer._ascend_shared_schedule, schedule)
            return NS(routed_out=kw["x"])

        layer.quant_method = NS(apply=Mock(side_effect=apply))
        schedule = object()
        done = object()
        with patch.object(torch.Tensor, "record_stream") as records:
            out = layer.forward_impl(
                hidden_states=states,
                router_logits=torch.ones(3, 6),
                precomputed_topk=(weights, ids, done),
                shared_expert_schedule=schedule,
            )
        self.assertEqual(out.shape, (3, 4))
        prepare.assert_called_once()
        main.wait_event.assert_called_once_with(done)
        self.assertEqual(records.call_count, 2)
        self.assertIsNone(layer.ascend_mc2_mask)
        self.assertIsNone(layer._ascend_shared_schedule)
        ns["_record_v2_eplb_load"].assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "padding"):
            layer.forward_impl(
                hidden_states=states, router_logits=torch.ones(3, 6), precomputed_topk=(weights[:3], ids[:3], done)
            )

    def test_real_dispatch_hook_brackets_shared_stages(self):
        order = []
        event = object()
        token_output = NS(combine_metadata=None, group_list_type=1, group_list=None)
        dispatcher = NS(
            token_dispatch=lambda **kw: (order.append("dispatch") or token_output),
            token_combine=lambda **kw: (order.append("combine") or kw["hidden_states"]),
        )
        ns = dict(
            torch=NS(
                **{
                    name: getattr(torch, name)
                    for name in ("float32", "float16", "bfloat16", "int8", "float8_e4m3fn", "uint8")
                },
                npu=NS(current_stream=lambda: NS(record_event=lambda: event)),
            ),
            _EXTRA_CTX=NS(moe_comm_method=object()),
            build_token_dispatch_input=lambda **kw: (order.append("prepare_dispatch") or object()),
            TokenDispatcherWithMC2=type("MC2", (), {}),
            build_mlp_compute_input=lambda **kw: object(),
            apply_moe_mlp=lambda *args: (order.append("gmm") or (torch.ones(1), event)),
            FusedExpertsResult=NS,
        )
        cls = load_class("vllm_ascend/ops/fused_moe/moe_comm_method.py", "MoECommMethod", {"fused_experts"}, ns)
        layer = cls()
        layer.token_dispatcher = dispatcher
        layer.moe_config = object()
        layer.shared_activation_overlaps_routed_gmm1 = False
        schedule = NS(
            before_dispatch=lambda e: (self.assertIs(e, event), order.append("gate_up")),
            after_dispatch=lambda: order.append("situ"),
        )
        layer.fused_experts(NS(hidden_states=torch.ones(1), shared_expert_schedule=schedule), object())
        self.assertEqual(order, ["prepare_dispatch", "gate_up", "dispatch", "situ", "gmm", "combine"])


if __name__ == "__main__":
    unittest.main()
