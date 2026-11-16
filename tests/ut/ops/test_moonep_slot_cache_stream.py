"""Exercise the real Slot cache/stream functions without loading vLLM workers."""

import ast
import dataclasses
import os
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


@dataclasses.dataclass
class Plan:
    experts_to_copy: object


@dataclasses.dataclass
class Compute:
    weights: object = None


def load_functions():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/moonep_balance.py"
    tree = ast.parse(source.read_text())
    names = {"_prepare_slot_cache", "_pf_stream_hook", "run_slot_forward"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = dict(torch=torch, dataclasses=dataclasses, os=os, time=time)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace


class CacheStreamTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions()
        self.events = []
        self.calls = []
        self.state = types.SimpleNamespace(
            caches={name: torch.tensor([[[[11.0]], [[33.0]]]]) for name in ("w13", "w2", "w13_scale", "w2_scale")},
            slots={name: torch.zeros(2, 1, 1) for name in ("w13", "w2", "w13_scale", "w2_scale")},
            cache_experts=[1, 3],
            cache_next=0,
            cache_stats=[0, 0],
            layer_index=0,
            cache_layer_index=0,
            compute_weights=lambda: None,
        )
        self.ctx = {}
        self.buffer = types.SimpleNamespace(_require_ctx=lambda: self.ctx)
        outer = self

        class Stream:
            def record_event(self):
                outer.events.append("record")
                return object()

            def wait_event(self, event):
                outer.events.append("wait")

        main = Stream()
        npu = types.SimpleNamespace(
            current_stream=lambda: main,
            Stream=Stream,
            stream=lambda stream: __import__("contextlib").nullcontext(),
        )
        self.fake_npu = types.SimpleNamespace(npu=npu)
        self.patch_modules = patch.dict(
            sys.modules,
            {
                "torch_npu": self.fake_npu,
                "vllm.distributed.parallel_state": types.SimpleNamespace(
                    get_ep_group=lambda: types.SimpleNamespace(rank_in_group=0)
                ),
            },
        )
        self.patch_modules.start()
        self.addCleanup(self.patch_modules.stop)

        def prefetch(buffer, state, plan):
            self.calls.append(plan.experts_to_copy.clone())
            self.events.append("prefetch")
            for slot, expert in enumerate(plan.experts_to_copy[0].tolist()):
                if expert >= 0:
                    for tensor in state.slots.values():
                        tensor[slot].fill_(expert * 11.0)

        def gemm(**kwargs):
            self.events.append("gemm")
            return self.state.slots["w13"].clone(), None

        self.buffer.prefetch_projections = lambda **kw: prefetch(self.buffer, self.state, kw["plan"])
        self.state.prefetch_views = lambda: []
        self.ns.update(
            _CACHED_EXPERT_COUNT=2,
            _SLOT_CACHE_STATS={},
            _PF_SKIPPED=[0],
            _PREFETCH_OVERLAP="dispatch",
            _PF_STREAMED=[0],
            _PF={"cur_key": 1, "stream": None, "fail": 0},
            _pf_state_for=lambda key: self.state,
            _pf_prefetch=prefetch,
            _EXTRA_CTX=types.SimpleNamespace(capturing=False),
            build_mlp_compute_input=lambda **kw: Compute(),
            apply_slot_mlp=lambda payload, method, **kwargs: gemm(mlp_compute_input=payload),
        )

    def forward(self, plan):
        return self.ns["run_slot_forward"](
            state=self.state,
            buffer=self.buffer,
            plan=plan,
            token_dispatch_output=None,
            fused_experts_input=None,
            use_fusion_ops=False,
        )[0]

    def hook(self, plan):
        self.ns["_pf_stream_hook"](self.buffer)(plan)
        self.assertEqual(self.ns["_PF"]["fail"], 0)

    def test_mixed_hits_are_masked_before_side_stream_get(self):
        plan = Plan(torch.tensor([[1, 2], [1, 2]], dtype=torch.int32))
        self.hook(plan)
        self.assertEqual(self.calls[0].tolist(), [[-1, 2], [1, 2]])
        self.assertEqual(plan.experts_to_copy.tolist(), [[1, 2], [1, 2]])
        self.assertEqual(self.forward(plan).flatten().tolist(), [11.0, 22.0])
        self.assertEqual(self.state.cache_stats, [1, 1])
        self.assertEqual(len(self.calls), 1)
        self.assertLess(max(i for i, e in enumerate(self.events) if e == "wait"), self.events.index("gemm"))
        self.assertNotIn("slot_prefetch_ready", self.ctx)
        for tensor in self.state.slots.values():
            self.assertEqual(tensor.flatten().tolist(), [11.0, 22.0])

    def test_all_hits_launch_no_remote_prefetch(self):
        plan = Plan(torch.tensor([[1, 3]], dtype=torch.int32))
        self.hook(plan)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.forward(plan).flatten().tolist(), [11.0, 33.0])
        self.assertEqual(self.state.cache_stats, [2, 0])
        self.assertEqual(self.ns["_PF_SKIPPED"], [1])

    def test_all_misses_fill_cache_and_next_step_hits(self):
        self.state.cache_experts = [-1, -1]
        plan = Plan(torch.tensor([[2, 4]], dtype=torch.int32))
        self.hook(plan)
        self.assertEqual(self.forward(plan).flatten().tolist(), [22.0, 44.0])
        self.hook(plan)
        self.assertEqual(self.forward(plan).flatten().tolist(), [22.0, 44.0])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.state.cache_stats, [2, 2])

    def test_first_use_falls_back_and_installs_hook(self):
        plan = Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.assertEqual(self.forward(plan).flatten().tolist(), [11.0, 22.0])
        self.assertEqual(self.calls[0].tolist(), [[-1, 2]])
        self.assertIn("after_planning_hook", self.ctx)

    def test_cache_disabled_still_uses_side_stream(self):
        self.ns["_CACHED_EXPERT_COUNT"] = 0
        plan = Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.hook(plan)
        self.assertEqual(self.calls[0].tolist(), [[1, 2]])
        self.forward(plan)
        self.assertEqual(self.state.cache_stats, [0, 0])

    def test_main_stream_cache_path(self):
        self.ns["_PREFETCH_OVERLAP"] = "none"
        plan = Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.assertEqual(self.forward(plan).flatten().tolist(), [11.0, 22.0])
        self.assertNotIn("after_planning_hook", self.ctx)

    def test_empty_slots_do_not_fetch(self):
        plan = Plan(torch.tensor([[-1, -1]], dtype=torch.int32))
        self.hook(plan)
        self.forward(plan)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state.cache_stats, [0, 0])

    def test_udma_without_dedicated_qps_rejected_before_submission(self):
        self.ctx["_prefetch_use_udma"] = True
        with self.assertRaisesRegex(RuntimeError, "dedicated QP"):
            self.ns["_pf_stream_hook"](self.buffer)(Plan(torch.tensor([[1, 2]])))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.events, [])

    def test_udma_with_dedicated_qps_preserves_event_join(self):
        self.ctx.update(_prefetch_use_udma=True, udma_prefetch_overlap_ready=True)
        self.ns["_CACHED_EXPERT_COUNT"] = 0
        plan = Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.hook(plan)
        self.assertEqual(len(self.calls), 1)
        self.forward(plan)
        self.assertLess(max(i for i, e in enumerate(self.events) if e == "wait"), self.events.index("gemm"))

    def test_udma_issue_error_is_not_swallowed_or_retried(self):
        self.ctx.update(_prefetch_use_udma=True, udma_prefetch_overlap_ready=True)
        self.ns["_CACHED_EXPERT_COUNT"] = 0

        def fail(*args):
            self.calls.append("issued")
            raise RuntimeError("injected partial issue failure")

        self.ns["_pf_prefetch"] = fail
        with self.assertRaisesRegex(RuntimeError, "UDMA side-stream prefetch failed"):
            self.ns["_pf_stream_hook"](self.buffer)(Plan(torch.tensor([[1, 2]])))
        self.assertEqual(self.calls, ["issued"])
        self.assertNotIn("slot_prefetch_ready", self.ctx)

    def test_wrong_layer_is_rejected(self):
        plan = Plan(torch.tensor([[1, 2]], dtype=torch.int32))
        self.hook(plan)
        ready = self.ctx["slot_prefetch_ready"]
        self.ctx["slot_prefetch_ready"] = (object(), *ready[1:])
        with self.assertRaisesRegex(RuntimeError, "different MoE layer"):
            self.forward(plan)


if __name__ == "__main__":
    unittest.main()
