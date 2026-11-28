# SPDX-License-Identifier: Apache-2.0
"""CPU checks for recipes registration and PR5/PR45 execution boundaries."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock

from test_kimi_v30_moe import load


class TestSituLoading(unittest.TestCase):
    def test_worker_loads_missing_ops_before_selection(self):
        for existing in ((), ("npu_situ_and_mul_sparse",)):
            with self.subTest(existing=existing):
                ops = NS(**{name: Mock() for name in existing})

                def register(name, ops=ops):
                    self.assertEqual(name, "custom_ops")
                    ops.npu_situ_and_mul_sparse = Mock()
                    ops.grouped_situ_mx_quant = Mock()

                ns = dict(torch=NS(ops=NS(custom=ops)), import_module=Mock(side_effect=register), logger=Mock())
                load("vllm_ascend/ops/kimi_situ.py", {"initialize_kimi_situ_ops", "SITU_OP_NAMES"}, ns)
                ns["initialize_kimi_situ_ops"]()
                ns["import_module"].assert_called_once_with("custom_ops")
                ns["logger"].warning_once.assert_not_called()
                # Already-registered extensions are not imported again.
                ns["initialize_kimi_situ_ops"]()
                ns["import_module"].assert_called_once()

    def test_missing_extension_warns_and_preserves_fallback(self):
        for error in (ModuleNotFoundError("custom_ops"), OSError("libcust_opapi.so"), None):
            with self.subTest(error=error):
                ns = dict(torch=NS(ops=NS(custom=NS())), import_module=Mock(side_effect=error), logger=Mock())
                load("vllm_ascend/ops/kimi_situ.py", {"initialize_kimi_situ_ops", "SITU_OP_NAMES"}, ns)
                ns["initialize_kimi_situ_ops"]()
                warning = ns["logger"].warning_once.call_args.args
                self.assertIn("native fallbacks", warning[0])
                self.assertIn("grouped_situ_mx_quant", warning[1])

    def test_prefill_gate_up_has_no_decode_core_limit(self):
        limit = Mock(side_effect=AssertionError("Decode limit leaked into Prefill"))
        ns = dict(
            nullcontext=nullcontext,
            SharedExpertMLPPath=NS(A8_INT_FUSED=0, A8_MXFP_FUSED=1),
            SubmittedSharedExpertGateUp=NS,
            torch_npu=NS(npu=NS(npugraph_ex=NS(scope=NS(limit_core_num=limit)))),
        )
        load("vllm_ascend/ops/fused_moe/shared_experts.py", {"enqueue_gate_up_after_router"}, ns, "AscendSharedExperts")
        shared = ns["AscendSharedExperts"]()
        shared._compute_stream_context = nullcontext
        shared._wait_for_milestone = Mock()
        shared._select_mlp_path = lambda: 1
        output = object()
        projection = Mock(return_value=(output,))
        shared.layer = NS(gate_up_proj=projection)
        prepared = NS(hidden_states=object(), gate_up_input=(object(), object()), local_dp_metadata=None)
        result = shared.enqueue_gate_up_after_router(prepared, None)
        self.assertIs(result.gate_up, output)
        projection.assert_called_once_with(prepared.gate_up_input)
        limit.assert_not_called()

    def test_pr45_topk_waits_for_prepare_and_dispatch_waits_for_topk(self):
        events = []
        main = NS(
            record_event=lambda: events.append("prepare_done") or "prepared",
            wait_event=lambda e: events.append("main_wait_" + e),
        )
        router = NS(
            record_event=lambda: events.append("topk_done") or "topk",
            wait_event=lambda e: events.append("router_wait_" + e),
        )
        current = [main]

        @contextmanager
        def switch(stream):
            self.assertIs(stream, router)
            current[0] = router
            try:
                yield
            finally:
                current[0] = main

        ctx = NS(moe_comm_type=1)
        ns = dict(
            torch=NS(npu=NS(current_stream=lambda: current[0])),
            _EXTRA_CTX=ctx,
            MoECommType=NS(MC2=1),
            npu_stream_switch=switch,
        )
        load("vllm_ascend/ops/fused_moe/routed_experts.py", {"_select_experts_with_stream"}, ns, "AscendRoutedExperts")
        layer = ns["AscendRoutedExperts"]()
        layer.moe_config = NS(is_sequence_parallel=True)
        weights, ids = Mock(), Mock()

        def select(*args):
            self.assertIs(current[0], router)
            events.append("select")
            return weights, ids

        layer._select_experts = Mock(side_effect=select)
        hidden, logits, input_ids = Mock(), Mock(), Mock()
        result = layer._select_experts_with_stream(hidden, logits, False, input_ids, router)
        self.assertEqual(result, (weights, ids))
        self.assertEqual(events, ["prepare_done", "router_wait_prepared", "select", "topk_done", "main_wait_topk"])
        for tensor in (hidden, logits, input_ids):
            tensor.record_stream.assert_called_once_with(router)
        for tensor in result:
            tensor.record_stream.assert_called_once_with(main)
        layer._select_experts = Mock(return_value=(weights, ids))
        for stream, comm, sp in ((None, 1, True), (router, 2, True), (router, 1, False)):
            events.clear()
            ctx.moe_comm_type = comm
            layer.moe_config.is_sequence_parallel = sp
            layer._select_experts_with_stream(hidden, logits, False, None, stream)
            self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
