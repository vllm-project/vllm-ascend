# SPDX-License-Identifier: Apache-2.0
"""Exercise graph replay metadata without importing an NPU runtime."""

from __future__ import annotations

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch


class TestDFlashDPMetadata(unittest.TestCase):
    def test_replay_updates_graph_with_host_metadata(self):
        source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        manager = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DFlashAclGraphManager"
        )
        replay = next(
            node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "run_fullgraph"
        )
        events = []
        metadata = {"draft": object()}
        context = object()
        stream = object()

        class GraphBase:
            def run_fullgraph(self, desc):
                events.append("replay")
                return "graph-output"

        @contextmanager
        def forward_context(*args, **kwargs):
            counts = kwargs["num_tokens_across_dp"]
            # Device metadata here would require a synchronization before the
            # pending graph's parameters can be updated. Meta tensors also
            # cannot be read as host scalars, so they expose that regression.
            self.assertEqual(counts.device.type, "cpu")
            self.assertEqual(counts.tolist(), [6, 6])
            self.assertEqual(counts[1].item(), kwargs["num_tokens"])
            events.append("context")
            yield

        def update(*args, **kwargs):
            self.assertIs(args[2], context)
            self.assertIs(kwargs["draft_attn_metadatas"], metadata)
            events.append("update")

        def wait_stream(other):
            self.assertIs(other, stream)
            events.append("wait")

        isolated_class = ast.ClassDef(
            name=manager.name,
            bases=[ast.Name(id="GraphBase", ctx=ast.Load())],
            keywords=[],
            body=[replay],
            decorator_list=[],
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), isolated_class],
            type_ignores=[],
        )
        namespace = {
            "GraphBase": GraphBase,
            "torch": SimpleNamespace(full=torch.full, npu=SimpleNamespace(current_stream=lambda: stream)),
            "set_forward_context": forward_context,
            "get_forward_context": lambda: context,
            "get_draft_graph_params": lambda: object(),
            "update_full_graph_params": update,
            "_EXTRA_CTX": SimpleNamespace(),
        }
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        instance = namespace[manager.name]()
        instance.device = torch.device("meta")
        instance.decode_query_len = 6
        instance.update_stream = SimpleNamespace(wait_stream=wait_stream)
        instance.vllm_config = object()
        instance._validate_graph_param_cardinality = lambda *args: None
        instance.speculator = SimpleNamespace(
            dp_size=2,
            num_query_per_req=6,
            input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=None),
            model_state=SimpleNamespace(attn_metadata={}),
            build_draft_attn_metadatas=lambda *args: metadata,
            attn_backend=object(),
            attn_backends={"draft": object()},
            speculative_config=object(),
        )
        # Explicit CPU placement must work even with a non-CPU default device.
        with torch.device("meta"):
            result = instance.run_fullgraph(SimpleNamespace(num_tokens=6, num_reqs=1, cg_mode="FULL"))
        self.assertEqual(result, "graph-output")
        self.assertEqual(events, ["wait", "replay", "context", "update"])


if __name__ == "__main__":
    unittest.main()
