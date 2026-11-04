# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for cancelled partial DSpark prefill bookkeeping.

Run: python tests/ut/worker/test_dspark_prefill_cleanup.py
Uses only the standard library and executes the complete checked-out sender
method. Connector and hardware dependencies are mocked.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestCancelledDSparkPrefillCleanup(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        runner_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUModelRunner")
        method = next(
            n for n in runner_class.body if isinstance(n, ast.FunctionDef) and n.name == "_maybe_send_draft_kv"
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        self.sender = MagicMock()
        self.metadata = SimpleNamespace(requests={})
        namespace = {
            "find_dspark_context_connector": MagicMock(return_value=(MagicMock(), self.metadata)),
            "get_kv_transfer_group": MagicMock(),
            "find_dspark_prefix_connector": MagicMock(return_value=None),
            "send_dspark_prefill_kv": self.sender,
        }
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        self.send = namespace["_maybe_send_draft_kv"]
        self.runner = SimpleNamespace(
            pd_dspark_aux_layer_ids=(1, 3),
            is_last_pp_rank=True,
            execute_model_state=None,
            speculator=MagicMock(),
            _dspark_prefill_progress={"cancelled": ("generation-a", 16), "still-live": ("generation-b", 16)},
        )
        self.scheduler_output = SimpleNamespace(
            kv_connector_metadata=SimpleNamespace(),
            finished_req_ids={"cancelled"},
            total_num_scheduled_tokens=0,
        )
        hardware = patch.dict("sys.modules", {"torch_npu": MagicMock(), "acl": MagicMock()})
        hardware.start()
        self.addCleanup(hardware.stop)

    def assert_cancelled_removed(self):
        self.assertNotIn("cancelled", self.runner._dspark_prefill_progress)
        self.assertEqual(self.runner._dspark_prefill_progress["still-live"], ("generation-b", 16))

    def test_finished_partial_prefill_is_removed_without_active_dspark_metadata(self):
        self.send(self.runner, self.scheduler_output)
        self.assert_cancelled_removed()
        self.sender.assert_not_called()

    def test_finished_partial_prefill_is_removed_without_execution_state(self):
        self.metadata.requests = {"still-live": SimpleNamespace(dspark_context_generation="generation-b")}
        self.send(self.runner, self.scheduler_output)
        self.assert_cancelled_removed()
        self.sender.assert_not_called()

    def test_cleanup_precedes_missing_auxiliary_state_error(self):
        self.metadata.requests = {"still-live": SimpleNamespace(dspark_context_generation="generation-b")}
        self.scheduler_output.total_num_scheduled_tokens = 1
        with self.assertRaisesRegex(RuntimeError, "no target auxiliary states"):
            self.send(self.runner, self.scheduler_output)
        self.assert_cancelled_removed()
        self.sender.assert_not_called()

    def test_empty_or_unknown_finished_ids_preserve_live_progress(self):
        expected = self.runner._dspark_prefill_progress.copy()
        for finished in (None, set(), {"not-in-progress"}):
            with self.subTest(finished=finished):
                self.scheduler_output.finished_req_ids = finished
                self.send(self.runner, self.scheduler_output)
                self.assertEqual(self.runner._dspark_prefill_progress, expected)
        self.sender.assert_not_called()

    def test_cleanup_preserves_sending_for_other_active_requests(self):
        self.metadata.requests = {"still-live": SimpleNamespace(dspark_context_generation="generation-b")}
        state = SimpleNamespace(input_batch=MagicMock(), aux_hidden_states=MagicMock())
        self.runner.execute_model_state = state
        self.send(self.runner, self.scheduler_output)
        self.assert_cancelled_removed()
        self.sender.assert_called_once()
        self.assertIs(self.sender.call_args.args[1], state.input_batch)
        self.assertIs(self.sender.call_args.args[2], state.aux_hidden_states)
        self.assertIs(self.sender.call_args.args[5], self.runner._dspark_prefill_progress)


if __name__ == "__main__":
    unittest.main()
