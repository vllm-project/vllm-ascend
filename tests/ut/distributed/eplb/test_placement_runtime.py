# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from tests.ut.distributed.eplb.test_placement import make_metadata
from vllm_ascend.distributed.eplb import placement_runtime as runtime
from vllm_ascend.distributed.eplb.placement import EplbPlacement, load_placement, save_placement


class TestPlacementRuntime(unittest.TestCase):
    def setUp(self):
        self.group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)

        def gather(output, value, *, group):
            output[:] = [value] * group.size()

        self.gather = patch.object(runtime.dist, "all_gather_object", side_effect=gather)
        self.gather.start()
        self.addCleanup(self.gather.stop)
        self.barrier = patch.object(runtime.dist, "barrier")
        self.barrier.start()
        self.addCleanup(self.barrier.stop)
        self.placement = EplbPlacement(make_metadata(), ((0, 1, 2, 0, 1, 3),), {"workload": "test"})

    def test_collective_validation_reports_remote_failure(self):
        def gather(output, value, *, group):
            output[:] = [None, "ValueError: wrong model"]

        with (
            patch.object(runtime.dist, "all_gather_object", side_effect=gather),
            self.assertRaisesRegex(ValueError, "rank 1.*wrong model"),
        ):
            runtime.collective_call(lambda: 42, self.group, "test")

    def test_consensus_rejects_different_rank_files(self):
        def gather(output, value, *, group):
            output[:] = [None, None] if value is None else [value, "different"]

        with (
            patch.object(runtime.dist, "all_gather_object", side_effect=gather),
            self.assertRaisesRegex(ValueError, "differs across"),
        ):
            runtime.verify_placement_consensus(self.placement, self.group)

    def test_export_finishes_commits_before_snapshot_and_writes_once(self):
        model_state = SimpleNamespace(physical_to_logical_map=torch.tensor([[0, 1, 2, 3, 0, 1]]))

        def finish():
            model_state.physical_to_logical_map.copy_(torch.tensor(self.placement.physical_to_logical))

        state = SimpleNamespace(
            model_states={"target": model_state},
            finish_pending_commits=MagicMock(side_effect=finish),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(policy="stair", window_size=2, step_interval=2)
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            result = runtime.export_committed_placement(state, make_metadata(), path, "calibration-A", self.group)
            self.assertEqual(load_placement(path).physical_to_logical, self.placement.physical_to_logical)
            self.assertEqual(load_placement(path).provenance["workload"], "calibration-A")
            self.assertEqual(result["sha256"], load_placement(path).digest())
            state.finish_pending_commits.assert_called_once_with()
            self.group.rank = lambda: 1
            with patch.object(runtime, "save_placement") as save:
                runtime.export_committed_placement(state, make_metadata(), path, "calibration-A", self.group)
                save.assert_not_called()

    def test_export_propagates_writer_failure(self):
        state = SimpleNamespace(
            model_states={
                "target": SimpleNamespace(physical_to_logical_map=torch.tensor(self.placement.physical_to_logical))
            },
            finish_pending_commits=MagicMock(),
            parallel_config=SimpleNamespace(
                eplb_config=SimpleNamespace(policy="stair", window_size=2, step_interval=2)
            ),
        )
        with (
            patch.object(runtime, "save_placement", side_effect=OSError("read-only directory")),
            self.assertRaisesRegex(ValueError, "placement writing.*read-only"),
        ):
            runtime.export_committed_placement(state, make_metadata(), "out.json", "calibration", self.group)
