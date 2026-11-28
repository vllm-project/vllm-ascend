# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch

from tests.ut.distributed.eplb.test_placement import make_metadata, write_placement
from vllm_ascend.distributed.eplb import placement_runtime as runtime
from vllm_ascend.distributed.eplb.placement import EplbPlacement


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

        self.leader = patch.object(runtime.dist, "get_global_rank", return_value=0)
        self.leader.start()
        self.addCleanup(self.leader.stop)

        def broadcast(object_list, *, src, group):
            # The leader already filled the payload in its own process.
            return None

        self.broadcast = patch.object(runtime.dist, "broadcast_object_list", side_effect=broadcast)
        self.broadcast.start()
        self.addCleanup(self.broadcast.stop)
        self.placement = EplbPlacement(make_metadata(), ((0, 1, 2, 0, 1, 3),), {"workload": "test"})

    def test_collective_validation_reports_remote_failure(self):
        def gather(output: list[Any], value: Any, *, group):
            output[:] = [None, "ValueError: wrong model"]

        with (
            patch.object(runtime.dist, "all_gather_object", side_effect=gather),
            self.assertRaisesRegex(ValueError, "rank 1.*wrong model"),
        ):
            runtime.collective_call(lambda: 42, self.group, "test")

    def test_consensus_rejects_different_rank_files(self):
        def gather(output: list[Any], value: Any, *, group):
            output[:] = [None, None] if value is None else [value, "different"]  # type: ignore[list-item]

        with (
            patch.object(runtime.dist, "all_gather_object", side_effect=gather),
            self.assertRaisesRegex(ValueError, "differs across"),
        ):
            runtime.verify_placement_consensus(self.placement, self.group)

    def test_loading_checks_model_before_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_placement(path, self.placement)
            metadata = make_metadata()
            metadata["model"] = "wrong-model"
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                runtime.read_placement_collectively(path, metadata, self.group)

    def test_loading_validates_draft_placement(self):
        draft_metadata = make_metadata()
        draft_metadata.update(
            model="draft-model",
            num_logical_experts=2,
            num_redundant_experts=0,
            layer_names=["layers.0.draft.moe"],
        )
        placement = EplbPlacement(
            self.placement.metadata,
            self.placement.physical_to_logical,
            self.placement.provenance,
            draft=EplbPlacement(draft_metadata, ((0, 1),), {"workload": "draft"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_placement(path, placement)
            loaded = runtime.read_placement_collectively(
                path,
                make_metadata(),
                self.group,
                draft_metadata=draft_metadata,
            )
            self.assertEqual(loaded, placement)
            wrong = dict(draft_metadata)
            wrong["model"] = "wrong-draft"
            with self.assertRaisesRegex(ValueError, "model mismatch"):
                runtime.read_placement_collectively(
                    path,
                    make_metadata(),
                    self.group,
                    draft_metadata=wrong,
                )

    def _patch_passthrough_collective(self):
        return patch.object(
            runtime,
            "collective_call",
            side_effect=lambda operation, group, description: operation(),
        )

    def test_loading_reads_only_on_the_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_placement(path, self.placement)
            with patch.object(
                runtime.dist,
                "broadcast_object_list",
                side_effect=lambda object_list, *, src, group: None,
            ) as broadcast:
                loaded = runtime.read_placement_collectively(path, make_metadata(), self.group)
            self.assertEqual(loaded, self.placement)
            self.assertEqual(broadcast.call_args.kwargs["src"], 0)

    def test_loading_does_not_read_on_a_non_leader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "deployment.json")
            write_placement(path, self.placement)
            text = Path(path).read_text(encoding="utf-8")
            self.group.rank = lambda: 1

            def broadcast(object_list, *, src, group):
                object_list[0] = text

            with patch.object(runtime.dist, "broadcast_object_list", side_effect=broadcast):
                loaded = runtime.read_placement_collectively(
                    str(Path(directory) / "absent.json"),
                    make_metadata(),
                    self.group,
                )
            self.assertEqual(loaded, self.placement)

    def test_loading_reports_missing_on_the_leader(self):
        with self.assertRaisesRegex(FileNotFoundError, "could not be read on EPLB rank 0"):
            runtime.read_placement_collectively(
                str(Path(tempfile.mkdtemp()) / "deployment.json"),
                make_metadata(),
                self.group,
            )

    def _make_install_state(self, rank, *, tensor_lists=False):
        old = torch.tensor([[0, 1, 2, 3, 0, 1]])
        global_weights = torch.arange(4, dtype=torch.float32).reshape(4, 1) + 10
        local = old[0, rank * 3 : (rank + 1) * 3]
        weights = [global_weights[local].clone(), (global_weights[local] * 2).clone()]
        buffers = [torch.zeros_like(weight) for weight in weights]
        if tensor_lists:
            weights = [list(weight.unbind()) for weight in weights]
            buffers = [list(buffer.unbind()) for buffer in buffers]

        class Communicator:
            def set_stream(self, stream):
                pass

            def set_transfer_context(self, mapping, layer_idx):
                pass

            def add_send(self, tensors, peer, expert):
                pass

            def add_recv(self, tensors, peer, expert):
                # Emulate the transport boundary using real peer expert tensors.
                for index, tensor in enumerate(tensors):
                    tensor.copy_(global_weights[expert] * (index + 1))

            def execute(self):
                pass

        model_state = SimpleNamespace(
            model=SimpleNamespace(num_logical_experts=4, expert_weights=[weights]),
            physical_to_logical_map=old.clone(),
            logical_to_physical_map=torch.full((1, 4, 3), -1, dtype=torch.long),
            logical_replica_count=torch.zeros((1, 4), dtype=torch.long),
            communicator=Communicator(),
            expert_buffer=buffers,
        )
        state = SimpleNamespace(
            model_states={"target": model_state},
            async_worker=None,
            prepare_movement_resources=MagicMock(),
            release_movement_resources=MagicMock(),
            placement_device_group=lambda: SimpleNamespace(size=lambda: 2, rank=lambda: rank),
            refresh_placement_layer=MagicMock(),
        )
        return state, model_state, global_weights

    def test_install_local_swap_remote_exchange_and_tensor_lists(self):
        target = (1, 0, 3, 2, 0, 1)
        placement = EplbPlacement(make_metadata(), (target,), {})
        for rank in range(2):
            for tensor_lists in (False, True):
                with self.subTest(rank=rank, tensor_lists=tensor_lists):
                    self.group.rank = lambda rank=rank: rank
                    state, model_state, global_weights = self._make_install_state(rank, tensor_lists=tensor_lists)
                    expected = global_weights[list(target[rank * 3 : (rank + 1) * 3])]

                    def refresh(committed, layer, expected=expected):
                        actual = torch.stack(list(committed.model.expert_weights[layer][0]))
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(committed.physical_to_logical_map.tolist(), [list(target)])

                    state.refresh_placement_layer.side_effect = refresh
                    runtime.install_placement(state, placement, self.group, MagicMock())
                    state.refresh_placement_layer.assert_called_once_with(model_state, 0)
                    self.assertEqual(model_state.logical_replica_count.tolist(), [[2, 2, 1, 1]])
                    for expert, count in enumerate(model_state.logical_replica_count[0].tolist()):
                        slots = model_state.logical_to_physical_map[0, expert, :count].tolist()
                        self.assertEqual(slots, [slot for slot, logical in enumerate(target) if logical == expert])
                    state.prepare_movement_resources.assert_called_once_with()
                    state.release_movement_resources.assert_called_once_with()

    def test_unchanged_install_does_not_initialize_transport(self):
        state, _, _ = self._make_install_state(0)
        unchanged = EplbPlacement(make_metadata(), ((0, 1, 2, 3, 0, 1),), {})
        runtime.install_placement(state, unchanged, self.group, MagicMock())
        state.prepare_movement_resources.assert_not_called()
        state.refresh_placement_layer.assert_not_called()

    def test_install_rejects_running_background_worker(self):
        state, _, _ = self._make_install_state(0)
        state.async_worker = object()
        with self.assertRaisesRegex(ValueError, "precede the background worker"):
            runtime.install_placement(state, self.placement, self.group, MagicMock())
        state.prepare_movement_resources.assert_not_called()

    def test_install_rejects_missing_draft_model_state(self):
        state, _, _ = self._make_install_state(0)
        draft_metadata = make_metadata()
        draft_metadata.update(
            model="draft-model",
            num_logical_experts=2,
            num_redundant_experts=0,
            layer_names=["layers.0.draft.moe"],
        )
        placement = EplbPlacement(
            self.placement.metadata,
            self.placement.physical_to_logical,
            self.placement.provenance,
            draft=EplbPlacement(draft_metadata, ((0, 1),), {}),
        )
        with self.assertRaisesRegex(ValueError, "missing its draft model state"):
            runtime.install_placement(
                state,
                placement,
                self.group,
                MagicMock(),
                model_states={"target": state.model_states["target"]},
            )
