# SPDX-License-Identifier: Apache-2.0
"""Accepted query transport graphs for the Qwen GDN tree target."""

from dataclasses import dataclass

import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats


class TreeQueryGraph:
    """Accepted hidden/aux rows and logical positions for the next draft."""

    def __init__(self, tensors, qsl, path_node_ids, active, positions, *, pool=None, stream=None, stats=None):
        from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_accepted_query_rows

        self.qsl = qsl.clone()
        self.path_node_ids = torch.full_like(path_node_ids, -1)
        self.active = torch.zeros_like(active)
        self.stats = stats if stats is not None else TreeGraphStats()

        def run():
            compact_accepted_query_rows(tensors, self.qsl, self.path_node_ids, self.active, positions)

        # Disabled request rows make warmup/capture non-mutating.
        with torch.inference_mode():
            run()
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, pool=pool, stream=stream):
                run()
        self.stats.record_capture("query")
        self.stats.set_workspace_bytes(
            "query",
            id(self),
            self.stats.tensor_bytes([self.qsl, self.path_node_ids, self.active]),
        )

    def replay(self, qsl, path_node_ids, active):
        self.qsl.copy_(qsl)
        self.path_node_ids.copy_(path_node_ids)
        self.active.copy_(active)
        self.graph.replay()
        self.stats.record_replay("query")


def _tensor_shape(tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _tensor_view(tensor):
    return tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device


@dataclass
class _QueryGraphEntry:
    graph: TreeQueryGraph
    # The recorded graph refers to these actual target outputs and positions.
    # Retain their storage even after the caller replaces its Python views.
    tensors: tuple[torch.Tensor, ...]
    positions: torch.Tensor


class QwenTreeQueryGraphs:
    """Reuse one private query graph for each stable target output layout."""

    def __init__(self, runner, *, stats=None):
        self.runner = runner
        self.stats = stats if stats is not None else TreeGraphStats()
        self.query_len = runner.speculator.tree_shape.target_verification_width
        self.graphs = {}

    def eligible(self, input_batch):
        # These fields are scheduler-owned CPU metadata, never device values.
        if input_batch is None or input_batch.num_reqs != 1 or input_batch.has_prefill:
            return False
        if input_batch.is_prefilling_np[:1].any():
            return False
        if input_batch.num_tokens != self.query_len:
            return False
        if not (input_batch.num_scheduled_tokens[:1] == self.query_len).all():
            return False
        qsl = input_batch.query_start_loc_np
        if qsl.shape != (2,) or qsl[1] - qsl[0] != self.query_len:
            return False
        # Configuring FULL is insufficient: its target output addresses must
        # belong to a captured FULL gear, rather than fresh eager allocations.
        manager = self.runner.cudagraph_manager
        return (
            manager is not None
            and manager._graphs_captured
            and any(
                desc.cg_mode == CUDAGraphMode.FULL
                and desc.num_reqs == 1
                and desc.num_tokens == self.query_len
                and desc.uniform_token_count == self.query_len
                for desc in manager.graphs
            )
        )

    def compact_queries(self, tensors, qsl, path_node_ids, active, positions):
        key = (
            tuple(_tensor_view(tensor) for tensor in tensors),
            _tensor_view(positions),
            _tensor_shape(qsl),
            _tensor_shape(path_node_ids),
            _tensor_shape(active),
        )
        entry = self.graphs.get(key)
        self.stats.record_lookup("query", entry is not None)
        if entry is None:
            # Inactive private rows protect the live target outputs during
            # warmup and capture. The first replay compacts the live rows.
            graph = TreeQueryGraph(tensors, qsl, path_node_ids, active, positions, stats=self.stats)
            entry = _QueryGraphEntry(graph, tuple(tensors), positions)
            self.graphs[key] = entry
        entry.graph.replay(qsl, path_node_ids, active)
