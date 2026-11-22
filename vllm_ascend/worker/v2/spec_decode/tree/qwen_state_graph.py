# SPDX-License-Identifier: Apache-2.0
"""Fixed-input graphs for committing the Qwen GDN tree's accepted state cursor."""

from dataclasses import dataclass

import torch

from vllm_ascend.worker.v2.spec_decode.tree.metrics import TreeGraphStats


def _signature(tensor):
    return None if tensor is None else (tuple(tensor.shape), tensor.dtype, tensor.device)


@dataclass
class _StateGraph:
    graph: object
    inputs: tuple[torch.Tensor | None, ...]
    outputs: tuple[torch.Tensor, ...]
    state_context: object | None


class QwenTreeStateGraphs:
    """Keep request IDs and accepted tree paths live behind stable graph inputs.

    Inactive request IDs make warmup and capture no-ops on the live state.
    The existing model-state postprocessor still owns the recurrent cursor and
    any block-aligned state transport; this class only records its operations.
    """

    def __init__(self, model_state, *, stats=None):
        self.model_state = model_state
        self.stats = stats if stats is not None else TreeGraphStats()
        self.graphs: dict[tuple, _StateGraph] = {}

    def run(self, idx_mapping, resume_state_cursor, path, num_sampled, tree_rows, num_computed_tokens):
        state = self.model_state
        # Recovery-by-recomputation has its own Python step lifecycle. Tree
        # state copying records the regular page-based postprocessor instead.
        if idx_mapping.device.type != "npu" or idx_mapping.shape != (1,) or state.recoverssm is not None:
            return False
        state._tree_buffers(idx_mapping.device)
        inputs = (idx_mapping, resume_state_cursor, path, num_sampled, tree_rows, num_computed_tokens)
        key = (
            tuple(_signature(tensor) for tensor in inputs),
            state.num_accepted_tokens_gpu.data_ptr(),
            state._tree_prev_path.data_ptr(),
            state._tree_prev_num_sampled.data_ptr(),
            id(state._mamba_ctx) if state._align_mode else None,
        )
        entry = self.graphs.get(key)
        self.stats.record_lookup("post", entry is not None)
        if entry is None:
            private = tuple(None if tensor is None else torch.zeros_like(tensor) for tensor in inputs)
            private[0].fill_(-1)
            graph = torch.npu.NPUGraph()
            with torch.inference_mode():
                state._postprocess_tree_state_tensors(*private)
                with torch.npu.graph(graph):
                    state._postprocess_tree_state_tensors(*private)
            entry = _StateGraph(
                graph,
                private,
                (state.num_accepted_tokens_gpu, state._tree_prev_path, state._tree_prev_num_sampled),
                state._mamba_ctx if state._align_mode else None,
            )
            self.graphs[key] = entry
            self.stats.record_capture("post")
            self.stats.set_workspace_bytes(
                "post",
                id(entry),
                self.stats.tensor_bytes([tensor for tensor in private if tensor is not None] + list(entry.outputs)),
            )
        for destination, source in zip(entry.inputs, inputs):
            if destination is not None:
                destination.copy_(source)
        entry.graph.replay()
        self.stats.record_replay("post")
        return True
