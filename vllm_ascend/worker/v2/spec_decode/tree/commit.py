"""Accepted-path commitment and query transport at the model boundary."""

from typing import TYPE_CHECKING

from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.spec_decode.tree.acceptance import TreeAcceptance
from vllm_ascend.worker.v2.spec_decode.tree.kv_layout import compact_accepted_query_rows

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_states.tree_mamba_hybrid import AscendTreeMambaHybridModelState


class TreeCommitCoordinator:
    """Commit accepted token caches and stage recurrent state on one stream."""

    def __init__(self, runner, model_state: "AscendTreeMambaHybridModelState"):
        from vllm_ascend.worker.v2.spec_decode.tree.qwen_query_graph import QwenTreeQueryGraphs

        self.compactor = runner.tree_kv_compact
        self.model_state = model_state
        self.qwen_query_graphs = None
        model_state.tree_graph_stats = runner.speculator.tree_graph_stats
        if runner.compilation_config.cudagraph_mode in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY):
            self.qwen_query_graphs = QwenTreeQueryGraphs(runner, stats=model_state.tree_graph_stats)

    def commit(self, idx_mapping, acceptance: TreeAcceptance, *, num_rejected) -> None:
        self.model_state.stage_tree_acceptance(acceptance, num_rejected)
        self.compactor.run(idx_mapping, acceptance)

    def compact_queries(self, tensors, query_start_loc, path_node_ids, active, positions, *, input_batch=None) -> None:
        """Prepare accepted hidden/aux rows and positions for the next draft."""
        if self.qwen_query_graphs is not None and self.qwen_query_graphs.eligible(input_batch):
            self.qwen_query_graphs.compact_queries(tensors, query_start_loc, path_node_ids, active, positions)
        else:
            compact_accepted_query_rows(tensors, query_start_loc, path_node_ids, active, positions)
