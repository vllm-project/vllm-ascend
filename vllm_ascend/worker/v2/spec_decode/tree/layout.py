from dataclasses import dataclass

import torch


@dataclass
class TreeLayout:
    """Flattened draft tree for a batch of parallel-draft logits.

    Naming:
        node_id: vertex id including root (root=0, draft nodes 1..N).
        slot: non-root layout index ``0..budget-1`` for node ``slot+1``.
        parent_id: parent ``node_id`` of a non-root node.
        num_nodes: number of valid non-root nodes.
        spec_num: draft depth dimension of ``draft_logits``.

    Slot-indexed (non-root): tokens, depths, parents, visibility, num_nodes.
    Node-id-indexed (includes root 0): values in parents, first_child,
    and next_sibling.

    Attributes:
        tokens: [R, budget] token id of non-root node i+1.
        depths: [R, budget] depth of non-root node i+1 (from 1).
        parents: [R, budget] parent node id of non-root node i+1.
        num_nodes: [R] number of valid non-root nodes.
        visibility: [R, budget, budget] whether non-root node i+1 can see
            non-root node j+1 (seeing root is implicit).
        first_child: [R, budget+1] first child node id of node p; -1 if none.
        next_sibling: [R, budget+1] next sibling node id of node c; -1 if none.
    """

    tokens: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor
    num_nodes: torch.Tensor
    visibility: torch.Tensor
    first_child: torch.Tensor
    next_sibling: torch.Tensor

    def verifiable_draft_node_counts(self, query_start_loc: torch.Tensor, cu_num_logits: torch.Tensor) -> torch.Tensor:
        """Bound each tree by its actual target query and logit rows on device.

        Root occupies row zero in both intervals. A truncated verification
        therefore exposes only a topological prefix of the stored draft tree.
        The links remain unchanged; traversal skips children outside this bound.
        """
        num_reqs = self.tokens.shape[0]
        query_counts = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        logit_counts = cu_num_logits[1 : num_reqs + 1] - cu_num_logits[:num_reqs]
        counts = torch.minimum(self.num_nodes, torch.minimum(query_counts, logit_counts) - 1)
        return counts.clamp(min=0, max=self.tokens.shape[1]).to(torch.int32)
