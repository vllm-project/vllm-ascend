# SPDX-License-Identifier: Apache-2.0
"""One verification step's logical GDN metadata, shared across cache groups."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.v2.spec_decode.tree.state_rollback import (
    TreeVerifyPlan,
    compute_tree_conv_rows,
    compute_tree_prefix_source,
)

if TYPE_CHECKING:
    from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_pages import GDNTreePageBatch


@dataclass(frozen=True)
class SharedGDNTreeMetadata:
    """Logical metadata for one complete BS1 tree, without physical state pages.

    A model state's prepare_attn owns this object for one step. Builders bind
    their own page tables, then retain their existing graph-stable copy targets.
    """

    slots: int
    conv_width: int
    init_state_columns: torch.Tensor
    conv_window_rows: torch.Tensor
    conv_prefix_src: torch.Tensor
    request_id: str | None = None
    page_batch: "GDNTreePageBatch | None" = field(default=None, repr=False, compare=False)
    _written_graph_inputs: set[int] = field(default_factory=set, init=False, repr=False, compare=False)

    @classmethod
    def create(
        cls,
        parents: torch.Tensor,
        num_nodes: torch.Tensor,
        depths: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        prev_path_node_ids: torch.Tensor,
        prev_num_sampled: torch.Tensor,
        conv_width: int,
        *,
        request_id: str | None = None,
        use_fused_topology: bool = False,
        page_batch: "GDNTreePageBatch | None" = None,
    ) -> "SharedGDNTreeMetadata":
        if parents.ndim != 2 or parents.shape[0] != 1 or parents.shape[1] < 1:
            raise ValueError("Shared GDN tree metadata requires BS1 rootless parents [1, budget].")
        if depths.shape != parents.shape:
            raise ValueError("TreeLayout depths must have the same rootless [1, budget] shape as parents.")
        if any(tensor.shape != (1,) for tensor in (num_nodes, num_accepted_tokens, prev_num_sampled)):
            raise ValueError("Shared GDN tree counts must have shape [1].")
        if prev_path_node_ids.ndim != 2 or prev_path_node_ids.shape[0] != 1:
            raise ValueError("The previous accepted path must have shape [1, path_capacity].")
        if conv_width < 1:
            raise ValueError("The GDN convolution width must be positive.")

        slots = parents.shape[1] + 1
        device = parents.device
        if (
            use_fused_topology
            and device.type == "npu"
            and slots <= 64
            and 2 <= conv_width <= 5
            and prev_path_node_ids.shape[1] >= slots - 1
            and all(
                t.device == device and t.dtype in (torch.int32, torch.int64)
                for t in (parents, num_nodes, depths, num_accepted_tokens, prev_path_node_ids, prev_num_sampled)
            )
        ):
            from vllm_ascend.ops.triton.spec_decode.tree.gdn_tree_topology import build_gdn_tree_state_and_conv_indices

            init_state_columns, conv_window_rows, conv_prefix_source_rows = build_gdn_tree_state_and_conv_indices(
                parents, num_nodes, depths, num_accepted_tokens, prev_path_node_ids, prev_num_sampled, conv_width
            )
            return cls(
                slots,
                conv_width,
                init_state_columns,
                conv_window_rows,
                conv_prefix_source_rows,
                request_id=request_id,
                page_batch=page_batch,
            )
        chain_parents = torch.arange(slots - 1, device=device, dtype=torch.long)
        parent_columns = torch.where(num_nodes[:, None] > 0, parents.to(torch.long), chain_parents[None, :])
        root_column = (num_accepted_tokens.to(torch.long) - 1).clamp(min=0).unsqueeze(1)
        init_state_columns = torch.cat((root_column, parent_columns), dim=1)
        return cls(
            slots=slots,
            page_batch=page_batch,
            conv_width=conv_width,
            request_id=request_id,
            init_state_columns=init_state_columns,
            conv_window_rows=compute_tree_conv_rows(
                parents,
                num_nodes,
                conv_width,
                conv_width - 1,
                tree_depths=depths,
            ),
            conv_prefix_src=compute_tree_prefix_source(prev_path_node_ids, prev_num_sampled, conv_width),
        )

    def _check_state_table(self, state_indices: torch.Tensor) -> None:
        if state_indices.shape != (1, self.slots):
            raise ValueError(
                f"Expected a complete BS1 state table [1, {self.slots}], got {tuple(state_indices.shape)}."
            )
        if state_indices.device != self.init_state_columns.device:
            raise ValueError("Shared GDN topology and group state pages must be on the same device.")

    def bind_init_state_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Resolve shared logical parents against this group's physical pages."""
        self._check_state_table(state_indices)
        return state_indices.gather(1, self.init_state_columns).to(torch.int32)

    @cached_property
    def _verify_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.init_state_columns.device
        token_indices = torch.arange(self.slots, dtype=torch.long, device=device)
        actual_seq_lengths = torch.zeros(2, dtype=torch.int32, device=device)
        actual_seq_lengths[1:].fill_(self.slots)
        return token_indices, actual_seq_lengths

    def bind_plan(
        self,
        state_indices: torch.Tensor,
        init_state_indices: torch.Tensor,
    ) -> TreeVerifyPlan:
        """Bind a complete row in token order to this group's physical pages."""
        self._check_state_table(state_indices)
        self._check_state_table(init_state_indices)
        token_indices, actual_seq_lengths = self._verify_geometry
        return TreeVerifyPlan(
            rows=1,
            slots=self.slots,
            token_indices=token_indices,
            state_indices=state_indices[0],
            init_state_indices=init_state_indices[0],
            out_indices=token_indices,
            actual_seq_lengths=actual_seq_lengths,
        )
