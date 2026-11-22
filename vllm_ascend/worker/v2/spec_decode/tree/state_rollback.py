"""GDN state rollback for draft-tree speculative decoding.

``tree_spec_config`` verifies a draft *tree*: the target model runs
``1 + budget`` tokens per request (the already-accepted root plus ``budget``
non-root nodes) and acceptance walks one root-to-leaf path through them. GDN
caches one recurrent state per page, so the state layout has to follow the tree
instead of the linear MTP chain:

* state slot ``j`` (0-based) holds the recurrent state *after* tree node ``j``;
  slot 0 is the root, whose state is the committed prefix state, and slot
  ordering matches node id ordering (every builder guarantees
  ``parent_id < node_id``);
* node ``j`` starts from its parent's slot, so dropping the rejected siblings is
  a pure index change -- no state copy and no recomputation;
* after sampling, the next step resumes from the verifier's accepted endpoint.
  The model state converts this zero-based node ID into a one-based state cursor.

The number of slots (``1 + budget``) is what
``vllm_ascend.worker.v2.attn_utils.get_kv_cache_spec`` widens ``MambaSpec`` to and
what ``AscendGDNAttentionMetadataBuilder`` sizes its spec buffers by. It is
larger than the chain width (``1 + num_speculative_tokens``), hence the state
memory and per-step state traffic grow with the tree width.

Only the Qwen GDN path (``MambaAttentionBackendEnum.GDN_ATTN``) is wired up so
far; :func:`validate_tree_state_slots` rejects a draft tree on any other state
backend (KDA, Mamba1/2, short-conv, pure linear attention) instead of silently
serving stale states.

The kernel side is tree-agnostic: :func:`compute_tree_init_state_indices` turns
the ``[R, slots]`` page table plus the tree's parent ids into a per-token table
of *page ids to load the initial state from*, which the recurrent op consumes
through its ``init_state_indices`` input. Node ``t`` therefore loads
its parent's page and writes its own, both indexed by the same slot table.
"""

from dataclasses import dataclass

import torch

# Keep the existing graph layouts: direct verification for small trees and
# fixed-width indexed verification for wider trees.
GDN_TREE_DIRECT_VERIFY_MAX_SLOTS = 16


def tree_state_slot_count(vllm_config) -> int | None:
    """Number of GDN state slots needed per request.

    Returns ``None`` when no draft tree is active, i.e. the caller must keep the
    linear-chain layout. Tree verification stores one recurrent snapshot per
    node, so its page table has one root slot plus the non-root node budget.
    """
    from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

    shape = TreeShape.from_config(vllm_config)
    if shape is None:
        return None
    if getattr(getattr(vllm_config, "cache_config", None), "mamba_cache_mode", None) == "align":
        # The align pre-copy kernel migrates the running state page and shifts
        # its short-conv window by ``num_accepted_tokens - 1``. A draft tree
        # cannot express that: the cursor is a node id and a node's window is
        # its own root-to-node chain instead of the previous batch entry, so the
        # shifted window would silently corrupt the conv history. The recurrent
        # state does not care: every node keeps its own page.
        raise NotImplementedError(
            "tree_spec_config is not supported with mamba_cache_mode='align' "
            "yet; the align pre-copy shift of the short-conv state has no "
            "draft-tree equivalent. Disable mamba_cache_mode='align' or "
            "omit tree_spec_config."
        )
    return shape.target_verification_width


def validate_tree_state_slots(slots: int, mamba_type) -> None:
    """Reject draft trees that the current state path cannot serve yet.

    Triton verification accepts the whole tree in one call; the slot count is
    bounded by allocated state memory rather than a kernel token limit.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if slots < 1:
        raise ValueError(f"a draft tree verifies at least one token per request, got slots={slots}.")
    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        raise NotImplementedError(
            f"tree_spec_config is enabled but mamba_type={mamba_type} does not "
            "support draft-tree state rollback yet; only GDN_ATTN (Qwen GDN) is "
            "wired up. Disable tree_spec_config for this model."
        )


def tree_conv_width(
    mamba_type,
    conv_state_shape: tuple[int, ...],
    num_spec: int,
) -> int | None:
    """Short-conv kernel width behind a GDN conv-state shape.

    The conv state is ``(width - 1 + num_spec) x conv_dim`` and the conv
    dimension (``2 * key_dim + value_dim``) is orders of magnitude larger than
    the state length, so the smaller side of the 2-D state shape is the length
    to undo the speculation width on. Returns ``None`` for layers without a
    short conv (Mamba2, KDA) or an unexpected shape.
    """
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if mamba_type is not MambaAttentionBackendEnum.GDN_ATTN:
        return None
    if len(conv_state_shape) != 2:
        return None
    width = min(conv_state_shape) - num_spec + 1
    if width < 2:
        raise ValueError(
            f"cannot recover the short-conv width from the state shape "
            f"{tuple(conv_state_shape)} with num_speculative_tokens={num_spec}."
        )
    return width


def tree_row_mask(
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
) -> torch.Tensor:
    """Rows of the batch whose step ran the tree op, in batch order, as a bool.

    ``num_sampled`` alone cannot tell the two rows apart that both report ``1``
    with an all ``-1`` path: a tree step that accepted the root only, and a row
    the scheduler gave no draft token (which the linear operator commits). They
    differ in how many tokens were verified: the tree op handles every row that
    checked more than its own token, while a single-token row keeps the
    committed short-conv history the linear operator left in place. The samplers
    set ``num_rejected = num_logits - num_sampled``
    (see ``spec_decode/tree/rejection_sampler.py``), so the sum is exactly the
    number of verified tokens and needs no extra tensor from the sampler.
    """
    return (num_sampled + num_rejected) > 1


def compute_tree_init_state_indices(
    state_indices: torch.Tensor,
    resume_state_cursor: torch.Tensor,
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
) -> torch.Tensor:
    """Per-token state page the recurrent kernel must load its initial state from.

    Args:
        state_indices: ``[R, slots]`` state pages written by this step, column
            ``j`` holding tree node ``j``'s state (``j == 0`` is the root, whose
            page carries the committed prefix state).
        resume_state_cursor: ``[R]`` resume cursor of the *previous* step, i.e.
            the accepted node ID plus one. ``slot = resume_state_cursor - 1``
            selects the page holding the state the next token continues from.
        tree_parents: ``[R, budget]`` parent node id of node ``slot + 1``
            (``0`` = root), straight from the tree layout.
        tree_num_nodes: ``[R]`` number of non-root nodes; ``0`` marks a row that
            was folded in from a prompt chunk rather than built as a tree.

    Returns:
        ``[R, slots]`` int32 page ids, in the same request/column order as
        ``state_indices``. The caller packs them token-major, because a row may
        verify fewer than ``slots`` tokens: see
        ``gdn_attn_builder._pack_tree_rows_to_tokens``.

    A row folded in from a prompt chunk is a linear sequence whose token ``t``
    follows token ``t - 1``: pointing the table at that token's own page makes
    the kernel's "parent page == previous token page" test hit and keeps its
    in-register carry, i.e. the exact linear-chain behaviour.
    """
    rows, slots = state_indices.shape
    if slots - 1 > tree_parents.shape[1]:
        raise ValueError(
            f"state table has {slots} slots but the tree layout only carries "
            f"{tree_parents.shape[1]} parents; tree_spec_config.budget must "
            "match the state slot count."
        )
    parents = tree_parents[:rows, : slots - 1].to(torch.long)
    # Chain rows: node ``t`` (1-based) continues the token before it.
    chain_parents = torch.arange(1, slots, device=state_indices.device, dtype=torch.long) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents[None, :])
    resume_state_cursor = resume_state_cursor[:rows].to(torch.long)
    accepted_slot = (resume_state_cursor - 1).clamp(min=0).unsqueeze(1)
    root_init = torch.gather(state_indices, 1, accepted_slot)
    node_init = torch.gather(state_indices, 1, parents)
    init_state_indices = torch.cat((root_init, node_init), dim=1)
    return init_state_indices.to(torch.int32)


def tree_node_parents_depths(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node parent ids and depths of the draft tree.

    Args:
        tree_parents: ``[R, budget]`` parent node id of node ``t + 1``, straight
            from the tree layout (see :func:`compute_tree_init_state_indices`).
        tree_num_nodes: ``[R]`` number of non-root nodes; ``0`` marks a row that
            was folded in from a prompt chunk and is a linear sequence.
        slots: number of nodes per request, i.e. ``1 + budget``.

    Returns:
        ``(parents, depths)``, both ``[R, slots]`` int64; the root's parent is
        ``-1``. Folded rows are chained (node ``t`` follows ``t - 1``) so they
        keep the batch-order window of the linear MTP chain.
    """
    rows = tree_parents.shape[0]
    device = tree_parents.device
    parents = torch.full((rows, slots), -1, dtype=torch.long, device=device)
    parents[:, 1:] = tree_parents[:rows, : slots - 1].to(torch.long)
    chain_parents = torch.arange(slots, dtype=torch.long, device=device) - 1
    parents = torch.where(tree_num_nodes[:rows, None] > 0, parents, chain_parents.expand(rows, slots))
    depths = torch.zeros((rows, slots), dtype=torch.long, device=device)
    for node in range(1, slots):
        # Every builder guarantees ``parent_id < node_id``, so a single forward
        # sweep resolves the whole depth table.
        parent_of_node = parents[:, node].clamp(min=0).unsqueeze(1)
        depths[:, node] = depths.gather(1, parent_of_node).squeeze(1) + 1
    return parents, depths


def _tree_conv_tap_rows(
    chain: torch.Tensor,
    depths: torch.Tensor,
    own_rows: torch.Tensor,
    request_ids: torch.Tensor,
    node_row_table: torch.Tensor,
    history_base: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Rows of the ``width`` conv taps of every target, oldest tap first.

    Args:
        chain: ``[N, width - 1]`` walk-up ancestors, column ``k`` being the
            ``k + 1``-th ancestor (parent first).
        depths: ``[N]`` depth of every target.
        own_rows: ``[N]`` row of the target's own activation.
        request_ids: ``[N]`` request each target belongs to.
        node_row_table: ``[R, slots]`` row of every node's activation.
        history_base: ``[R]`` row of the oldest committed activation.

    The rows index ``cat([committed, activations])``: the committed
    activations of the previous steps first (oldest first), this step's
    activations after.  A tap that falls before the root resolves to the
    committed history, which keeps the window of shallow nodes identical to the
    linear-chain window.
    """
    taps = width - 1
    tap_ids = torch.arange(taps, device=chain.device).unsqueeze(0)  # [1, K]
    # Result column j is tap j (oldest first), which walks the (width - 1 - j)
    # -th ancestor, i.e. chain column width - 2 - j: the walk-up table is read
    # with its columns reversed (column 0 of the reversed table is the deepest
    # ancestor). Clamped in range where the ancestor is missing (those entries
    # are discarded by the depth mask below).
    ancestor = node_row_table[
        request_ids.unsqueeze(1).expand(-1, taps),
        chain.flip(dims=[1]).clamp(min=0),
    ]
    # committed[n, j] = the committed-history row tap j lands on when the
    # chain of n runs past the root.
    committed = history_base[request_ids].unsqueeze(1) + (depths.unsqueeze(1) + tap_ids).clamp(max=width - 2)
    owned = depths.unsqueeze(1) >= width - 1 - tap_ids
    window = torch.where(owned, ancestor, committed)
    return torch.cat((window, own_rows.unsqueeze(1)), dim=1)


def compute_tree_conv_rows(
    tree_parents: torch.Tensor,
    tree_num_nodes: torch.Tensor,
    width: int,
    node_row_offset: int,
    row_widths: torch.Tensor | None = None,
    *,
    tree_depths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token conv window of every node of a draft tree.

    Args:
        tree_parents: ``[R, budget]`` tree parents, as above.
        tree_num_nodes: ``[R]`` non-root node count, as above.
        width: short-conv kernel width; the window holds ``width`` taps.
        node_row_offset: first row of this step's activations inside the gather
            source; the ``R * (width - 1)`` committed rows precede it.
        row_widths: ``[R]`` tokens each request verifies, i.e. ``1 +`` its node
            count. ``None`` assumes the full ``1 + budget`` width. A row may
            verify fewer tokens -- the scheduler truncates a draft near
            ``max_model_len`` and the tree search can run out of candidates --
            and the layer reads the table by global token offset, so the rows
            are packed back to back.
        tree_depths: optional rootless ``[R, budget]`` depths from TreeLayout.
            Supplying them avoids reconstructing depths by walking every node.

    Returns:
        ``[R * slots, width]`` rows of the taps (oldest first) consumed by
        ``tree_causal_conv1d``, indexing
        ``cat([committed history, this step's activations])``. Only the first
        ``row_widths.sum()`` rows are meaningful; the layer slices the table by
        the token count of the spec batch, and for the full width every row is
        meaningful.

    Node ``t`` convolves over its own root-to-node chain: its parent, its
    grandparent, ... and, once the chain runs past the root, the committed
    activations of the previous steps.  Sibling and unrelated nodes never enter
    a node's window, which is what the linear-chain conv kernel gets wrong.
    """
    rows = tree_parents.shape[0]
    slots = tree_parents.shape[1] + 1
    device = tree_parents.device
    if row_widths is None:
        row_widths = torch.full((rows,), slots, dtype=torch.long, device=device)
    else:
        # Validated before the device transfer (a CPU tensor here), because a
        # device-side ``bool(...)`` would synchronize the stream.
        row_widths = row_widths.reshape(-1)
        if row_widths.numel() != rows:
            raise ValueError(f"row_widths must hold {rows} widths, got {row_widths.numel()}.")
        if bool(((row_widths < 1) | (row_widths > slots)).any()):
            raise ValueError(f"every draft-tree row verifies between 1 and {slots} tokens, got {row_widths.tolist()}.")
        row_widths = row_widths.to(device=device, dtype=torch.long)
    if tree_depths is None:
        parents, depths = tree_node_parents_depths(tree_parents, tree_num_nodes, slots)
    else:
        if tree_depths.shape != tree_parents.shape:
            raise ValueError("TreeLayout depths must match the rootless parent table shape.")
        parents = torch.full((rows, slots), -1, dtype=torch.long, device=device)
        parents[:, 1:] = tree_parents.to(torch.long)
        chain_parents = torch.arange(slots, dtype=torch.long, device=device) - 1
        is_tree = tree_num_nodes[:rows, None] > 0
        parents = torch.where(is_tree, parents, chain_parents.expand(rows, slots))
        depths = torch.cat(
            (torch.zeros((rows, 1), dtype=torch.long, device=device), tree_depths.to(device=device, dtype=torch.long)),
            dim=1,
        )
        depths = torch.where(is_tree, depths, (chain_parents + 1).expand(rows, slots))
    # Walk-up ancestors: column ``k`` is the ``k + 1``-th ancestor. ``clamp``
    # keeps the gather in range where an ancestor is missing; those entries are
    # discarded by the depth test in _tree_conv_tap_rows.
    chain = [parents]
    for _ in range(width - 2):
        chain.append(torch.gather(parents, 1, chain[-1].clamp(min=0)))
    chain = torch.stack(chain, dim=2)

    # Token-major packing: the ``p``-th token of the step is node
    # ``p - start(r)`` of row ``r``. The tables are built on the fixed
    # ``R * slots`` capacity -- so the result keeps its shape and the cost does
    # not depend on the widths -- and only the first ``row_widths.sum()`` rows
    # hold windows the layer reads.
    starts = torch.cumsum(row_widths, dim=0) - row_widths
    ends = starts + row_widths
    positions = torch.arange(rows * slots, device=device)
    token_rows = (positions.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1).clamp(max=rows - 1)
    token_cols = (positions - starts[token_rows]).clamp(max=slots - 1)
    node_row_table = node_row_offset + starts.unsqueeze(1) + torch.arange(slots, device=device).unsqueeze(0)

    window_rows = _tree_conv_tap_rows(
        chain=chain[token_rows, token_cols],
        depths=depths[token_rows, token_cols],
        own_rows=node_row_offset + positions,
        request_ids=token_rows,
        node_row_table=node_row_table,
        history_base=torch.arange(rows, device=device) * (width - 1),
        width=width,
    )
    return window_rows.to(torch.int32)


def compute_tree_prefix_source(
    path_node_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Committed-history columns the next step has to read its window from.

    The committed history is the ``width - 1`` activations *ending at* the token
    the next step's root continues from, i.e. the last node the sampler accepted.
    ``num_sampled`` counts the accepted
    tokens including the trailing resample, so that node sits at tree depth
    ``num_sampled - 1``: depth ``0`` is the root, once no draft node was accepted.
    ``path_node_ids[req, k]`` holds the accept-walk node of depth ``k + 1``, i.e.
    the last accepted node at index ``num_sampled - 2``.

    A tap of the window is therefore resolved by its tree depth ``d``:

    * ``d >= 1``: the accepted node of that depth, stored by this step in node
      column ``width - 1 + node id``;
    * ``d == 0``: the root, node column ``width - 1`` -- its activation is this
      step's first token, so it is *not* one of the committed columns;
    * ``d < 0``: the activation ``-d`` tokens before the root, i.e. committed
      column ``width - 1 + d``, which already holds it.

    A row the sampler left without a path (``num_sampled == 0``, a prompt chunk
    folded into the spec batch) maps the committed columns onto themselves: the
    chain op already left the history its root continues from.

    Args:
        path_node_ids: ``[R, path_capacity]`` accepted node IDs in depth order.
        num_sampled: ``[R]`` accepted token count including the resample.
        width: short-conv kernel width.

    Returns:
        ``[R, width - 1]`` int32 conv-state columns, oldest first.
    """
    rows = num_sampled.shape[0]
    device = num_sampled.device
    # Depth of the last accepted node; ``-1`` marks a row the sampler skipped, so
    # that its taps land on the committed columns themselves.
    last_depth = (num_sampled.to(torch.long) - 1).clamp(min=-1).unsqueeze(1)
    depth = last_depth - (width - 2) + torch.arange(width - 1, device=device).unsqueeze(0)
    # Only the node taps (``depth >= 1``) are looked up in the path: its entry
    # ``depth - 1`` is exactly the node of that depth. The lower entries are
    # masked out below, so the clamped index only has to stay in range.
    node_ids = path_node_ids[:rows].to(torch.long).gather(1, (depth - 1).clamp(min=0))
    return torch.where(depth >= 1, width - 1 + node_ids, width - 1 + depth).to(torch.int32)


# ---------------------------------------------------------------------------
# Indexed verification plans for draft trees
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeVerifyPlan:
    """One indexed recurrent verify of a fixed-width [rows, slots] token grid.

    The index tables are flat, but each row occupies ``slots`` cells. Active
    cells map to packed tokens and physical state pages. Padding cells carry
    invalid state page IDs and point past the real output into scratch space.
    ``actual_seq_lengths`` stores a leading zero followed by one token count per
    row, so the kernel skips padding and zero-length graph rows.
    """

    rows: int
    slots: int
    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    actual_seq_lengths: torch.Tensor

    def copy_into(self, destination: "TreeVerifyPlan") -> None:
        """Refresh the data that a full graph reads by address."""
        destination.token_indices.copy_(self.token_indices)
        # Uniform BS1 graphs may already use the same group's state-page buffers.
        if destination.state_indices.data_ptr() != self.state_indices.data_ptr():
            destination.state_indices.copy_(self.state_indices)
        if destination.init_state_indices.data_ptr() != self.init_state_indices.data_ptr():
            destination.init_state_indices.copy_(self.init_state_indices)
        destination.out_indices.copy_(self.out_indices)
        destination.actual_seq_lengths.copy_(self.actual_seq_lengths)


@dataclass
class TreeVerifyPlanBuffers:
    """Graph-owned buffers sized to the builder's maximum row capacity."""

    token_indices: torch.Tensor
    state_indices: torch.Tensor
    init_state_indices: torch.Tensor
    out_indices: torch.Tensor
    actual_seq_lengths: torch.Tensor

    def views(self, rows: int, slots: int) -> TreeVerifyPlan:
        cells = rows * slots
        return TreeVerifyPlan(
            rows=rows,
            slots=slots,
            token_indices=self.token_indices[:cells],
            state_indices=self.state_indices[:cells],
            init_state_indices=self.init_state_indices[:cells],
            out_indices=self.out_indices[:cells],
            actual_seq_lengths=self.actual_seq_lengths[: rows + 1],
        )

    def zero_lengths(self) -> None:
        """Skip all rows when replaying an inactive speculative branch."""
        self.actual_seq_lengths.zero_()


def allocate_tree_verify_plan_buffers(slots: int, rows: int, device: torch.device) -> TreeVerifyPlanBuffers:
    cells = rows * slots
    return TreeVerifyPlanBuffers(
        token_indices=torch.empty(cells, dtype=torch.long, device=device),
        state_indices=torch.empty(cells, dtype=torch.int32, device=device),
        init_state_indices=torch.empty(cells, dtype=torch.int32, device=device),
        out_indices=torch.empty(cells, dtype=torch.long, device=device),
        actual_seq_lengths=torch.empty(rows + 1, dtype=torch.int32, device=device),
    )


def compute_tree_verify_plan(
    row_widths: torch.Tensor,
    state_indices: torch.Tensor,
    init_state_indices: torch.Tensor,
    scratch_base: int,
) -> TreeVerifyPlan:
    """Map a ragged packed batch onto the indexed kernel's fixed-width grid.

    The builder validates each active row length on the host and supplies a
    width of zero for graph padding. Both state tables are in packed token
    order. The unused grid cells can point to token zero for input and unique
    scratch indices for output because the kernel never visits them.
    """
    if state_indices.ndim != 2 or init_state_indices.shape != state_indices.shape:
        raise ValueError(
            f"the state tables must be a pair of [rows, slots] tensors, got "
            f"{tuple(state_indices.shape)} and {tuple(init_state_indices.shape)}."
        )
    if scratch_base < 0:
        raise ValueError(f"the output scratch base must be non-negative, got {scratch_base}.")

    rows, slots = state_indices.shape
    device = state_indices.device
    widths = row_widths.to(device=device, dtype=torch.long).reshape(-1)
    if widths.numel() != rows:
        raise ValueError(f"got {widths.numel()} row widths for {rows} state rows.")

    token_starts = torch.cumsum(widths, dim=0) - widths
    columns = torch.arange(slots, device=device, dtype=torch.long)
    active = (columns < widths[:, None]).reshape(-1)
    packed_tokens = (token_starts[:, None] + columns).reshape(-1)
    token_indices = torch.where(active, packed_tokens, 0)
    scratch_indices = torch.arange(rows * slots, device=device, dtype=torch.long) + scratch_base
    actual_seq_lengths = torch.cat((torch.zeros(1, dtype=torch.int32, device=device), widths.to(torch.int32)))

    return TreeVerifyPlan(
        rows=rows,
        slots=slots,
        token_indices=token_indices,
        state_indices=torch.where(active, state_indices.reshape(-1).index_select(0, token_indices), -1),
        init_state_indices=torch.where(active, init_state_indices.reshape(-1).index_select(0, token_indices), -1),
        out_indices=torch.where(active, token_indices, scratch_indices),
        actual_seq_lengths=actual_seq_lengths,
    )
