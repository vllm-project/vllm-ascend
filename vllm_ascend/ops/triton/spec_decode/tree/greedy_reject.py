from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_reqs", "budget", "max_draft_depth"])
def _verify_tree_greedy_kernel(
    tokens_ptr,
    first_child_ptr,
    next_sibling_ptr,
    target_ids_ptr,
    logits_cu_ptr,
    num_verifiable_draft_nodes_ptr,
    sampled_ptr,
    path_ptr,
    num_sampled_ptr,
    endpoint_ptr,
    active_ptr,
    num_reqs,
    budget,
    max_draft_depth,
    stride_tokens_r,
    stride_fc_r,
    stride_ns_r,
    stride_sampled_r,
    stride_path_r,
    BLOCK_OUT: tl.constexpr,
):
    # One request per program. Dynamic loops terminate as soon as a branch
    # rejects or its sibling list ends; they do not unroll budget * depth.
    req = tl.program_id(0)
    if req >= num_reqs:
        return
    columns = tl.arange(0, BLOCK_OUT)
    tl.store(sampled_ptr + req * stride_sampled_r + columns, -1, columns <= max_draft_depth)
    tl.store(path_ptr + req * stride_path_r + columns, -1, columns < max_draft_depth)
    start = tl.load(logits_cu_ptr + req).to(tl.int64)
    verified = tl.load(num_verifiable_draft_nodes_ptr + req)
    alive = tl.load(active_ptr + req) != 0
    current = tl.full((), 0, tl.int64)
    accepted = tl.full((), 0, tl.int32)
    sampled = tl.full((), 0, tl.int32)
    while alive & (accepted < max_draft_depth):
        target = tl.load(target_ids_ptr + start + current)
        tl.store(sampled_ptr + req * stride_sampled_r + accepted, target)
        sampled += 1
        child = tl.load(first_child_ptr + req * stride_fc_r + current).to(tl.int64)
        found = tl.full((), -1, tl.int64)
        visited = tl.full((), 0, tl.int32)
        while (child > 0) & (child <= budget) & (found < 0) & (visited < budget):
            eligible = child <= verified
            token = tl.load(tokens_ptr + req * stride_tokens_r + child - 1, mask=eligible, other=-1)
            found = tl.where(eligible & (token == target), child, found)
            child = tl.load(next_sibling_ptr + req * stride_ns_r + child).to(tl.int64)
            visited += 1
        alive = found > 0
        if alive:
            tl.store(path_ptr + req * stride_path_r + accepted, found)
            current = found
            accepted += 1
    if alive:
        target = tl.load(target_ids_ptr + start + current)
        tl.store(sampled_ptr + req * stride_sampled_r + max_draft_depth, target)
        sampled += 1
    tl.store(num_sampled_ptr + req, sampled)
    tl.store(endpoint_ptr + req, current)


def verify_tree_greedy(
    tokens,
    first_child,
    next_sibling,
    target_token_ids,
    cu_num_logits,
    num_verifiable_draft_nodes,
    sampled_token_ids,
    accepted_path_node_ids,
    num_sampled,
    last_accepted_node_id,
    active,
) -> None:
    num_reqs, budget = tokens.shape
    max_draft_depth = accepted_path_node_ids.shape[1]
    _verify_tree_greedy_kernel[(max(num_reqs, 1),)](
        tokens,
        first_child,
        next_sibling,
        target_token_ids,
        cu_num_logits,
        num_verifiable_draft_nodes,
        sampled_token_ids,
        accepted_path_node_ids,
        num_sampled,
        last_accepted_node_id,
        active,
        num_reqs,
        budget,
        max_draft_depth,
        tokens.stride(0),
        first_child.stride(0),
        next_sibling.stride(0),
        sampled_token_ids.stride(0),
        accepted_path_node_ids.stride(0),
        triton.next_power_of_2(max_draft_depth + 1),
    )
