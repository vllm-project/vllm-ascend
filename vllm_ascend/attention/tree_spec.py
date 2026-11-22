import torch
from vllm.config import VllmConfig

from vllm_ascend.ascend_config import get_tree_spec_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX


def align_up(value: int, alignment: int = 128) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def tree_spec_enabled(vllm_config: VllmConfig | None = None) -> bool:
    """Whether a valid ``additional_config.tree_spec_config`` was supplied."""
    return get_tree_spec_config(vllm_config) is not None


def tree_mask_query_capacity() -> int | None:
    """Storage/visibility capacity B+1, independent of scheduled target rows.

    Target capture and dispatch use TreeShape.target_verification_width. This helper
    sizes root-inclusive visibility storage, including for a truncated tree.
    """
    tree_config = get_tree_spec_config()
    return None if tree_config is None else tree_config.budget + 1


def tree_decode_threshold(vllm_config: VllmConfig | None = None) -> int:
    """Target decode capacity including the root token."""
    if vllm_config is not None:
        from vllm_ascend.worker.v2.spec_decode.tree.shape import TreeShape

        shape = TreeShape.from_config(vllm_config)
        if shape is not None:
            return shape.target_verification_width
        spec = vllm_config.speculative_config
        return 1 + int(spec.num_speculative_tokens) if spec is not None else 1
    query_len = tree_mask_query_capacity()
    return query_len if query_len is not None else 1


def dummy_tree_visibility(
    num_decode: int,
    device: torch.device,
    budget: int | None = None,
) -> torch.Tensor:
    """Identity visibility for FULL capture when the dummy batch has no tree."""
    if budget is None:
        query_len = tree_mask_query_capacity()
        budget = query_len - 1 if query_len is not None else 1
    vis = torch.eye(budget, dtype=torch.bool, device=device)
    return vis.unsqueeze(0).expand(num_decode, -1, -1).contiguous()


def tree_verify_shape(num_tokens: int) -> tuple[int, int] | None:
    """``(num_decode, 1+budget)`` when the batch is tree-verify shaped."""
    tree_q = tree_mask_query_capacity()
    if tree_q is None or tree_q <= 0 or num_tokens < tree_q or num_tokens % tree_q != 0:
        return None
    n_dec = num_tokens // tree_q
    if n_dec <= 0:
        return None
    return n_dec, tree_q


def is_tree_verification_capture_shape(num_tokens: int, num_reqs: int) -> bool:
    """Dummy FULL capture only needs the tree layout for ``k × (1+budget)``."""
    shaped = tree_verify_shape(num_tokens)
    return shaped is not None and shaped[0] == num_reqs


def seq_lens_are_tree_query(seq_lens: torch.Tensor | None, num_decode: int) -> bool:
    """Dummy FULL capture sets seq_len == query_len; tree verify is 1+budget."""
    tree_q = tree_mask_query_capacity()
    if tree_q is None or seq_lens is None or num_decode <= 0:
        return False
    sl = seq_lens[:num_decode]
    if sl.numel() != num_decode or sl.device.type != "cpu":
        return False
    return all(v == tree_q for v in sl.tolist())


def is_draft_forward() -> bool:
    """True only inside a draft forward that already has a forward context.

    Graph ``prepare_attn`` runs before ``set_forward_context``; treat that as
    target so dummy tree visibility can still be injected.
    """
    from vllm.forward_context import is_forward_context_available

    return is_forward_context_available() and bool(_EXTRA_CTX.is_draft_model)


def need_dummy_tree_visibility_for_capture() -> bool:
    """Target FULL capture only; draft decode must keep its own mask/indices."""
    from vllm.forward_context import is_forward_context_available

    if not is_forward_context_available():
        return False
    if not _EXTRA_CTX.capturing or is_draft_forward():
        return False
    return tree_mask_query_capacity() is not None
