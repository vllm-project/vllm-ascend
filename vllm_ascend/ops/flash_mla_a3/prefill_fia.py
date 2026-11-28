"""One right-aligned causal FIA for current tokens and a single projected history chunk."""

from dataclasses import dataclass

import torch
import torch_npu


@dataclass(frozen=True)
class PrefillFiaSegments:
    q_ends: tuple[int, ...]
    kv_ends: tuple[int, ...]
    # Each entry selects a contiguous token range from history (True) or current (False).
    pieces: tuple[tuple[bool, int, int], ...]
    history_tokens: int


def prepare_prefill_fia_segments(q_ends, history_ends=None):
    """Prepare once per scheduler step from host metadata; both inputs contain cumulative lengths."""
    queries = tuple(q_ends)
    histories = tuple(history_ends) if history_ends is not None else (0,) * len(queries)
    if not queries or len(queries) != len(histories):
        raise ValueError("Query and history endpoints must cover the same nonempty batch")
    q_start = h_start = 0
    pieces = []
    for q_end, h_end in zip(queries, histories):
        if q_end <= q_start or h_end < h_start:
            raise ValueError("Each request needs positive query length and nonnegative history")
        if h_end > h_start:
            pieces.append((True, h_start, h_end))
        pieces.append((False, q_start, q_end))
        q_start, h_start = q_end, h_end
    return PrefillFiaSegments(queries, tuple(q + h for q, h in zip(queries, histories)), tuple(pieces), h_start)


def _interleave(history, current, pieces):
    return torch.cat([(history if historical else current)[start:end] for historical, start, end in pieces], dim=0)


def prefill_fia_once(
    q,
    q_rope,
    current_k,
    current_v,
    current_k_rope,
    history_k=None,
    history_v=None,
    history_k_rope=None,
    *,
    q_ends=None,
    history_ends=None,
    segments=None,
    mask,
    scale,
    before_attention=None,
):
    """Return BF16 [T,H,Dv]. The supplied KV contains expanded, projected attention heads."""
    if segments is None:
        segments = prepare_prefill_fia_segments(q_ends, history_ends)
    heads = q.shape[1]
    current_r = current_k_rope.expand(-1, heads, -1)
    if segments.history_tokens:
        history_r = history_k_rope.expand(-1, heads, -1)
        k = _interleave(history_k, current_k, segments.pieces)
        v = _interleave(history_v, current_v, segments.pieces)
        r = _interleave(history_r, current_r, segments.pieces)
    else:
        k, v, r = current_k.contiguous(), current_v.contiguous(), current_r.contiguous()
    if before_attention is not None:
        before_attention()
    out, _ = torch_npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=heads,
        num_key_value_heads=heads,
        input_layout="TND",
        query_rope=q_rope,
        key_rope=r,
        scale=scale,
        atten_mask=mask,
        sparse_mode=3,
        actual_seq_lengths=list(segments.q_ends),
        actual_seq_lengths_kv=list(segments.kv_ends),
        softmax_lse_flag=False,
    )
    return out
