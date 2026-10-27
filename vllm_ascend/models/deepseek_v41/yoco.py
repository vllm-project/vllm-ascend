# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YOCO consumer prefill truncation (mask / compact) for DeepSeek-V4.1.

Ports the DeepInfer keep-tail SWA visibility and compact boundary gather into
pure Python so Ascend can:

* validate mask vs compact equivalence on CPU (L1),
* gather hidden/residual rows at the first YOCO consumer layer (CP=1),
* apply an Ascend-native SWA length mask outside the keep-tail.

This module does **not** depend on ``draft_window_size`` / decoder bounded
replay; those address draft KV and prefix-cache replay respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

YocoTruncationMode = Literal["off", "mask", "compact"]

# DeepSeek-V4.1 Flash defaults (SWA 128 + 12 speculative backtrack margin).
DEFAULT_YOCO_KEEP_TOKENS = 140
DEFAULT_YOCO_MAX_BACKTRACK = 12


@dataclass(frozen=True)
class YocoSeqSpec:
    """One request's shape on a single prefill chunk.

    ``N``
        Absolute sequence length (rectangle anchor; may include generated tokens).
    ``q_start``
        Absolute start position of this chunk.
    ``q_len``
        Number of query tokens in this chunk (``seqlen_k = q_start + q_len``).
    ``valid_start``
        SWA state validity floor (normally 0; raised after zero-state recompute).
    ``state_id``
        Per-request ring id used for compact SWA store slots.
    """

    N: int
    q_start: int
    q_len: int
    valid_start: int = 0
    state_id: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.q_len <= self.N and self.q_start + self.q_len <= self.N):
            raise ValueError(f"invalid YocoSeqSpec: {self}")
        if self.valid_start < 0 or self.valid_start > self.q_start:
            raise ValueError(f"valid_start out of range for {self}")


@dataclass(frozen=True)
class YocoSwaVisibility:
    """Visible SWA key ranges inside the original (section1 + section2) layout."""

    start_slot: int
    sec1_len: int
    k_lo: int
    new_len: int


@dataclass
class YocoTruncationMetadata:
    """CPU metadata produced for mask or compact consumer prefill."""

    mode: YocoTruncationMode
    keep_tokens: int
    window_size: int
    clip_prefixes: list[int]
    cu_seqlens_q: torch.Tensor  # int32 [n_seq + 1]; compact keep lengths (mask: full)
    total_q: int
    swa_index_pivots: torch.Tensor  # int32 [total_q, 4]
    # Compact-only fields (None in mask mode).
    query_indices: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    slot_mapping_store_swa: torch.Tensor | None = None
    ks: dict[str, torch.Tensor] | None = None
    ke: dict[str, torch.Tensor] | None = None
    kv_ca_index_offset_len: dict[str, torch.Tensor] | None = None


def yoco_enabled(config: Any) -> bool:
    """Whether the HF text config describes a YOCO producer/consumer split."""
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    n_prefill = getattr(config, "n_prefill_layer", None)
    if n_prefill is None:
        return False
    return int(n_prefill) < n_layers


def resolve_yoco_consumer_start(config: Any) -> int | None:
    """First YOCO consumer layer index, or ``None`` if the model is not YOCO."""
    if not yoco_enabled(config):
        return None
    return int(config.n_prefill_layer)


def yoco_swa_visibility(
    pos: int,
    q_start: int,
    j: int,
    window_size: int,
    rect_lo: int,
    valid_start: int = 0,
) -> YocoSwaVisibility:
    """Return the keep-tail ∩ sliding-window visibility for one query token.

    Mirrors DeepInfer ``yoco_swa_visibility`` in ``prepare_tensor_input_v4.cpp``.
    ``abs_lo == pos + 1`` means an empty visible domain (token fully outside the
    keep rectangle).
    """
    abs_lo = min(max(pos - window_size + 1, rect_lo, valid_start, 0), pos + 1)
    sec1_lo_abs = min(abs_lo, q_start)
    start_slot = sec1_lo_abs - (q_start - window_size)
    if not (0 <= start_slot <= window_size):
        raise ValueError(f"yoco start_slot out of range: {start_slot}")
    k_lo = max(abs_lo - q_start, 0)
    new_len = j - k_lo + 1
    if new_len < 0:
        raise ValueError(f"yoco section2 new_len is negative: {new_len}")
    return YocoSwaVisibility(start_slot, window_size - start_slot, k_lo, new_len)


def reference_visible_keys(spec: YocoSeqSpec, pos: int, keep_tokens: int, window: int) -> list[int]:
    """Analytical keep-tail ∩ sliding-window key positions for ``pos``."""
    lo = max(pos - window + 1, spec.N - keep_tokens, 0)
    return list(range(lo, pos + 1)) if lo <= pos else []


def expand_swa_pivots(
    row: Sequence[int],
    *,
    sec1_start: int,
    sec2_start: int,
    spec: YocoSeqSpec,
    window: int,
    sec2_base_abs: int,
) -> list[int]:
    """Expand one pivot row ``[hist_start, hist_len, new_start, new_len]`` to absolute keys."""
    p0, p1, p2, p3 = (int(x) for x in row)
    keys: list[int] = []
    if p1 > 0:
        s0 = p0 - sec1_start
        keys.extend(spec.q_start - window + s for s in range(s0, s0 + p1))
    if p3 > 0:
        k0 = p2 - sec2_start
        keys.extend(sec2_base_abs + k for k in range(k0, k0 + p3))
    return keys


def clip_prefix_for_seq(spec: YocoSeqSpec, keep_tokens: int, *, compact: bool) -> int:
    """Tokens dropped from the chunk head under compact mode (0 under mask)."""
    if not compact:
        return 0
    # Keep at least the last token of the chunk as a placeholder when the whole
    # chunk lies before the keep rectangle (DeepInfer clamp to q_len - 1).
    return min(max(spec.N - keep_tokens - spec.q_start, 0), spec.q_len - 1)


def build_yoco_truncation_metadata(
    seqs: Sequence[YocoSeqSpec],
    *,
    keep_tokens: int,
    window_size: int,
    mode: YocoTruncationMode,
    swa_block_size: int | None = None,
    sparse_compress_ratios: Sequence[int] = (),
    dense_compress_ratios: Sequence[int] = (),
    max_backtrack_tokens: int = DEFAULT_YOCO_MAX_BACKTRACK,
) -> YocoTruncationMetadata:
    """Build mask or compact YOCO consumer metadata for a prompt batch (CP=1)."""
    if mode == "off":
        raise ValueError("build_yoco_truncation_metadata requires mode mask or compact")
    if keep_tokens <= 0:
        raise ValueError(f"keep_tokens must be > 0; got {keep_tokens}")
    if keep_tokens < window_size + max_backtrack_tokens:
        raise ValueError(
            f"keep_tokens ({keep_tokens}) must be >= window_size ({window_size}) + "
            f"max_backtrack_tokens ({max_backtrack_tokens})"
        )
    if swa_block_size is None:
        swa_block_size = window_size
    if swa_block_size < window_size:
        raise ValueError(f"swa_block_size ({swa_block_size}) must be >= window_size ({window_size})")

    compact = mode == "compact"
    n_seq = len(seqs)
    clip_prefixes = [clip_prefix_for_seq(spec, keep_tokens, compact=compact) for spec in seqs]
    keep_lens = [spec.q_len - clip for spec, clip in zip(seqs, clip_prefixes)]
    cu = torch.zeros(n_seq + 1, dtype=torch.int32)
    for i, keep_len in enumerate(keep_lens):
        cu[i + 1] = cu[i] + keep_len
    total_q = int(cu[-1].item())

    pivots = torch.empty((total_q, 4), dtype=torch.int32)
    query_indices = torch.empty(total_q, dtype=torch.int32) if compact else None
    positions = torch.empty(total_q, dtype=torch.int32) if compact else None
    slot_store = torch.empty(total_q, dtype=torch.int32) if compact else None

    full_bases: list[int] = []
    base = 0
    for spec in seqs:
        full_bases.append(base)
        base += spec.q_len

    i_qc = 0
    for i_seq, spec in enumerate(seqs):
        clip = clip_prefixes[i_seq]
        sec1_start = i_seq * window_size
        sec2_start_c = n_seq * window_size + int(cu[i_seq].item())
        rect_lo = spec.N - keep_tokens
        for j in range(clip, spec.q_len):
            pos = spec.q_start + j
            if compact:
                assert query_indices is not None and positions is not None and slot_store is not None
                query_indices[i_qc] = full_bases[i_seq] + j
                positions[i_qc] = pos
                in_rect = pos >= rect_lo
                if in_rect and spec.q_len - j <= swa_block_size:
                    slot_store[i_qc] = spec.state_id * swa_block_size + pos % swa_block_size
                else:
                    slot_store[i_qc] = -1

            vis = yoco_swa_visibility(pos, spec.q_start, j, window_size, rect_lo, spec.valid_start)
            if vis.new_len > 0 and vis.k_lo < clip:
                raise ValueError("yoco consumer visible key before keep range")
            pivots[i_qc, 0] = sec1_start + vis.start_slot
            pivots[i_qc, 1] = vis.sec1_len
            pivots[i_qc, 2] = sec2_start_c + (vis.k_lo - clip)
            pivots[i_qc, 3] = vis.new_len
            i_qc += 1
    if i_qc != total_q:
        raise RuntimeError(f"pivot row count mismatch: wrote {i_qc}, expected {total_q}")

    ks: dict[str, torch.Tensor] | None = None
    ke: dict[str, torch.Tensor] | None = None
    ca: dict[str, torch.Tensor] | None = None
    if compact:
        ks, ke, ca = _compact_compress_maps(seqs, clip_prefixes, total_q, sparse_compress_ratios, dense_compress_ratios)

    return YocoTruncationMetadata(
        mode=mode,
        keep_tokens=keep_tokens,
        window_size=window_size,
        clip_prefixes=clip_prefixes,
        cu_seqlens_q=cu,
        total_q=total_q,
        swa_index_pivots=pivots,
        query_indices=query_indices,
        positions=positions,
        slot_mapping_store_swa=slot_store,
        ks=ks,
        ke=ke,
        kv_ca_index_offset_len=ca,
    )


def _compact_compress_maps(
    seqs: Sequence[YocoSeqSpec],
    clip_prefixes: Sequence[int],
    total_q: int,
    sparse_ratios: Sequence[int],
    dense_ratios: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    def for_each_kept_row(ratio: int, emit) -> None:
        i = 0
        base = 0
        for spec, clip in zip(seqs, clip_prefixes):
            seqlen_k = spec.q_start + spec.q_len
            for j in range(clip, spec.q_len):
                emit(i, base, (spec.q_start + j + 1) // ratio)
                i += 1
            base += max(1, seqlen_k // ratio)
        if i != total_q:
            raise RuntimeError(f"compress map row count mismatch: {i} != {total_q}")

    ks: dict[str, torch.Tensor] = {}
    ke: dict[str, torch.Tensor] = {}
    for ratio in sparse_ratios:
        ks_t = torch.empty(total_q, dtype=torch.int32)
        ke_t = torch.empty(total_q, dtype=torch.int32)

        def _emit_sparse(i: int, base: int, klen: int, _ks=ks_t, _ke=ke_t) -> None:
            _ks[i] = base
            _ke[i] = base + klen

        for_each_kept_row(int(ratio), _emit_sparse)
        name = str(int(ratio))
        ks[name] = ks_t
        ke[name] = ke_t

    ca: dict[str, torch.Tensor] = {}
    for ratio in dense_ratios:
        offset_len = torch.empty((total_q, 2), dtype=torch.int32)

        def _emit_dense(i: int, base: int, klen: int, _ol=offset_len) -> None:
            _ol[i, 0] = base
            _ol[i, 1] = klen

        for_each_kept_row(int(ratio), _emit_dense)
        ca[str(int(ratio))] = offset_len
    return ks, ke, ca


def compact_yoco_boundary(x: torch.Tensor, query_indices: torch.Tensor) -> torch.Tensor:
    """Gather per-token ``x`` onto compact consumer rows (CP=1).

    ``x`` must hold one row per full-layout token along dim 0. CP>1
    redistribution is intentionally out of scope for the first PR.
    """
    if query_indices.ndim != 1:
        raise ValueError(f"query_indices must be rank 1, got {tuple(query_indices.shape)}")
    idx = query_indices.to(device=x.device, dtype=torch.long)
    return x.index_select(0, idx)


def apply_yoco_mask_to_window_lengths(
    positions: torch.Tensor,
    lengths: torch.Tensor,
    *,
    seq_lens: Sequence[int] | torch.Tensor,
    query_start_loc: Sequence[int] | torch.Tensor,
    keep_tokens: int,
) -> torch.Tensor:
    """Clip SWA window lengths to keep-tail ∩ causal window (DeepInfer mask).

    For each query row at absolute ``pos`` with sequence length ``N``:

    * ``rect_lo = N - keep_tokens``
    * visible count = ``max(0, min(length, pos - rect_lo + 1))``

    So rows before the keep-tail become empty, and retained rows whose raw SWA
    still reaches before ``rect_lo`` are shortened (e.g. N=1000, keep=140,
    W=128, pos=860 → length 1, not 128). Returns a clone; does not mutate
    ``lengths``. Callers that also pass sparse indices must rebuild those
    indices to match — length clipping alone is not enough for index-driven
    kernels.

    Fully vectorized: no per-token ``.item()`` host syncs.
    """
    if keep_tokens <= 0:
        raise ValueError(f"keep_tokens must be > 0; got {keep_tokens}")
    out = lengths.clone()
    if out.numel() == 0:
        return out

    device = out.device
    if isinstance(query_start_loc, torch.Tensor):
        query_start_loc_t = query_start_loc.to(device=device, dtype=torch.long)
    else:
        query_start_loc_t = torch.tensor(query_start_loc, device=device, dtype=torch.long)
    if isinstance(seq_lens, torch.Tensor):
        seq_lens_t = seq_lens.to(device=device, dtype=torch.long)
    else:
        seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)
    if query_start_loc_t.numel() != seq_lens_t.numel() + 1:
        raise ValueError("query_start_loc length must be num_seqs + 1")

    q_lens = query_start_loc_t[1:] - query_start_loc_t[:-1]
    if positions.numel() == 0:
        # Reconstruct absolute positions from chunk layout: q_start = N - q_len.
        local_idx = torch.arange(out.shape[0], device=device, dtype=torch.long) - torch.repeat_interleave(
            query_start_loc_t[:-1], q_lens
        )
        q_starts = seq_lens_t - q_lens
        positions_t = local_idx + torch.repeat_interleave(q_starts, q_lens)
    else:
        positions_t = positions.to(device=device, dtype=torch.long)

    # Cap = keys in [rect_lo, pos]; empty when pos < rect_lo.
    rect_lo = torch.repeat_interleave(seq_lens_t - keep_tokens, q_lens)
    keep_cap = torch.clamp(positions_t - rect_lo + 1, min=0).to(dtype=out.dtype)
    flat = out.view(out.shape[0], -1)
    flat.copy_(torch.minimum(flat, keep_cap.unsqueeze(-1).expand_as(flat)))
    return out


@dataclass
class YocoPrefillRuntime:
    """Per-forward YOCO truncation state attached to the model (CP=1).

    Executable mask/compact paths are not wired yet (no runner metadata attach,
    no attention length/index rewrite). Constructing this for a non-``off`` mode
    raises via :func:`make_yoco_prefill_runtime`.
    """

    mode: YocoTruncationMode
    keep_tokens: int
    consumer_start: int
    metadata: YocoTruncationMetadata | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def compaction_enabled(self) -> bool:
        return self.mode == "compact" and self.metadata is not None and self.metadata.query_indices is not None

    @property
    def mask_enabled(self) -> bool:
        return self.mode == "mask" and self.metadata is not None


YOCO_RUNTIME_NOT_WIRED = (
    "yoco_prefill_truncation_mode={mode!r} is not executable yet: the runner never "
    "calls prepare_yoco_prefill() and consumer attention does not apply YOCO SWA "
    "masking/compaction. CPU metadata helpers remain for unit tests; leave the "
    "setting at 'off' until the wiring follow-up lands."
)


def make_yoco_prefill_runtime(config: Any, ascend_config: Any) -> YocoPrefillRuntime | None:
    """Build a model-scoped YOCO runtime, or ``None`` if off.

    Non-``off`` modes raise :class:`NotImplementedError` until runner/attention
    wiring exists. Prefer AscendConfig normalization for keep/backtrack defaults.
    """
    mode: YocoTruncationMode = getattr(ascend_config, "yoco_prefill_truncation_mode", "off")
    if mode == "off":
        return None
    # Defense in depth if config validation was bypassed.
    raise NotImplementedError(YOCO_RUNTIME_NOT_WIRED.format(mode=mode))
