# SPDX-License-Identifier: Apache-2.0
"""Reuse geometry and batch FP32 normalization for the eager SWA prefix."""

import torch

from ..attention_logits import build_mla_logits
from ..prefill_metadata import logical_positions, reuse_prefill_metadata

# Bound the live QK/PV intermediates to four existing attention chunks.
_SOFTMAX_BATCH_CHUNKS = 4
_MAX_GATHER_BYTES = 72 * 1024 * 1024


def _window_geometry(device, first, rows, history_start, width, group_rows):
    def build():
        positions = logical_positions(first, first + rows, device)
        offsets = torch.arange(1 - width, 1, device=device)
        selected = (positions[:, None] + offsets).clamp_min(-1)
        indices = selected - history_start if history_start else selected
        indices = indices.clamp_min(0).flatten()
        return tuple(zip(selected.split(group_rows), indices.split(group_rows * width)))

    return reuse_prefill_metadata(
        device, ("early_swa", first, rows, history_start, width, group_rows), build
    )


def attend_window_chunks(query, keys, first, history_start, width, sink, scale, latent_dim, chunk, dtype):
    """Keep each QK/PV matrix size and batch the independent softmax rows."""
    rows, heads, dim = query.shape
    keys = keys.float()
    chunk_bytes = chunk * width * dim * keys.element_size()
    group_chunks = min(_SOFTMAX_BATCH_CHUNKS, max(1, _MAX_GATHER_BYTES // chunk_bytes))
    group_rows = group_chunks * chunk
    geometry = _window_geometry(query.device, first, rows, history_start, width, group_rows)
    for group, (selected, indices) in enumerate(geometry):
        first_row = group * group_rows
        count = selected.size(0)
        gathered = keys.index_select(0, indices).view(count, width, dim)
        scores = torch.empty((count, heads, width), dtype=torch.float32, device=query.device)
        key_chunks = gathered.split(chunk)
        for j, (score_rows, selected_kv) in enumerate(zip(scores.split(chunk), key_chunks)):
            offset = j * chunk
            q = query[first_row + offset : first_row + offset + score_rows.size(0)]
            torch.bmm(q, selected_kv.transpose(1, 2), out=score_rows)
        logits = build_mla_logits(scores, selected, sink, scale)
        probabilities = logits.softmax(dim=-1)
        for j, selected_kv in enumerate(key_chunks):
            offset = j * chunk
            probs = probabilities[offset : offset + selected_kv.size(0), :, :-1]
            context = torch.bmm(probs, selected_kv[..., :latent_dim]).to(dtype)
            yield first_row + offset, context
