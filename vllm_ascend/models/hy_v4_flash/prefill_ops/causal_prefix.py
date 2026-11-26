# SPDX-License-Identifier: Apache-2.0
"""FP32 causal attention for the early C8 Prefill rows."""

import torch
from vllm.triton_utils import tl, triton

from ..c8_read import gather_c8_rows
from ..prefill_metadata import logical_positions


@triton.jit(do_not_specialize=["rows", "keys", "first", "scale"])
def _causal_prefix_softmax(scores, sink, probs, rows, keys, first, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    key = tl.arange(0, BLOCK)
    live = key < keys
    address = (head * rows + row) * keys + key
    score = tl.load(scores + address, live, 0.0)
    logit = tl.where(live & (key <= first + row), score * scale, -float("inf"))
    sink_logit = tl.load(sink + head)
    peak = tl.maximum(tl.max(logit, axis=0), sink_logit)
    numerator = tl.exp(logit - peak)
    denominator = tl.sum(numerator, axis=0) + tl.exp(sink_logit - peak)
    tl.store(probs + address, numerator / denominator, live)


def attend_causal_prefix(query, cache, block_table, block_size, seq_len, sink, scale):
    """All earlier tokens fit the layer's recall budget in this interval."""
    positions = logical_positions(0, seq_len, query.device)
    keys = gather_c8_rows(cache, block_table, block_size, positions)
    values = keys[:, :512].contiguous()
    query = query.float().transpose(0, 1)
    first = seq_len - query.size(1)
    out = []
    # Bound the temporary FP32 QK/P workspace at 2 * 512 * 8 * 2048 elements.
    for offset in range(0, query.size(1), 512):
        q = query[:, offset : offset + 512]
        valid_keys = first + offset + q.size(1)
        scores = (q @ keys[:valid_keys].T).contiguous()
        probs = torch.empty_like(scores)
        _causal_prefix_softmax[(q.size(1), q.size(0))](
            scores, sink, probs, q.size(1), valid_keys, first + offset, scale,
            BLOCK=triton.next_power_of_2(valid_keys),
        )
        out.append((probs @ values[:valid_keys]).transpose(0, 1).to(torch.bfloat16))
    return out[0] if len(out) == 1 else torch.cat(out, dim=0)
