# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Protocol

import numpy as np
import torch
from vllm.v1.core.sched.output import GrammarOutput


class _GrammarBitmaskBatch(Protocol):
    """Minimal runner batch fields required to place grammar masks."""

    req_ids: list[str]
    cu_num_logits_np: np.ndarray


def apply_grammar_bitmask(
    logits: torch.Tensor,
    input_batch: _GrammarBitmaskBatch,
    grammar_output: GrammarOutput,
) -> None:
    """Apply scheduler masks in-place on the logits device without Triton.

    Masks have a fixed stride per grammar request; logits contain only the
    positions actually scheduled, in runner batch order. The mask excludes
    the first invalid draft token; normal rejection sampling then discards
    the remaining draft suffix without an additional acceptance bound.
    """
    grammar_req_ids = grammar_output.structured_output_request_ids
    if not grammar_req_ids or logits.shape[0] == 0:
        return None

    bitmask = grammar_output.grammar_bitmask
    mask_stride = bitmask.shape[0] // len(grammar_req_ids)
    sorted_bitmask = torch.full((logits.shape[0], bitmask.shape[1]), -1, dtype=torch.int32)
    req_id_to_idx = {req_id: i for i, req_id in enumerate(input_batch.req_ids)}
    for grammar_idx, req_id in enumerate(grammar_req_ids):
        req_idx = req_id_to_idx.get(req_id)
        if req_idx is None:
            continue
        start, end = map(int, input_batch.cu_num_logits_np[req_idx : req_idx + 2])
        mask_start = grammar_idx * mask_stride
        sorted_bitmask[start:end].copy_(torch.from_numpy(bitmask[mask_start : mask_start + end - start]))
    # Scheduler masks already live on CPU. Unpack their little-endian words
    # without reading logits back from the device; transfer only the mask.
    packed = sorted_bitmask.numpy().astype("<i4", copy=False).view(np.uint8)
    allowed = np.unpackbits(packed, axis=-1, bitorder="little")[:, : logits.shape[1]]
    blocked = torch.from_numpy(np.ascontiguousarray(allowed == 0))
    logits.masked_fill_(blocked.to(device=logits.device, non_blocking=True), -torch.inf)
