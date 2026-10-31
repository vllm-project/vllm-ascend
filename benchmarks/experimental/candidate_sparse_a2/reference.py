# SPDX-License-Identifier: Apache-2.0
"""CPU oracle for physically compact candidate QLI (no torch required)."""

import math

BLOCK_SIZE = 8
NEG_HUGE = -1e30


def block_plan(candidates, visible, topk=512):
    """Union candidates with a short prefix that preserves R11 fill behavior.

    Candidate IDs come from the existing source: unique, with -1 padding.
    The prefix covers at least topk positions unless the context is shorter.
    Thus omitted, penalized outside positions cannot improve a TopK score.
    """
    prefix = math.ceil(topk / BLOCK_SIZE)
    valid = sorted(block for block in candidates if 0 <= block * BLOCK_SIZE < visible)
    prefix_count = min(prefix, math.ceil(visible / BLOCK_SIZE))
    blocks = list(range(prefix_count)) + [block for block in valid if block >= prefix]
    relative = {block: column for column, block in enumerate(blocks)}
    mapped_candidates = [relative[block] for block in valid]
    positions = [
        position
        for block in blocks
        for position in range(block * BLOCK_SIZE, (block + 1) * BLOCK_SIZE)
        if position < visible
    ]
    return blocks, mapped_candidates, positions


def select(scores, candidates, topk):
    candidate_set = set(candidates)
    penalized = [score if pos // BLOCK_SIZE in candidate_set else NEG_HUGE for pos, score in enumerate(scores)]
    result = sorted(range(len(scores)), key=lambda pos: (-penalized[pos], pos))[:topk]
    return result + [-1] * (topk - len(result))


def compact_select(scores, candidates, topk):
    _, mapped, positions = block_plan(candidates, len(scores), topk)
    packed_scores = [scores[pos] for pos in positions]
    selected = select(packed_scores, mapped, topk)
    return [positions[pos] if pos >= 0 else -1 for pos in selected]
