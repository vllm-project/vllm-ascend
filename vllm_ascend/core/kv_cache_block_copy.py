# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for carrying DSv4 KV block copy pairs through SchedulerOutput.

``SchedulerOutput`` is defined in upstream vLLM. To keep vLLM-Ascend from
adding fields to that dataclass, copy-pairs are packed into the existing
``new_block_ids_to_zero`` list using an integer sentinel. The worker unpacks
them before forwarding the remaining block ids to upstream zeroing.
"""

from __future__ import annotations


KV_CACHE_COPY_PAIR_SENTINEL = -1
KV_CACHE_COPY_PAIR_WIDTH = 6
KVCacheCopyPair = tuple[int, int, int, int, int]


def attach_kv_cache_block_copy_pairs(scheduler_output, copy_pairs: list[KVCacheCopyPair] | None) -> None:
    if not copy_pairs:
        return

    encoded = list(scheduler_output.new_block_ids_to_zero or [])
    for group_id, src_block_id, dst_block_id, compressed_slots, original_slots in copy_pairs:
        encoded.extend(
            [
                KV_CACHE_COPY_PAIR_SENTINEL,
                group_id,
                src_block_id,
                dst_block_id,
                compressed_slots,
                original_slots,
            ]
        )
    scheduler_output.new_block_ids_to_zero = encoded


def extract_kv_cache_block_copy_pairs(scheduler_output) -> list[KVCacheCopyPair] | None:
    block_ids = scheduler_output.new_block_ids_to_zero
    if not block_ids:
        return getattr(scheduler_output, "kv_cache_block_copy_pairs", None)

    copy_pairs = list(getattr(scheduler_output, "kv_cache_block_copy_pairs", None) or [])
    zero_block_ids: list[int] = []
    i = 0
    while i < len(block_ids):
        item = block_ids[i]
        if item == KV_CACHE_COPY_PAIR_SENTINEL:
            if i + KV_CACHE_COPY_PAIR_WIDTH > len(block_ids):
                raise RuntimeError("Malformed KV cache block copy-pair payload")
            copy_pairs.append(tuple(block_ids[i + 1 : i + KV_CACHE_COPY_PAIR_WIDTH]))  # type: ignore[arg-type]
            i += KV_CACHE_COPY_PAIR_WIDTH
        else:
            zero_block_ids.append(item)
            i += 1

    scheduler_output.new_block_ids_to_zero = zero_block_ids or None
    return copy_pairs or None
