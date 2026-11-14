# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from vLLM PR #66881 (vllm/v1/core/hybrid_checkpoint/keys.py).
"""Prefix-cache keys of frontier checkpoints."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.multimodal.utils import get_mm_features_in_window
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request

# Separates frontier keys from block hashes of the same prefix.
FRONTIER_KEY_MARKER = "mamba-frontier-checkpoint"
# Separates frontier locator keys from frontier keys and block hashes.
FRONTIER_LOCATOR_MARKER = "mamba-frontier-locator"
# Tokens after a locator's anchor that its key chains; a frontier shorter
# than this has no locator.
#
# A locator is the only key an external store can find a frontier by without
# knowing its length, so it has to be long enough that two requests sharing it
# really do share a prefix, and short enough that a request computes it before
# it reaches the frontier. 16 is one default hash unit of tokens
# (``hash_block_size``), which makes the locator exactly one block's worth of
# hashing work -- the granularity the block hash itself is computed at -- and
# keeps its extra-key window inside one multimodal feature window. A frontier
# of fewer than 16 tokens cannot hold a whole hash unit, so chaining 16 tokens
# would pull in tokens the frontier does not own; such a frontier is simply
# not locatable by an external store, which costs it a store hit but never
# correctness (in-process hits use ``frontier_key`` and its length).
FRONTIER_LOCATOR_TOKENS = 16


def frontier_key(
    hash_fn: Callable[[Any], bytes],
    request: "Request",
    num_tokens: int,
    hash_block_size: int,
    last_token: int | None = None,
) -> BlockHash:
    """Key of the state after ``request``'s first ``num_tokens`` tokens, or
    after those tokens with the last one replaced by ``last_token``.

    It chains the request's block hash at ``h * hash_block_size`` (``h =
    num_tokens // hash_block_size``) with the tokens after it, their extra
    keys (multimodal inputs, LoRA, cache salt) and ``num_tokens``, so equal
    keys mean equal prefixes and lengths.
    """
    h = num_tokens // hash_block_size
    start = h * hash_block_size
    parent = request.block_hashes[h - 1] if h else None
    extra = _extra_keys(request, start, num_tokens)
    tokens = list(request.all_token_ids[start:num_tokens])
    if last_token is not None:
        tokens[-1] = last_token
    return hash_block_tokens(
        hash_fn,
        parent,
        tokens,
        (*(extra or ()), (FRONTIER_KEY_MARKER, num_tokens)),
    )


def frontier_locator_unit(num_tokens: int, hash_block_size: int) -> int:
    """The hash unit whose start anchors the locator of a frontier of
    ``num_tokens >= FRONTIER_LOCATOR_TOKENS`` tokens: the one holding token
    ``num_tokens - FRONTIER_LOCATOR_TOKENS``, so the locator's tokens lie
    inside the frontier."""
    return (num_tokens - FRONTIER_LOCATOR_TOKENS) // hash_block_size


def frontier_locator_key(
    hash_fn: Callable[[Any], bytes],
    request: "Request",
    h: int,
    hash_block_size: int,
) -> BlockHash:
    """Key of the frontier locator anchored at hash unit ``h``.

    An external store finds the frontiers it holds through locators: a frontier
    key names its length, which a reader does not know, while a locator key
    chains the request's block hash at ``h * hash_block_size`` with the next
    ``FRONTIER_LOCATOR_TOKENS`` tokens and their extra keys only, so every
    request sharing those tokens computes it.
    """
    start = h * hash_block_size
    end = start + FRONTIER_LOCATOR_TOKENS
    parent = request.block_hashes[h - 1] if h else None
    extra = _extra_keys(request, start, end)
    tokens = list(request.all_token_ids[start:end])
    return hash_block_tokens(hash_fn, parent, tokens, (*(extra or ()), (FRONTIER_LOCATOR_MARKER,)))


def _extra_keys(request: "Request", start: int, end: int) -> tuple[Any, ...] | None:
    """Prefix-cache extra keys (multimodal inputs, LoRA, cache salt) of
    tokens ``[start, end)``."""
    mm_idx = 0
    mm_features = request.mm_features
    if start > 0 and mm_features:
        last = mm_features[-1].mm_position
        if last.offset + last.length > start:
            mm_idx, _ = get_mm_features_in_window(mm_features, start, end)
    extra, _ = generate_block_hash_extra_keys(request, start, end, mm_idx)
    return extra
