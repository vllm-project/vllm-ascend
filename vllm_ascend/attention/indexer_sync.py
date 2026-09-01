# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Callable

# At ~1M-token prefills the lightning indexer can return before its result is
# visible to the stage that publishes the shared index cache. Shorter prefills
# finish before the cache consumer catches up; long ones do not, and the layers
# that reuse the cached top-k then read a partially written buffer.
_LONG_INDEXER_SYNC_TOKENS = 1_000_000


def synchronize_long_indexer_if_needed(
    *,
    seq_len: int,
    num_query_tokens: int,
    pp_world_size: int,
    use_index_cache: bool,
    synchronize: Callable[[], None],
) -> bool:
    """Fence the indexer before the shared index cache is published.

    Only the reproduced case is fenced - a ~1M prefill chunk on a pipeline
    parallel deployment that reuses the cached top-k - so decode steps and
    short prefills keep their current behaviour.
    """
    should_synchronize = (
        seq_len >= _LONG_INDEXER_SYNC_TOKENS and num_query_tokens > 64 and pp_world_size > 1 and use_index_cache
    )
    if should_synchronize:
        synchronize()
    return should_synchronize
