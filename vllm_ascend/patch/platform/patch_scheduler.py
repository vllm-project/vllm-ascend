from collections.abc import Iterable

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def _mamba_block_aligned_split(
    self,
    request: Request,
    num_new_tokens: int,
    num_new_local_computed_tokens: int = 0,
    num_external_computed_tokens: int = 0,
) -> int:
    num_computed_tokens = request.num_computed_tokens + num_new_local_computed_tokens + num_external_computed_tokens
    # Perform block-aligned splitting at prefill phase, including:
    # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
    # * resumed requests: num_computed_tokens < (
    #                       num_prompt_tokens + num_output_tokens
    #                     )
    # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
    if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
        # To enable block-aligned caching of the Mamba state, `num_new_tokens`
        # must be a multiple of `block_size`.
        # As an exception, if `num_new_tokens` is less than `block_size`, the
        # state is simply not cached, requiring no special handling.
        # Additionally, when Eagle mode is enabled, FullAttn prunes the last
        # matching block. To prevent this from causing a Mamba cache miss, the
        # last chunk must be not smaller than `block_size`.
        block_size = self.cache_config.block_size
        last_cache_position = request.num_tokens - request.num_tokens % block_size
        # eagle prune
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
        num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
        if num_computed_tokens_after_sched < last_cache_position:
            # align to block_size
            num_new_tokens = num_new_tokens // block_size * block_size
        elif num_computed_tokens < last_cache_position < num_computed_tokens_after_sched:
            # force to cache the last chunk
            num_new_tokens = last_cache_position - num_computed_tokens
        else:
            # prefill the last few tokens
            pass
    return num_new_tokens


def _update_requests_with_invalid_blocks(
    self,
    requests: Iterable[Request],
    invalid_block_ids: set[int],
    num_scheduled_tokens: dict[str, int],
    evict_blocks: bool = True,
) -> tuple[set[str], int, set[int]]:
    affected_req_ids: set[str] = set()
    total_affected_tokens = 0
    blocks_to_evict: set[int] = set()
    marked_invalid_block_ids: set[int] = set()
    kv_cache_groups = getattr(self.kv_cache_config, "kv_cache_groups", ())
    coordinator = getattr(self.kv_cache_manager, "coordinator", None)
    get_effective_block_size = getattr(coordinator, "_get_effective_block_size", None)

    for request in requests:
        is_affected = False
        rewind_num_computed_tokens: int | None = None
        req_id = request.request_id
        req_block_ids_by_group = self.kv_cache_manager.get_block_ids(req_id)
        req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
        group_block_sizes: list[int] = []

        for group_id, req_block_ids in enumerate(req_block_ids_by_group):
            block_size = self.block_size
            if group_id < len(kv_cache_groups):
                kv_cache_spec = kv_cache_groups[group_id].kv_cache_spec
                block_size = (
                    get_effective_block_size(kv_cache_spec)
                    if get_effective_block_size is not None
                    else getattr(kv_cache_spec, "block_size", block_size)
                )
            group_block_sizes.append(block_size)
            req_num_computed_blocks = (req_num_computed_tokens + block_size - 1) // block_size

            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True
                if block_id in marked_invalid_block_ids:
                    continue

                marked_invalid_block_ids.add(block_id)
                invalid_token_pos = idx * block_size
                if rewind_num_computed_tokens is None or invalid_token_pos < rewind_num_computed_tokens:
                    rewind_num_computed_tokens = invalid_token_pos

        if is_affected:
            if rewind_num_computed_tokens is None:
                total_affected_tokens += request.num_computed_tokens - req_num_computed_tokens
                request.num_computed_tokens = req_num_computed_tokens
            else:
                request.num_computed_tokens = rewind_num_computed_tokens
                total_affected_tokens += req_num_computed_tokens - request.num_computed_tokens
                if evict_blocks:
                    for block_ids, block_size in zip(req_block_ids_by_group, group_block_sizes):
                        blocks_to_evict.update(block_ids[rewind_num_computed_tokens // block_size :])
            affected_req_ids.add(request.request_id)

    return affected_req_ids, total_affected_tokens, blocks_to_evict


Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
Scheduler._update_requests_with_invalid_blocks = _update_requests_with_invalid_blocks
