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


Scheduler._mamba_block_aligned_split = _mamba_block_aligned_split
