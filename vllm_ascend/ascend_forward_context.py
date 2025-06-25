from contextlib import contextmanager
from typing import Any, Optional

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, set_forward_context

from vllm_ascend.utils import get_fused_moe_state


@contextmanager
def set_ascend_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        with_prefill: bool = True,
        in_profile_run: bool = False):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(attn_metadata,
                             vllm_config,
                             virtual_engine=virtual_engine,
                             num_tokens=num_tokens,
                             num_tokens_across_dp=num_tokens_across_dp):
        forward_context = get_forward_context()
        forward_context.with_prefill = with_prefill

        ep_size = torch.distributed.get_world_size(
        ) if vllm_config.parallel_config.enable_expert_parallel else 1
        fused_moe_state = get_fused_moe_state(ep_size, with_prefill)
        forward_context.fused_moe_state = fused_moe_state

        forward_context.in_profile_run = in_profile_run

        try:
            yield
        finally:
            pass
