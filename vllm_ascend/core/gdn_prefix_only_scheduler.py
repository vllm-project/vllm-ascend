# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
from vllm.v1.request import Request

from vllm_ascend.core.single_type_kv_cache_manager import AscendGDNPrefixOnlyManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.ascend_config import AscendConfig


class GDNPrefixOnlyScheduler(Scheduler):
    """Skip an otherwise unnecessary in-page prefill checkpoint split."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        managers = self.kv_cache_manager.coordinator.single_type_managers
        self._prefix_only_managers = tuple(
            manager for manager in managers if isinstance(manager, AscendGDNPrefixOnlyManager)
        )
        if not self._prefix_only_managers or any(
            isinstance(manager, MambaManager)
            and (not isinstance(manager, AscendGDNPrefixOnlyManager) or manager.has_prefill_checkpoint_blocks)
            for manager in managers
        ):
            raise ValueError(
                "GDN prefix_only requires GDN state in every Mamba cache group and internal exports disabled."
            )

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        aligned_tokens = super()._mamba_block_aligned_split(
            request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens
        )
        if not self.mamba_partial_cache_hit or aligned_tokens == num_new_tokens or num_external_computed_tokens:
            return aligned_tokens

        start = request.num_computed_tokens + num_new_local_computed_tokens
        end = start + num_new_tokens
        block_size = self.cache_config.block_size
        # Preserve all physical-page, token-budget, replay and junction stops,
        # including those that the final tail stop may otherwise hide.
        if (
            end != request.num_prompt_tokens
            or request.num_tokens != end
            or start // block_size != (end - 1) // block_size
            or start < request.shared_prefix_boundary < end
            or any(manager.block_size != block_size for manager in self._prefix_only_managers)
        ):
            return aligned_tokens

        tail = get_mamba_prefill_checkpoint_position(
            request.num_prompt_tokens,
            self.hash_block_size,
            self.use_eagle_block_drop,
            num_hash_lookahead_tokens=self.num_hash_lookahead_tokens,
        )
        if not start < tail < end or start + aligned_tokens != tail:
            return aligned_tokens

        # Waiting requests expose the hit from this lookup before allocation.
        # Continued chunks use the manager's record of the accepted hit.
        if num_new_local_computed_tokens > 0 or all(
            manager.has_local_prefix_hit(request.request_id) for manager in self._prefix_only_managers
        ):
            return num_new_tokens
        return aligned_tokens


class AsyncGDNPrefixOnlyScheduler(GDNPrefixOnlyScheduler, AsyncScheduler):
    """Run the same tail policy with upstream async bookkeeping."""


def configure_gdn_prefix_only_scheduler(vllm_config: "VllmConfig", ascend_config: "AscendConfig") -> None:
    cache_config = vllm_config.cache_config
    if (
        not vllm_config.use_v2_model_runner
        or not cache_config.enable_prefix_caching
        or cache_config.mamba_cache_mode != "align"
        or vllm_config.kv_transfer_config is not None
        or vllm_config.speculative_config is not None
        or ascend_config.gdn_internal_prefill_checkpoints
    ):
        raise ValueError(
            "GDN prefix_only requires Model Runner V2, align-mode local prefix caching, "
            "and disabled speculative decoding and internal prefill checkpoints."
        )
    if ascend_config.scheduler_config.enable_balance_scheduling:
        raise ValueError("GDN prefix_only and balance scheduling are mutually exclusive.")

    scheduler_config = vllm_config.scheduler_config
    scheduler_cls = scheduler_config.get_scheduler_cls()
    use_async = bool(scheduler_config.async_scheduling)
    # A scheduler composed through inheritance can supply the same policy.
    if issubclass(scheduler_cls, GDNPrefixOnlyScheduler) and issubclass(scheduler_cls, AsyncScheduler) == use_async:
        return
    if scheduler_cls not in (Scheduler, AsyncScheduler):
        raise ValueError("GDN prefix_only requires the default scheduler or a compatible GDN prefix-only subclass.")
    scheduler_config.scheduler_cls = AsyncGDNPrefixOnlyScheduler if use_async else GDNPrefixOnlyScheduler
