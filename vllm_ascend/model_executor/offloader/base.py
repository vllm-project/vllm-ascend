from typing import TYPE_CHECKING

from vllm.model_executor.offloader.base import BaseOffloader, NoopOffloader

from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader

if TYPE_CHECKING:
    from vllm.config import OffloadConfig


def create_offloader(offload_config: "OffloadConfig | None") -> BaseOffloader:
    """Create an Ascend-aware offloader while preserving vLLM defaults."""

    if offload_config is None:
        return NoopOffloader()

    backend = offload_config.offload_backend
    prefetch = offload_config.prefetch

    if backend == "prefetch":
        return AscendPrefetchOffloader(
            group_size=prefetch.offload_group_size,
            num_in_group=prefetch.offload_num_in_group,
            prefetch_step=prefetch.offload_prefetch_step,
            offload_params=prefetch.offload_params,
            mode="cpu",
        )
    return NoopOffloader()
