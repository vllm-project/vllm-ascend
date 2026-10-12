# SPDX-License-Identifier: Apache-2.0

from copy import copy
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointBuilder,
    MambaPrefillCheckpointMetadata,
)
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _effective_eagle_block_drop

_ORIGINAL_BUILD = MambaPrefillCheckpointBuilder.build


class _CheckpointSpeculativeConfig:
    """Change checkpoint replay only, preserving the actual draft configuration."""

    def __init__(self, speculative_config: Any, drop_eagle_block: bool) -> None:
        self._speculative_config = speculative_config
        self._drop_eagle_block = drop_eagle_block

    def use_eagle_block_drop(self) -> bool:
        return self._drop_eagle_block

    def __getattr__(self, name: str) -> Any:
        return getattr(self._speculative_config, name)


class _CheckpointVllmConfig:
    def __init__(self, vllm_config: VllmConfig, drop_eagle_block: bool) -> None:
        self._vllm_config = vllm_config
        self.speculative_config = _CheckpointSpeculativeConfig(vllm_config.speculative_config, drop_eagle_block)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vllm_config, name)


def build_mamba_prefill_checkpoint(
    self: MambaPrefillCheckpointBuilder,
    m: CommonAttentionMetadata,
    request_rows: list[int],
) -> MambaPrefillCheckpointMetadata | None:
    if self.kv_cache_spec.num_prefill_checkpoint_blocks == 0:
        return _ORIGINAL_BUILD(self, m, request_rows)
    speculative_config = self.vllm_config.speculative_config
    if speculative_config is None:
        return _ORIGINAL_BUILD(self, m, request_rows)
    use_eagle_block_drop = speculative_config.use_eagle_block_drop()
    effective_drop = _effective_eagle_block_drop(use_eagle_block_drop, self.vllm_config.kv_transfer_config)
    if effective_drop == use_eagle_block_drop:
        return _ORIGINAL_BUILD(self, m, request_rows)

    # Workers receive VllmConfig independently of the scheduler's KV cache
    # config. Derive the role here and delegate upstream checkpoint validity,
    # offsets and block-table gathering using a private, read-only config view.
    # Never change the shared speculative config or disable actual MTP.
    checkpoint_builder = copy(self)
    checkpoint_builder.vllm_config = cast(VllmConfig, _CheckpointVllmConfig(self.vllm_config, effective_drop))
    return _ORIGINAL_BUILD(checkpoint_builder, m, request_rows)


# Patch the method rather than replace the class: attention backends may have
# already imported the builder class before worker patches are loaded.
MambaPrefillCheckpointBuilder.build = build_mamba_prefill_checkpoint
