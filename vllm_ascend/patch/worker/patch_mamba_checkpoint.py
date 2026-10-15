# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import copy
from functools import wraps
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.v1.attention.backend import AttentionMetadataBuilder, CommonAttentionMetadata

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _effective_eagle_block_drop

try:
    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointBuilder,
        MambaPrefillCheckpointMetadata,
    )
except ModuleNotFoundError as exc:
    # v0.30.0 implements checkpoints inline in the Kimi K3 metadata builder.
    # Missing dependencies of an existing generic module are real failures.
    if exc.name != "vllm.model_executor.layers.mamba.checkpoint":
        raise
    HAS_GENERIC_CHECKPOINT_BUILDER = False
else:
    HAS_GENERIC_CHECKPOINT_BUILDER = True


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


def _configure_legacy_kda_checkpoint_policy(builder: AttentionMetadataBuilder, vllm_config: VllmConfig) -> None:
    if vllm_config.cache_config.mamba_cache_mode != "align":
        return
    if getattr(builder.kv_cache_spec, "num_prefill_checkpoint_blocks", 0) == 0:
        return
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return
    use_eagle_block_drop = speculative_config.use_eagle_block_drop()
    effective_drop = _effective_eagle_block_drop(use_eagle_block_drop, vllm_config.kv_transfer_config)
    if effective_drop == use_eagle_block_drop:
        return

    # Legacy KDA builds checkpoint metadata inline and retains state such as
    # its RecoverSSM context across calls. Keep the same builder, and change
    # only the config read used by that checkpoint calculation. The shared
    # config and speculative fields cached by GDN initialization stay intact.
    builder.vllm_config = cast(VllmConfig, _CheckpointVllmConfig(vllm_config, effective_drop))


def install_legacy_kda_checkpoint_policy(
    builder_base: type[AttentionMetadataBuilder] = AttentionMetadataBuilder,
) -> None:
    """Adapt legacy KDA when it is instantiated, without importing CUDA models."""
    original_init = builder_base.__init__
    if getattr(original_init, "_ascend_legacy_kda_checkpoint_policy", False):
        return

    @wraps(original_init)
    def init(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs):
        original_init(self, kv_cache_spec, layer_names, vllm_config, device, *args, **kwargs)
        if any(
            cls.__module__ == "vllm.models.kimi_k3.nvidia.kda_metadata" and cls.__name__ == "KimiK3KDAMetadataBuilder"
            for cls in type(self).__mro__
        ):
            _configure_legacy_kda_checkpoint_policy(self, vllm_config)

    # Repeated adapt_patch calls must not add another constructor wrapper.
    init._ascend_legacy_kda_checkpoint_policy = True  # type: ignore[attr-defined]
    builder_base.__init__ = init  # type: ignore[method-assign]


if HAS_GENERIC_CHECKPOINT_BUILDER:
    _ORIGINAL_BUILD = MambaPrefillCheckpointBuilder.build

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

    # Preserve aliases already imported by attention backends.
    MambaPrefillCheckpointBuilder.build = build_mamba_prefill_checkpoint
else:
    install_legacy_kda_checkpoint_policy()
