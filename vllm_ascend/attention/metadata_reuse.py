# SPDX-License-Identifier: Apache-2.0
"""Ascend policy for the upstream attention metadata update protocol.

The builder interface predates vLLM #58762. Keep the NPU policy here rather
than depending on the CUDA runner's reuse loop or its aligned-index buffers.
"""

from typing import Any

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.ascend_config import validate_additional_config_bool


def kv_cache_group_reuse_enabled(vllm_config) -> bool:
    """Use the owning builder's config, including in an upstream runner loop."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return validate_additional_config_bool(
        additional_config.get("reuse_kv_cache_groups", False),
        "additional_config.reuse_kv_cache_groups",
    )


def _metadata_reuse_value_key(value: Any) -> Any:
    """Identify identical input views without reading device tensor values."""
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), value.shape, value.stride(), value.dtype, value.device)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    # Unknown model-specific inputs must not be assumed interchangeable.
    raise TypeError("Unsupported metadata reuse input")


def _metadata_reuse_key(builder, causal, common_kwargs, build_kwargs, group_spec=None):
    if not builder.supports_update_block_table or not isinstance(
        builder.kv_cache_spec, (MambaSpec, FullAttentionSpec, SlidingWindowSpec)
    ):
        return None
    try:
        key = (
            type(builder),
            builder.kv_cache_spec,
            group_spec,
            id(builder.vllm_config),
            builder.kernel_block_size,
            builder.reorder_batch_threshold,
            causal,
            tuple((name, _metadata_reuse_value_key(value)) for name, value in sorted(common_kwargs.items())),
            tuple((name, _metadata_reuse_value_key(value)) for name, value in sorted(build_kwargs.items())),
        )
        hash(key)
        return key
    except TypeError:
        return None


class KVCacheGroupMetadataReuse:
    """One build invocation's cache; never share it across steps or captures.

    Ascend needs stricter compatibility than the upstream Mamba-only cache:
    causality, kernel layout and model-specific inputs must match as well.
    Backends own all device-specific updates through update_block_table().
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._metadata: dict[Any, Any] = {}

    def build(self, builder, common_metadata, *, group_spec, common_kwargs, build_kwargs, for_capture):
        key = None
        if self.enabled and getattr(builder, "supports_update_block_table", False) is True:
            key = _metadata_reuse_key(builder, common_metadata.causal, common_kwargs, build_kwargs, group_spec)
        if key is not None and key in self._metadata:
            return builder.update_block_table(
                self._metadata[key], common_metadata.block_table_tensor, common_metadata.slot_mapping
            )
        if for_capture:
            metadata = builder.build_for_cudagraph_capture(common_metadata, **build_kwargs)
        else:
            metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_metadata, **build_kwargs)
        if key is not None:
            self._metadata[key] = metadata
        return metadata
