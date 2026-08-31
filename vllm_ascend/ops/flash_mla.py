"""A5-only FlashMLA bindings compiled with vLLM Ascend."""

from typing import Any

import torch


def _flash_mla_ops():
    # Importing the extension registers the A5-only _C_ascend schemas and
    # PrivateUse1 implementations.  The custom kernels themselves are bundled
    # in vllm_ascend/_cann_ops_custom by csrc/build_aclnn.sh.
    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, PLC0415

    return torch.ops._C_ascend


def flash_mla_metadata_size(batch_size: int) -> int:
    """Return the int32 element count required by FlashMLA metadata."""
    metadata_size = ((36 + 72) * batch_size + 1) * 16
    return ((metadata_size + 4095) // 4096) * 4096


def build_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    *,
    output_buffer: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Build FlashMLA tiling metadata outside the attention graph."""
    metadata = _flash_mla_ops().flash_mla_with_kvcache_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        **kwargs,
    )
    if output_buffer is None:
        return metadata
    if output_buffer.shape != metadata.shape:
        raise ValueError(
            "FlashMLA metadata buffer shape mismatch: "
            f"expected {tuple(metadata.shape)}, got {tuple(output_buffer.shape)}."
        )
    output_buffer.copy_(metadata)
    return output_buffer


def run_flash_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the bundled A5 FlashMLA operator with prebuilt metadata."""
    return _flash_mla_ops().flash_mla_with_kvcache(query, kv_cache, **kwargs)
