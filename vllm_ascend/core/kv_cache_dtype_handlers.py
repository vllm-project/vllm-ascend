# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend KV-cache dtype handlers registered via vLLM's
``register_kv_cache_dtype`` mechanism (vLLM >= releases/v0.33.0).

Importing this module eagerly decorates the handlers, which appends each name
to ``KV_CACHE_DTYPES``, injects the torch dtype into
``STR_DTYPE_TO_TORCH_DTYPE`` in place, and stores the handler for later
``quant_mode()`` / ``is_quantized()`` queries. The in-place dict mutation means
that once this module is imported (which happens in every process that builds a
VllmConfig, via the platform's quant-config registration), the upstream
``kv_cache_dtype_str_to_dtype`` lookup resolves ``"fp8"`` to
``torch.float8_e4m3fn`` rather than the upstream ``torch.uint8``.
"""

import inspect

import torch
from vllm.config.cache import register_kv_cache_dtype
from vllm.v1.kv_cache_interface import KVQuantMode


@register_kv_cache_dtype("fp8")
class Fp8AscendHandler:
    """fp8 KV cache stored as ``torch.float8_e4m3fn`` on Ascend.

    The upstream builtin ``"fp8"`` maps to ``torch.uint8`` (raw fp8 bytes
    consumed by NVIDIA cutlass kernels). Ascend's MLA/DSA kernels operate
    on native ``torch.float8_e4m3fn`` directly, so re-registering ``"fp8"``
    here flips that mapping to ``float8_e4m3fn``. The decorator logs an
    ``already exists and will be overwritten`` warning; this is expected
    and intentional on the Ascend platform.

    No per-token scale: ``quant_mode`` returns ``KVQuantMode.CUSTOM`` so
    the Ascend backend fully self-manages kernel dispatch rather than
    reusing an upstream generic fp8/per-token-head kernel path.
    """

    name = "fp8"

    def torch_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def quant_mode(self):
        return KVQuantMode.CUSTOM


@register_kv_cache_dtype("int8")
class Int8AscendHandler:
    """int8 KV cache stored as ``torch.int8`` on Ascend."""

    name = "int8"

    def torch_dtype(self) -> torch.dtype:
        return torch.int8

    def quant_mode(self):
        return KVQuantMode.CUSTOM


@register_kv_cache_dtype("mxfp8")
class Mxfp8AscendHandler:
    """C8-MXFP KV cache payload stored as ``torch.float8_e4m3fn``.

    The C8 backend swaps in its own cache class and owns the E8M0 scale
    layout, so the generic dtype/quant-mode tables only need to expose the
    fp8 payload storage that the cache constructor resolves first.
    """

    name = "mxfp8"

    def torch_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def quant_mode(self):
        return KVQuantMode.CUSTOM


@register_kv_cache_dtype("hif8")
class Hif8AscendHandler:
    """HIF8 latent payload stored as raw uint8 bytes in the MLA page layout."""

    name = "hif8"

    def torch_dtype(self) -> torch.dtype:
        return torch.uint8

    def quant_mode(self):
        return KVQuantMode.CUSTOM


@register_kv_cache_dtype("mxfp4")
class Mxfp4AscendHandler:
    """Packed MXFP4 main KV cache stored as raw uint8 bytes.

    The packed row is ``head_size * 5 / 8`` bytes (512 -> 320): a 4-bit E2M1
    payload plus one BF16 group scale per 16 values. The SFA FP4 path owns the
    in-page layout, so only the byte view and dispatch mode are registered.
    """

    name = "mxfp4"

    def torch_dtype(self) -> torch.dtype:
        return torch.uint8

    def quant_mode(self):
        return KVQuantMode.CUSTOM


def _patch_mxfp4_mla_page_size() -> None:
    """Report the packed MXFP4 row size to upstream block alignment.

    ``Platform._align_hybrid_block_size`` sizes a one-token MLA page as
    ``head_size * dtype_size``. The packed MXFP4 row is 320 B for the 512-wide
    latent, not the 512 B implied by uint8. Only that temporary one-token spec
    carries ``cache_dtype_str == "mxfp4"`` with ``head_size == 512``; the real
    Ascend spec publishes 320 as its head size and keeps the normal path.
    """
    try:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
    except ImportError:  # pragma: no cover - depends on the vllm build
        return

    descriptor = inspect.getattr_static(MLAAttentionSpec, "page_size_bytes")
    original = descriptor.fget if isinstance(descriptor, property) else descriptor
    if getattr(original, "_ascend_mxfp4", False):
        return

    def page_size_bytes(self):
        if (
            getattr(self, "cache_dtype_str", None) == "mxfp4"
            and getattr(self, "head_size", 0) == 512
            and getattr(self, "page_size_padded", None) is None
        ):
            # 512 one-byte lanes pack into 320 bytes (5/8), independently of
            # how many states the probe page describes.
            return original(self) * 5 // 8
        return original(self)

    page_size_bytes._ascend_mxfp4 = True  # type: ignore[attr-defined]
    if isinstance(descriptor, property):
        MLAAttentionSpec.page_size_bytes = property(page_size_bytes)
    else:
        MLAAttentionSpec.page_size_bytes = page_size_bytes


_patch_mxfp4_mla_page_size()
