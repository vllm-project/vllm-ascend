# SPDX-License-Identifier: Apache-2.0
"""Contiguous C8/Mamba cache sections for the V1 model runner."""

import math

import torch

# Each shared allocation is partitioned across all scheduler blocks:
# [Conv][SSM/K][V][K scale][V scale]. Only SSM and K alias. In particular,
# Mamba writes cannot overwrite the statically initialized V scale section.
# The scheduler block size makes one K block exactly as large as one SSM
# state; attention still uses 512-token physical kernel blocks.


def use_contiguous_c8_hybrid_layout(config):
    return (
        config.cache_config.cache_dtype == "mxfp8" and config.model_config.is_hybrid and not config.use_v2_model_runner
    )


def configure_contiguous_c8_hybrid_layout(config):
    """Set geometry both during config setup and after backend selection."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.utils.torch_utils import get_dtype_size

    model_cls, _ = ModelRegistry.resolve_model_cls(config.model_config.architecture, model_config=config.model_config)
    shapes = model_cls.get_mamba_state_shape_from_config(config)
    dtypes = model_cls.get_mamba_state_dtype_from_config(config)
    if len(shapes) != 2:
        raise ValueError("C8 hybrid contiguous layout requires Conv and SSM states.")
    conv_bytes, ssm_bytes = [math.prod(s) * get_dtype_size(d) for s, d in zip(shapes, dtypes)]
    heads = config.model_config.get_num_kv_heads(config.parallel_config)
    dim = config.model_config.get_head_size()
    block_size, remainder = divmod(ssm_bytes, heads * dim)
    if remainder or block_size % 512:
        raise ValueError("C8 hybrid contiguous layout requires K == SSM and whole 512-token blocks.")
    config.cache_config.block_size = block_size
    config.cache_config.mamba_page_size_padded = conv_bytes + 2 * ssm_bytes + 2 * ssm_bytes // 32
    if config.cache_config.mamba_cache_mode == "align":
        config.cache_config.mamba_block_size = block_size


def _state_bytes(spec):
    if len(spec.shapes) != 2:
        raise ValueError("C8 hybrid contiguous layout requires exactly two Mamba states.")
    return tuple(math.prod(s) * torch.empty((), dtype=d).element_size() for s, d in zip(spec.shapes, spec.dtypes))


def _validate_raw(raw, page_bytes):
    if raw.dtype != torch.int8 or raw.ndim != 1 or not raw.is_contiguous():
        raise ValueError("C8 hybrid contiguous layout requires contiguous raw int8 storage.")
    if raw.numel() % page_bytes:
        raise ValueError("C8 hybrid contiguous layout allocation must contain complete pages.")
    return raw.numel() // page_bytes


def c8_contiguous_views(raw, spec, kernel_block_size, mamba_spec):
    conv_bytes, ssm_bytes = _state_bytes(mamba_spec)
    if kernel_block_size != 512 or spec.block_size % kernel_block_size:
        raise ValueError("C8 hybrid contiguous layout requires 512-token kernel blocks.")
    if spec.head_size != spec.head_size_v or spec.num_kv_heads * spec.block_size * spec.head_size != ssm_bytes:
        raise ValueError(
            "C8 hybrid contiguous layout requires equal K/V dimensions and K block bytes == SSM bytes: "
            f"block_size={spec.block_size}, num_kv_heads={spec.num_kv_heads}, "
            f"head_size={spec.head_size}, head_size_v={spec.head_size_v}, ssm_bytes={ssm_bytes}."
        )
    page = conv_bytes + 2 * ssm_bytes + 2 * ssm_bytes // 32
    if spec.page_size_bytes != page or mamba_spec.page_size_bytes != page:
        raise ValueError("Planner changed the C8 hybrid contiguous layout page geometry.")
    blocks = _validate_raw(raw, page)
    kernels = blocks * (spec.block_size // kernel_block_size)
    n, d, bs = spec.num_kv_heads, spec.head_size, kernel_block_size
    if d % 64:
        raise ValueError("C8 hybrid contiguous layout requires head dim divisible by 64.")
    shapes = (
        (kernels, n, d // 32, bs, 32),
        (kernels, n, d // 32, bs, 32),
        (kernels, n, bs // 16, d // 64, 16, 2),
        (kernels, n, d // 16, bs // 64, 16, 2),
    )
    offset = blocks * conv_bytes
    views = []
    for shape, dtype in zip(shapes, (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.uint8, torch.uint8)):
        size = math.prod(shape)
        views.append(raw[offset : offset + size].view(dtype).view(shape))
        offset += size
    assert offset == raw.numel()
    return tuple(views)


def mamba_contiguous_views(raw, spec):
    conv_bytes, ssm_bytes = _state_bytes(spec)
    page = conv_bytes + 2 * ssm_bytes + 2 * ssm_bytes // 32
    if spec.page_size_bytes != page:
        raise ValueError("Planner changed the Mamba contiguous layout page geometry.")
    blocks = _validate_raw(raw, page)
    views = []
    offset = 0
    for shape, dtype, size in zip(spec.shapes, spec.dtypes, (conv_bytes, ssm_bytes)):
        end = offset + blocks * size
        views.append(raw[offset:end].view(dtype).view(blocks, *shape))
        offset = end
    return tuple(views)
