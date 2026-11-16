# mypy: ignore-errors

import torch
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.mamba_utils import MambaCopyBuffers

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel

# Upstream uses 16 temporal-copy tiles to saturate H100/GB200. K3 already
# exposes 138 independent state programs per request, while Triton-Ascend
# flattens all grid dimensions into a coreDim that cannot exceed 65535. Keep
# the pre-tiling launch shape on Ascend: it has enough state-level parallelism
# and remains valid at the configured request limit (for example,
# 32 * 138 * 1 instead of 32 * 138 * 16).
#
# Consumed by the V2 Mamba-Hybrid state, which reads ``_TEMPORAL_TILES`` and
# ``precopy_mamba_align_fused_kernel`` off this module at call time (the precopy
# kernel itself is rebound by ``patch_triton.py``), and launches
# ``postprocess_mamba_fused_kernel`` from
# ``MambaSpecDecodeGPUContext.run_fused_postprocess_align``.
mamba_utils._TEMPORAL_TILES = 1


def _can_launch_triton_batch_memcpy() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.TRITON_BATCH_MEMCPY)


def _batch_memcpy_triton(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[(batch,)](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


# Upstream's batch_memcpy_kernel fails to lower on Ascend. The port above keeps
# the same signature and raw-pointer semantics, so do_mamba_copy_block and any
# other caller reached through the module attribute pick it up unchanged.
if _can_launch_triton_batch_memcpy():
    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy_triton
    mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel

# Ascend NPU does not support DT_UINT64 in aclnnInplaceZero.
# MambaCopyBuffers.create() uses torch.uint64 for src_ptrs/dst_ptrs,
# which triggers a runtime error. Remap to int64 at the source.
_original_create = MambaCopyBuffers.create


@classmethod
def _patched_create(cls, max_num_reqs, kv_cache_config, copy_funcs, make_buffer):
    return _original_create(
        max_num_reqs,
        kv_cache_config,
        copy_funcs,
        lambda n, dtype: make_buffer(n, dtype=torch.int64 if dtype == torch.uint64 else dtype),
    )


MambaCopyBuffers.create = _patched_create
