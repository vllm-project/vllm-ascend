# mypy: ignore-errors

from vllm.v1.worker import mamba_utils

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel

# Upstream uses 16 temporal-copy tiles to saturate H100/GB200. K3 already
# exposes 138 independent state programs per request, while Triton-Ascend
# flattens all grid dimensions into a coreDim that cannot exceed 65535. Keep
# the pre-tiling launch shape on Ascend: it has enough state-level parallelism
# and remains valid at the configured request limit (for example,
# 32 * 138 * 1 instead of 32 * 138 * 16).
#
# Shared by both model runners: the V2 Mamba-Hybrid state launches
# ``preprocess_mamba_align_fused_kernel`` / ``precopy_mamba_align_fused_kernel``
# from this module, and both read ``_TEMPORAL_TILES`` at call time. The remaining
# MRv1-only state-copy overrides live in
# ``vllm_ascend.patch.worker.patch_v1.patch_mamba_utils``.
mamba_utils._TEMPORAL_TILES = 1


def _can_launch_triton_batch_memcpy() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.TRITON_BATCH_MEMCPY)


if _can_launch_triton_batch_memcpy():
    mamba_utils.postprocess_mamba_fused_kernel = postprocess_mamba_fused_kernel
