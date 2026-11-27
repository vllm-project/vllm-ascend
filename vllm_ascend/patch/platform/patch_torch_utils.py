import torch
import vllm.utils.torch_utils as upstream_torch_utils

try:
    from vllm.model_executor.offloader import uva as upstream_uva_offloader
except ImportError:
    upstream_uva_offloader = None


def get_npu_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Return an NPU UVA view of a pinned CPU tensor."""
    if not cpu_tensor.is_pinned():
        raise RuntimeError("NPU UVA requires pinned host memory")

    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu_tensor)
    if view is None:
        raise RuntimeError("NPU UVA requires PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True")
    return view


def _supports_weight_uva_offload() -> bool:
    # Pinning is supported even when the driver cannot map pinned memory onto the NPU.
    # Use the upstream offloader's explicit-copy path for Ascend model weights.
    return False


upstream_torch_utils.get_accelerator_view_from_cpu_tensor = get_npu_view_from_cpu_tensor
if upstream_uva_offloader is not None:
    upstream_uva_offloader.is_uva_available = _supports_weight_uva_offload
