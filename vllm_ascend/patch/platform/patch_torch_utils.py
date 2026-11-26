import torch
import vllm.utils.torch_utils as upstream_torch_utils


def get_npu_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """Return an NPU UVA view of a pinned CPU tensor."""
    if not cpu_tensor.is_pinned():
        raise RuntimeError("NPU UVA requires pinned host memory")

    import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401

    view = torch.ops._C_ascend.get_npu_view_from_cpu_tensor(cpu_tensor)
    if view is None:
        raise RuntimeError("NPU UVA requires PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True")
    return view


upstream_torch_utils.get_accelerator_view_from_cpu_tensor = get_npu_view_from_cpu_tensor
