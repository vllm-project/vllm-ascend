from functools import lru_cache

import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

SUPPORTED_HIDDEN_SIZES = frozenset((128, 256, 2048))
# The kernel supports all sizes above, but the model dispatcher only selects
# shapes that beat the stock Graph and Eager paths by a useful margin.
PROFITABLE_HIDDEN_SIZES = frozenset((256,))


@lru_cache(maxsize=1)
def _get_adn_rms_norm_op():
    if not enable_custom_op():
        return None
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None:
        return None
    return getattr(ascend_ops, "adn_rms_norm", None)


def adn_rms_norm_or_fallback(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    if (
        x.dim() > 0
        and x.numel() > 0
        and x.dtype == torch.float16
        and gamma.dtype == torch.float16
        and x.device.type == "npu"
        and gamma.device == x.device
        and x.shape[-1] in PROFITABLE_HIDDEN_SIZES
        and gamma.dim() == 1
        and gamma.shape[0] == x.shape[-1]
        and x.is_contiguous()
        and gamma.is_contiguous()
    ):
        candidate = _get_adn_rms_norm_op()
        if candidate is not None:
            return candidate(x, gamma, epsilon)

    return torch_npu.npu_rms_norm(x, gamma, epsilon)[0]
