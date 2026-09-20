import torch
from vllm.logger import logger

_SUPPORTED_INPUT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _get_fused_gdn_gating_op():
    """Return the vLLM Ascend fused gating op when available."""
    ascend_ops = getattr(torch.ops, "_C_ascend", None)
    if ascend_ops is None:
        return None
    op = getattr(ascend_ops, "fused_gdn_gating", None)
    is_available = getattr(ascend_ops, "is_fused_gdn_gating_available", None)
    if op is None or is_available is None or not is_available():
        return None
    return op


def _can_use_fused_gdn_gating_op(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> bool:
    if A_log.device.type != "npu" or any(tensor.device != A_log.device for tensor in (a, b, dt_bias)):
        return False
    if not all(tensor.dtype in _SUPPORTED_INPUT_DTYPES for tensor in (A_log, a, b, dt_bias)):
        return False
    if A_log.ndim != 1 or dt_bias.shape != A_log.shape:
        return False
    return a.ndim == 2 and b.shape == a.shape and a.shape[1] == A_log.shape[0]


def fused_gdn_gating_pytorch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch to the external fused_gdn_gating op on supported 310P inputs.
    Fall back to the PyTorch implementation when the op is unavailable.

    Args:
        A_log: Log of A parameter, shape [num_heads]
        a: a parameter, shape [batch, num_heads]
        b: b parameter, shape [batch, num_heads]
        dt_bias: dt bias, shape [num_heads]
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        g: gating parameter, shape [1, batch, num_heads]
        beta_output: sigmoid(b), shape [1, batch, num_heads]
    """
    op = _get_fused_gdn_gating_op()
    if op is not None and _can_use_fused_gdn_gating_op(A_log, a, b, dt_bias):
        logger.info_once(
            "[wxc][310P] Using torch.ops._C_ascend.fused_gdn_gating with FP16 inputs "
            "(source dtypes: A_log=%s, a=%s, b=%s, dt_bias=%s).",
            A_log.dtype,
            a.dtype,
            b.dtype,
            dt_bias.dtype,
        )
        return op(
            A_log.to(torch.float16).contiguous(),
            a.to(torch.float16).contiguous(),
            b.to(torch.float16).contiguous(),
            dt_bias.to(torch.float16).contiguous(),
            beta,
            threshold,
        )

    logger.info_once(
        "[wxc][310P] Falling back to the PyTorch fused_gdn_gating implementation because "
        "torch.ops._C_ascend.fused_gdn_gating is unavailable or the inputs are unsupported."
    )
    batch, num_heads = a.shape
    del num_heads
    # Keep nonlinear gating math in fp32 for stability.
    compute_dtype = torch.float32
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    # Expand A_log and dt_bias to match a shape.
    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    # Compute x = a + dt_bias.
    x = a_f + dt_bias_expanded

    # Compute softplus(x).
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )

    # Compute g = -exp(A_log) * softplus(x).
    g = -torch.exp(A_log_expanded) * softplus_x

    # Add sequence dimension.
    g = g.unsqueeze(0)

    # Match Triton kernel: sigmoid in fp32, then cast to input b dtype.
    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)

    return g, beta_output
