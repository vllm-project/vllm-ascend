# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import re
from functools import cache
from importlib import import_module

import torch
from vllm.logger import logger

from vllm_ascend.utils import bootstrap_custom_op_env


def _check_paired_page_kernel(op, device: torch.device, num_heads: int) -> str | None:
    # Logical tokens 7 and 137 map to different physical pages in reverse order.
    # Each page stores the entire latent plane first, then the entire RoPE plane.
    block = 128
    host_pages = torch.zeros((2, block * 576), dtype=torch.bfloat16, device="cpu")
    latent = host_pages[:, : block * 512].view(2, block, 512)
    rope = host_pages[:, block * 512 :].view(2, block, 64)
    latent[1, 7].fill_(1)
    latent[0, 9].fill_(3)
    rope[1, 7, 0] = 2
    rope[0, 9, 0] = -2
    pages = host_pages.view(2, block, 1, 576).to(device)
    midpoint = num_heads // 2
    query = torch.zeros((1, num_heads, 512), dtype=torch.bfloat16, device="cpu")
    query[..., 0] = 1
    query_rope = torch.zeros((1, num_heads, 64), dtype=torch.bfloat16, device="cpu")
    query_rope[:, :midpoint, 0] = 1
    query_rope[:, midpoint:, 0] = -1
    indices = torch.full((1, 1, 2048), -1, dtype=torch.int32, device="cpu")
    indices[0, 0, 0] = 7
    indices[0, 0, 1] = 137
    try:
        output, maximum, total = op(
            query=query.to(device),
            key=pages,
            value=pages,
            sparse_indices=indices.to(device),
            scale_value=1.0,
            block_table=torch.tensor([[1, 0]], dtype=torch.int32, device=device),
            actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32, device=device),
            actual_seq_lengths_kv=torch.tensor([138], dtype=torch.int32, device=device),
            query_rope=query_rope.to(device),
            key_rope=pages,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_KR_BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
        )
    except RuntimeError as exc:
        # An old tiler may win CANN's operator lookup even when the new ACLNN
        # symbols exist. Workspace validation completes before OpCommand queues work.
        message = str(exc).partition("TraceBack (most recent call last):")[0].strip()
        status = re.match(r"call aclnnVllmSfaPairedPageGetWorkspaceSize failed with status (\d+)", message)
        # ACLNN parameter, infer-shape, tiling and kernel lookup failures describe
        # incompatible installed operator components. Runtime/executor failures propagate.
        if status is None or status[1] not in {"161002", "561001", "561002", "561003"}:
            raise
        return message

    # The first half has scores [3,1], the second half [-1,5]. Copying the
    # results to CPU also surfaces asynchronous failures before enabling the layout.
    output, maximum, total = (tensor.cpu().float() for tensor in (output, maximum, total))
    if output.shape != (1, num_heads, 512) or maximum.shape != (1, 1, num_heads) or total.shape != maximum.shape:
        return f"unexpected output/statistic shapes {output.shape}, {maximum.shape}, {total.shape}"
    for heads, expected_max, e0, e1 in (
        (slice(0, midpoint), 3, 1, math.exp(-2)),
        (slice(midpoint, num_heads), 5, math.exp(-6), 1),
    ):
        expected_output = (e0 + 3 * e1) / (e0 + e1)
        for label, actual, expected, atol in (
            ("output", output[:, heads], expected_output, 0.015),
            ("softmax_max", maximum[..., heads], expected_max, 0.005),
            ("softmax_sum", total[..., heads], e0 + e1, 0.005),
        ):
            if not torch.allclose(actual, torch.full_like(actual, expected), atol=atol, rtol=0):
                return f"paired-page {label} mismatch for heads {heads.start}..{heads.stop - 1}"
    return None


@cache
def paired_page_sfa_available(device: torch.device) -> bool:
    """Verify the installed ACLNN, tiler and A5 kernel once per worker device before capture."""
    bootstrap_custom_op_env()
    try:
        extension = import_module("vllm_ascend.vllm_ascend_C")
    except (ImportError, OSError) as exc:
        logger.warning("Kimi K3.1 DCP SFA stage=probe layout=PA_KR_BSND device=%s enabled=False reason=%s", device, exc)
        return False

    ops = torch.ops._C_ascend
    symbols = getattr(ops, "is_sfa_paired_page_available", None)
    op = getattr(ops, "npu_sparse_flash_attention", None)
    if symbols is None or op is None or not op.default.has_kernel_for_dispatch_key("PrivateUse1"):
        reason = "extension lacks the paired-page probe or SFA NPU binding"
    elif not symbols():
        reason = "OPP lacks matching aclnnVllmSfaPairedPage and GetWorkspaceSize symbols"
    else:
        # G=128 (including padded Kimi G=96) and G<=64 use different tiling keys.
        for num_heads in (128, 32):
            reason = _check_paired_page_kernel(op, device, num_heads)
            if reason:
                reason = f"heads={num_heads}: {reason}"
                break
    if reason:
        logger.warning(
            "Kimi K3.1 DCP SFA stage=probe layout=PA_KR_BSND device=%s extension=%s enabled=False reason=%s",
            device,
            extension.__file__,
            reason,
        )
        return False
    logger.info(
        "Kimi K3.1 DCP SFA stage=probe layout=PA_KR_BSND device=%s extension=%s enabled=True "
        "op=aclnnVllmSfaPairedPage heads=128,32 verified=output,softmax_max,softmax_sum",
        device,
        extension.__file__,
    )
    return True
