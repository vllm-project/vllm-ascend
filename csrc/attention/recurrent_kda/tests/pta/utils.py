# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""
Utility functions for accuracy testing.
"""

import torch


def compare_tensors_by_ratio(
    golden: torch.Tensor,
    actual: torch.Tensor,
    name: str = "tensor",
    rtol: float = 0.01,
    atol: float = 0.004,
) -> bool:
    """
    Compare two tensors using combined tolerance: pass if |diff| <= atol + rtol * |golden|.
    Default: rtol=1%, atol=0.004 (roughly 1 ULP for bf16 near zero).
    Returns True if all points pass.
    """
    if golden.shape != actual.shape:
        print(f"  [{name}] shape mismatch: golden {golden.shape} vs actual {actual.shape}")
        return False

    golden_f = golden.float()
    actual_f = actual.float()
    diff = torch.abs(golden_f - actual_f)
    threshold = atol + rtol * torch.abs(golden_f)
    mask = diff > threshold
    failed_count = int(mask.sum().item())
    total_count = golden.numel()

    max_abs = diff.max().item()
    # Relative error only where golden is significant
    significant = torch.abs(golden_f) > atol
    if significant.any():
        rel_on_sig = diff[significant] / torch.abs(golden_f[significant])
        max_rel_sig = rel_on_sig.max().item()
        mean_rel_sig = rel_on_sig.mean().item()
    else:
        max_rel_sig = 0.0
        mean_rel_sig = 0.0

    if failed_count == 0:
        print(
            f"  [{name}] PASS  total={total_count}  "
            f"max_abs={max_abs:.6f}  "
            f"max_rel(significant)={max_rel_sig:.6f}  "
            f"mean_rel(significant)={mean_rel_sig:.6f}"
        )
        return True
    else:
        print(
            f"  [{name}] FAIL  total={total_count}  "
            f"failed={failed_count} ({failed_count / total_count * 100:.4f}%)  "
            f"max_abs={max_abs:.6f}  "
            f"max_rel(significant)={max_rel_sig:.6f}"
        )
        failed_indices = torch.nonzero(mask, as_tuple=False)
        for i in range(min(5, failed_count)):
            idx = tuple(failed_indices[i].tolist())
            g = golden_f[idx].item()
            a = actual_f[idx].item()
            d = diff[idx].item()
            t = threshold[idx].item()
            print(f"    pos{idx}: golden={g:.8f}  actual={a:.8f}  diff={d:.8f}  threshold={t:.8f}")
        return False
