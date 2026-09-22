# SPDX-License-Identifier: Apache-2.0
"""Correct the v0.23.0 fused RMS quantization Meta without changing its kernel."""

import torch


def rms_norm_dynamic_quant_meta(x, gamma, smooth_scale=None, beta=None, epsilon=1e-6):
    return torch.empty_like(x, dtype=torch.int8), x.new_empty(x.shape[:-1], dtype=torch.float32)


# Keep the Library alive. Registration can be retried if an earlier import
# happened before the extension was available.
_META_LIB = None


def register_rms_quant_meta():
    global _META_LIB
    if _META_LIB is not None or not hasattr(torch.ops._C_ascend, "npu_rms_norm_dynamic_quant"):
        return
    lib = torch.library.Library("_C_ascend", "IMPL", "Meta")
    lib.impl("npu_rms_norm_dynamic_quant", rms_norm_dynamic_quant_meta, allow_override=True)
    _META_LIB = lib
