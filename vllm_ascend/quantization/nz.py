# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Named FRACTAL_NZ cast recipes shared by weight preparation and its consumers.

``npu_format_cast`` derives the C0 tiling of FRACTAL_NZ from the dtypes it is
given, so a weight cast with ``customize_dtype``/``input_dtype`` reports a C0
variant such as FRACTAL_NZ_C0_16 rather than FRACTAL_NZ. ``copy_`` refuses
internal-format tensors whose storage descriptions disagree, so anything that
rebuilds such a weight elsewhere, EPLB above all, has to repeat the same cast.

Keeping the recipes here lets a quant method and the code that rebuilds its
weights share one definition instead of restating the dtypes side by side,
where they can drift apart.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch_npu


@dataclass(frozen=True)
class NzCastSpec:
    """The parameters one FRACTAL_NZ cast was performed with.

    torch_npu exposes some dtypes as integer enums rather than
    ``torch.dtype``, so the fields accept both.
    """

    customize_dtype: torch.dtype | int | None = None
    input_dtype: torch.dtype | int | None = None

    def cast_kwargs(self) -> dict[str, Any]:
        """Keyword arguments reproducing this cast in ``npu_format_cast``."""
        kwargs: dict[str, Any] = {}
        if self.customize_dtype is not None:
            kwargs["customize_dtype"] = self.customize_dtype
        if self.input_dtype is not None:
            kwargs["input_dtype"] = self.input_dtype
        return kwargs


# The dtypes resolve on first use: torch_npu's FP4 enum is absent from
# environments without the NPU extension, and a module-level constant would
# also freeze a dtype that tests substitute.
def fp4_packed_nz() -> NzCastSpec:
    """FP4 weights read as the packed two-values-per-byte view."""
    return NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)


def fp4_semantic_nz() -> NzCastSpec:
    """FP4 weights read with the semantic dtype the SiTU operator requires."""
    return NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
