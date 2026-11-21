# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Named FRACTAL_NZ cast recipes shared by weight preparation and its consumers.

``npu_format_cast`` derives the C0 tiling of FRACTAL_NZ from the dtypes it is
given, so a weight cast with ``customize_dtype``/``input_dtype`` reports a C0
variant such as FRACTAL_NZ_C0_16 rather than FRACTAL_NZ. ``copy_`` refuses
internal-format tensors whose storage descriptions disagree, so anything that
rebuilds such a weight elsewhere, EPLB above all, has to repeat the same cast.

Every parameterized cast in the tree goes through a recipe named here, and
performs it through the recipe too, so a quant method and the code that rebuilds
its weights cannot restate the dtypes side by side and drift apart.

A cast whose parameters are read off a destination at runtime, rather than
chosen for a weight role, is not a recipe and stays where it is.
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


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

    def cast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` cast to FRACTAL_NZ with this recipe."""
        return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_NZ, **self.cast_kwargs())

    def cast_(self, tensor: torch.Tensor) -> torch.Tensor:
        """Cast ``tensor`` to FRACTAL_NZ in place with this recipe."""
        torch_npu.npu_format_cast_(tensor, ACL_FORMAT_FRACTAL_NZ, **self.cast_kwargs())
        return tensor


# The dtypes resolve on first use: torch_npu's FP4 enum is absent from
# environments without the NPU extension, and a module-level constant would
# also freeze a dtype that tests substitute.
def fp4_packed_nz() -> NzCastSpec:
    """FP4 weights read as the packed two-values-per-byte view."""
    return NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2)


def fp4_semantic_nz() -> NzCastSpec:
    """FP4 weights read with the semantic dtype the SiTU operator requires."""
    return NzCastSpec(customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)


def fp4_nz_for_dtype(dtype: torch.dtype) -> NzCastSpec:
    """The FP4 recipe matching how ``dtype`` already presents its values.

    For code that rebuilds a weight it did not prepare, and so has the tensor
    rather than the layer that chose between the two recipes.
    """
    return fp4_semantic_nz() if dtype == torch.float4_e2m1fn_x2 else fp4_packed_nz()


def mxfp8_nz() -> NzCastSpec:
    """MXFP8 weights, whose C0 tiling follows the FP8 element size."""
    return NzCastSpec(customize_dtype=torch.float8_e4m3fn)


def bf16_nz() -> NzCastSpec:
    """Weights unpacked to BF16 before their int4 repack."""
    return NzCastSpec(customize_dtype=torch.bfloat16)
