#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""GroupedMatmulSituQuant A5 fused custom op (MXFP8 x MXFP4 GMM + SiTU +
dynamic MX quant, single launch), Ascend950PR (arch35) only.

Fuses the production split chain ``npu_grouped_matmul + situ_mx_quant`` with
bit-exact outputs. The native entries are registered when
``vllm_ascend.vllm_ascend_C`` is imported::

    torch.ops._C_ascend.grouped_matmul_situ_quant(...)
    torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz(...)

each with a ``.list`` overload for per-expert TensorLists.  Prefer this module
over raw torch.ops calls: it pins the supported quant combo, hides nullable
V2-aligned parameters, handles ND/NZ dispatch and provides the one-time
weight NZ cast helper.

Only the MX A8W4 combination is implemented (dequantMode=1 joint MX
data+scale antiquant, dequantDtype=0 BF16 intermediate, quantMode=1 dynamic
MX quant to FP8-E4M3 + E8M0 scale) — the Kimi w4a8 scenario.  ``bias`` /
``smoothScale`` are unsupported by design; other values raise.
"""

import torch

__version__ = "0.1.0"

# Kernel source generation stamp (evolution tree: p0_entries, R3 kernel).
# Bump together with csrc/grouped_matmul_situ_quant/sync_from_p0.sh runs.
KERNEL_STAMP = "gmsq-r3-p0-20260825"

# Parameter values of the implemented MX A8W4 combo (op-level constants; the
# official aclnn enum tables are not published, these are this kernel's
# contract — see csrc/grouped_matmul_situ_quant/op_host/gmm_situ_quant_entries.cpp).
DEQUANT_MODE_MX_JOINT = 1
DEQUANT_DTYPE_BF16 = 0
QUANT_MODE_DYNAMIC_MX = 1


def _soc_supported() -> bool:
    try:
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        return get_ascend_device_type() == AscendDeviceType.A5
    except Exception:
        return False


def _load_native_extension() -> None:
    if not torch.npu.is_available():
        raise RuntimeError("grouped_matmul_situ_quant: NPU is not available")
    if not _soc_supported():
        raise RuntimeError("grouped_matmul_situ_quant requires Ascend950PR (arch35); use the split chain on other SOCs")
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    if not hasattr(torch.ops._C_ascend, "grouped_matmul_situ_quant"):
        raise RuntimeError(
            "grouped_matmul_situ_quant is not present in vllm_ascend_C; "
            "reinstall vllm-ascend with SOC_VERSION=ascend950"
        )


def is_available() -> bool:
    """Whether this SOC is supported and the native extension has the op."""
    if not (torch.npu.is_available() and _soc_supported()):
        return False
    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._C_ascend, "grouped_matmul_situ_quant")


def to_weight_nz(w_nd: torch.Tensor) -> torch.Tensor:
    """One-time model-load cast: ND fp4x2 packed (E, N, K/2) -> NZ form.

    Returns the NZ byte stream the kernel consumes natively ((E, K/2, N) view
    of format-29 storage).  Do this once at weight load and keep the result
    resident; the NZ entry then pays zero per-step conversion.  Requires
    torch_npu.
    """
    import torch_npu

    # NOTE: no allow_internal_format toggling needed — the explicit
    # customize_dtype/input_dtype arguments drive the cast directly (the
    # config flag is accepted-but-reverted to False on current torch_npu
    # builds and the cast works regardless, as verified by the acceptance
    # suite).
    return torch_npu.npu_format_cast(
        w_nd,
        29,
        customize_dtype=torch.float8_e4m3fn,
        input_dtype=torch_npu.float4_e2m1fn_x2,
    ).transpose(1, 2)


def to_weight_nz_list(w_nd: torch.Tensor) -> list[torch.Tensor]:
    """Per-expert NZ cast for the TensorList form of the weight.

    Each element is independently allocated ((1, K/2, N) cast of expert ei's
    (1, N, K/2) slice). The runtime passes these storages directly through an
    ACL dynamic-input address table; no forward-time cat or format cast occurs.
    """
    import torch_npu

    out = []
    for ei in range(w_nd.shape[0]):
        # offset-0 fresh storage required: dim-0 slices keep a storage offset
        # into the stacked tensor and the format cast rejects that.
        elem = w_nd[ei].unsqueeze(0).view(torch.uint8).clone().view(torch.float4_e2m1fn_x2)
        out.append(
            torch_npu.npu_format_cast(
                elem, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch_npu.float4_e2m1fn_x2
            )
        )
    return out


WeightType = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]

def _align_weight_scale(t: torch.Tensor) -> torch.Tensor:
    """Restore the contiguous N-major view of an MX weight scale (zero-copy).

    The loader stores scales as N-major base bytes ([E, N, kb, 2] — the order
    the kernel addresses: ``swishNOffset = nOffset * (K/64)``) and hands the op
    the transposed logical view (E, kb, N, 2).  ``transpose(-3, -2)`` relabels
    that view back onto its contiguous base: same data_ptr, same byte order,
    and aclnn accepts it as contiguous.

    Materializing with ``.contiguous()`` is wrong twice over: it reorders the
    bytes into kb-major (scrambling per-channel scales), and the resulting E8M0
    copy dispatches an AICPU Transpose that CANN does not implement for
    DT_FLOAT8_E8M0FN — an async aicpu exception (507018) that kills the stream
    and is attributed to whatever op runs next.
    """
    t = t.transpose(-3, -2)
    if not t.is_contiguous():
        raise ValueError(
            "weight_scale must be the loader's (E, kb, N, 2) transposed view of "
            "N-major bytes, or an already-contiguous N-major (E, N, kb, 2) "
            f"tensor; got shape {tuple(t.shape)} strides {tuple(t.stride())}"
        )
    return t


def _restore_mxfp_semantic_dtype(tensors: WeightType, semantic_dtype: torch.dtype) -> WeightType:
    """View byte-backed MX tensors with the dtype required by ACLNN.

    Only plain ND-format uint8 tensors can be re-viewed in place. Weights
    that already went through npu_format_cast carry their semantic dtype in
    the cast metadata, and their padded fractal storage no longer matches
    the logical shape, so view(dtype) would raise — pass those through.
    """

    def restore_dtype(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            return tensor
        # Only real ND-format NPU tensors can be checked/re-viewed in place;
        # weights already npu_format_cast-ed (fractal storage) pass through.
        if isinstance(tensor, torch.Tensor):
            import torch_npu

            if int(torch_npu.get_npu_format(tensor)) != int(torch_npu.Format.ND):
                return tensor
        return tensor.view(semantic_dtype)

    if isinstance(tensors, list):
        return [restore_dtype(tensor) for tensor in tensors]
    if isinstance(tensors, tuple):
        return tuple(restore_dtype(tensor) for tensor in tensors)
    return restore_dtype(tensors)


def grouped_matmul_situ_quant(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: WeightType,
    weight_scale: WeightType,
    group_list: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
    group_list_type: int = 1,
    weight_format: str = "nz",
):
    """Fused GroupedMatmul(MXFP8xMXFP4) + SiTU + dynamic MX quant.

    Args:
        x:          fp8_e4m3 (M_cap, K), rows assigned to experts in order.
        x_scale:    e8m0 (M_cap, ceil(K/64), 2) MX scales of ``x``.
        weight:     NZ form (E, K/2, N) fp4x2 [default], ND packed
                    (E, N, K/2), or a per-expert list of either (``.list``).
                    NZ list elements are passed directly through the custom
                    op's dynamic address table; their storage is not
                    concatenated or reformatted.
        weight_scale: e8m0 MX scales, stacked or per-expert list, in either the
            loader's transposed logical view (E, ceil(K/64), N, 2) of N-major
            bytes (non-contiguous; realigned zero-copy here) or an already
            contiguous N-major (E, N, ceil(K/64), 2) tensor.  The kernel reads
            N-major physical bytes — a kb-major contiguous tensor is silently
            wrong and cannot be detected here.
        group_list: int64 (E,) **device** tensor; type1 = per-expert row
            counts, type0 = cumsum.  Device-side decode keeps the op
            graph-capturable (torch.npu.graph replay reads current values).
        beta, linear_beta: SiTU parameters.
        weight_format: "nz" (recommended; weights cast once via to_weight_nz)
            or "nd" (entry-layer cast, per-call cost).

    Returns:
        (output, output_scale): fp8_e4m3 (M_cap, N/2) and e8m0
        (M_cap, ceil((N/2)/64), 2).  Rows of inactive experts (count 0) are
        undefined.
    """
    _load_native_extension()
    if weight_format not in ("nz", "nd"):
        raise ValueError(f"weight_format must be 'nz' or 'nd', got {weight_format!r}")

    # npu_format_cast can leave production packed MX tensors exposed as
    # uint8. Restore their logical dtype at this custom-op boundary, so
    # OpDef and ACLNN see MXFP4/E8M0 rather than ordinary byte tensors.
    # NOTE: must use the NATIVE torch dtypes here — torch_npu's registered
    # float4_e2m1fn_x2/float8_e8m0fnu carry broken itemsize metadata on this
    # torch_npu build, and view() against them always fails with
    # "shape '[296]' is invalid for input of size <N>".
    weight = _restore_mxfp_semantic_dtype(weight, torch.float4_e2m1fn_x2)
    weight_scale = _restore_mxfp_semantic_dtype(weight_scale, torch.float8_e8m0fnu)
    # dynamic_scale arrives from upstream quant as raw uint8 bytes (ND); the
    # split chain covers this with the per_token_scale_dtype kwarg, but the
    # fused op has no such argument, so aclnn would reject DT_UINT8 (EZ1009).
    x_scale = _restore_mxfp_semantic_dtype(x_scale, torch.float8_e8m0fnu)

    is_list = isinstance(weight, (list, tuple))
    # aclnn rejects non-contiguous MX inputs (EZ1009 family), so hand it views
    # whose metadata matches their bytes — zero-copy relabelings, never copies:
    #  - weight: transpose back to the cast-order view — same NZ storage/bytes,
    #    and both tiling and kernel derive geometry from numel + dim0 only;
    #  - weight_scale: restore the contiguous N-major view (see
    #    _align_weight_scale — .contiguous() here scrambles bytes AND crashes
    #    the stream on an unsupported AICPU E8M0 Transpose).
    if not is_list and not weight.is_contiguous():
        weight = weight.transpose(1, 2)
    if isinstance(weight_scale, (list, tuple)):
        weight_scale = [t if t.is_contiguous() else _align_weight_scale(t) for t in weight_scale]
    elif not weight_scale.is_contiguous():
        weight_scale = _align_weight_scale(weight_scale)
    op = (
        torch.ops._C_ascend.grouped_matmul_situ_quant
        if weight_format == "nd"
        else torch.ops._C_ascend.grouped_matmul_situ_quant_weight_nz
    )
    if is_list:
        op = op.list
    return op(
        x,
        weight,
        weight_scale,
        None,  # weightAssistMatrix: accepted-and-ignored by the kernel
        None,  # bias: unsupported by design (real values raise in-kernel)
        x_scale,
        None,  # smoothScale: unsupported by design
        group_list,
        DEQUANT_MODE_MX_JOINT,
        DEQUANT_DTYPE_BF16,
        QUANT_MODE_DYNAMIC_MX,
        group_list_type,
        None,  # tuningConfigOptional: kernel uses its own tiling policy
        beta,
        linear_beta,
    )
