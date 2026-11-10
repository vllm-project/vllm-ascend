#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""FP8 scaled-MM kernels for Ascend, executed as MXFP8 or INT8.

Upstream vLLM's ``Fp8PtpcOnlineLinearMethod`` (per-output-channel weight
scale + dynamic per-token activation scale) needs a platform FP8 scaled-MM
kernel. Ascend has no native scaled-MM kernel that consumes that layout, so
this module maps the PTPC layout onto the existing Ascend W8A8 execution
paths, picking the format the hardware supports:

* ``AscendFP8ScaledMMLinearKernel`` — MXFP8 execution on devices with the
  ``ENGRAM_MXFP8`` capability (A5 / Ascend 950 or later). The PTPC weights
  are resolved into a dense model-dtype matrix, re-quantized to MXFP8 (one
  E8M0 scale per 32 elements along the reduction dim), and handed to the
  MXFP8 weight layout transform; activations are quantized dynamically per
  token with ``npu_dynamic_mx_quant``.
* ``AscendFP8ScaledMMInt8LinearKernel`` — INT8 execution on devices
  without ``ENGRAM_MXFP8`` but with the standard quantization backend
  family (A3 / A2). The PTPC weights are re-quantized to per-channel
  symmetric INT8 and handed to the W8A8 dynamic method's layout transform;
  activations are quantized dynamically per token with
  ``npu_dynamic_quant``.

Both kernels only implement the PTPC QuantKey pair, so registering them on
the ``OOT`` platform cannot accidentally capture other online FP8 methods
(their per-tensor / per-block keys fail ``can_implement``). Their
``is_supported`` gates are mutually exclusive on the ``ENGRAM_MXFP8``
capability, so upstream kernel selection resolves the int8-vs-mxfp8
execution format from the hardware alone; the config interface stays
``draft_lm_head_quantization="fp8"`` (upstream only accepts fp8/nvfp4).

Both paths re-quantize weights that PTPC has already quantized once, so the
copy carries slightly more error than a direct BF16 -> MXFP8/INT8
quantization would. Only the drafter reads the copy, so the extra error can
move the acceptance length but never the target's verification output.
"""

import logging

import torch
import torch_npu
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)

logger = logging.getLogger(__name__)

_MXFP8_GROUP_SIZE = 32

# Bounds the transient dequantization memory to roughly this many rows of K
# float32 values regardless of head size, mirroring
# fp8_block._ROWS_PER_DEQUANT_STEP.
_ROWS_PER_DEQUANT_STEP = 1024

# INT8 symmetric quantization bounds for the per-channel requantization.
_INT8_MIN = -128
_INT8_MAX = 127
# Floor for the per-channel scale so an all-zero row never divides by zero;
# real lm_head channels are orders of magnitude above it.
_MIN_CHANNEL_SCALE = 1e-12


def _ptpc_config_failure(c: FP8ScaledMMLinearLayerConfig) -> str | None:
    """Return why ``c`` is not the PTPC QuantKey pair, or None if it is."""
    if c.weight_quant_key != kFp8StaticChannelSym:
        return "requires a static per-channel weight scale."
    if c.activation_quant_key != kFp8DynamicTokenSym:
        return "requires a dynamic per-token activation scale."
    if c.out_dtype not in (torch.bfloat16, torch.float16):
        return "Only bfloat16/float16 output dtype supported."
    return None


def _iter_ptpc_row_chunks(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
):
    """Yield ``(row_start, row_end, chunk)`` of the resolved PTPC matrix.

    ``weight`` is the PTPC ``[K, N]`` fp8 matrix and ``weight_scale`` the
    ``[N, 1]`` fp32 per-output-channel scale; each ``chunk`` is
    ``[rows, K]`` float32 with the scales already applied, like
    fp8_block.resolve_block_scales (the scales stay float32 so a channel's
    scale is not itself rounded). Chunking bounds the transient to
    ~``_ROWS_PER_DEQUANT_STEP * K * 4`` bytes for any head size.
    """
    weight_t = weight.t()
    n = weight_t.shape[0]
    for row_start in range(0, n, _ROWS_PER_DEQUANT_STEP):
        row_end = min(row_start + _ROWS_PER_DEQUANT_STEP, n)
        chunk = weight_t[row_start:row_end].to(torch.float32)
        scales = weight_scale[row_start:row_end].to(torch.float32)
        yield row_start, row_end, chunk * scales


def _resolve_ptpc_scales(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Expand the PTPC layout into a dense ``out_dtype`` matrix.

    ``weight`` is the PTPC ``[K, N]`` fp8 matrix and ``weight_scale`` the
    ``[N, 1]`` fp32 per-output-channel scale; the result is ``[N, K]``.
    """
    n, k = weight.t().shape
    resolved = torch.empty((n, k), dtype=out_dtype, device=weight.device)
    for row_start, row_end, chunk in _iter_ptpc_row_chunks(weight, weight_scale):
        resolved[row_start:row_end] = chunk
    return resolved


def _requantize_ptpc_to_int8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize the PTPC layout into per-channel symmetric INT8.

    ``weight`` is the PTPC ``[K, N]`` fp8 matrix and ``weight_scale`` the
    ``[N, 1]`` fp32 per-output-channel scale. Returns ``(q, scale)`` where
    ``q`` is ``[N, K]`` int8 and ``scale`` is ``[N, 1]`` ``out_dtype`` —
    the layout ``AscendW8A8DynamicLinearMethod`` consumes from its
    checkpoints before its own ``process_weights_after_loading`` transform.

    Rows are processed in chunks like ``_resolve_ptpc_scales``, so the fp32
    transient stays ~``_ROWS_PER_DEQUANT_STEP * K * 4`` bytes and the only
    full-size buffer is the final int8 weight. The per-channel scale is
    computed in float32, cast to ``out_dtype`` first, and only then used
    for the division, so dequantization with the stored scale reproduces
    the rounded values exactly.
    """
    n, k = weight.t().shape
    q = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    scale = torch.empty((n, 1), dtype=out_dtype, device=weight.device)
    for row_start, row_end, chunk in _iter_ptpc_row_chunks(weight, weight_scale):
        amax = chunk.abs().amax(dim=1, keepdim=True)
        row_scale = (amax / _INT8_MAX).clamp_min(_MIN_CHANNEL_SCALE).to(out_dtype)
        q[row_start:row_end] = (
            torch.round(chunk / row_scale.to(torch.float32))
            .clamp_(_INT8_MIN, _INT8_MAX)
            .to(torch.int8)
        )
        scale[row_start:row_end] = row_scale
    return q, scale


class AscendFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """PTPC FP8 scaled-MM kernel executed through the Ascend MXFP8 path.

    The kernel is selected by upstream ``init_fp8_linear_kernel`` when the
    platform FP8 kernel table (``_POSSIBLE_FP8_KERNELS``) lists it on the OOT
    platform; ``register_ascend_fp8_lm_head_kernel`` (invoked by the
    ``patch_fp8_lm_head_kernel`` worker patch) puts it there.
    ``process_weights_after_loading`` re-quantizes the PTPC-produced weights
    to MXFP8 and delegates the layout transform to
    ``AscendW8A8MXFP8DynamicLinearMethod``; ``apply_weights`` delegates the
    per-token MXFP8 activation quantization and quantized matmul.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "requires an out-of-tree (Ascend NPU) platform."
        # Resolving the profile reads the device configuration, which is only
        # valid once the NPU runtime is initialized; import lazily so this
        # module stays importable in the engine process.
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            get_current_hardware_profile,
        )

        if not get_current_hardware_profile().supports(
            HardwareCapability.ENGRAM_MXFP8
        ):
            return False, (
                "requires an Ascend device with native MXFP8 support "
                "(A5 / 950 or later)."
            )
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        failure = _ptpc_config_failure(c)
        if failure is not None:
            return False, failure
        n, k = c.weight_shape
        if k % _MXFP8_GROUP_SIZE != 0:
            return False, (
                f"requires the weight reduction dim (K={k}) to be a multiple "
                f"of the MXFP8 group size {_MXFP8_GROUP_SIZE}."
            )
        return True, None

    def __init__(self, c: FP8ScaledMMLinearLayerConfig, layer_param_names):
        # The base class builds a QuantFP8 for the activation key. It is not
        # used here (in-kernel MXFP8 quantization), but is cheap and harmless.
        super().__init__(c, layer_param_names)
        self.mxfp8_method = AscendW8A8MXFP8DynamicLinearMethod()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Re-quantize the PTPC layout to MXFP8, then apply the MXFP8 layout.

        PTPC's ``process_weights_after_loading`` has already replaced the
        layer's weight with ``[K, N]`` fp8 and ``weight_scale`` with ``[N, 1]``
        fp32 (per-output-channel). Resolve those back into a dense
        model-dtype matrix, MX-quantize it, and delegate the transpose/NZ
        weight and scale layout to the MXFP8 method.
        """
        weight = getattr(layer, self.layer_param_names[0])
        weight_scale = getattr(layer, self.layer_param_names[1])
        # weight: [K, N] fp8 * weight_scale: [N, 1] -> dense [N, K] model dtype.
        resolved = _resolve_ptpc_scales(
            weight, weight_scale, self.config.input_dtype
        )

        quantized, mx_scale = torch_npu.npu_dynamic_mx_quant(
            resolved,
            dst_type=torch.float8_e4m3fn,
            scale_alg=self.mxfp8_method.dynamic_mx_quant_scale_alg,
        )
        # CANN 9.1 packs two E8M0 scales per entry on a 2D weight, giving
        # [N, K // 64, 2]; the MXFP8 method consumes [N, K // 32]. view() keeps
        # the bytes, reshape() restores the expected layout (see fp8_block).
        mx_scale = mx_scale.view(torch.uint8).reshape(resolved.shape[0], -1)
        replace_parameter(layer, "weight", torch.nn.Parameter(quantized, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(mx_scale, requires_grad=False))
        self.mxfp8_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.mxfp8_method.apply(layer, x, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "AscendFP8ScaledMMLinearKernel overrides apply_weights directly."
        )


class AscendFP8ScaledMMInt8LinearKernel(FP8ScaledMMLinearKernel):
    """PTPC FP8 scaled-MM kernel executed through the Ascend INT8 W8A8 path.

    Selected on Ascend devices without native MXFP8 support (A3 / A2, where
    ``ENGRAM_MXFP8`` is absent and the quantization backend family is
    STANDARD): the OOT FP8 kernel table lists both kernels and their
    ``is_supported`` gates are mutually exclusive on the MXFP8 capability,
    so upstream kernel selection resolves the int8-vs-mxfp8 execution
    format from the hardware alone.

    ``process_weights_after_loading`` re-quantizes the PTPC-produced weights
    to per-channel symmetric INT8 and delegates the transpose/NZ layout
    transform to ``AscendW8A8DynamicLinearMethod``; ``apply_weights``
    delegates the per-token INT8 activation quantization
    (``npu_dynamic_quant``) and the ``npu_quant_matmul`` matmul.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_out_of_tree():
            return False, "requires an out-of-tree (Ascend NPU) platform."
        # Resolving the profile reads the device configuration, which is only
        # valid once the NPU runtime is initialized; import lazily so this
        # module stays importable in the engine process.
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability,
            QuantizationBackendFamily,
            get_current_hardware_profile,
        )

        profile = get_current_hardware_profile()
        if profile.supports(HardwareCapability.ENGRAM_MXFP8):
            return False, (
                "not used on devices with native MXFP8 support; the MXFP8 "
                "execution kernel is selected instead."
            )
        if profile.quantization_backend_family is not QuantizationBackendFamily.STANDARD:
            return False, "requires the standard quantization backend family."
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        failure = _ptpc_config_failure(c)
        if failure is not None:
            return False, failure
        return True, None

    def __init__(self, c: FP8ScaledMMLinearLayerConfig, layer_param_names):
        # The base class builds a QuantFP8 for the activation key. It is not
        # used here (in-kernel INT8 quantization), but is cheap and harmless.
        super().__init__(c, layer_param_names)
        self.int8_method = AscendW8A8DynamicLinearMethod()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Re-quantize the PTPC layout to INT8, then apply the W8A8 layout.

        PTPC's ``process_weights_after_loading`` has already replaced the
        layer's weight with ``[K, N]`` fp8 and ``weight_scale`` with ``[N, 1]``
        fp32 (per-output-channel). Re-quantize those into per-channel
        symmetric INT8 — ``weight [N, K]`` int8 + ``weight_scale [N, 1]``
        model dtype + a zero ``weight_offset``, the layout the W8A8 dynamic
        method expects from its checkpoints — and delegate the transpose/NZ
        weight and scale-flatten transform to the INT8 method.
        """
        weight = getattr(layer, self.layer_param_names[0])
        weight_scale = getattr(layer, self.layer_param_names[1])
        q, scale = _requantize_ptpc_to_int8(
            weight, weight_scale, self.config.input_dtype
        )
        replace_parameter(layer, "weight", torch.nn.Parameter(q, requires_grad=False))
        replace_parameter(layer, "weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        # The W8A8 dynamic transform reads weight_offset; PTPC only creates
        # weight/weight_scale, so register the symmetric zero offset here.
        replace_parameter(
            layer,
            "weight_offset",
            torch.nn.Parameter(torch.zeros_like(scale), requires_grad=False),
        )
        self.int8_method.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.int8_method.apply(layer, x, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "AscendFP8ScaledMMInt8LinearKernel overrides apply_weights directly."
        )


def register_ascend_fp8_lm_head_kernel() -> bool:
    """Register the Ascend kernels on the OOT platform for PTPC online FP8.

    Upstream ``init_fp8_linear_kernel`` selects a platform FP8 kernel from
    ``_POSSIBLE_FP8_KERNELS``, which has no entry for the OOT (Ascend)
    platform; without one, ``Fp8PtpcOnlineLinearMethod`` (used by the
    quantized draft lm_head copy, ``draft_lm_head_quantization="fp8"``) raises
    ``ValueError``. This uses upstream's own OOT registration API
    (``register_linear_kernel`` with ``kernel_type="fp8"`` on
    ``PlatformEnum.OOT``) so the upstream platform FP8 kernel table selects
    the kernels; ``can_implement`` only matches the PTPC QuantKey pair
    (static per-channel weight + dynamic per-token activation), so other
    online FP8 methods are never captured. The two kernels' ``is_supported``
    gates are mutually exclusive on the ``ENGRAM_MXFP8`` hardware
    capability, so selection resolves the int8-vs-mxfp8 execution format
    from the device: MXFP8 on A5 (Ascend 950) and later, INT8 on A3/A2.

    It also makes ``fp8_scaled_mm_is_batch_invariant`` return True for both
    kernels. Under ``VLLM_BATCH_INVARIANT``, ``Fp8PtpcOnlineLinearMethod.apply``
    otherwise falls back to a BF16 dequant path that assumes the PTPC
    ``[K, N]`` weight + ``[N, 1]`` scale layout, which no longer holds after
    the kernels rewrite the layer to the MXFP8/INT8 layouts. Both paths
    quantize activations with per-token scales (MXFP8 group scales, INT8
    per-token amax), which are batch-invariant, so the kernels may safely
    run directly. Both the ``vllm.model_executor.kernels.linear`` module
    attribute and the ``online.fp8`` module binding must be replaced:
    ``online.fp8`` imports the function at module import time, which happens
    before worker patches run.

    Returns:
        False on vLLM builds without the OOT registration API (< 0.26.0),
        where the OOT FP8 kernel table cannot be extended and
        draft_lm_head_quantization stays unsupported; upstream's kernel
        selection then raises its own ValueError.
    """
    try:
        from vllm.model_executor.kernels.linear import register_linear_kernel
        from vllm.platforms.interface import PlatformEnum
    except ImportError:
        # register_linear_kernel exists since vLLM 0.26.0 ([OOT] Add OOT
        # support for linear kernel, #37989); it is present in both v0.31.0
        # and main.
        logger.debug(
            "vLLM build has no OOT linear-kernel registration API; "
            "draft_lm_head_quantization stays unsupported."
        )
        return False

    register_linear_kernel(
        AscendFP8ScaledMMLinearKernel,
        PlatformEnum.OOT,
        kernel_type="fp8",
    )
    register_linear_kernel(
        AscendFP8ScaledMMInt8LinearKernel,
        PlatformEnum.OOT,
        kernel_type="fp8",
    )

    # ``fp8_scaled_mm_is_batch_invariant`` only exists on vLLM builds that ship
    # the batch-invariance branch in ``online/fp8.py`` (v0.32.0+/main). On
    # v0.31.0 PTPC's apply() never consults it, so nothing needs patching.
    import vllm.model_executor.kernels.linear as _kernels_linear  # noqa: E402

    _fp8_batch_invariant = getattr(_kernels_linear, "fp8_scaled_mm_is_batch_invariant", None)
    if _fp8_batch_invariant is not None:

        def _ascend_fp8_batch_invariant(kernel):
            if isinstance(
                kernel,
                (AscendFP8ScaledMMLinearKernel, AscendFP8ScaledMMInt8LinearKernel),
            ):
                return True
            return _fp8_batch_invariant(kernel)

        _kernels_linear.fp8_scaled_mm_is_batch_invariant = _ascend_fp8_batch_invariant
        try:
            import vllm.model_executor.layers.quantization.online.fp8 as _online_fp8  # noqa: E402

            if hasattr(_online_fp8, "fp8_scaled_mm_is_batch_invariant"):
                _online_fp8.fp8_scaled_mm_is_batch_invariant = _ascend_fp8_batch_invariant
        except ImportError:
            # online/fp8.py may not be imported yet (or at all) on this build;
            # when it is imported later it resolves the function from the
            # kernels.linear module attribute patched above.
            pass

    logger.info(
        "Registered %s (MXFP8 execution) and %s (INT8 execution) on the OOT "
        "platform for FP8 PTPC online quantization "
        "(draft_lm_head_quantization='fp8'); is_supported resolves the "
        "execution format from the hardware.",
        AscendFP8ScaledMMLinearKernel.__name__,
        AscendFP8ScaledMMInt8LinearKernel.__name__,
    )
    return True
