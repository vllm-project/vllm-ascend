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
"""FP8 scaled-MM kernel for Ascend, backed by the MXFP8 execution path.

Upstream vLLM's ``Fp8PtpcOnlineLinearMethod`` (per-output-channel weight scale
+ dynamic per-token activation scale) needs a platform FP8 scaled-MM kernel.
Ascend has no native scaled-MM kernel that consumes that layout, so this kernel
maps the PTPC layout onto the MXFP8 path already used by
``AscendW8A8MXFP8DynamicLinearMethod``:

* weights: PTPC produces ``weight [K, N]`` fp8 + ``weight_scale [N, 1]`` fp32.
  The scale is resolved into a dense model-dtype matrix, re-quantized to MXFP8
  (one E8M0 scale per 32 elements along the reduction dim), and handed to the
  MXFP8 weight layout transform.
* activations: quantized dynamically per token with ``npu_dynamic_mx_quant``,
  matching MXFP8's per-token scale semantics.

The remap re-quantizes weights that PTPC has already quantized once, so the
copy carries slightly more error than a direct BF16 -> MXFP8 quantization
would. Only the drafter reads the copy, so the extra error can move the
acceptance length but never the target's verification output.

This kernel only implements the PTPC QuantKey pair, so registering it on the
``OOT`` platform cannot accidentally capture other online FP8 methods (their
per-tensor / per-block keys fail ``can_implement``). ``is_supported`` gates on
the ``ENGRAM_MXFP8`` hardware capability: the execution path runs on the
native MXFP8 matmul, available on A5 (Ascend 950) and later devices.
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

from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DynamicLinearMethod,
)

logger = logging.getLogger(__name__)

_MXFP8_GROUP_SIZE = 32

# Bounds the transient dequantization memory to roughly this many rows of K
# float32 values regardless of head size, mirroring
# fp8_block._ROWS_PER_DEQUANT_STEP.
_ROWS_PER_DEQUANT_STEP = 1024


def _resolve_ptpc_scales(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Expand the PTPC layout into a dense ``out_dtype`` matrix.

    ``weight`` is the PTPC ``[K, N]`` fp8 matrix and ``weight_scale`` the
    ``[N, 1]`` fp32 per-output-channel scale; the result is ``[N, K]``.
    Rows are processed in chunks and the scales stay in float32, like
    fp8_block.resolve_block_scales, so a channel's scale is not itself
    rounded and the transient stays ~``_ROWS_PER_DEQUANT_STEP * K * 4`` bytes
    for any head size.
    """
    weight_t = weight.t()
    n, k = weight_t.shape
    resolved = torch.empty((n, k), dtype=out_dtype, device=weight.device)
    for row_start in range(0, n, _ROWS_PER_DEQUANT_STEP):
        row_end = min(row_start + _ROWS_PER_DEQUANT_STEP, n)
        chunk = weight_t[row_start:row_end].to(torch.float32)
        scales = weight_scale[row_start:row_end].to(torch.float32)
        resolved[row_start:row_end] = chunk * scales
    return resolved


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
        if c.weight_quant_key != kFp8StaticChannelSym:
            return False, "requires a static per-channel weight scale."
        if c.activation_quant_key != kFp8DynamicTokenSym:
            return False, "requires a dynamic per-token activation scale."
        if c.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "Only bfloat16/float16 output dtype supported."
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


def register_ascend_fp8_lm_head_kernel() -> bool:
    """Register this kernel on the OOT platform for PTPC online FP8.

    Upstream ``init_fp8_linear_kernel`` selects a platform FP8 kernel from
    ``_POSSIBLE_FP8_KERNELS``, which has no entry for the OOT (Ascend)
    platform; without one, ``Fp8PtpcOnlineLinearMethod`` (used by the
    quantized draft lm_head copy, ``draft_lm_head_quantization="fp8"``) raises
    ``ValueError``. This uses upstream's own OOT registration API
    (``register_linear_kernel`` with ``kernel_type="fp8"`` on
    ``PlatformEnum.OOT``) so the upstream platform FP8 kernel table selects
    it; ``can_implement`` only matches the PTPC QuantKey pair (static
    per-channel weight + dynamic per-token activation), so other online FP8
    methods are never captured.

    It also makes ``fp8_scaled_mm_is_batch_invariant`` return True for this
    kernel. Under ``VLLM_BATCH_INVARIANT``, ``Fp8PtpcOnlineLinearMethod.apply``
    otherwise falls back to a BF16 dequant path that assumes the PTPC
    ``[K, N]`` weight + ``[N, 1]`` scale layout, which no longer holds after
    the kernel rewrites the layer to the MXFP8 layout. MXFP8 per-token
    activation scales are batch-invariant, so the kernel may safely run
    directly. Both the ``vllm.model_executor.kernels.linear`` module attribute
    and the ``online.fp8`` module binding must be replaced: ``online.fp8``
    imports the function at module import time, which happens before worker
    patches run.

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

    # ``fp8_scaled_mm_is_batch_invariant`` only exists on vLLM builds that ship
    # the batch-invariance branch in ``online/fp8.py`` (v0.32.0+/main). On
    # v0.31.0 PTPC's apply() never consults it, so nothing needs patching.
    import vllm.model_executor.kernels.linear as _kernels_linear  # noqa: E402

    _fp8_batch_invariant = getattr(_kernels_linear, "fp8_scaled_mm_is_batch_invariant", None)
    if _fp8_batch_invariant is not None:

        def _ascend_fp8_batch_invariant(kernel):
            if isinstance(kernel, AscendFP8ScaledMMLinearKernel):
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
        "Registered %s on the OOT platform for FP8 PTPC online quantization "
        "(draft_lm_head_quantization='fp8').",
        AscendFP8ScaledMMLinearKernel.__name__,
    )
    return True
