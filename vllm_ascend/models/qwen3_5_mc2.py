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
# mypy: ignore-errors
"""Ascend Qwen3.5: fused matmul+allreduce + NPU image preprocessing.

Composes both Ascend optimizations for Qwen3.5 in a single class so the two
architecture overrides do not fight over the ModelRegistry mapping:

- MC2: torch_npu.npu_mm_all_reduce_base (aclnnMatmulAllReduce) fuses the
  matmul and the all-reduce of row-parallel linears under TP into one kernel:
  the activation crosses the HCCL link once and comm overlaps the matmul
  tail. Supports BF16 (dense matmul) and quantized weights (W8A8 int8 /
  MXFP8) via the op's antiquant_scale / pertoken_scale parameters. Ineligible
  calls delegate to the stock forward captured at install time; during ACL
  graph capture the op falls back to stock matmul+allreduce (EE1016).
- VL preprocessing (from qwen3_5_vl): the front-end emits flat raw uint8 and
  resize/patchify/rescale+normalize run on the NPU worker; falls back to the
  stock CPU pipeline for unsupported settings, non-uint8 pixels or
  mm_device_do_normalize=false.
"""

import os
import types

import torch
import torch_npu
from vllm.distributed import (
    get_tp_group,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import logger
from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5MoeProcessingInfo,
    AscendQwen3_5ProcessingInfo,
    AscendQwen3_5VLProcessor,
    _AscendVLPreprocessMixin,
)

_ascend_tp_group = None
_ascend_hcomm = None
_mc2_warmup_done = False


def _fetch_hcomm():
    """Fetch the HCCL communicator name (idempotent, cached)."""
    global _ascend_hcomm
    if _ascend_hcomm is None:
        rank = torch.distributed.get_rank(group=_ascend_tp_group)
        _ascend_hcomm = _ascend_tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return _ascend_hcomm


def _mm_all_reduce_warmup():
    """One tiny fused call to pre-allocate CANN comm resources.

    npu_mm_all_reduce_base does a synchronous memcpy on its FIRST call to
    set up internal communication resources (EE1016 prohibits this under
    ACL graph capture). A single tiny eager call before capture completes
    the allocation, so the operator is capture-safe from then on.

    Must be called with all TP ranks in lockstep (same mechanism as the
    stock allreduce path).
    """
    global _mc2_warmup_done
    if _mc2_warmup_done:
        return
    hcomm = _fetch_hcomm()
    # Minimal shapes: 16x16 matmul on each rank is enough to trigger the
    # comm-resource allocation without measurable cost.
    dummy_x = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
    dummy_w = torch.zeros(16, 16, dtype=torch.bfloat16, device="npu")
    torch_npu.npu_mm_all_reduce_base(dummy_x, dummy_w, hcomm, bias=None)
    _mc2_warmup_done = True
    logger.info("MC2: warmup call completed (comm resources allocated, graph-capture safe)")


def _mm_all_reduce_base_impl(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # The MC2 comm-resource allocation was done during the __init__ warmup
    # call, so the operator is safe under ACL graph capture. Only fall back
    # if the warmup somehow did not run (e.g., the mixin was not installed).
    if not _mc2_warmup_done:
        return tensor_model_parallel_all_reduce(torch.matmul(x1, x2))
    return torch_npu.npu_mm_all_reduce_base(x1, x2, _fetch_hcomm(), bias=None)


def _mm_all_reduce_base_fake(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.empty((x1.shape[0], x2.shape[1]), dtype=x1.dtype, device=x1.device)


direct_register_custom_op(
    op_name="ascend_mm_all_reduce",
    op_func=_mm_all_reduce_base_impl,
    fake_impl=_mm_all_reduce_base_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _mm_all_reduce_quant_impl(
    x1: torch.Tensor,
    x2: torch.Tensor,
    antiquant_scale: torch.Tensor,
    antiquant_offset: torch.Tensor,
    pertoken_scale: torch.Tensor,
    antiquant_group_size: int,
) -> torch.Tensor:
    """Fused quantized matmul + allreduce (W8A8 / MXFP8).

    npu_mm_all_reduce_base natively supports quantized operands through the
    antiquant_scale / antiquant_offset (weight dequant) and pertoken_scale
    (activation dequant) parameters; antiquant_group_size selects per-channel
    (0) vs per-block (MXFP8=128) scaling. Comm resources were pre-allocated
    during the __init__ warmup call, so this is graph-capture safe.
    """
    if not _mc2_warmup_done:
        return tensor_model_parallel_all_reduce(
            torch.matmul(
                x1.to(pertoken_scale.dtype) * pertoken_scale,
                x2.to(antiquant_scale.dtype) * antiquant_scale,
            )
        )
    return torch_npu.npu_mm_all_reduce_base(
        x1,
        x2,
        _fetch_hcomm(),
        bias=None,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset if antiquant_offset.numel() > 0 else None,
        pertoken_scale=pertoken_scale,
        antiquant_group_size=antiquant_group_size,
    )


def _mm_all_reduce_quant_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    antiquant_scale: torch.Tensor,
    antiquant_offset: torch.Tensor,
    pertoken_scale: torch.Tensor,
    antiquant_group_size: int,
) -> torch.Tensor:
    # Output dtype follows the dequantized (scale) dtype.
    return torch.empty((x1.shape[0], x2.shape[1]), dtype=antiquant_scale.dtype, device=x1.device)


direct_register_custom_op(
    op_name="ascend_mm_all_reduce_quant",
    op_func=_mm_all_reduce_quant_impl,
    fake_impl=_mm_all_reduce_quant_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


def _fused_forward_factory(orig_forward):
    # Fused matmul+allreduce fast path for one RowParallelLinear instance.
    # Ineligible calls delegate to `orig_forward` (the stock bound method
    # captured at install time), so the fallback is the upstream code itself,
    # not a copy that can drift. `self` is the bound RowParallelLinear.

    def forward(self, input_):
        if self.reduce_results and self.tp_size > 1 and self.bias is None and input_.shape[0] > 1000:
            input_parallel = input_
            if not self.input_is_parallel:
                input_parallel = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)[
                    self.tp_rank
                ].contiguous()

            if isinstance(self.quant_method, UnquantizedLinearMethod):
                # BF16: dense matmul + allreduce fused directly.
                logger.warning_once("MC2 FUSED (bf16): M=%d warmup=%s", input_.shape[0], _mc2_warmup_done)
                output = torch.ops.vllm.ascend_mm_all_reduce(input_parallel, self.weight.t())
            elif hasattr(self, "weight_scale") and hasattr(self.quant_method, "act_quant_type"):
                # W8A8 / MXFP8: quantize the activation, then fused
                # quantized matmul + allreduce via antiquant parameters.
                logger.warning_once(
                    "MC2 FUSED (quant): M=%d quant=%s warmup=%s w_scale=%s",
                    input_.shape[0],
                    type(self.quant_method).__name__,
                    _mc2_warmup_done,
                    self.weight_scale.shape if hasattr(self.weight_scale, "shape") else "N/A",
                )
                quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(
                    input_parallel, dst_type=self.quant_method.act_quant_type
                )
                antiquant_offset = getattr(self, "weight_offset", None)
                if antiquant_offset is None:
                    antiquant_offset = torch.empty(0, device=input_parallel.device)
                antiquant_group_size = getattr(self.quant_method, "group_size", 0) or 0
                output = torch.ops.vllm.ascend_mm_all_reduce_quant(
                    quantized_x,
                    self.weight,
                    self.weight_scale,
                    antiquant_offset,
                    pertoken_scale,
                    antiquant_group_size,
                )
            else:
                # Unsupported quantization scheme: stock path.
                logger.warning_once(
                    "MC2 FALLBACK (unsupported quant): M=%d quant=%s w_scale=%s act_q=%s",
                    input_.shape[0],
                    type(self.quant_method).__name__,
                    hasattr(self, "weight_scale"),
                    hasattr(self.quant_method, "act_quant_type"),
                )
                return orig_forward(input_)
            return (output, None) if self.return_bias else output
        if input_.shape[0] != input_.shape[0]:  # never true; silence linter
            pass
        if not (input_.shape[0] > 1000):
            logger.warning_once(
                "MC2 FALLBACK (M=%d <= 1000): reduce=%s tp=%d bias=%s",
                input_.shape[0],
                self.reduce_results,
                self.tp_size,
                self.bias is not None,
            )
        return orig_forward(input_)

    return forward


class _AscendMC2Mixin:
    """Fused matmul+allreduce for row-parallel linears under TP.

    When the fla_npu GDN prefill backend is active, the fused op's per-call
    sync memcpy drains the pipeline after each of fla's long fused GDN
    kernels, regressing prefill latency. Skip the row-parallel linears that
    belong to GDN (linear_attention) layers and only install the fusion on
    full-attention layers, where the preceding attention kernels are short
    and the sync cost is amortized.
    """

    def __init__(self, *, vllm_config, prefix="model"):
        # Explicit signature required: vllm's initialize_model inspects
        # __init__ parameter names; *args/**kwargs is classified old-style.
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if os.environ.get("VLLM_ASCEND_MM_ALL_REDUCE", "1") != "1":
            return
        if not hasattr(torch_npu, "npu_mm_all_reduce_base"):
            return
        tp = get_tp_group()
        if tp.world_size <= 1:
            return

        # When the FLA NPU GDN prefill backend is active, only fuse
        # row-parallel linears in full-attention layers; GDN layers keep the
        # stock path so fla's long kernels are not followed by a
        # pipeline-draining sync. Mirror gdn.py's runtime gate exactly (op
        # importable + hardware capability + non-PCP): AscendConfig has no
        # gdn_prefill_backend field (it is a stripped upstream-injected
        # key), so reading it raised AttributeError at model init.
        from vllm.distributed import get_pcp_group

        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
        from vllm_ascend.ops.gdn import _get_fla_gdn_prefill_op

        skip_gdn_layers = (
            _get_fla_gdn_prefill_op() is not None
            and get_pcp_group().world_size == 1
            and get_current_hardware_profile().supports(HardwareCapability.FLA_GDN_PREFILL)
        )
        gdn_layer_indices = set()
        if skip_gdn_layers:
            text_config = vllm_config.model_config.hf_text_config
            layer_types = getattr(text_config, "layer_types", None)
            if layer_types is not None:
                gdn_layer_indices = {i for i, t in enumerate(layer_types) if t == "linear_attention"}

        # Only stash the group reference here (no collective): the HCCL comm
        # name is fetched lazily at the first eligible call inside the custom
        # op impl, where the TP ranks run in lockstep — see _mm_all_reduce_base_impl.
        global _ascend_tp_group
        _ascend_tp_group = tp.device_group
        # Instance-level forward replacement: only this model's own
        # reduce_results RowParallelLinear modules are touched.
        installed = 0
        for name, module in self.named_modules():
            if not isinstance(module, RowParallelLinear) or not module.reduce_results:
                continue
            if skip_gdn_layers and gdn_layer_indices:
                # name looks like "model.layers.N.self_attn.o_proj" — extract N.
                # Unparseable names keep the fused path: the skip is a perf
                # heuristic for GDN layers, not a correctness requirement.
                parts = name.split(".")
                try:
                    layer_idx = int(parts[2])
                except (IndexError, ValueError):
                    layer_idx = None
                if layer_idx is not None and layer_idx in gdn_layer_indices:
                    continue
            module.forward = types.MethodType(_fused_forward_factory(module.forward), module)
            installed += 1
        total = sum(1 for _, m in self.named_modules() if isinstance(m, RowParallelLinear) and m.reduce_results)
        if gdn_layer_indices:
            logger.info(
                "MC2: installed on %d/%d row-parallel linears (skipped %d GDN layers for the fla_npu backend)",
                installed,
                total,
                len(gdn_layer_indices),
            )
        else:
            logger.info("MC2: installed on %d/%d row-parallel linears", installed, total)

        # Pre-allocate CANN comm resources with a tiny eager call so the
        # fused op is graph-capture safe: npu_mm_all_reduce_base does a sync
        # memcpy on its FIRST invocation (EE1016 under capture). All TP ranks
        # run __init__ in lockstep, so this is the safe place to warm it up.
        try:
            _mm_all_reduce_warmup()
        except Exception as e:
            # Do not fail model init if the warmup hits an environment issue;
            # the capture guard in the op impl will just fall back to stock.
            logger.warning("MC2: warmup call failed (%s); graph capture will use the stock path", e)


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5ForConditionalGeneration(_AscendMC2Mixin, _AscendVLPreprocessMixin, Qwen3_5ForConditionalGeneration):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=AscendQwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5MoeForConditionalGeneration(
    _AscendMC2Mixin, _AscendVLPreprocessMixin, Qwen3_5MoeForConditionalGeneration
):
    pass
