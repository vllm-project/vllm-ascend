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
  tail. Supports BF16 (dense matmul) and quantized weights with float scales
  (W8A8 int8 / W8A8FP8 dynamic; float32 scales are cast to bfloat16 in-flight)
  via the op's antiquant_scale parameters (W8A16 pattern: BF16 activation,
  weight dequantized in-flight). MXFP8's packed uint8 E8M0 scales are rejected
  by the op (EZ1001) and keep the stock path. Ineligible
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
_mc2_comm_ready = False


def _ensure_comm_ready():
    """Lazily allocate CANN comm resources on the first eager call.

    npu_mm_all_reduce_base does a synchronous memcpy on its FIRST invocation
    to set up internal communication resources (EE1016 prohibits this under
    ACL graph capture). vLLM's startup guarantees an eager profile/warmup
    forward BEFORE graph capture, and during that forward all TP ranks run
    in lockstep — so the first real call is the safe place to initialize.
    After that, the operator is graph-capture safe.

    Must NOT be called during model __init__: ranks are skewed across model
    construction (EI0015 RootInfoDetect), and the collective would deadlock.
    """
    global _mc2_comm_ready, _ascend_hcomm
    if _mc2_comm_ready:
        return True
    if torch.npu.is_current_stream_capturing():
        # Captured before any eager call (shouldn't happen in vLLM's
        # profile-run-before-capture sequence, but guard anyway).
        return False
    if _ascend_hcomm is None:
        rank = torch.distributed.get_rank(group=_ascend_tp_group)
        _ascend_hcomm = _ascend_tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    _mc2_comm_ready = True
    logger.info("MC2: comm resources ready (first eager call, graph-capture safe)")
    return True


def _mm_all_reduce_base_impl(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if not _ensure_comm_ready():
        return tensor_model_parallel_all_reduce(torch.matmul(x1, x2))
    return torch_npu.npu_mm_all_reduce_base(x1, x2, _ascend_hcomm, bias=None)


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
    antiquant_group_size: int,
) -> torch.Tensor:
    """Fused weight-quantized matmul + allreduce (W8A16 antiquant pattern).

    npu_mm_all_reduce_base's quantized variant is W8A16: x1 is a BF16/FP16
    activation (NOT int8 — the kernel rejects DT_INT8 for x1), x2 is the
    quantized weight, and antiquant_scale / antiquant_offset dequantize the
    weight in-flight during the fused matmul+allreduce. This skips the
    separate npu_dynamic_quant step and the separate allreduce kernel, at
    the cost of a wider (BF16 vs int8) matmul operand.
    """
    if not _ensure_comm_ready():
        return tensor_model_parallel_all_reduce(torch.matmul(x1, x2.to(x1.dtype) * antiquant_scale.to(x1.dtype)))
    # aclnnWeightQuantMatmulAllReduce requires FP16/BF16 scales (EZ1001); the
    # W8A8FP8 dynamic scheme stores them as float32.
    if antiquant_scale.dtype == torch.float32:
        antiquant_scale = antiquant_scale.to(torch.bfloat16)
    return torch_npu.npu_mm_all_reduce_base(
        x1,
        x2,
        _ascend_hcomm,
        bias=None,
        antiquant_scale=antiquant_scale,
        antiquant_offset=antiquant_offset if antiquant_offset.numel() > 0 else None,
        antiquant_group_size=antiquant_group_size,
    )


def _mm_all_reduce_quant_fake(
    x1: torch.Tensor,
    x2: torch.Tensor,
    antiquant_scale: torch.Tensor,
    antiquant_offset: torch.Tensor,
    antiquant_group_size: int,
) -> torch.Tensor:
    # Output dtype follows the activation (BF16) dtype.
    return torch.empty((x1.shape[0], x2.shape[1]), dtype=x1.dtype, device=x1.device)


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
                output = torch.ops.vllm.ascend_mm_all_reduce(input_parallel, self.weight.t())
            elif hasattr(self, "weight_scale"):
                # W8A16 antiquant pattern: BF16 activation × quantized weight,
                # weight dequantized in-flight by the fused kernel (no
                # separate npu_dynamic_quant or allreduce kernel needed).
                antiquant_offset = getattr(self, "weight_offset", None)
                if antiquant_offset is None:
                    antiquant_offset = torch.empty(0, device=input_parallel.device)
                # group_size lives on the concrete scheme, which may be
                # wrapped by the AscendLinearMethod delegator.
                scheme = getattr(self.quant_method, "quant_method", self.quant_method)
                antiquant_group_size = getattr(scheme, "group_size", 0) or 0
                output = torch.ops.vllm.ascend_mm_all_reduce_quant(
                    input_parallel,
                    self.weight,
                    self.weight_scale,
                    antiquant_offset,
                    antiquant_group_size,
                )
            else:
                # Unsupported quantization scheme: stock path.
                return orig_forward(input_)
            return (output, None) if self.return_bias else output
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
            # W8A16 antiquant gate: aclnnWeightQuantMatmulAllReduce only
            # accepts float scales (EZ1001 rejects DT_UINT8). MXFP8
            # checkpoints store packed uint8 E8M0 scales — those linears keep
            # the stock forward. float32 scales (the W8A8FP8 dynamic scheme)
            # are cast to bfloat16 inside the op impl.
            if not isinstance(module.quant_method, UnquantizedLinearMethod):
                scale = getattr(module, "weight_scale", None)
                if scale is None or scale.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    logger.warning_once(
                        "MC2: quantized weight_scale dtype %s is not supported by the fused "
                        "antiquant op (needs float16/bfloat16; MXFP8 packed uint8 E8M0 scales "
                        "are rejected) — these linears keep the stock path",
                        scale.dtype if scale is not None else "missing",
                    )
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

        # NOTE: comm resources are NOT allocated here. Model __init__ runs
        # with skewed TP ranks (EI0015); a collective here would deadlock.
        # The _ensure_comm_ready() call inside the op impl handles it lazily
        # on the first eager forward (vLLM's profile run), where all ranks
        # are in lockstep and the sync memcpy is safe.


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
