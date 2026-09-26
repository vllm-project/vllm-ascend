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
"""Ascend Qwen3.5: fused matmul+allreduce for row-parallel linears under TP.

Uses torch_npu.npu_mm_all_reduce_base (aclnnMatmulAllReduce) to compute both
in one kernel: activation crosses the HCCL link once, comm overlaps the
matmul tail. Ineligible calls delegate to the stock forward captured at
install time; during ACL graph capture the op falls back to stock
matmul+allreduce (EE1016).
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
    Qwen3_5MoeProcessingInfo,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.torch_utils import direct_register_custom_op

_ascend_tp_group = None
_ascend_hcomm = None


def _mm_all_reduce_base_impl(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    # Capture guard must live here (not in outer Python): dynamo specializes
    # outer guards away at compile time, so only this impl runs during ACL
    # capture. The MC2 comm-resource alloc does a sync memcpy, prohibited
    # under capture modes GLOBAL/MAX (EE1016).
    if torch.npu.is_current_stream_capturing():
        return tensor_model_parallel_all_reduce(torch.matmul(x1, x2))
    global _ascend_hcomm
    if _ascend_hcomm is None:
        # Fetch lazily at the first eager call, where the TP ranks run in
        # lockstep (same mechanism as the stock allreduce path). Fetching at
        # model-init time triggered the TP group's HCCL comm creation
        # (hcclCommInitRootInfoConfig) while the ranks were skewed across
        # model construction, failing with EI0015 RootInfoDetect.
        rank = torch.distributed.get_rank(group=_ascend_tp_group)
        _ascend_hcomm = _ascend_tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
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


def _fused_forward_factory(orig_forward):
    # Fused matmul+allreduce fast path for one RowParallelLinear instance.
    # Ineligible calls delegate to `orig_forward` (the stock bound method
    # captured at install time), so the fallback is the upstream code itself,
    # not a copy that can drift. `self` is the bound RowParallelLinear.

    def forward(self, input_):
        if (
            self.reduce_results
            and self.tp_size > 1
            and self.bias is None
            and isinstance(self.quant_method, UnquantizedLinearMethod)
            and input_.shape[0] > 1000
        ):
            input_parallel = input_
            if not self.input_is_parallel:
                input_parallel = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)[
                    self.tp_rank
                ].contiguous()
            output = torch.ops.vllm.ascend_mm_all_reduce(input_parallel, self.weight.t())
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

        # When fla_npu is active, only fuse row-parallel linears in
        # full-attention layers; GDN layers keep the stock path so fla's
        # long kernels are not followed by a pipeline-draining sync.
        from vllm_ascend.ascend_config import get_ascend_config

        skip_gdn_layers = get_ascend_config().gdn_prefill_backend == "fla_npu"
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
                # name looks like "model.layers.N.self_attn.o_proj" — extract N
                parts = name.split(".")
                try:
                    layer_idx = int(parts[2])
                except (IndexError, ValueError):
                    continue
                if layer_idx in gdn_layer_indices:
                    continue
            module.forward = types.MethodType(_fused_forward_factory(module.forward), module)
            installed += 1
        if skip_gdn_layers and gdn_layer_indices:
            total = sum(1 for _, m in self.named_modules() if isinstance(m, RowParallelLinear) and m.reduce_results)
            logger.info(
                "MC2: installed on %d/%d row-parallel linears (skipped %d GDN layers for fla_npu backend)",
                installed,
                total,
                len(gdn_layer_indices),
            )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5ForConditionalGeneration(_AscendMC2Mixin, Qwen3_5ForConditionalGeneration):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoeProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendQwen3_5MoeForConditionalGeneration(_AscendMC2Mixin, Qwen3_5MoeForConditionalGeneration):
    pass
