# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
from importlib import import_module

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.config import get_current_vllm_config_or_none

from vllm_ascend.quantization.quant_type import QuantType

COMM_STREAM = None

_CANN_ACL_INT8 = 258
_CANN_ACL_INT4 = 285
_CANN_MEGA_MOE_QUANT_MODE_None = 0
_CANN_MEGA_MOE_QUANT_MODE_INT8 = 2


def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.npu.current_device(),
        )

    if event:
        # multi stream wait event
        global COMM_STREAM
        if COMM_STREAM is None:
            COMM_STREAM = torch_npu.npu.Stream(device=torch.npu.current_device())
        with torch_npu.npu.stream(COMM_STREAM):
            event.wait()
            handle = dist.all_to_all_single(
                a2a_out,
                input_.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
                async_op=True,
            )
    else:
        handle = dist.all_to_all_single(
            a2a_out,
            input_.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    return input_, a2a_out, handle


def _gather_along_first_dim(input_, group, output_split_sizes=None):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor):
            A tensor to be gathered.
        output_split_sizes (List[int], optional):
            A list specifying the sizes of the output splits along the first dimension.
            If None, equal splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """
    world_size = torch.distributed.get_world_size(group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.npu.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def gather_from_sequence_parallel_region(
    input_,
    group,
    output_split_sizes=None,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    return _gather_along_first_dim(input_, group, output_split_sizes)


def load_cann_mega_moe_ops():
    ops_module = import_module("cann_ops_transformer.ops")
    get_symm_buffer_for_mega_moe = ops_module.get_symm_buffer_for_mega_moe
    mega_moe = ops_module.mega_moe
    return get_symm_buffer_for_mega_moe, mega_moe


def _get_cann_mega_moe_quant_settings(quant_type: QuantType) -> tuple[int, int | None, int | None]:
    # Returns (dispatch_quant_mode, dispatch_quant_out_dtype, weight_type).
    # The current custom op package still requires explicit INT4 for W4A8
    # packed weights; otherwise it derives W4A8's packed N as an INT8 N and
    # rejects weight2.
    #
    # dispatch_quant_out_dtype: the doc types this as torch.dtype (torch.int8 /
    # torch.float8_e4m3fn). We pass the ACL enum ints (258 / 24) because W8A8
    # was validated end-to-end this way in PD; switching W4A8 to torch.int8 did
    # NOT fix the W4A8 accuracy issue and slowed graph capture (see bug_a3.md),
    # so keep the working values until the W4A8 accuracy root cause is found on
    # the operator side.
    if quant_type == QuantType.W8A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT8)
    if quant_type == QuantType.W4A8:
        return (_CANN_MEGA_MOE_QUANT_MODE_INT8, _CANN_ACL_INT8, _CANN_ACL_INT4)
    if quant_type == QuantType.NONE:
        return (_CANN_MEGA_MOE_QUANT_MODE_None, None, None)
    raise RuntimeError(
        f"MegaMoe integration supports W8A8/W4A8/BF16 on A2/A3 MegaMoe platforms. Unsupported quant type: {quant_type}."
    )


# Token-dim padding via `torch.cat` with a persistent zero block, replacing
# `nn.functional.pad` in the MoE prepare paths. On NPU each `F.pad` lowers to
# two device kernels (a full-output MemSet plus a PadV3 copy) while the padded
# data is only a few KB in decode, so the pair is pure launch overhead
# (~44us per MoE layer in a Kimi K3 TP16 decode profile). The cat variant is
# one ConcatD kernel per tensor and is safe under cudagraph replay: the cat
# output is a fresh tensor per call (each captured graph owns its output
# buffer), and the zero block is never written after allocation, so the
# zero-tail property is structural — no cross-call bookkeeping.

# (width..., dtype, device) -> zero block of [tp_size, width...]. Both MoE
# prepare paths pad to a multiple of tp_size by fewer than tp_size rows (MC2
# pads the local token count up to the DP-uniform padded_num_tokens; All2All
# pads up to exactly tp_size), so one tp_size-row block per tensor width
# serves every pad and is allocated exactly once — never replaced or freed,
# which is what makes it safe for captured graphs to reference.
#
# Outside a worker context there is no current vllm config (e.g. unit tests
# calling prepare() directly): tp_size then reads as 0 and every pad takes
# the F.pad fallback below.
#
# A pad wider than tp_size can only happen outside that invariant (e.g. an
# eager call with zero tokens); it falls back to plain `nn.functional.pad`
# instead of growing the entry, keeping the cache static.
#
# Memory footprint: the key drops the token dim, so the cache holds one entry
# per distinct (trailing shape, dtype, device) — a set fixed by the model
# config, independent of batch size and cudagraph capture sizes (two entries
# for Kimi K3 TP16: hidden 3584, experts 896), each tp_size * width *
# dtype.itemsize bytes, so the whole cache peaks at O(0.1) MB.
_PAD_ZERO_BLOCKS: dict[tuple, torch.Tensor] = {}


def _pad_tokens_with_cat(x: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Token-dim padding of `x` ([n, ...] -> [padded_len, ...]) by concatenating
    a slice of a cached zero block: value-equivalent to
    `F.pad(x, (0, 0, 0, padded_len - n))` at one kernel instead of two."""
    pad_rows = padded_len - x.shape[0]
    assert pad_rows >= 0, f"padded_len ({padded_len}) is smaller than the input's token dim ({x.shape[0]})"
    vllm_config = get_current_vllm_config_or_none()
    tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config is not None else 0
    if pad_rows > tp_size:
        return nn.functional.pad(x, (0, 0, 0, pad_rows))
    key = (*x.shape[1:], x.dtype, str(x.device))
    zero_block = _PAD_ZERO_BLOCKS.get(key)
    if zero_block is None:
        zero_block = torch.zeros((tp_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        _PAD_ZERO_BLOCKS[key] = zero_block
    return torch.cat([x, zero_block[:pad_rows]], dim=0)
