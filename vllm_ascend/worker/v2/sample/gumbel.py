# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/gumbel.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.utils import vllm_version_is


@triton.jit(do_not_specialize=["logits_stride", "vocab_size"])
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    """
    Args:
        logits: Tensor of shape (num_tokens, vocab_size) containing the logits.
        expanded_idx_mapping: Tensor containing the mapping from token index
            to request index of tensor temperature.
        temperature: Tensor containing the temperature value for each request.
    """
    num_tokens, vocab_size = logits.shape
    # BLOCK_SIZE keeps BF16/FP16 logits and the FP32 working vector within
    # the Ascend A2/A3 UB limit of 1572864 bits (192 KB):
    #   32768 * (16 + 32) bits = 1572864 bits
    # The previous value 44032 overflowed UB when vLLM v0.26+ stopped
    # upcasting logits to FP32 upstream and the kernel started receiving
    # BF16/FP16 logits (44032 * 48 bits = 2113536 bits > 1572864).
    if vllm_version_is("0.26.0"):
        BLOCK_SIZE = 32768
    else:
        BLOCK_SIZE = 44032
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_tokens, num_blocks)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        multibuffer=False,
    )


@triton.jit(
    do_not_specialize=[
        "local_argmax_stride",
        "local_max_stride",
        "logits_cache_stride",
        "logits_stride",
        "vocab_size",
    ]
)
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_cache_ptr,
    logits_cache_stride,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
    STORE_PRE_TEMP: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)

    if temp != 0.0 and APPLY_TEMPERATURE:
        if not STORE_PRE_TEMP:
            # Release: cache temperature-applied logits (matches _temperature_kernel).
            logits = logits / temp

    if logits_cache_ptr is not None:
        # Store the logits *before* temperature on main so the rejection
        # sampler divides by the same temperature on load (bitwise-exact).
        # On release the cached logits are already temperature-applied.
        if logits_cache_col_ptr is not None:
            if PER_TOKEN_COL:
                col = tl.load(logits_cache_col_ptr + token_idx)
            else:
                col = tl.load(logits_cache_col_ptr)
        else:
            col = 0
        tl.store(
            logits_cache_ptr + req_state_idx * logits_cache_stride + col * vocab_size + block,
            logits,
            mask=mask,
        )

    if temp != 0.0 and APPLY_TEMPERATURE:
        if STORE_PRE_TEMP:
            # Main: apply temperature after caching the pre-temperature logits.
            logits = logits / temp

    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        # NOTE(Ronald1995): change pos's dtype to tl.int32, because triton-ascend's
        # compiler doesn't support uint64 of pos arg.
        pos = tl.load(pos_ptr + token_idx).to(tl.int32)
        gumbel_seed = tl.randint(seed, pos)

        # NOTE(Ronald1995): r is tl.float64 in vllm, change it to tl.float32,
        # because triton-ascend's compiler does not support float64.
        r = tl.rand(gumbel_seed, block).to(tl.float32)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + token_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + token_idx * local_max_stride + block_idx, value)


def _gumbel_sample(
    logits: torch.Tensor,  # [num_tokens, vocab_size]
    expanded_idx_mapping: torch.Tensor,  # [num_tokens]
    temperature: torch.Tensor,  # [max_num_reqs]
    seed: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [num_tokens]
    apply_temperature: bool,
    logits_cache: torch.Tensor | None,  # [max_num_reqs, num_cols, vocab_size]
    logits_cache_col: torch.Tensor | None,  # scalar or [num_tokens]
    use_fp64: bool,
    store_pre_temp: bool,
) -> torch.Tensor:
    if use_fp64:
        raise NotImplementedError("FP64 Gumbel sampling is not supported on NPU.")
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_tokens,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        PER_TOKEN_COL=per_token_col,
        STORE_PRE_TEMP=store_pre_temp,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


# main2main compat: upstream `gumbel_sample` post-0.26.0 renamed
# `output_processed_logits`/`output_processed_logits_col` to
# `logits_cache`/`logits_cache_col` and now stores the logits *before*
# temperature so the rejection sampler re-applies it on load. The release
# callers pass the old names.
if vllm_version_is("0.26.0"):

    def gumbel_sample(
        logits: torch.Tensor,  # [num_tokens, vocab_size]
        expanded_idx_mapping: torch.Tensor,  # [num_tokens]
        temperature: torch.Tensor,  # [max_num_reqs]
        seed: torch.Tensor,  # [max_num_reqs]
        pos: torch.Tensor,  # [num_tokens]
        apply_temperature: bool,
        output_processed_logits: torch.Tensor | None = None,
        output_processed_logits_col: torch.Tensor | None = None,
        use_fp64: bool = False,
    ) -> torch.Tensor:
        return _gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            output_processed_logits,
            output_processed_logits_col,
            use_fp64,
            store_pre_temp=False,
        )

else:

    def gumbel_sample(  # type: ignore[misc]
        logits: torch.Tensor,  # [num_tokens, vocab_size]
        expanded_idx_mapping: torch.Tensor,  # [num_tokens]
        temperature: torch.Tensor,  # [max_num_reqs]
        seed: torch.Tensor,  # [max_num_reqs]
        pos: torch.Tensor,  # [num_tokens]
        apply_temperature: bool,
        logits_cache: torch.Tensor | None = None,  # [max_num_reqs, num_cols, V]
        logits_cache_col: torch.Tensor | None = None,  # scalar or [num_tokens]
        use_fp64: bool = False,
    ) -> torch.Tensor:
        return _gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature,
            logits_cache,
            logits_cache_col,
            use_fp64,
            store_pre_temp=True,
        )
