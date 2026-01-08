# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.worker.ubatch_utils import (UBatchSlice, UBatchSlices,
                                         check_ubatch_thresholds)

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.utils import dbo_current_stream
from vllm_ascend.worker.npu_ubatch_wrapper import NPUCoreControlContextManager

logger = init_logger(__name__)


def is_last_ubatch_empty(orig_num_tokens: int,
                         padded_num_tokens: int,
                         num_ubatches: int = 2) -> bool:
    return (padded_num_tokens // num_ubatches) * (num_ubatches -
                                                  1) >= orig_num_tokens


def should_ubatch_across_dp(
        should_ubatch: bool, orig_num_tokens_per_ubatch: int,
        padded_num_tokens_per_ubatch: int, dp_size: int,
        dp_rank: int) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]
    """

    device = current_platform.device_type
    tensor = torch.zeros(3, dp_size, device=device, dtype=torch.int32)
    tensor[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor[2][dp_rank] = 1 if should_ubatch else 0

    from vllm.distributed.parallel_state import get_dp_group
    dist.all_reduce(tensor, group=get_dp_group().device_group)

    result: bool = bool(torch.all(tensor[2] == 1).item())
    if not result:
        return result, None

    orig_num_tokens_tensor = tensor[0, :]
    padded_num_tokens_tensor = tensor[1, :]

    orig_min_num_tokens = int(orig_num_tokens_tensor.min().item())
    padded_max_num_tokens = int(padded_num_tokens_tensor.max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
        logger.debug("Aborting ubatching %s %s", orig_min_num_tokens,
                     padded_max_num_tokens)
        return False, None
    return result, padded_num_tokens_tensor.cpu()


def should_ubatch_with_num_tokens(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    vllm_config: VllmConfig,
) -> tuple[bool, Optional[torch.Tensor]]:
    dp_size = vllm_config.parallel_config.data_parallel_size
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    return should_ubatch_across_dp(should_ubatch, orig_num_tokens_per_ubatch,
                                   padded_num_tokens_per_ubatch, dp_size,
                                   dp_rank)


def get_dp_padding_ubatch(
        num_tokens_unpadded: int, num_tokens_padded: int,
        should_attempt_ubatching: bool,
        vllm_config: VllmConfig) -> tuple[bool, Optional[torch.Tensor]]:
    """
    1. Decides if each DP rank is going to microbatch. Either all ranks
    run with microbatching or none of them do. If this function decides
    not to run with microbatching. It will "abort" meaning that no padding
    information will be returned to the caller. It will return (False, None)

    2. Determines the total number of tokens that each rank will run.
    All ranks will be padded out so that the run with the same number
    of tokens

    Returns: tuple[
        should_ubatch: Are all DP ranks going to microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if should_ubatch if False
    ]

    """
    assert num_tokens_padded >= num_tokens_unpadded
    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        # Early exit.
        tokens_per_ubatch = torch.tensor([num_tokens_padded // 2])
        #return False, None
        return True, tokens_per_ubatch

    # If this DP rank doesn't want to attempt microbatching
    if not should_attempt_ubatching:
        (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
            False, 0, 0, vllm_config)
        assert should_ubatch is False
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    # Round up to the next multiple of two for even divisibility
    num_tokens_padded = round_up(num_tokens_padded, 2)
    num_tokens_per_ubatch = num_tokens_padded // 2
    should_ubatch = True

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_last_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(
            "Empty second µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        should_ubatch = False

    # Note that we compute the number of padded tokens per ubatch
    (should_ubatch, num_tokens_across_dp) = should_ubatch_with_num_tokens(
        should_ubatch, num_tokens_unpadded // 2, num_tokens_per_ubatch,
        vllm_config)
    if not should_ubatch:
        assert num_tokens_across_dp is None
        return should_ubatch, num_tokens_across_dp

    assert num_tokens_across_dp is not None

    max_tokens_across_dp_cpu = int(torch.max(num_tokens_across_dp).item())
    num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                            dp_size,
                                            device="cpu",
                                            dtype=torch.int32)
    return should_ubatch, num_tokens_after_padding

def create_ubatch_slices(num_scheduled_tokens: np.ndarray, token_split_points: list[int], request_level_split: bool = False) \
    -> UBatchSlices:
    cu_num_tokens = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, dtype=np.int32, out=cu_num_tokens[1:])

    ubatch_slices = []
    start_token = 0
    # Add the end point to the split points to make iteration easier
    all_points = token_split_points + [cu_num_tokens[-1]]
    if not request_level_split:
        for end_token in all_points:
            token_slice = slice(start_token, end_token)
            # Determine request slices using exclusive stop semantics
            # Ubatch includes requests whose tokens overlap [start_token, end_token)

            # Start at the request that contains the start_token
            # or the request starting exactly at start_token (if on boundary)
            req_start = int(
                np.searchsorted(cu_num_tokens, start_token, side="right") - 1)

            # Stop at the request that starts at or after end_token
            req_stop = int(
                np.searchsorted(cu_num_tokens, end_token, side="left"))

            req_slice = slice(req_start, req_stop)
            ubatch_slices.append(UBatchSlice(req_slice, token_slice))

            start_token = end_token

    else:
        # if split by requests
        for end_token in all_points:
            req_start = int(
                np.searchsorted(cu_num_tokens, start_token, side="right") - 1)
            req_stop = int(
                np.searchsorted(cu_num_tokens, end_token, side="right") - 1)
            req_slice = slice(req_start, req_stop)
            token_slice = slice(cu_num_tokens[req_start],
                                cu_num_tokens[req_stop])
            ubatch_slices.append(UBatchSlice(req_slice, token_slice))

    return ubatch_slices


def check_enable_ubatch(num_scheduled_tokens_per_request: np.ndarray,
                        num_tokens_unpadded: int,
                        uniform_decode: bool,
                        vllm_config: VllmConfig,
                        moe_comm_type: Optional[MoECommType],
                        request_level_split: bool = False) -> bool:
    parallel_config = vllm_config.parallel_config
    # Check preconditions for microbatching
    should_attempt_ubatching = check_ubatch_thresholds(
        parallel_config,
        num_tokens_unpadded,
        uniform_decode=uniform_decode,
    )

    if not parallel_config.enable_dbo or not should_attempt_ubatching or moe_comm_type == MoECommType.MC2:
        return False

    # check second batch
    num_tokens_padded = round_up(num_tokens_unpadded, 2)

    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_last_ubatch_empty(num_tokens_unpadded, num_tokens_padded):
        logger.debug(
            "Empty second µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        return False

    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        token_split_point = int(num_tokens_unpadded) // 2
    else:
        token_split_point = int(num_tokens_padded) // 2

    total_tokens = int(num_scheduled_tokens_per_request.sum())
    if token_split_point >= total_tokens:
        return False
    if request_level_split:
        # using request-level splitting
        cu_num_tokens = np.zeros(len(num_scheduled_tokens_per_request) + 1,
                                 dtype=np.int32)
        np.cumsum(num_scheduled_tokens_per_request,
                  dtype=np.int32,
                  out=cu_num_tokens[1:])

        request_split_point = int(
            np.searchsorted(cu_num_tokens, token_split_point, side="right") -
            1)
        imbalance_ratio = (
            token_split_point -
            cu_num_tokens[request_split_point]) / cu_num_tokens[-1]
        if len(num_scheduled_tokens_per_request) == 1 or imbalance_ratio > 0.5:
            return False
        # first/second batch should not be empty
        if request_split_point == 0 or request_split_point == len(
                cu_num_tokens) - 1:
            return False

    return True


def maybe_create_ubatch_slices(
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_unpadded: int,
    vllm_config: VllmConfig,
    request_level_split: bool = False,
) -> Optional[UBatchSlices]:
    """
    Coordinates amongst all DP ranks to determine if and how the full batch
    should be split into microbatches.

    Returns: tuple[
        ubatch_slices: if this is set then all DP ranks have agreed to 
        microbatch
        num_tokens_after_padding: A tensor containing the total number of
        tokens per-microbatch for each DP rank including padding. Will be
        None if ubatch_slices is None
    ]

    """
    if hasattr(vllm_config.parallel_config, 'num_ubatches'):
        num_ubatches = vllm_config.parallel_config.num_ubatches
    else:
        num_ubatches = 2
    num_tokens_padded = round_up(num_tokens_unpadded, num_ubatches)

    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        token_split_point = int(num_tokens_unpadded) // num_ubatches
    else:
        token_split_point = int(num_tokens_padded) // num_ubatches

    token_split_points = [
        token_split_point * i for i in range(1, num_ubatches)
    ]
    ubatch_slices = create_ubatch_slices(num_scheduled_tokens_per_request,
                                         token_split_points,
                                         request_level_split)
    return ubatch_slices


def create_core_control_context(aic_core: int, aiv_core: int):
    comm_aic_core = aic_core
    comm_aiv_core = aiv_core
    current_stream = dbo_current_stream()

    return NPUCoreControlContextManager(comm_aiv_core=comm_aiv_core,
                                        comm_aic_core=comm_aic_core,
                                        curren_stream=current_stream)
