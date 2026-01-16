# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import numpy as np
from vllm.config import VllmConfig
from vllm.logger import init_logger
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


# This pads the last ubatch slice out to the total number of tokens
# (num_tokens + padding) since we do `create_ubatch_slices` before applying DP padding.
def _pad_out_ubatch_slices(ubatch_slices: UBatchSlices, num_total_tokens: int,
                           num_reqs_padded: int) -> UBatchSlices:
    last_slice = ubatch_slices[-1]
    padded_last_request_slice = slice(last_slice.request_slice.start,
                                      num_reqs_padded)
    padded_last_token_slice = slice(last_slice.token_slice.start,
                                    num_total_tokens)

    return ubatch_slices[:-1] + [
        UBatchSlice(padded_last_request_slice, padded_last_token_slice)
    ]


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
                        num_tokens_padded: int,
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

    if hasattr(vllm_config.parallel_config, 'num_ubatches'):
        num_ubatches = vllm_config.parallel_config.num_ubatches
    else:
        num_ubatches = 2
    # Sanity Check that the existing padding isn't giving us an empty second
    # ubatch. Abort if so
    if is_last_ubatch_empty(num_tokens_unpadded, num_tokens_padded,
                            num_ubatches):
        logger.debug(
            "Empty last µbatch detected: unpadded tokens: %s, padded "
            "tokens: %s", num_tokens_unpadded, num_tokens_padded)
        return False

    dp_size = vllm_config.parallel_config.data_parallel_size
    if dp_size == 1:
        token_split_point = int(num_tokens_unpadded) // 2
    else:
        token_split_point = int(num_tokens_padded) // 2

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
    should_ubatch: bool,
    num_scheduled_tokens_per_request: np.ndarray,
    num_tokens_padded: int,
    num_reqs_padded: int,
    vllm_config: VllmConfig,
    request_level_split: bool = False,
) -> tuple[UBatchSlices | None, UBatchSlices | None]:
    if not should_ubatch:
        return None, None

    if hasattr(vllm_config.parallel_config, 'num_ubatches'):
        num_ubatches = vllm_config.parallel_config.num_ubatches
    else:
        num_ubatches = 2

    split_point = int(num_tokens_padded) // num_ubatches
    token_split_points = [split_point * i for i in range(1, num_ubatches)]

    ubatch_slices = create_ubatch_slices(num_scheduled_tokens_per_request,
                                         token_split_points,
                                         request_level_split)

    ubatch_slices_padded = _pad_out_ubatch_slices(ubatch_slices,
                                                  num_tokens_padded,
                                                  num_reqs_padded)
    return ubatch_slices, ubatch_slices_padded


def create_core_control_context(aic_core: int, aiv_core: int):
    comm_aic_core = aic_core
    comm_aiv_core = aiv_core
    current_stream = dbo_current_stream()

    return NPUCoreControlContextManager(comm_aiv_core=comm_aiv_core,
                                        comm_aic_core=comm_aic_core,
                                        curren_stream=current_stream)
