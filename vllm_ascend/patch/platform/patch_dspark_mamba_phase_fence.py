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
"""Fence DSpark padding at a partial Mamba verification boundary.

On a PD decode node, the first locally scheduled token can be the one-token
tail reconstructed after remote prefill. vLLM pads that token to ``1 + K`` for
speculative verification before Mamba alignment is applied. If alignment
clips the padded width, downstream DSpark metadata still describes all K
placeholder drafts while the model runner only sees the clipped physical
chunk.

Keep the upstream Mamba boundary rule intact. When this exact transition is
detected, schedule the real target token only and remove its synthetic draft
placeholders before request data and connector metadata are constructed. The
next scheduler iteration can then form a complete ``1 + K`` decode window.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from vllm.logger import logger

_DEFERRED_REQUEST_IDS_ATTR = "_vllm_ascend_dspark_deferred_padding_request_ids"


def _is_pd_decode_dspark(scheduler: Any) -> bool:
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None:
        return False

    speculative_config = getattr(vllm_config, "speculative_config", None)
    if getattr(speculative_config, "method", None) != "dspark":
        return False

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return False

    kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role is not None:
        return kv_role == "kv_consumer"

    return bool(getattr(kv_transfer_config, "is_kv_consumer", False)) and not bool(
        getattr(kv_transfer_config, "is_kv_producer", False)
    )


def _is_padded_transition_window(
    scheduler: Any,
    request: Any,
    num_new_tokens: int,
    num_new_local_computed_tokens: int,
    num_external_computed_tokens: int,
) -> bool:
    if not _is_pd_decode_dspark(scheduler):
        return False

    num_spec_tokens = int(getattr(scheduler, "num_spec_tokens", 0))
    if num_spec_tokens <= 0 or num_new_tokens != 1 + num_spec_tokens:
        return False

    start = (
        request.num_computed_tokens
        + num_new_local_computed_tokens
        + num_external_computed_tokens
    )
    return request.num_tokens - start == 1


def _make_mamba_split_wrapper(original: Callable[..., int]) -> Callable[..., int]:
    @wraps(original)
    def _patched_mamba_block_aligned_split(
        self,
        request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        clipped_tokens = original(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if clipped_tokens == num_new_tokens or not _is_padded_transition_window(
            self,
            request,
            num_new_tokens,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        ):
            return clipped_tokens

        target_tokens = original(
            self,
            request,
            1,
            num_new_local_computed_tokens,
            num_external_computed_tokens,
        )
        if target_tokens != 1:
            return target_tokens

        deferred_request_ids = getattr(self, _DEFERRED_REQUEST_IDS_ATTR, None)
        if deferred_request_ids is None:
            deferred_request_ids = set()
            setattr(self, _DEFERRED_REQUEST_IDS_ATTR, deferred_request_ids)
        deferred_request_ids.add(request.request_id)
        logger.debug(
            "Deferring DSpark padding for request %s: Mamba alignment clipped "
            "the %d-token window to %d; scheduling the target token first",
            request.request_id,
            num_new_tokens,
            clipped_tokens,
        )
        return target_tokens

    _patched_mamba_block_aligned_split._vllm_ascend_dspark_phase_fence = True  # type: ignore[attr-defined]
    return _patched_mamba_block_aligned_split


def _make_cached_request_data_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def _patched_make_cached_request_data(
        self,
        running_reqs,
        resumed_reqs,
        num_scheduled_tokens,
        spec_decode_tokens,
        req_to_new_blocks,
    ):
        deferred_request_ids = getattr(self, _DEFERRED_REQUEST_IDS_ATTR, None)
        if deferred_request_ids:
            try:
                for request_id in deferred_request_ids:
                    placeholder_tokens = spec_decode_tokens.get(request_id)
                    if (
                        num_scheduled_tokens.get(request_id) == 1
                        and placeholder_tokens
                        and all(token_id == -1 for token_id in placeholder_tokens)
                    ):
                        spec_decode_tokens.pop(request_id)
            finally:
                deferred_request_ids.clear()

        return original(
            self,
            running_reqs,
            resumed_reqs,
            num_scheduled_tokens,
            spec_decode_tokens,
            req_to_new_blocks,
        )

    _patched_make_cached_request_data._vllm_ascend_dspark_phase_fence = True  # type: ignore[attr-defined]
    return _patched_make_cached_request_data


def _apply_patch() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    mamba_split = Scheduler._mamba_block_aligned_split
    if not getattr(mamba_split, "_vllm_ascend_dspark_phase_fence", False):
        Scheduler._mamba_block_aligned_split = _make_mamba_split_wrapper(mamba_split)

    make_cached_request_data = Scheduler._make_cached_request_data
    if not getattr(
        make_cached_request_data,
        "_vllm_ascend_dspark_phase_fence",
        False,
    ):
        Scheduler._make_cached_request_data = _make_cached_request_data_wrapper(
            make_cached_request_data
        )


_apply_patch()
