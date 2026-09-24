#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""DEBUG ``[SamplingMeta]`` dump (log-level gated; not RuntimeConfig)."""

from __future__ import annotations

import logging
from typing import Any

from vllm.distributed.parallel_state import get_pp_group

from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)


def log_sampling_meta_debug(runner: Any, req_ids: list[str] | None) -> None:
    """DEBUG ``[SamplingMeta]`` for local-batch reqs (TP0 + last PP).

    Gated by logger level only (no JSON switch). Skips ``.item()`` work when
    DEBUG is off. Soft-fail: never raise into the sample path.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        _emit_sampling_meta_debug(runner, req_ids)
    except Exception as exc:
        logger.debug("[runtime_guard] SamplingMeta log failed: %s", exc, exc_info=True)


def _emit_sampling_meta_debug(runner: Any, req_ids: list[str] | None) -> None:
    if runner is None:
        return
    if int(getattr(runner, "tp_rank", 0)) != 0:
        return
    if not get_pp_group().is_last_rank:
        return

    input_batch = getattr(runner, "input_batch", None)
    if input_batch is None:
        return
    sampling_metadata = getattr(input_batch, "sampling_metadata", None)
    if sampling_metadata is None:
        return

    batch_ids = list(getattr(input_batch, "req_ids", None) or [])
    want = {str(r) for r in (req_ids or []) if r} if req_ids else set(batch_ids)
    if not want:
        return

    for req_idx, req_id in enumerate(batch_ids):
        if not req_id or str(req_id) not in want:
            continue

        temp = sampling_metadata.temperature[req_idx].item() if sampling_metadata.temperature is not None else None
        topk = sampling_metadata.top_k[req_idx].item() if sampling_metadata.top_k is not None else None
        topp = sampling_metadata.top_p[req_idx].item() if sampling_metadata.top_p is not None else None

        freq_pen = sampling_metadata.frequency_penalties[req_idx].item()
        pres_pen = sampling_metadata.presence_penalties[req_idx].item()
        rep_pen = sampling_metadata.repetition_penalties[req_idx].item()

        bad_words = sampling_metadata.bad_words_token_ids
        req_bad_words = bad_words.get(req_idx, []) if bad_words else []
        req_output_tokens = (
            sampling_metadata.output_token_ids[req_idx]
            if sampling_metadata.output_token_ids and req_idx < len(sampling_metadata.output_token_ids)
            else []
        )
        req_spec_tokens = (
            sampling_metadata.spec_token_ids[req_idx]
            if sampling_metadata.spec_token_ids and req_idx < len(sampling_metadata.spec_token_ids)
            else None
        )
        if sampling_metadata.logprob_token_ids:
            req_logprob_tokens = sampling_metadata.logprob_token_ids.get(req_idx, [])
        else:
            req_logprob_tokens = None

        logger.debug(
            "[SamplingMeta] req_id=%s req_idx=%d "
            "dp_rank=%d tp_rank=%d "
            "temperature=%.4f top_k=%s top_p=%.4f "
            "freq_pen=%.4f pres_pen=%.4f rep_pen=%.4f "
            "bad_words_group_num=%d output_tokens_len=%d spec_tokens_len=%s logprob_target_tokens_len=%s "
            "all_greedy=%s all_random=%s max_num_logprobs=%s",
            req_id,
            req_idx,
            runner.dp_rank,
            runner.tp_rank,
            temp if temp is not None else -1,
            topk,
            topp if topp is not None else 1.0,
            freq_pen,
            pres_pen,
            rep_pen,
            len(req_bad_words),
            len(req_output_tokens),
            len(req_spec_tokens) if req_spec_tokens else None,
            len(req_logprob_tokens) if req_logprob_tokens else None,
            sampling_metadata.all_greedy,
            sampling_metadata.all_random,
            sampling_metadata.max_num_logprobs,
        )
