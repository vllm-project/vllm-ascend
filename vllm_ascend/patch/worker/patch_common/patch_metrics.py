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
#

from typing import Callable

import torch
from vllm.spec_decode.metrics import AsyncMetricsCollector

Timer = Callable[[], float]


def _copy_rejsample_metrics_async(self) -> torch.npu.Event:
    """
    TODO: torch.cuda.xxx --> torch.npu.xxx
    Copy rejection/typical-acceptance sampling metrics
    (number of accepted tokens, etc) to CPU asynchronously.

    Returns a NPU event recording when the copy is complete.
    """
    assert self._copy_stream is not None
    self._copy_stream.wait_stream(torch.npu.current_stream())

    with torch.npu.stream(self._copy_stream):
        self._aggregate_num_accepted_tokens.copy_(
            self.spec_decode_sampler.num_accepted_tokens, non_blocking=True)
        self._aggregate_num_emitted_tokens.copy_(
            self.spec_decode_sampler.num_emitted_tokens, non_blocking=True)
        # Number of draft tokens is calculated on CPU, so no copy is
        # required.
        self._aggregate_num_draft_tokens = (
            self.spec_decode_sampler.num_draft_tokens)

    aggregate_metrics_ready = torch.npu.Event()
    aggregate_metrics_ready.record(self._copy_stream)

    return aggregate_metrics_ready


AsyncMetricsCollector._copy_rejsample_metrics_async = _copy_rejsample_metrics_async
