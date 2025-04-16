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

from typing import Callable, Optional, Union

import torch
import torch_npu
from vllm.spec_decode.metrics import (AsyncMetricsCollector,
                                      SpecDecodeWorkerMetrics)

Timer = Callable[[], float]

# TODO: revert this patch when the cuda hard code is removed in vllm
# init_tensors: Modified the hard-coded cuda judgment logic to npu;
# maybe_collect_rejsample_metrics: Removed the check for current_platform.is_cuda_alike()


def init_tensors(self,
                 rank: int,
                 device_type: Union[torch.device, str] = 'npu') -> None:
    self._rank = rank
    if isinstance(device_type, torch.device):
        device_type = device_type.type
    if device_type == 'npu':
        self._copy_stream = torch_npu.npu.Stream()


def maybe_collect_rejsample_metrics(
        self, k: int) -> Optional[SpecDecodeWorkerMetrics]:

    # If a copy was initiated in the previous call, collect and return.
    if self._in_flight_copy is not None:
        ready_event = self._in_flight_copy
        self._in_flight_copy = None
        return self._collect_rejsample_metrics(k, ready_event)

    # Otherwise, check if we should start a new copy.
    if self._should_collect_rejsample_metrics(self._timer()):
        assert self._in_flight_copy is None
        self._in_flight_copy = self._copy_rejsample_metrics_async()

    return None


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


AsyncMetricsCollector.init_tensors = init_tensors
AsyncMetricsCollector.maybe_collect_rejsample_metrics = maybe_collect_rejsample_metrics
AsyncMetricsCollector._copy_rejsample_metrics_async = _copy_rejsample_metrics_async
