# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Ascend-local PLE layer contract.

The CUDA implementation added by vLLM PR #53899 uses CUDA IPC and driver
semaphores.  Qwen4Exp still needs the common layer contract when CPU offload
is disabled, so the plugin owns this device-resident implementation instead
of modifying vLLM 0.26.0.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

PLE_CPU_OFFLOAD = False


def is_offload_process() -> bool:
    return False


class PleOffloadLayer(nn.Module, ABC):
    """Base class for device-resident PLE layers on Ascend."""

    @classmethod
    def get_target_device(cls) -> torch.device:
        return torch.device("npu", torch.npu.current_device())

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        return default_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(self, stream: object | None = None) -> None:
        del stream
