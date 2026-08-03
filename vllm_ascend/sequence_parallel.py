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

from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

DENSE_SP_MIN_TOKENS = 1000
MOE_SP_MIN_TOKENS = 1


class SequenceParallelActivationState(Enum):
    """Ownership of an activation at an SP-aware layer boundary."""

    FULL = "full"
    TP_PARTIAL = "tp_partial"
    SEQUENCE_SHARDED = "sequence_sharded"


_VALID_ACTIVATION_TRANSITIONS = {
    SequenceParallelActivationState.FULL: {
        SequenceParallelActivationState.TP_PARTIAL,
        SequenceParallelActivationState.SEQUENCE_SHARDED,
    },
    SequenceParallelActivationState.TP_PARTIAL: {
        SequenceParallelActivationState.FULL,
        SequenceParallelActivationState.SEQUENCE_SHARDED,
    },
    SequenceParallelActivationState.SEQUENCE_SHARDED: {
        SequenceParallelActivationState.FULL,
    },
}


@dataclass(frozen=True)
class SequenceParallelRuntimeState:
    """Per-forward SP geometry and current activation ownership."""

    active: bool
    world_size: int
    num_tokens: int
    padded_num_tokens: int
    local_num_tokens: int
    pad_size: int
    activation: SequenceParallelActivationState = SequenceParallelActivationState.FULL

    @classmethod
    def create(
        cls,
        *,
        active: bool,
        world_size: int,
        num_tokens: int,
        max_num_tokens: int | None = None,
    ) -> "SequenceParallelRuntimeState":
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if max_num_tokens is None:
            max_num_tokens = num_tokens
        if max_num_tokens < num_tokens:
            raise ValueError("max_num_tokens must not be smaller than num_tokens")
        if active and world_size == 1:
            raise ValueError("active sequence parallelism requires world_size > 1")

        if active:
            padded_num_tokens = ((max_num_tokens + world_size - 1) // world_size) * world_size
            local_num_tokens = padded_num_tokens // world_size
            pad_size = padded_num_tokens - num_tokens
        else:
            padded_num_tokens = num_tokens
            local_num_tokens = num_tokens
            pad_size = 0

        return cls(
            active=active,
            world_size=world_size,
            num_tokens=num_tokens,
            padded_num_tokens=padded_num_tokens,
            local_num_tokens=local_num_tokens,
            pad_size=pad_size,
        )

    def transition_to(
        self,
        activation: SequenceParallelActivationState,
    ) -> "SequenceParallelRuntimeState":
        """Return a new state after validating an explicit layer transition."""
        if activation is self.activation:
            return self
        if not self.active:
            raise ValueError("activation cannot be sharded while sequence parallelism is inactive")
        if activation not in _VALID_ACTIVATION_TRANSITIONS[self.activation]:
            raise ValueError(f"invalid sequence-parallel transition: {self.activation.value} -> {activation.value}")
        return replace(self, activation=activation)


@dataclass(frozen=True)
class SequenceParallelPolicy:
    """Resolved sequence-parallel configuration for the Ascend runtime."""

    enabled: bool
    min_tokens: int
    legacy_flashcomm_enabled: bool
    compile_pass_enabled: bool

    @property
    def configured(self) -> bool:
        return self.legacy_flashcomm_enabled or self.compile_pass_enabled

    def should_shard(self, num_tokens: int) -> bool:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        return self.enabled and num_tokens > 0 and num_tokens >= self.min_tokens


def resolve_sequence_parallel_policy(
    vllm_config: "VllmConfig",
    *,
    legacy_flashcomm_enabled: bool,
) -> SequenceParallelPolicy:
    """Resolve legacy FlashComm1 and compile-pass SP into one policy."""

    model_config = vllm_config.model_config
    pass_config = vllm_config.compilation_config.pass_config
    compile_pass_enabled = bool(model_config is not None and not model_config.enforce_eager and pass_config.enable_sp)
    legacy_flashcomm_enabled = bool(legacy_flashcomm_enabled)
    configured = legacy_flashcomm_enabled or compile_pass_enabled
    tp_size = vllm_config.parallel_config.tensor_parallel_size

    min_tokens = pass_config.sp_min_token_num
    if min_tokens is None:
        is_moe = bool(model_config is not None and model_config.is_moe)
        min_tokens = MOE_SP_MIN_TOKENS if is_moe else DENSE_SP_MIN_TOKENS
    if not isinstance(min_tokens, int) or isinstance(min_tokens, bool):
        raise TypeError("sp_min_token_num must be an integer")
    if min_tokens < 0:
        raise ValueError("sp_min_token_num must be non-negative")

    return SequenceParallelPolicy(
        enabled=configured and tp_size > 1,
        min_tokens=min_tokens,
        legacy_flashcomm_enabled=legacy_flashcomm_enabled,
        compile_pass_enabled=compile_pass_enabled,
    )
