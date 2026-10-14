# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speculative KV capacity accounting shared by the Ascend schedulers.

Backport vLLM #59880 through subclass overrides on both pinned vLLM baselines.
Reservations gate admission; they do not allocate blocks.
"""

from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request

from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler as _SchedulerBase
else:
    _SchedulerBase = object


class SpecDecodeReservationMixin(_SchedulerBase):
    """Cooperative overrides for the pinned schedulers' KV admission helpers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        initialize_spec_reservation(self)

    def _spec_decode_step_blocks(self) -> int:
        return spec_step_blocks(self)

    def _request_remaining_blocks(self, request: Request) -> int:
        return request_remaining_blocks(self, request)

    def _inflight_prefill_reserved_blocks(self) -> int:
        reserved_blocks = inflight_reserved_blocks(self)
        # v0.30.0 calls this helper only at async-load admission, without a
        # separate current-request margin. Bridge that legacy call here. Main
        # already adds _spec_decode_step_blocks() in schedule(). Ascend-owned
        # admissions use async_load_reserved_blocks(), bypassing this bridge.
        if vllm_version_is("0.30.0"):
            reserved_blocks += spec_step_blocks(self)
        return reserved_blocks


def initialize_spec_reservation(self) -> None:
    """Cache the configured margin once, including for schedulers without the mixin."""
    if hasattr(self, "_ascend_spec_step_blocks"):
        return
    self._ascend_spec_margin_tokens = (
        1 + self.num_spec_tokens + self.num_lookahead_tokens if self.num_spec_tokens else 0
    )
    self._ascend_spec_step_blocks = sum(
        cdiv(self._ascend_spec_margin_tokens, manager.block_size)
        for manager in self.kv_cache_manager.coordinator.single_type_managers
        if not isinstance(manager.kv_cache_spec, MambaSpec)
    )


def spec_step_blocks(self) -> int:
    """Use each attention manager's effective block size, excluding Mamba states."""
    initialize_spec_reservation(self)
    return self._ascend_spec_step_blocks


def request_remaining_blocks(self, request: Request) -> int:
    initialize_spec_reservation(self)
    full_sequence_tokens = min(request.num_tokens, self.max_model_len)
    # Let each manager account for blocks already held by this request. Mamba
    # manages its speculative states separately, using the main-model length.
    return self.kv_cache_manager.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=full_sequence_tokens + self._ascend_spec_margin_tokens,
        new_computed_blocks=self.kv_cache_manager.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0,
        total_computed_tokens=request.num_computed_tokens,
        num_local_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=full_sequence_tokens,
        apply_admission_cap=True,
    )


def async_load_reserved_blocks(self) -> int:
    """Reserve other in-flight requests plus this admission's decode margin once."""
    # Copied schedulers may have captured the upstream base before platform
    # setup. Call the shared implementation directly, regardless of their MRO.
    return inflight_reserved_blocks(self) + spec_step_blocks(self)


def inflight_reserved_blocks(self) -> int:
    """Count only requests already holding blocks, on either upstream baseline."""
    return sum(request_remaining_blocks(self, request) for request in self._inflight_prefills)
