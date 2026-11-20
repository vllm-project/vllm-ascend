from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class V2MoEBatchState:
    vllm_config: VllmConfig
    has_real_work: bool
    is_profile: bool
    num_actual_tokens: int
    allow_megamoe: bool = True
    batch_contains_decode: bool = True
    prefill_ready: bool = False
    padded_num_tokens: int = 0


class V2MoEPhaseMetadata(NamedTuple):
    num_tokens: int
    has_real_work: bool
    batch_contains_decode: bool
    is_profile: bool
    eligible: bool
    capacity: int


_V2_MOE_BATCH_STATE: ContextVar[V2MoEBatchState | None] = ContextVar("_V2_MOE_BATCH_STATE", default=None)


@contextmanager
def v2_moe_batch_context(
    vllm_config: VllmConfig,
    num_tokens: int,
    *,
    dummy_run: bool,
    is_profile: bool,
    allow_megamoe: bool = True,
):
    """Scope target-only MoE metadata to one V2 execute_model call."""
    state = V2MoEBatchState(
        vllm_config=vllm_config,
        has_real_work=not dummy_run,
        is_profile=is_profile,
        num_actual_tokens=num_tokens if not dummy_run or is_profile else 0,
        allow_megamoe=allow_megamoe,
        batch_contains_decode=not is_profile,
    )
    token = _V2_MOE_BATCH_STATE.set(state)
    try:
        yield
    finally:
        _V2_MOE_BATCH_STATE.reset(token)


def get_v2_moe_batch_state(vllm_config: VllmConfig | None = None) -> V2MoEBatchState | None:
    state = _V2_MOE_BATCH_STATE.get()
    if state is not None and vllm_config is not None and state.vllm_config is not vllm_config:
        return None
    return state


def record_v2_moe_batch_phase(batch_req_state) -> None:
    state = get_v2_moe_batch_state()
    if state is not None and state.has_real_work and batch_req_state is not None:
        _V2_MOE_BATCH_STATE.set(
            replace(
                state,
                num_actual_tokens=batch_req_state.num_tokens,
                batch_contains_decode=not bool(batch_req_state.is_prefilling_np.all()),
            )
        )


def clear_v2_moe_padding(input_buffers, num_tokens: int, padded_num_tokens: int) -> None:
    state = get_v2_moe_batch_state()
    if state is not None and state.prefill_ready and padded_num_tokens > num_tokens:
        input_buffers.input_ids[num_tokens:padded_num_tokens].zero_()
        input_buffers.positions[num_tokens:padded_num_tokens].zero_()


def agree_v2_moe_batch(metadata: list[list[int]], tp_size: int) -> V2MoEBatchState:
    """Agree the phase and physical input size from CPU-group metadata."""
    state = get_v2_moe_batch_state()
    assert state is not None
    peers = [V2MoEPhaseMetadata(*values) for values in metadata]
    active = [peer for peer in peers if peer.has_real_work]
    batch_contains_decode = (
        any(peer.batch_contains_decode for peer in active) if active else not all(peer.is_profile for peer in peers)
    )
    max_tokens = max(peer.num_tokens for peer in peers)
    padded_num_tokens = (max_tokens + tp_size - 1) // tp_size * tp_size
    prefill_ready = (
        not batch_contains_decode
        and padded_num_tokens > 0
        and all(peer.eligible and padded_num_tokens <= peer.capacity for peer in peers)
    )
    state = replace(
        state,
        batch_contains_decode=batch_contains_decode,
        prefill_ready=prefill_ready,
        padded_num_tokens=padded_num_tokens,
    )
    _V2_MOE_BATCH_STATE.set(state)
    return state
