# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend asynchronous EPLB execution with zero-change layer elision."""

# TODO(upstream-eplb): Delete this local worker fork once vLLM exposes hooks
# for changed-layer selection and explicit zero-transfer cycle completion.
# Until then, changes to the upstream async-worker protocol must be mirrored
# here and covered by contract tests so this copy cannot silently drift.

import threading
from typing import TYPE_CHECKING

import torch
from vllm.distributed.eplb.eplb_utils import CpuGpuEvent
from vllm.distributed.eplb.rebalance_execute import AsyncEplbLayerResult, transfer_layer
from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import logger

if TYPE_CHECKING:
    from vllm.distributed.eplb.eplb_state import EplbModelState

    from vllm_ascend.distributed.eplb_state import AscendEplbState


class NoTransferCycleComplete:
    """Marker consumed by the main thread when a cycle has no final transfer."""


NO_TRANSFER_CYCLE_COMPLETE = NoTransferCycleComplete()


def start_async_worker(
    state: "AscendEplbState",
    is_profile: bool = False,
) -> threading.Thread:
    """Start the state-owned Ascend transfer worker."""
    rank = get_eplb_group().device_group.rank()
    device_index: int | None = state.cuda_device_index
    assert state.is_async

    def thread_target() -> None:
        assert device_index is not None
        torch.accelerator.set_device_index(device_index)
        stream = torch.cuda.Stream(device=device_index)
        try:
            transfer_run_periodically(state=state, cuda_stream=stream, is_profile=is_profile)
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.exception("async loop error (Rank %d): %s", rank, str(exc))

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread


def _run_rebalance_experts(
    model_state: "EplbModelState",
    state: "AscendEplbState",
    physical_to_logical_map_cpu: torch.Tensor,
    cuda_stream: torch.cuda.Stream,
) -> torch.Tensor:
    assert model_state.eplb_stats is not None
    stats = model_state.eplb_stats
    with torch.cuda.stream(cuda_stream):
        load_window_cpu = stats.global_expert_load_window.cpu()
    new_mapping = state.policy.rebalance_experts(
        load_window_cpu,
        stats.num_replicas,
        stats.num_groups,
        stats.num_nodes,
        stats.num_gpus,
        physical_to_logical_map_cpu,
    )
    if new_mapping.device != torch.device("cpu"):
        raise RuntimeError("STAIR must return a CPU physical-to-logical map.")
    model_state._ascend_eplb_policy_load = load_window_cpu
    return new_mapping


def _publish_result(
    model_state: "EplbModelState",
    layer_idx: int,
    new_mapping: torch.Tensor,
    transfer_metadata,
    cuda_stream: torch.cuda.Stream,
) -> None:
    consumed_event = CpuGpuEvent()
    model_state.pending_result = AsyncEplbLayerResult(
        layer_idx=layer_idx,
        new_physical_to_logical_map=new_mapping,
        transfer_metadata=transfer_metadata,  # type: ignore[arg-type]
        consumed_event=consumed_event,
    )
    consumed_event.wait(stream=cuda_stream)
    assert model_state.pending_result is None


def transfer_run_periodically(
    state: "AscendEplbState",
    cuda_stream: torch.cuda.Stream,
    is_profile: bool = False,
) -> None:
    """Transfer only changed layers and publish one explicit cycle completion."""
    while True:
        state.rearrange_event.wait(stream=cuda_stream)

        eplb_coordinator = get_eplb_group()
        eplb_group = eplb_coordinator.device_group
        eplb_cpu_group = eplb_coordinator.cpu_group
        ep_rank = eplb_group.rank()

        for model_state in state.model_states.values():
            model_state.communicator.set_stream(cuda_stream)
            with torch.cuda.stream(cuda_stream):
                old_mapping = model_state.physical_to_logical_map.cpu()
            new_mapping = _run_rebalance_experts(model_state, state, old_mapping, cuda_stream)
            changed_layers: list[int] = (
                torch.any(new_mapping != old_mapping, dim=1).nonzero(as_tuple=False).flatten().tolist()
            )

            for layer_idx in changed_layers:
                flag = torch.tensor([int(model_state.rebalanced)], dtype=torch.int32, device="cpu")
                torch.distributed.all_reduce(flag, group=eplb_cpu_group)
                flag_sum = int(flag.item())
                if flag_sum != eplb_cpu_group.size():
                    logger.warning(
                        "async worker (rank=%d): layer %d coordinated stop (flag_sum=%d, group_size=%d)",
                        ep_rank,
                        layer_idx,
                        flag_sum,
                        eplb_cpu_group.size(),
                    )
                    model_state.rebalanced = False
                    break

                transfer_metadata = transfer_layer(
                    old_layer_indices=old_mapping[layer_idx],
                    new_layer_indices=new_mapping[layer_idx],
                    expert_weights=model_state.model.expert_weights[layer_idx],
                    expert_weights_buffer=model_state.expert_buffer,
                    communicator=model_state.communicator,
                    ep_group=eplb_group,
                    is_profile=is_profile,
                    cuda_stream=cuda_stream,
                    layer_idx=layer_idx,
                )
                cuda_stream.synchronize()
                _publish_result(
                    model_state,
                    layer_idx,
                    new_mapping[layer_idx],
                    transfer_metadata,
                    cuda_stream,
                )

            if model_state.rebalanced:
                _publish_result(
                    model_state,
                    model_state.model.num_moe_layers - 1,
                    new_mapping[-1],
                    NO_TRANSFER_CYCLE_COMPLETE,
                    cuda_stream,
                )
