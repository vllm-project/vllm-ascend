# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""STAIR EPLB policy integration and distributed planning."""

import os
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from vllm.distributed import get_eplb_group
from vllm.distributed.eplb.policy import AbstractEplbPolicy

from vllm_ascend.ascend_config import StairConfig
from vllm_ascend.distributed.eplb.layer_sharding import all_gather_layer_shards, assigned_layer_ids
from vllm_ascend.distributed.eplb.policy import PreparedLoadStats
from vllm_ascend.distributed.eplb.policy._stair_planner import (
    LayerPlan,
    PlacementImbalance,
    PlacementPlan,
    StairPlan,
    StairPlanner,
    planner_config_values,
    receive_planner_response,
    send_planner_request,
)

__all__ = (
    "LayerPlan",
    "PlacementImbalance",
    "PlacementPlan",
    "StairEplbPolicy",
    "StairPlan",
)

_PLANNER_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_PLANNER_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class StairEplbPolicy(StairPlanner, AbstractEplbPolicy):
    """STAIR load statistics, isolated planning, and EPLB integration."""

    def __init__(self, config: StairConfig) -> None:
        self.config = config
        self._planner_process: subprocess.Popen[bytes] | None = None
        self._planner_socket: socket.socket | None = None
        self._planner_stream: BinaryIO | None = None

    def _start_planner_process(self) -> None:
        process = self._planner_process
        if process is not None and process.poll() is None:
            return
        self._stop_planner_process()

        parent_socket, child_socket = socket.socketpair()
        environment = os.environ.copy()
        environment.update(_PLANNER_THREAD_ENV)
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_stair_planner.py")),
                    str(child_socket.fileno()),
                ],
                pass_fds=(child_socket.fileno(),),
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=environment,
            )
        except Exception:
            parent_socket.close()
            child_socket.close()
            raise
        child_socket.close()
        self._planner_process = process
        self._planner_socket = parent_socket
        self._planner_stream = parent_socket.makefile("rwb")

    def _stop_planner_process(self) -> None:
        stream = getattr(self, "_planner_stream", None)
        planner_socket = getattr(self, "_planner_socket", None)
        process = getattr(self, "_planner_process", None)
        self._planner_stream = None
        self._planner_socket = None
        self._planner_process = None

        if stream is not None:
            stream.close()
        if planner_socket is not None:
            planner_socket.close()
        if process is None:
            return
        try:
            process.wait(timeout=_PLANNER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=_PLANNER_SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stderr is not None:
            process.stderr.close()

    def _planner_exit_details(self) -> str:
        process = self._planner_process
        if process is None or process.poll() is None or process.stderr is None:
            return ""
        return process.stderr.read().decode(errors="replace").strip()

    def _plan_in_subprocess(
        self,
        logical_load_values: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        last_committed_mean_ratios: np.ndarray,
        rank_node_ids: np.ndarray,
        config: StairConfig,
        layer_ids: Sequence[int] | None = None,
        sample_counts: np.ndarray | None = None,
    ) -> StairPlan:
        """Run one CPU plan in STAIR's persistent lightweight subprocess."""

        self._start_planner_process()
        stream = self._planner_stream
        if stream is None:
            raise RuntimeError("STAIR planner subprocess did not provide a socket")
        try:
            send_planner_request(
                stream,
                logical_load_values,
                current_rank_expert_ids,
                last_committed_mean_ratios,
                rank_node_ids,
                planner_config_values(config),
                layer_ids,
                sample_counts,
            )
            remote_error_type, remote_error, plan_fields = receive_planner_response(stream)
        except (EOFError, OSError, TypeError, ValueError) as error:
            details = self._planner_exit_details()
            self._stop_planner_process()
            message = "STAIR planner subprocess terminated unexpectedly"
            if details:
                message += f": {details}"
            raise RuntimeError(message) from error
        if remote_error is not None:
            error_class = ValueError if remote_error_type == "ValueError" else RuntimeError
            raise error_class(f"STAIR planner subprocess failed:\n{remote_error.rstrip()}")
        if plan_fields is None:
            raise RuntimeError("STAIR planner subprocess returned no plan")
        return StairPlan(*plan_fields)

    def prepare_local_load_stats(self, load_samples: torch.Tensor) -> PreparedLoadStats:
        """Compress local temporal samples into STAIR's weighted bins."""
        boundaries, samples_per_bin = self._load_bin_boundaries(load_samples.shape[0], self.config.load_window_bins)
        load_sums_per_bin = torch.stack(
            [load_samples[start:end].sum(dim=0) for start, end in zip(boundaries[:-1], boundaries[1:])]
        )
        return PreparedLoadStats(load_sums_per_bin, samples_per_bin)

    def rebalance_experts(
        self,
        weight: torch.Tensor | PreparedLoadStats,
        num_replicas: int,
        num_groups: int,
        num_nodes: int,
        num_ranks: int,
        old_global_expert_indices: torch.Tensor | None = None,
        *,
        last_committed_mean_ratios: np.ndarray | None = None,
        rank_node_ids: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Plan through the upstream policy contract with STAIR context.

        A tensor is either ``[layers, experts]`` or an uncompressed
        ``[samples, layers, experts]`` window. A :class:`PreparedLoadStats`
        carries pre-binned sums and sample counts; its values must already
        reside on CPU, where planning runs. Optional anchors and node IDs
        enable the full STAIR path while preserving the upstream positional
        contract. ``num_groups`` is validated for that contract but does not
        constrain STAIR placement. The CPU result is ``[layers, num_replicas]``;
        its source-coordinate attributes are ``[layers, ranks, slots]`` and
        predicted ratios are ``[layers]``.
        """
        controls = num_replicas, num_groups, num_nodes, num_ranks
        invalid_type = any(isinstance(value, bool) or not isinstance(value, int) for value in controls)
        if invalid_type or min(controls) < 1:
            raise ValueError("STAIR topology values must be positive integers")
        if num_replicas % num_ranks:
            raise ValueError("STAIR requires equal rank capacity")
        if old_global_expert_indices is None:
            raise ValueError("STAIR requires the current expert placement")

        current_map = old_global_expert_indices.cpu()
        if current_map.ndim != 2 or current_map.shape[1] != num_replicas:
            raise ValueError("current expert placement must be [layers, num_replicas]")
        if isinstance(weight, PreparedLoadStats):
            if weight.values.device.type != "cpu":
                raise ValueError("prepared STAIR load statistics must be on CPU")
            logical_load_values = weight.values.to(dtype=torch.float64).numpy()
            sample_counts = weight.sample_counts
            if sample_counts is None:
                raise ValueError("prepared STAIR load statistics require sample counts")
        else:
            logical_load_values = weight.to(device="cpu", dtype=torch.float64).numpy()
            logical_load_values = (
                logical_load_values[None, ...] if logical_load_values.ndim == 2 else logical_load_values
            )
            sample_counts = None
        if logical_load_values.ndim != 3 or logical_load_values.shape[1] != current_map.shape[0]:
            raise ValueError("weight must be [samples, layers, logical_experts] and match the placement")

        current_placement = current_map.numpy().reshape(current_map.shape[0], num_ranks, num_replicas // num_ranks)
        if last_committed_mean_ratios is None:
            last_committed_mean_ratios = np.full(current_placement.shape[0], np.nan)
        if rank_node_ids is None:
            if num_ranks % num_nodes:
                raise ValueError("STAIR cannot infer topology with unequal ranks per node")
            node_ids = np.arange(num_ranks, dtype=np.int64) // (num_ranks // num_nodes)
        else:
            node_ids = np.asarray(rank_node_ids)
            if node_ids.shape != (num_ranks,) or not np.issubdtype(node_ids.dtype, np.integer) or np.any(node_ids < 0):
                raise ValueError("rank_node_ids must contain one non-negative integer per rank")
            if len(np.unique(node_ids)) != num_nodes:
                raise ValueError("num_nodes must match the distinct rank_node_ids")
        cpu_group = get_eplb_group().cpu_group
        if cpu_group.size() != num_ranks:
            raise RuntimeError("STAIR topology does not match the stage-local EPLB group")
        if num_ranks == 1:
            plan = self._plan_in_subprocess(
                logical_load_values=logical_load_values,
                current_rank_expert_ids=current_placement,
                last_committed_mean_ratios=last_committed_mean_ratios,
                rank_node_ids=node_ids,
                config=self.config,
                sample_counts=sample_counts,
            )
        else:
            plan = self.plan_sharded_rebalance(
                logical_load_values=logical_load_values,
                current_rank_expert_ids=current_placement,
                last_committed_mean_ratios=last_committed_mean_ratios,
                rank_node_ids=node_ids,
                config=self.config,
                cpu_group=cpu_group,
                sample_counts=sample_counts,
                planner=self._plan_in_subprocess,
            )
        self.validate_plan(
            current_placement,
            plan,
            logical_load_values.shape[2],
            rank_node_ids=node_ids,
            rank_transfer_limit=self.config.rank_transfer_limit,
            cross_node_transfer_limit=self.config.cross_node_transfer_limit,
        )
        planned_map = plan.rank_expert_ids.reshape(current_map.shape)
        target = torch.from_numpy(planned_map).to(dtype=current_map.dtype)
        # The async migration patch consumes STAIR's exact source plan.
        target.source_rank_ids = plan.source_rank_ids
        target.source_slot_ids = plan.source_slot_ids
        target.predicted_mean_ratios = plan.predicted_mean_ratios
        if sample_counts is None:
            load_bins, bin_counts = self.compress_load_window(logical_load_values, self.config.load_window_bins)
        else:
            bin_counts = np.asarray(sample_counts)
            load_bins = logical_load_values / bin_counts[:, None, None]
        # Report model-wide averages of the per-layer temporal mean and p95 ratios.
        before, after = [], []
        for layer_id in range(current_placement.shape[0]):
            before.append(self.placement_imbalance(load_bins[:, layer_id], bin_counts, current_placement[layer_id]))
            after.append(self.placement_imbalance(load_bins[:, layer_id], bin_counts, plan.rank_expert_ids[layer_id]))
        target.predicted_imbalance_summary = (
            float(np.mean([score.mean_ratio for score in before])),
            float(np.mean([score.p95_ratio for score in before])),
            float(np.mean([score.mean_ratio for score in after])),
            float(np.mean([score.p95_ratio for score in after])),
        )
        return target

    @classmethod
    def plan_sharded_rebalance(
        cls,
        logical_load_values: np.ndarray,
        current_rank_expert_ids: np.ndarray,
        last_committed_mean_ratios: np.ndarray,
        rank_node_ids: np.ndarray,
        config: StairConfig,
        cpu_group: torch.distributed.ProcessGroup,
        sample_counts: np.ndarray | None = None,
        planner: Callable[..., StairPlan] | None = None,
    ) -> StairPlan:
        """Plan round-robin layer shards and gather one stage-local plan.

        Every rank in ``cpu_group`` must call this method with identical model
        inputs and configuration. The group must contain only the current PP
        stage's EPLB ranks. All ranks receive and validate the same complete
        plan; no coordinator rank assembles the result. ``planner`` is an
        internal execution override used by STAIR's isolated CPU subprocess.
        Input and result shapes follow :meth:`plan_rebalance` and
        :class:`StairPlan`.
        """
        group_size = cpu_group.size()
        local_error = None
        local_plan_fields = None
        current = None
        try:
            current = np.asarray(current_rank_expert_ids)
            if current.ndim != 3:
                raise ValueError("current_rank_expert_ids must be a [layers, ranks, slots] array")
            num_layers = current.shape[0]
            owned_layer_ids = assigned_layer_ids(num_layers, cpu_group.rank(), group_size)
            plan_local_layers = cls.plan_rebalance if planner is None else planner
            local_plan = plan_local_layers(
                logical_load_values,
                current,
                last_committed_mean_ratios,
                rank_node_ids,
                config,
                layer_ids=owned_layer_ids,
                sample_counts=sample_counts,
            )
            owned_indices = np.fromiter(owned_layer_ids, dtype=np.int64, count=len(owned_layer_ids))
            local_plan_fields = tuple(
                torch.from_numpy(plan_field[owned_indices])
                for plan_field in (
                    local_plan.rank_expert_ids,
                    local_plan.source_rank_ids,
                    local_plan.source_slot_ids,
                    local_plan.predicted_mean_ratios,
                )
            )
            num_experts = np.asarray(logical_load_values).shape[2]
        except Exception as error:
            local_error = error

        planning_succeeded = torch.tensor(local_error is None, dtype=torch.int32)
        if group_size > 1:
            torch.distributed.all_reduce(
                planning_succeeded,
                op=torch.distributed.ReduceOp.MIN,
                group=cpu_group,
            )
        if not planning_succeeded.item():
            if local_error is not None:
                raise local_error
            raise RuntimeError("STAIR layer shard planning failed on another EPLB rank")
        if current is None or local_plan_fields is None:
            raise RuntimeError("STAIR layer shard planning produced no local plan")

        def gather_owned_field(local_values: torch.Tensor) -> np.ndarray:
            return all_gather_layer_shards(local_values, num_layers, cpu_group).numpy()

        plan = StairPlan(
            rank_expert_ids=gather_owned_field(local_plan_fields[0]),
            source_rank_ids=gather_owned_field(local_plan_fields[1]),
            source_slot_ids=gather_owned_field(local_plan_fields[2]),
            predicted_mean_ratios=gather_owned_field(local_plan_fields[3]),
        )
        cls.validate_plan(
            current,
            plan,
            num_experts,
            rank_node_ids,
            config.rank_transfer_limit,
            config.cross_node_transfer_limit,
        )
        return plan

    @classmethod
    def validate_plan(
        cls,
        current_rank_expert_ids: np.ndarray,
        plan: StairPlan,
        num_experts: int,
        rank_node_ids: np.ndarray,
        rank_transfer_limit: int,
        cross_node_transfer_limit: int,
    ) -> None:
        """Validate a fixed-shape plan against its current placement.

        Current and planned arrays are ``[layers, ranks, slots]``. Every source
        coordinate must own its target expert in the current placement;
        retained experts must keep their rank and slot. Rank and cross-node
        transfer usage is counted independently for each layer. Predicted mean
        ratios are ``[layers]``: changed layers require a finite value and
        unchanged layers require NaN.
        """
        current = np.asarray(current_rank_expert_ids)
        target = np.asarray(plan.rank_expert_ids)
        source_ranks = np.asarray(plan.source_rank_ids)
        source_slots = np.asarray(plan.source_slot_ids)
        if current.ndim != 3 or 0 in current.shape or not np.issubdtype(current.dtype, np.integer):
            raise ValueError("current_rank_expert_ids must be a non-empty integer [layers, ranks, slots] array")
        if target.shape != current.shape or source_ranks.shape != current.shape or source_slots.shape != current.shape:
            raise ValueError("STAIR plan placement and source arrays must match the current placement shape")
        if not all(np.issubdtype(values.dtype, np.integer) for values in (target, source_ranks, source_slots)):
            raise ValueError("STAIR plan placement and source arrays must contain integers")
        node_ids = np.asarray(rank_node_ids)
        if node_ids.shape != (current.shape[1],) or not np.issubdtype(node_ids.dtype, np.integer):
            raise ValueError("rank_node_ids must contain one integer per rank")
        controls = num_experts, rank_transfer_limit, cross_node_transfer_limit
        invalid_type = any(isinstance(value, bool) or not isinstance(value, int) for value in controls)
        invalid_limits = rank_transfer_limit < -1 or rank_transfer_limit == 0 or cross_node_transfer_limit < -1
        if invalid_type or num_experts < 1 or invalid_limits:
            raise ValueError("STAIR expert count and transfer limits are invalid")

        ratios = np.asarray(plan.predicted_mean_ratios)
        if ratios.shape != (current.shape[0],) or not np.issubdtype(ratios.dtype, np.floating):
            raise ValueError("predicted_mean_ratios must be a floating-point value per layer")
        ratios = ratios.astype(np.float64, copy=False)
        if np.any(~np.isnan(ratios) & (~np.isfinite(ratios) | (ratios < 1))):
            raise ValueError("predicted mean ratios must be NaN or finite values no smaller than one")
        if (
            np.any(source_ranks < 0)
            or np.any(source_ranks >= current.shape[1])
            or np.any(source_slots < 0)
            or np.any(source_slots >= current.shape[2])
        ):
            raise ValueError("STAIR plan contains an out-of-range source coordinate")

        for layer_id, target_layer in enumerate(target):
            current_layer = current[layer_id]
            cls.placement_replica_counts(current_layer, num_experts)
            cls.placement_replica_counts(target_layer, num_experts)
            changed = not np.array_equal(target_layer, current_layer)
            has_candidate = not np.isnan(ratios[layer_id])
            if changed != has_candidate:
                raise ValueError("predicted_mean_ratios must be finite for changed layers and NaN for unchanged layers")

            outgoing = np.zeros(current.shape[1], dtype=np.int64)
            incoming = np.zeros(current.shape[1], dtype=np.int64)
            cross_out: dict[int, int] = {}
            cross_in: dict[int, int] = {}
            for dst_rank, target_experts in enumerate(target_layer):
                current_slots = {int(expert): slot for slot, expert in enumerate(current_layer[dst_rank])}
                for dst_slot, expert in enumerate(target_experts):
                    src_rank = int(source_ranks[layer_id, dst_rank, dst_slot])
                    src_slot = int(source_slots[layer_id, dst_rank, dst_slot])
                    if current_layer[src_rank, src_slot] != expert:
                        raise ValueError("STAIR source does not own the target expert")
                    retained_slot = current_slots.get(int(expert))
                    if retained_slot is not None:
                        if (src_rank, src_slot, dst_slot) != (dst_rank, retained_slot, retained_slot):
                            raise ValueError("retained experts must keep their current rank and slot")
                        continue
                    outgoing[src_rank] += 1
                    incoming[dst_rank] += 1
                    if rank_transfer_limit != -1 and (
                        outgoing[src_rank] > rank_transfer_limit or incoming[dst_rank] > rank_transfer_limit
                    ):
                        raise ValueError("STAIR plan exceeds a per-rank transfer limit")
                    src_node, dst_node = int(node_ids[src_rank]), int(node_ids[dst_rank])
                    if src_node != dst_node:
                        cross_out[src_node] = cross_out.get(src_node, 0) + 1
                        cross_in[dst_node] = cross_in.get(dst_node, 0) + 1
                        if cross_node_transfer_limit != -1 and (
                            cross_out[src_node] > cross_node_transfer_limit
                            or cross_in[dst_node] > cross_node_transfer_limit
                        ):
                            raise ValueError("STAIR plan exceeds a per-node cross-node transfer limit")
