# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import compute_logical_maps
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.eplb.global_pool import GlobalExpertPoolPlanner
from vllm_ascend.distributed.eplb.shared_weights import SharedExpertWeights, plan_shared_slot_transfers
from vllm_ascend.distributed.eplb.state import AscendEplbState, refresh_model_routing_tables
from vllm_ascend.ops.fused_moe.eplb import EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS
from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method, setup_moe_comm_method

_SLOTS_PER_TRANSFER_STEP = 4


class GlobalPoolState(AscendEplbState):

    def __init__(self, parallel_config, device, slots_per_rank):
        super().__init__(parallel_config, device)
        self.slots_per_rank = slots_per_rank
        self.planner = GlobalExpertPoolPlanner(slots_per_rank, routing_table_rows=EXPERT_REPLICA_ROUTING_TABLE_NUM_ROWS)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eplb-global")
        self._plan = None
        self._steps = []
        self._pending = None
        self._requests = []
        self._started_at = None

    @property
    def collecting(self):
        return self._pending is None and not self._steps

    def add_model(self, model, model_config):
        if self.model_states:
            raise ValueError("Global Policy4 currently accepts one target model, without a draft model")
        if model.num_redundant_experts:
            raise ValueError("Global Policy4 requires zero native per-layer redundant experts")
        if not hasattr(model, "moe_mlp_layers"):
            raise ValueError("Global Policy4 currently requires the DeepSeek MoE metadata interface")
        group = get_ep_group()
        if group.cpu_group is None or not 1 < group.world_size <= 32:
            raise ValueError("Global Policy4 requires an EP CPU group and 2..32 expert-parallel ranks")
        layers = [getattr(layer, "routed_experts", layer) for layer in model.moe_layers]
        if any(getattr(layer, "expert_placement_strategy", "linear") != "linear" for layer in layers):
            raise ValueError("Global Policy4 requires linear immutable base placement")
        base_slots = model.num_local_physical_experts
        if base_slots * group.world_size != model.num_logical_experts:
            raise ValueError("The logical expert count must divide the EP size exactly")
        super().add_model(model, model_config)
        self.is_async = False
        self.storage = SharedExpertWeights(layers, self.slots_per_rank)
        self._model_state = next(iter(self.model_states.values()))
        local_slots = base_slots + self.slots_per_rank
        physical_experts = local_slots * group.world_size

        configs = {id(layer.moe_config): layer.moe_config for layer in layers}
        for config in configs.values():
            config.num_local_experts = local_slots
            config.num_experts = physical_experts
            config.global_redundant_expert_num = self.slots_per_rank * group.world_size
        for layer in layers:
            layer.local_num_experts = local_slots
            layer.global_num_experts = physical_experts
        model.num_local_physical_experts = local_slots
        model.update_physical_experts_metadata(physical_experts, local_slots)
        setup_moe_comm_method(layers[0].moe_config)
        alltoall = get_moe_comm_method(MoECommType.ALLTOALL)
        expert_ids = getattr(alltoall.token_dispatcher, "expert_ids_per_ep_rank", None)
        if expert_ids is not None:
            for layer in layers:
                layer.expert_ids_per_ep_rank = expert_ids

        shape = (len(layers), physical_experts)
        state = self._model_state
        state.physical_to_logical_map_buffer = torch.full(shape, -1, dtype=torch.long, device=self.device)
        state.physical_to_logical_map = state.physical_to_logical_map_buffer
        state.expert_load_pass_buffer = torch.zeros(shape, dtype=torch.int32, device=self.device)
        state.expert_load_pass = state.expert_load_pass_buffer
        state.expert_load_window = torch.zeros(
            (self.expert_load_window_size, *shape), dtype=torch.int32, device=self.device
        )
        self.layout = np.full((len(layers), group.world_size, local_slots), -1, dtype=np.int64)
        self.layout[:, :, :base_slots] = np.arange(model.num_logical_experts).reshape(group.world_size, base_slots)
        for layer in range(len(layers)):
            self._publish_layer(layer)
        model.set_eplb_state(state.expert_load_pass_buffer, state.logical_to_physical_map, state.logical_replica_count)
        self._propagate_shared_tensors(model, state.num_unpadded_tokens_tensors)
        refresh_model_routing_tables(state)
        dist.all_reduce(torch.zeros(1, device=self.device), group=group.device_group)

    def _publish_layer(self, layer):
        state = self._model_state
        mapping = torch.from_numpy(self.layout[layer].reshape(-1))
        logical_map, counts = compute_logical_maps(mapping, state.model.num_logical_experts)
        state.physical_to_logical_map[layer].copy_(mapping)
        target = state.logical_to_physical_map[layer]
        target.fill_(-1)
        target[:, : logical_map.shape[1]].copy_(logical_map)
        state.logical_replica_count[layer].copy_(counts)
        refresh_model_routing_tables(state, layer)

    def rearrange(self, is_profile=False, rank_mapping=None):
        if rank_mapping is not None:
            raise ValueError("Global Policy4 does not support elastic expert parallelism")
        if is_profile or self._plan is not None or not self.collecting:
            return None
        if not self._has_global_fresh_recorded_load():
            return None
        loads = self._allreduce_list([self._model_state.expert_load_window.sum(dim=0)])[0]
        workload = loads.cpu().numpy().reshape(self.layout.shape)
        self._has_fresh_recorded_load = False
        if get_ep_group().rank_in_group == 0:
            rank_loads = workload.sum(axis=-1)
            means = rank_loads.mean(axis=-1)
            ratios = np.divide(rank_loads.max(axis=-1), means, out=np.ones_like(means), where=means > 0)
            shared_load = workload[:, :, self.storage.base_slots :].sum()
            logger.info(
                "[eplb/global] Observed mean=%.6f max=%.6f shared_load_fraction=%.6f active_slots=%d",
                float(ratios.mean()),
                float(ratios.max()),
                float(shared_load / max(1, workload.sum())),
                int((self.layout[:, :, self.storage.base_slots :] >= 0).sum()),
            )
        self._plan = self._executor.submit(self.planner.plan, self.layout.copy(), workload)
        return None

    def _poll_plan(self):
        if self._plan is None:
            return
        group = get_ep_group()
        ready = torch.tensor(int(self._plan.done()), dtype=torch.int32)
        dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if not ready.item():
            return
        decision = self._plan.result()
        self._plan = None
        if not decision.should_apply:
            if group.rank_in_group == 0:
                logger.info("[eplb/global] Skip reason=%s gain=%.6f", decision.reason, decision.gain)
            return
        self._candidate = np.asarray(decision.placement, dtype=np.int64)
        transfers = plan_shared_slot_transfers(self.layout, self._candidate, self.slots_per_rank)
        for start in range(0, self.slots_per_rank, _SLOTS_PER_TRANSFER_STEP):
            step = [item for item in transfers if start <= item.shared_slot < start + _SLOTS_PER_TRANSFER_STEP]
            if step:
                self._steps.append(step)
        self._started_at = time.monotonic()
        if group.rank_in_group == 0:
            logger.info(
                "[eplb/global] Apply gain=%.6f changed_slots=%d steps=%d",
                decision.gain,
                decision.changed_slots,
                len(self._steps),
            )

    def _start_transfer(self):
        if self._pending is not None or not self._steps:
            return
        group = get_ep_group()
        torch.npu.synchronize()
        self._pending = self._steps.pop(0)
        changed_layers = set()
        for item in self._pending:
            if item.old_layer is not None:
                self.layout[item.old_layer, item.destination_rank, self.storage.base_slots + item.shared_slot] = -1
                changed_layers.add(item.old_layer)
        for layer in sorted(changed_layers):
            self._publish_layer(layer)
        torch.npu.current_stream().synchronize()
        operations = []
        for item in self._pending:
            if item.new_layer is None:
                continue
            if group.rank_in_group == item.source_rank:
                for tensor in self.storage.source(item.new_layer, item.source_slot):
                    operations.append(
                        dist.P2POp(
                            dist.isend,
                            tensor,
                            dist.get_global_rank(group.device_group, item.destination_rank),
                            group.device_group,
                        )
                    )
            if group.rank_in_group == item.destination_rank:
                for tensor in self.storage.destination(item.shared_slot):
                    operations.append(
                        dist.P2POp(
                            dist.irecv,
                            tensor,
                            dist.get_global_rank(group.device_group, item.source_rank),
                            group.device_group,
                        )
                    )
        self._requests = dist.batch_isend_irecv(operations) if operations else []
        self._clear_loads()

    def _clear_loads(self):
        self._model_state.expert_load_pass.zero_()
        self._model_state.expert_load_window.zero_()
        self._has_fresh_recorded_load = False

    def _finish_transfer(self):
        if self._pending is None:
            return
        for request in self._requests:
            request.wait()
        dist.barrier(group=get_ep_group().cpu_group)
        changed_layers = set()
        for item in self._pending:
            if item.new_layer is not None:
                slot = self.storage.base_slots + item.shared_slot
                self.layout[item.new_layer, item.destination_rank, slot] = self._candidate[
                    item.new_layer, item.destination_rank, slot
                ]
                changed_layers.add(item.new_layer)
        for layer in sorted(changed_layers):
            self._publish_layer(layer)
        self._pending = None
        self._requests = []
        self._clear_loads()
        if not self._steps:
            if not np.array_equal(self.layout, self._candidate):
                raise RuntimeError("Global Policy4 migration did not reach its planned placement")
            if get_ep_group().rank_in_group == 0:
                logger.info(
                    "[eplb/global] Migration completed elapsed_ms=%.3f", (time.monotonic() - self._started_at) * 1000
                )

    def step(self, is_dummy=False, is_profile=False, log_stats=False):
        if is_profile:
            self._clear_loads()
            return
        self._finish_transfer()
        super().step(is_dummy, is_profile, log_stats=log_stats)
        self._poll_plan()
        self._start_transfer()
