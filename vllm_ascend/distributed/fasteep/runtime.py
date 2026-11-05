# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-side retained-state resize; all RPCs involve every startup rank."""

import json
from dataclasses import replace

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed import parallel_state as ascend_parallel_state
from vllm_ascend.distributed.fasteep.communication import ReclaimableCommunicators
from vllm_ascend.distributed.fasteep.memory import PeerMemory
from vllm_ascend.distributed.fasteep.migration import ExpertMigration
from vllm_ascend.distributed.fasteep.placement import plan_active_placement, validate_placement
from vllm_ascend.distributed.fasteep.policy import communication_method, validate_config, validate_target
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    _moe_config_key,
    _MoECommMethods,
)
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather


class FastEEPRuntime:
    def __init__(self, worker):
        validate_config(worker.vllm_config)
        if get_ascend_config().eplb_config.dynamic_eplb or get_ascend_config().enable_fused_mc2:
            raise ValueError("FastEEP requires paused EPLB and unfused MoE.")
        self.worker = worker
        self.parent_dp = parallel_state.get_dp_group()
        self.parent_ep = parallel_state.get_ep_group()
        self.maximum = self.parent_dp.world_size
        self.rank = self.parent_dp.rank_in_group
        self.active = self.maximum
        self.failed = False
        self.verified_migration_tensors = 0
        self.layers = [module for module in worker.get_model().modules() if isinstance(module, AscendRoutedExperts)]
        if not self.layers or any(layer.global_expert_map is None or layer.log2phy is None for layer in self.layers):
            raise ValueError("FastEEP requires explicit redundant static expert maps for every MoE layer.")
        self.maps = [layer.global_expert_map.cpu().tolist() for layer in self.layers]
        for placement in self.maps:
            validate_placement(placement)
        self.full_maps = [layer.global_expert_map.cpu().tolist() for layer in self.layers]
        self.memory = PeerMemory(worker.get_model())
        self.memory.storages()
        self.captured_graphs = self._captured_graphs()
        if not worker.vllm_config.model_config.enforce_eager and not self._graph_pool_bytes():
            raise RuntimeError("FastEEP graph capture did not allocate a reclaimable graph pool.")
        self.comms = None

    @property
    def inactive(self):
        return self.rank >= self.active

    def _graph_pool_bytes(self):
        return sum(data.handle[1] for data in self.memory.allocator.pointer_to_data.values() if data.tag == "graphs")

    @staticmethod
    def _captured_graphs():
        from vllm_ascend.compilation.acl_graph import _acl_graph_wrappers

        return {
            id(entry.aclgraph)
            for wrapper in _acl_graph_wrappers
            for entry in wrapper.concrete_aclgraph_entries.values()
            if entry.aclgraph is not None
        }

    def state(self):
        return {
            "rank": self.rank,
            "active": self.active,
            "maximum": self.maximum,
            "sleeping": self.memory.sleeping,
            "failed": self.failed,
            "graph_pool_bytes": self._graph_pool_bytes(),
            "captured_graphs": len(self._captured_graphs()),
            "verified_migration_tensors": self.verified_migration_tensors,
            "graph_offloaded_bytes": self.memory.graph_offloaded_bytes,
            "communication_memory_delta_bytes": self.comms.last_memory_delta if self.comms is not None else 0,
        }

    def validate(self, target):
        if self.failed:
            raise RuntimeError("FastEEP worker is in failed state; restart required.")
        validate_target(target, self.maximum)
        return [
            plan_active_placement(current, template, target) for current, template in zip(self.maps, self.full_maps)
        ]

    def _clear_finished(self):
        runner = self.worker.model_runner
        if self.worker.vllm_config.use_v2_model_runner:
            for request_id in sorted(runner.req_states.req_id_to_index):
                runner._remove_request(request_id)
            return
        # Admission has drained complete responses and generation is paused.
        # Finished IDs may still await the next scheduler-output notification.
        for request_id in list(runner.requests):
            runner.requests.pop(request_id)
            runner.num_prompt_logprobs.pop(request_id, None)
            runner.input_batch.remove_request(request_id)
        runner.input_batch.condense()
        runner.input_batch.refresh_metadata()
        runner.input_batch.prev_req_id_to_index = None
        runner.input_batch.prev_sampled_token_ids = None

    def _refill(self, target):
        comms = self.comms
        assert comms is not None
        # Dense parameters have identical layouts across DP ranks. Expert
        # slots are overwritten individually by ExpertMigration below.
        donor = 0
        joining = range(self.active, target)
        if self.rank in joining:
            self.memory.map()
        manifests = comms.gather_objects(self.memory.metadata())
        for rank in joining:
            if manifests[rank] != manifests[donor]:
                raise RuntimeError("Peer refill storage layouts differ.")
        for rank in joining:
            if self.rank in (rank, donor):
                for storage in self.memory.storages():
                    comms.transfer(
                        storage["pointer"],
                        storage["bytes"],
                        rank if self.rank == donor else donor,
                        send=self.rank == donor,
                    )
        torch.npu.synchronize()
        if self.rank in joining:
            self.memory.finish()
        comms.barrier()

    def _activate(self, target, group, placements):
        if self.rank >= target:
            return
        parallel_state._DP = group
        parallel_state._EP = group
        config = self.worker.vllm_config
        config.parallel_config.data_parallel_size = target
        if not config.use_v2_model_runner:
            self.worker.model_runner.dp_size = target
        for layer, placement in zip(self.layers, placements):
            layer.global_expert_map = torch.tensor(
                placement, dtype=layer.global_expert_map.dtype, device=layer.global_expert_map.device
            )
            log2phy = layer.log2phy
            assert log2phy is not None
            routing_rank = self.rank if communication_method(config) == "alltoall" else 0
            log2phy.copy_(generate_log2phy_map(layer.global_expert_map, routing_rank, tp_size=1).to(log2phy.device))
            moe = layer.moe_config
            local = moe.num_local_experts
            physical = local * target
            layer.global_num_experts = physical
            layer.global_redundant_expert_num = physical - moe.num_logical_experts
            moe.num_experts = physical
            moe.global_redundant_expert_num = layer.global_redundant_expert_num
            moe.moe_parallel_config = replace(
                moe.moe_parallel_config, dp_size=target, ep_size=target, ep_rank=self.rank
            )
            moe.dp_group = parallel_state._DP
            moe.ep_group = parallel_state._EP
            alltoall = communication_method(config) == "alltoall"
            kind = MoECommType.ALLTOALL if alltoall else MoECommType.ALLGATHER
            with set_current_vllm_config(config):
                comm = AlltoAllCommImpl(moe) if alltoall else AllGatherCommImpl(moe)
            # Each physical expert contributes on exactly one active rank.
            if isinstance(comm.token_dispatcher, TokenDispatcherWithAllGather):
                comm.token_dispatcher.fasteep_physical_routing = True
            _MoECommMethods[_moe_config_key(kind, moe)] = comm

    def resize(self, target):
        placements = self.validate(target)
        if self.active == target:
            return self.state()
        try:
            torch.npu.synchronize()
            self._clear_finished()
            captured = self.captured_graphs
            if self._captured_graphs() != captured:
                raise RuntimeError("FastEEP inference replaced or recaptured a graph after initialization.")
            if self.comms is None:
                self.comms = ReclaimableCommunicators(self.parent_dp)
            self.comms.begin_transition()
            if target > self.active:
                self._refill(target)
            verify = get_ascend_config().fasteep.verify_migration
            migration = ExpertMigration(self.layers, self.rank, self.comms, verify=verify)
            self.verified_migration_tensors = migration.execute(self.maps, placements)
            group = self.comms.activate(target)
            self._release_initial_device_groups()
            self._activate(target, group, placements)
            if self.active > target and target <= self.rank < self.active:
                self.memory.sleep()
            self.comms.barrier()
            self.maps = placements
            self.active = target
            if self._captured_graphs() != captured:
                raise RuntimeError("FastEEP resize replaced a captured graph.")
            state = self.state()
            logger.info("FastEEP resize committed: %s", json.dumps(state))
            return state
        except BaseException:
            self.failed = True
            raise

    def _release_initial_device_groups(self):
        # The CPU groups continue to coordinate all startup ranks. Only the
        # replaced native device communicators (and their buffers) are freed.
        groups = [self.parent_dp, self.parent_ep]
        if ascend_parallel_state.model_parallel_initialized():
            # MoE setup also initializes an MC2 handle. FastEEP uses the
            # replaceable AllGather/AllToAll group, so this handle is unused.
            groups.append(ascend_parallel_state.get_mc2_group())
        seen = set()
        for group in groups:
            device_group = getattr(group, "device_group", None)
            if device_group is None:
                continue
            release = getattr(group, "destroy_hccl", None)
            if callable(release):
                # The coordinator owns reference counts for shared HCCL
                # groups. Release through it so shutdown cannot destroy twice.
                release()
                continue
            if id(device_group) not in seen:
                if getattr(group, "tcp_store_group", None) is not None:
                    from vllm.distributed.utils import stateless_destroy_torch_distributed_process_group

                    stateless_destroy_torch_distributed_process_group(device_group)
                else:
                    torch.distributed.destroy_process_group(device_group)
                seen.add(id(device_group))
            del group.device_group
            group.device_communicator = None

    def close(self):
        parallel_state._DP = self.parent_dp
        parallel_state._EP = self.parent_ep
        if self.comms is not None:
            self.comms.close()


class FastEEPWorkerExtension:
    """Enable with --worker-extension-cls and the FastEEP ASGI middleware."""

    def fasteep_state(self):
        if getattr(self, "_fasteep_runtime", None) is None:
            self._fasteep_runtime = FastEEPRuntime(self)
        return self._fasteep_runtime.state()

    def fasteep_validate(self, target):
        self.fasteep_state()
        self._fasteep_runtime.validate(target)
        return True

    def fasteep_resize(self, target):
        self.fasteep_state()
        return self._fasteep_runtime.resize(target)
