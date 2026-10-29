# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Worker-side retained-state resize; all RPCs involve every startup rank."""

from dataclasses import replace

import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.distributed.fasteep.hccl import RetainedCommunicators
from vllm_ascend.distributed.fasteep.memory import PeerMemory
from vllm_ascend.distributed.fasteep.policy import replica_donors, validate_config, validate_target
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, _moe_config_key, _MoECommMethods
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts


class FastEEPRuntime:
    def __init__(self, worker):
        validate_config(worker.vllm_config)
        if get_ascend_config().eplb_config.dynamic_eplb or get_ascend_config().enable_fused_mc2:
            raise ValueError("FastEEP requires static placement and unfused AllGather MoE.")
        self.worker = worker
        self.parent_dp = parallel_state.get_dp_group()
        self.parent_ep = parallel_state.get_ep_group()
        self.maximum = self.parent_dp.world_size
        self.rank = self.parent_dp.rank_in_group
        self.active = self.maximum
        self.failed = False
        self.layers = [module for module in worker.get_model().modules() if isinstance(module, AscendRoutedExperts)]
        if not self.layers or any(layer.global_expert_map is None or layer.log2phy is None for layer in self.layers):
            raise ValueError("FastEEP requires explicit redundant static expert maps for every MoE layer.")
        self.maps = [layer.global_expert_map.cpu().tolist() for layer in self.layers]
        replica_donors(self.maps, self.maximum)
        self.full_maps = [layer.global_expert_map.clone() for layer in self.layers]
        self.memory = PeerMemory(worker.get_model())
        self.memory.storages()
        self.comms = None

    @property
    def inactive(self):
        return self.rank >= self.active

    def state(self):
        return {
            "rank": self.rank,
            "active": self.active,
            "maximum": self.maximum,
            "sleeping": self.memory.sleeping,
            "failed": self.failed,
        }

    def validate(self, target):
        if self.failed:
            raise RuntimeError("FastEEP worker is in failed state; restart required.")
        validate_target(target, self.maximum)
        return replica_donors(self.maps, target)

    def _clear_finished(self):
        runner = self.worker.model_runner
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
        donors = replica_donors(self.maps, self.active)
        joining = range(self.active, target)
        if self.rank in joining:
            self.memory.map()
        manifests = self.comms.gather_objects(self.memory.metadata())
        for rank in joining:
            donor = donors[rank]
            if manifests[rank] != manifests[donor]:
                raise RuntimeError("Peer refill storage layouts differ.")
        for rank in joining:
            donor = donors[rank]
            if self.rank in (rank, donor):
                for storage in self.memory.storages():
                    self.comms.transfer(
                        storage["pointer"],
                        storage["bytes"],
                        rank if self.rank == donor else donor,
                        send=self.rank == donor,
                    )
        torch.npu.synchronize()
        if self.rank in joining:
            self.memory.finish()
        self.comms.barrier()

    def _activate(self, target, group):
        if self.rank >= target:
            return
        parallel_state._DP = self.parent_dp if target == self.maximum else group
        parallel_state._EP = self.parent_ep if target == self.maximum else group
        config = self.worker.vllm_config
        config.parallel_config.data_parallel_size = target
        self.worker.model_runner.dp_size = target
        for layer, full in zip(self.layers, self.full_maps):
            layer.global_expert_map = full[:target].clone()
            layer.log2phy.copy_(
                generate_log2phy_map(layer.global_expert_map, self.rank, tp_size=1).to(layer.log2phy.device)
            )
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
            with set_current_vllm_config(config):
                comm = AllGatherCommImpl(moe)
            # Routing maps logical IDs to physical IDs before dispatch. Only
            # the local physical interval may contribute to the output sum.
            comm.token_dispatcher.fasteep_physical_routing = True
            _MoECommMethods[_moe_config_key(MoECommType.ALLGATHER, moe)] = comm

    def resize(self, target):
        self.validate(target)
        if self.active == target:
            return self.state()
        try:
            torch.npu.synchronize()
            self._clear_finished()
            if self.comms is None:
                self.comms = RetainedCommunicators(self.parent_dp)
            if target > self.active:
                self._refill(target)
            child = self.comms.child(target) if target < self.maximum else None
            self._activate(target, child)
            if self.active > target and target <= self.rank < self.active:
                self.memory.sleep()
            self.comms.barrier()
            self.active = target
            return self.state()
        except BaseException:
            self.failed = True
            raise

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
