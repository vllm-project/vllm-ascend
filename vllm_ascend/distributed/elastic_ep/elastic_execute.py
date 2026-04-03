import copy
import gc
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from torch.distributed import P2POp
from vllm.compilation.counter import compilation_counter
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import (
    CompilationMode,
    set_current_vllm_config,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
)
from vllm.distributed.elastic_ep.elastic_execute import ElasticEPScalingExecutor
from vllm.distributed.elastic_ep.standby_state import (
    create_standby_groups,
    get_standby_dp_group,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.layer import FusedMoEParallelConfig
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.parallel_state import (
    _replace_ascend_active_groups,
    get_dynamic_eplb_group,
    get_mc2_group,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod


def batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    device_comm = dp_group.device_communicator
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            expert_weights_set.add(weight.data_ptr())

    state_dict = model.state_dict()
    all_params = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        if param.data_ptr() not in expert_weights_set:
            all_params.append(param.data)

    quant_weight_names = ["aclnn_input_scale", "aclnn_input_scale_reciprocal", "aclnn_input_offset"]
    for module in model.modules():
        for name in quant_weight_names:
            if (param := getattr(module, name, None)) is not None:
                all_params.append(param)

    assert len(all_params) > 0
    p2p_ops = []
    for param in all_params:
        op = object.__new__(P2POp)
        if is_sender:
            op.op = torch.distributed.isend
            op.tensor = param
        else:
            op.op = torch.distributed.irecv
            op.tensor = param
        op.group_peer = peer_rank
        p2p_ops.append(op)

    device_comm.batch_isend_irecv(p2p_ops)


def broadcast_expert_mapping(
    expert_maps: torch.Tensor | None,
    group: StatelessGroupCoordinator,
    src_rank: int = 0,
):
    if group.rank_in_group == src_rank:
        assert expert_maps is not None
        shape_tensor = torch.tensor(list(expert_maps.shape), dtype=torch.int64, device="cpu")
    else:
        shape_tensor = torch.empty(3, dtype=torch.int64, device="cpu")

    shape_tensor = group.tcp_store_group.broadcast(shape_tensor, src_rank)

    if group.rank_in_group != src_rank:
        expert_maps = torch.empty(
            tuple(shape_tensor.tolist()),
            dtype=torch.int64,
            device="cpu",
        )

    assert expert_maps is not None
    expert_maps = group.tcp_store_group.broadcast(expert_maps, src_rank)

    return expert_maps


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    def __init__(self, worker):
        super().__init__(worker)

    def load_model(self) -> None:
        (
            expert_maps,
            num_local_experts,
            num_logical_experts,
        ) = self.worker.elastic_ep_executor.receive_expert_mapping()
        dp_size = self.worker.parallel_config.data_parallel_size
        tp_size = self.worker.parallel_config.tensor_parallel_size
        pcp_size = self.worker.parallel_config.prefill_context_parallel_size
        ep_size = dp_size * tp_size * pcp_size
        get_ascend_config().eplb_config.num_redundant_experts = ep_size * num_local_experts - num_logical_experts
        if get_ascend_config().eplb_config.dynamic_eplb:
            self.worker.model_runner.shared_dict["expert_maps"] = expert_maps
            self.worker.model_runner.shared_dict["old_ep_size"] = expert_maps.shape[1]
        self.worker.load_model(load_dummy_weights=True)

    def create_standby_groups(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        world_size = self.worker.vllm_config.parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(self.worker.vllm_config.parallel_config)
        updated_config.parallel_config.data_parallel_size = new_dp_size
        with set_current_vllm_config(updated_config):
            create_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
                enable_eplb=updated_config.parallel_config.enable_eplb,
            )
            create_ascend_standby_groups(
                new_dp_size=new_dp_size,
                new_world_size_across_dp=new_world_size_across_dp,
                master_ip=reconfig_request.new_data_parallel_master_ip,
                coord_store_port=reconfig_request.coord_store_port,
            )

    def transfer_weights(self, old_dp_size: int, new_dp_size: int) -> None:
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        # Broadcast old_dp_size to all workers in standby group
        if standby_dp_group.rank_in_group < old_dp_size:
            old_dp_size_tensor = torch.tensor([old_dp_size], dtype=torch.int64, device="cpu")
        else:
            old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = standby_dp_group.tcp_store_group.broadcast(old_dp_size_tensor, 0)

        num_new_workers = new_dp_size - old_dp_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        # Sender-receiver pairing: the first new_workers % old_dp_size
        # senders get (k+1) contiguous receivers, the rest get k
        # receivers.
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if dp_rank < remainder:
            recv_begin = dp_rank * (num_dst_per_sender + 1)
            recv_end = recv_begin + num_dst_per_sender + 1
        else:
            recv_begin = remainder * (num_dst_per_sender + 1) + (dp_rank - remainder) * num_dst_per_sender
            recv_end = recv_begin + num_dst_per_sender

        ranks_to_send = list(range(old_dp_size + recv_begin, old_dp_size + recv_end))

        model = self.worker.model_runner.get_model()
        for new_worker_rank in sorted(ranks_to_send):
            batch_transfer_weights(
                model=model,
                is_sender=True,
                peer_rank=new_worker_rank,
                dp_group=standby_dp_group,
                expert_weights=model.expert_weights,
            )
        torch.accelerator.synchronize()

    def broadcast_expert_mapping(self):
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        expert_maps = self.worker.model_runner.shared_dict["expert_maps"]
        broadcast_expert_mapping(
            expert_maps=expert_maps,
            group=standby_dp_group,
            src_rank=0,
        )

    def _release_acl_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def switch_and_remove(self) -> None:
        self._release_acl_graphs()
        _replace_active_groups(world=None, dp=None, ep=None, eplb=None, node_count=None)
        _replace_ascend_active_groups(mc2=None, dynamic_eplb=None, fc3_quant_x=None)

    def switch_and_prepare(self) -> None:
        old_ep_size = get_ep_group().world_size
        self.worker.model_runner.shared_dict["old_ep_size"] = old_ep_size

        self._release_acl_graphs()
        _replace_active_groups(**pop_standby_groups())
        _replace_ascend_active_groups(**pop_ascend_standby_groups())

        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        new_ep_size = get_ep_group().world_size

        parallel_config.data_parallel_size = new_dp_size

        if reconfig_request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = reconfig_request.new_data_parallel_master_port
        self.worker.model_runner.dp_size = new_dp_size
        self.worker.model_runner.eplb_updator.comm_group = get_dynamic_eplb_group()
        self.worker.model_runner.eplb_updator.world_size = get_dynamic_eplb_group().world_size
        self.worker.model_runner.eplb_updator.cur_iterations = 0
        self.worker.model_runner.eplb_loader.comm_group = get_dynamic_eplb_group()

        # Reconfigure MoE modules with new EP size
        moe_modules = [
            module
            for module in self.worker.model_runner.model.modules()
            if (module.__class__.__name__ == "AscendFusedMoE" or module.__class__.__name__ == "AscendSharedFusedMoE")
        ]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts for module in moe_modules), (
            "All MoE modules must have the same number of experts"
        )
        for module in moe_modules:
            # module.local_num_experts = module.w2_weight.shape[0]
            num_logical_experts = self.worker.model_runner.shared_dict["expert_maps"].shape[-1]
            module.global_redundant_expert_num = module.local_num_experts * new_ep_size - num_logical_experts
            module.moe_config.num_experts = num_local_experts * new_ep_size
            module.global_num_experts = module.moe_config.num_experts
            tp_size = get_tp_group().world_size
            is_sequence_parallel = parallel_config.use_sequence_parallel_moe
            sp_size = tp_size if is_sequence_parallel else 1
            module.moe_parallel_config = FusedMoEParallelConfig.make(
                tp_size_=tp_size,
                pcp_size_=get_pcp_group().world_size,
                dp_size_=get_dp_group().world_size,
                sp_size_=sp_size,
                vllm_parallel_config=parallel_config,
            )
            module.moe_config.moe_parallel_config = module.moe_parallel_config

            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()

            with set_current_vllm_config(self.worker.vllm_config):
                if hasattr(module.quant_method, "quant_method") and isinstance(
                    module.quant_method.quant_method, AscendW8A8DynamicFusedMoEMethod
                ):
                    module.quant_method.quant_method = AscendW8A8DynamicFusedMoEMethod()
                setup_moe_comm_method(module.moe_config)

        if self.worker.vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE:
            # NOTE(yongji): when using stock torch.compile,
            # torch.compile is triggered during GPUModelRunner's load_model()
            # TODO(yongji):check do we need to re-trigger torch.compile here?
            # any changes to the tensor shapes in execution should already
            # be handled internally by torch.compile.
            backend = self.worker.vllm_config.compilation_config.init_backend(self.worker.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            self.worker.model_runner.model.compile(fullgraph=True, backend=backend)

        multi_block_table = self.worker.model_runner.input_batch.block_table
        saved_block_tables: list[tuple[torch.Tensor, torch.Tensor]] = []
        for bt in multi_block_table.block_tables:
            saved_block_tables.append((bt.block_table.gpu.clone(), bt.block_table.cpu.clone()))
        multi_block_table.clear()

        unlock_workspace()
        self.worker.compile_or_warm_up_model()
        lock_workspace()

        for bt, (saved_gpu, saved_cpu) in zip(multi_block_table.block_tables, saved_block_tables):
            bt.block_table.gpu.copy_(saved_gpu)
            bt.block_table.cpu.copy_(saved_cpu)

    def _perform_eplb_reshuffle(self):
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding...")
        eplb_loader = self.worker.model_runner.eplb_loader
        eplb_adaptor = self.worker.model_runner.eplb_adaptor
        eplb_updator = self.worker.model_runner.eplb_updator

        eplb_updator.compute_and_set_moe_load()
        # Wake up the EPLB worker to retrieve expert placement update information
        eplb_updator.wakeup_eplb_worker()
        # Retrieve the blocking update queue containing expert resharding information
        eplb_updator.update_info_all = eplb_updator.eplb_process.block_update_q.get()
        # Process each layer's expert redistribution information
        while eplb_updator.update_info_all:
            (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = (
                eplb_updator.update_info_all.pop(0)
            )
            # Convert logical to physical expert mapping to tensor for this rank
            log2phy_map_this_rank = torch.from_numpy(np.array(log2phy_map))
            eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            # Convert updated expert mapping to tensor for this rank
            updated_expert_map_this_rank = torch.from_numpy(np.array(updated_expert_map))
            # Get global expert map for this layer from shared dictionary
            # updated_global_expert_map_this_rank = self.worker.model_runner.shared_dict["expert_maps"][layer_id]
            # Generate device-to-device transfer tasks for expert weights
            eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info,
                expert_recv_info,
                updated_expert_map_this_rank,
                layer_id + eplb_adaptor.num_dense_layers,
            )
            # Execute asynchronous expert weight transfer
            reqs = []
            eplb_loader.asyn_expert_weight_transfer(reqs)
            # Update expert mapping and apply transferred weights
            eplb_loader.update_expert_map_and_weight(reqs)

        # Clear all MoE load statistics after resharding
        eplb_adaptor.model.clear_all_moe_loads()
        # Reset iteration counter for the updator
        eplb_updator.cur_iterations = 0
        # Synchronize NPU to ensure all transfers are complete
        torch_npu.npu.synchronize()

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed")

        self.worker.model_runner.shared_dict["scale"] = False
        self.worker.model_runner.shared_dict["old_ep_size"] = None
        self.worker.model_runner.shared_dict["new_ep_size"] = None

    def perform_eplb_reshuffle(self) -> None:
        new_ep_size = get_ep_group().world_size
        self.worker.model_runner.shared_dict["scale"] = True
        self.worker.model_runner.shared_dict["new_ep_size"] = new_ep_size

        self._perform_eplb_reshuffle()

    def perform_scale_down_eplb_reshuffle(self, new_dp_size: int) -> None:
        old_ep_size = get_ep_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        new_ep_size = new_dp_size * tp_size

        self.worker.model_runner.shared_dict["scale"] = True
        self.worker.model_runner.shared_dict["old_ep_size"] = old_ep_size
        self.worker.model_runner.shared_dict["new_ep_size"] = new_ep_size

        self._perform_eplb_reshuffle()

    def receive_weights(self) -> None:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        new_dp_size = dp_group.world_size
        dp_rank = self.worker.vllm_config.parallel_config.data_parallel_rank

        # Receive old_dp_size broadcasted during transfer_weights
        old_dp_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        old_dp_size_tensor = dp_group.tcp_store_group.broadcast(old_dp_size_tensor, 0)
        old_dp_size = int(old_dp_size_tensor[0].item())

        # Calculate which existing worker will send to this new worker
        num_new_workers = new_dp_size - old_dp_size
        new_worker_idx = dp_rank - old_dp_size
        num_dst_per_sender = num_new_workers // old_dp_size
        remainder = num_new_workers % old_dp_size

        if new_worker_idx < remainder * (num_dst_per_sender + 1):
            sender_rank = new_worker_idx // (num_dst_per_sender + 1)
        else:
            sender_rank = remainder + (new_worker_idx - remainder * (num_dst_per_sender + 1)) // num_dst_per_sender

        model = self.worker.model_runner.get_model()
        batch_transfer_weights(
            model=model,
            is_sender=False,
            peer_rank=sender_rank,
            dp_group=dp_group,
            expert_weights=model.expert_weights,
        )
        torch.accelerator.synchronize()

    def receive_expert_mapping(self) -> tuple[torch.Tensor, int, int]:
        dp_group = get_dp_group()
        assert isinstance(dp_group, StatelessGroupCoordinator)
        expert_maps = broadcast_expert_mapping(
            expert_maps=None,
            group=dp_group,
            src_rank=0,
        )
        num_local_experts = (expert_maps[0, 0] != -1).sum().item()
        num_logical_experts = expert_maps.shape[-1]

        return expert_maps, num_local_experts, num_logical_experts

    def prepare_new_worker(self) -> None:
        moe_modules = [
            module
            for module in self.worker.model_runner.model.modules()
            if (module.__class__.__name__ == "AscendFusedMoE" or module.__class__.__name__ == "AscendSharedFusedMoE")
        ]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                if hasattr(module.quant_method, "quant_method") and isinstance(
                    module.quant_method.quant_method, AscendW8A8DynamicFusedMoEMethod
                ):
                    try:
                        device_group = get_mc2_group().device_group
                        # TODO: Try local_rank = ep_group.rank_in_group
                        local_rank = get_mc2_group().rank_in_group
                        backend = device_group._get_backend(torch.device("npu"))
                        module.quant_method.quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(
                            local_rank
                        )
                    except AttributeError:
                        module.quant_method.quant_method.moe_all_to_all_group_name = ""
                setup_moe_comm_method(module.moe_config)
