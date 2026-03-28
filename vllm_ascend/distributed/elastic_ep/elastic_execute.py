# NOTE:
# This file is adapted from vLLM's elastic_execute.py
#
# Key differences:
# 1. Device-specific adaptations: Replaces CUDA-specific operations with NPU (Ascend) equivalents
#    - Uses `torch_npu` instead of CUDA APIs
#    - Replaces `torch.accelerator.synchronize()` with `torch.npu.synchronize()`
#    - Replaces `torch.accelerator.empty_cache()` with `torch.npu.empty_cache()`
#    - Uses `ACLGraphWrapper` instead of `CUDAGraphWrapper` for graph management
#
# 2. Custom weight transfer implementation: Implements `ascend_batch_transfer_weights()`
#    - Adds support for quantized weight names (aclnn_input_scale, aclnn_input_scale_reciprocal, aclnn_input_offset)
#    - Uses threading lock (`_PATCH_LOCK`) for thread-safe weight transfer patching
#
# 3. Enhanced broadcast_expert_mapping: Simplified signature and implementation
#    - Removed `physical_to_logical`, `num_local_physical_experts`, `num_logical_experts` parameters
#    - Uses `expert_maps` tensor directly for broadcasting
#
# 4. Extended AscendElasticEPScalingExecutor class:
#    - Adds `_use_ascend_transfer_impl()` context manager for patching weight transfer
#    - Implements `_release_acl_graphs()` to clear ACL graphs instead of CUDA graphs
#    - Adds `_replace_ascend_active_groups()` calls for Ascend-specific group management
#    - Integrates with `create_ascend_standby_groups()` and `pop_ascend_standby_groups()`
#    - Adds support for Ascend-specific MoE modules (AscendFusedMoE, AscendSharedFusedMoE)
#    - Handles Ascend-specific quantization method (AscendW8A8DynamicFusedMoEMethod)
#    - Integrates with `get_mc2_group()` and `get_dynamic_eplb_group()` for Ascend communication
#    - Adds `setup_moe_comm_method()` calls for MoE communication setup
#
# 5. EPLB (Expert Parallel Load Balancing) adaptations:
#    - Uses `eplb_loader`, `eplb_adaptor`, `eplb_updator` from model_runner
#    - Implements `_perform_eplb_reshuffle()` with expert resharding logic
#    - Handles dynamic EPLB configuration via `get_ascend_config().eplb_config`
#
# ============================================================

import copy
import gc
import os
import threading
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn as nn
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
    get_standby_ep_group,
    pop_standby_groups,
)
from vllm.distributed.parallel_state import _replace_active_groups
from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
from vllm.model_executor.layers.fused_moe.layer import FusedMoE, FusedMoEParallelConfig
from vllm.v1.attention.backend import AttentionImplBase
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.workspace import lock_workspace, unlock_workspace

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation.acl_graph import (
    ACLGraphWrapper,
    reset_graph_params,
    set_draft_graph_params,
    set_graph_params,
)
from vllm_ascend.distributed.elastic_ep.standby_state import (
    create_ascend_standby_groups,
    pop_ascend_standby_groups,
)
from vllm_ascend.distributed.parallel_state import (
    _replace_ascend_active_groups,
    get_mc2_group,
)
from vllm_ascend.distributed.utils import use_stateless_pg_with_world_registration
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method
from vllm_ascend.quantization.methods import AscendW4A8DynamicFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

from .eplb_manager import ElasticEplbManager, generate_expert_maps_file

_PATCH_LOCK = threading.Lock()


def ascend_batch_transfer_weights(
    model: nn.Module,
    is_sender: bool,
    peer_rank: int,
    dp_group: StatelessGroupCoordinator,
    expert_weights: Sequence[Iterable[torch.Tensor]],
) -> None:
    device_comm = dp_group.device_communicator
    tcp_store_group = dp_group.tcp_store_group
    if device_comm is None:
        raise ValueError("No device communicator found")

    expert_weights_set = set()
    for weight_group in expert_weights:
        for weight in weight_group:
            expert_weights_set.add(weight.data_ptr())

    state_dict = model.state_dict()
    all_params = []
    all_params_ptrs = set()
    all_params_name = []

    for name, param in state_dict.items():
        if name.endswith("expert_map"):
            continue
        ptr = param.data_ptr()
        if ptr not in expert_weights_set and ptr not in all_params_ptrs:
            if param.device.type == "npu":
                all_params.append(param.data)
                all_params_ptrs.add(ptr)
                all_params_name.append(name)

    def handle_sub_module(submodule, submodule_name):
        for attr_name, attr_value in submodule.__dict__.items():
            if attr_name.endswith("expert_map"):
                continue
            if isinstance(attr_value, torch.Tensor):
                data_ptr = attr_value.data_ptr()
                if data_ptr not in expert_weights_set and data_ptr not in all_params_ptrs:
                    if attr_value.device.type == "npu":
                        all_params.append(attr_value)
                        all_params_ptrs.add(data_ptr)
                        all_params_name.append(submodule_name + "." + attr_name)
            if isinstance(attr_value, AttentionImplBase):
                handle_sub_module(attr_value, submodule_name + "." + attr_name)

    for module_name, module in model.named_modules():
        handle_sub_module(module, module_name)

    expert_map_params = []
    if not get_ascend_config().eplb_config.dynamic_eplb:
        for module_name, module in model.named_modules():
            if (expert_map := getattr(module, "expert_map", None)) is not None:
                if isinstance(expert_map, torch.Tensor):
                    expert_map_params.append((expert_map.npu(), expert_map))
                    all_params_name.append(module_name + "." + "expert_map")
        all_params.extend([npu_tensor for npu_tensor, _ in expert_map_params])

    if is_sender:
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
    else:
        peer_rank_all_params_name = tcp_store_group.recv_obj(src=peer_rank)
        tcp_store_group.send_obj(all_params_name, dst=peer_rank)

    if len(all_params_name) != len(peer_rank_all_params_name):
        common = list(set(all_params_name) & set(peer_rank_all_params_name))
        ids = [all_params_name.index(name) for name in common]
        all_params = [param for idx, param in enumerate(all_params) if idx in ids]

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

    if len(expert_map_params) > 0:
        for npu_tensor, cpu_tensor in expert_map_params:
            cpu_tensor.copy_(npu_tensor)


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


def setup_moe_comm_and_quant_method(module: nn.Module) -> None:
    if isinstance(
        quant_method := getattr(module.quant_method, "quant_method", None),
        (AscendW8A8DynamicFusedMoEMethod, AscendW4A8DynamicFusedMoEMethod),
    ):
        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = get_mc2_group().rank_in_group
            backend = device_group._get_backend(torch.device("npu"))
            quant_method.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            quant_method.moe_all_to_all_group_name = ""
    setup_moe_comm_method(module.moe_config)


class AscendElasticEPScalingExecutor(ElasticEPScalingExecutor):
    def __init__(self, worker):
        super().__init__(worker)
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        if not self.dynamic_eplb and os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", "0") != "1":
            get_ascend_config().eplb_config.expert_map_path = generate_expert_maps_file()
        self._eplb_manager: ElasticEplbManager | None = None
        self.old_ep_size: int | None = None

    def init_eplb_manager(self):
        self._eplb_manager = ElasticEplbManager(self.worker)

    @property
    def eplb_manager(self) -> ElasticEplbManager:
        if self._eplb_manager is None:
            raise RuntimeError("eplb_manager is not initialized")
        return self._eplb_manager

    @contextmanager
    def _use_ascend_transfer_impl(self):
        with patch(
            "vllm.distributed.elastic_ep.elastic_execute.batch_transfer_weights", new=ascend_batch_transfer_weights
        ):
            yield

    def load_model(self) -> None:
        (
            expert_maps,
            num_local_experts,
            num_logical_experts,
        ) = self.receive_expert_mapping()
        n_layers, old_ep_size, n_experts = expert_maps.shape
        new_ep_size = get_ep_group().world_size
        n_redundant = new_ep_size * num_local_experts - num_logical_experts
        get_ascend_config().eplb_config.num_redundant_experts = n_redundant
        if self.dynamic_eplb:
            self.worker.model_runner.shared_dict["expert_maps"] = expert_maps
        else:
            with set_current_vllm_config(self.worker.vllm_config):
                get_ascend_config().eplb_config.expert_map_path = generate_expert_maps_file()
        self.old_ep_size = old_ep_size
        self.worker.load_model(load_dummy_weights=True)
        self.eplb_manager.expert_maps = expert_maps

    def create_standby_groups(self, reconfig_request: ReconfigureDistributedRequest) -> None:
        self.reconfig_request = reconfig_request
        new_dp_size = reconfig_request.new_data_parallel_size
        world_size = self.worker.vllm_config.parallel_config.world_size
        new_world_size_across_dp = world_size * new_dp_size
        updated_config = copy.copy(self.worker.vllm_config)
        updated_config.parallel_config = copy.deepcopy(self.worker.vllm_config.parallel_config)
        updated_config.parallel_config.data_parallel_size = new_dp_size
        with set_current_vllm_config(updated_config), use_stateless_pg_with_world_registration():
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
        model = self.worker.model_runner.get_model()
        if self.dynamic_eplb:
            model.expert_weights = [item[1] for item in self.worker.model_runner.eplb_adaptor.param_dict.items()]
        else:
            model.expert_weights = []
        get_standby_dp_group().barrier()
        get_tp_group().barrier()
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().transfer_weights(old_dp_size=old_dp_size, new_dp_size=new_dp_size)

    def broadcast_expert_mapping(self):
        standby_dp_group = get_standby_dp_group()
        assert standby_dp_group is not None
        expert_maps = self.eplb_manager.get_expert_maps()
        broadcast_expert_mapping(
            expert_maps=expert_maps,
            group=standby_dp_group,
            src_rank=0,
        )

    def _release_cuda_graphs(self) -> None:
        if isinstance(self.worker.model_runner.model, UBatchWrapper):
            raise RuntimeError("DBO is not yet supported in elastic EP")

        ACLGraphWrapper.clear_all_graphs()

        torch.compiler.reset()
        with set_current_vllm_config(self.worker.vllm_config):
            reset_compile_wrapper(self.worker.model_runner.get_model())

        reset_graph_params()

        capture_descs = self.worker.model_runner.cudagraph_dispatcher.get_capture_descs()
        capture_sizes = sorted({desc.num_tokens for _, descs in capture_descs for desc in descs})
        if self.worker.model_runner.use_aclgraph:
            set_graph_params(capture_sizes)
            if self.worker.model_runner.speculative_config:
                set_draft_graph_params(capture_sizes)

        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()

    def switch_and_remove(self) -> None:
        self._release_cuda_graphs()
        with use_stateless_pg_with_world_registration():
            _replace_active_groups(world=None, dp=None, ep=None, eplb=None, node_count=None)
            _replace_ascend_active_groups(mc2=None, dynamic_eplb=None, fc3_quant_x=None)

    def switch_and_prepare(self) -> None:
        self.old_ep_size = get_ep_group().world_size

        self._release_cuda_graphs()

        parallel_config = self.worker.vllm_config.parallel_config
        reconfig_request = self.reconfig_request
        assert reconfig_request is not None
        new_dp_size = reconfig_request.new_data_parallel_size
        new_ep_size = get_standby_ep_group().world_size
        self.new_ep_size = new_ep_size
        parallel_config.data_parallel_size = new_dp_size

        with use_stateless_pg_with_world_registration():
            _replace_active_groups(**pop_standby_groups())
            _replace_ascend_active_groups(**pop_ascend_standby_groups())

        if reconfig_request.new_data_parallel_rank != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if reconfig_request.new_data_parallel_rank_local != ReconfigureRankType.KEEP_CURRENT_RANK:
            parallel_config.data_parallel_rank_local = reconfig_request.new_data_parallel_rank_local
        parallel_config.data_parallel_master_ip = reconfig_request.new_data_parallel_master_ip
        parallel_config.data_parallel_master_port = reconfig_request.new_data_parallel_master_port
        self.worker.model_runner.dp_size = new_dp_size
        self.eplb_manager.reset_eplb_updator()
        self.eplb_manager.set_new_comm_group()

        # Reconfigure MoE modules with new EP size
        moe_modules = [module for module in self.worker.model_runner.model.modules() if isinstance(module, FusedMoE)]
        num_local_experts = moe_modules[0].moe_config.num_local_experts
        assert all(module.moe_config.num_local_experts == num_local_experts for module in moe_modules), (
            "All MoE modules must have the same number of experts"
        )
        for module in moe_modules:
            num_logical_experts = self.eplb_manager.get_expert_maps().shape[-1]
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
                setup_moe_comm_and_quant_method(module)

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

    def perform_eplb_reshuffle(self) -> None:
        new_ep_size = get_ep_group().world_size
        self.eplb_manager.eplb(self.old_ep_size, new_ep_size)

    def perform_scale_down_eplb_reshuffle(self, new_dp_size: int) -> None:
        old_ep_size = get_ep_group().world_size
        parallel_config = self.worker.vllm_config.parallel_config
        tp_size = parallel_config.tensor_parallel_size
        new_ep_size = new_dp_size * tp_size

        self.eplb_manager.eplb(old_ep_size, new_ep_size)

    def receive_weights(self) -> None:
        model = self.worker.model_runner.get_model()
        if self.dynamic_eplb:
            model.expert_weights = [item[1] for item in self.worker.model_runner.eplb_adaptor.param_dict.items()]
        else:
            model.expert_weights = []
        get_dp_group().barrier()
        get_tp_group().barrier()
        with _PATCH_LOCK, self._use_ascend_transfer_impl():
            super().receive_weights()

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
        moe_modules = [module for module in self.worker.model_runner.model.modules() if isinstance(module, FusedMoE)]
        for module in moe_modules:
            with set_current_vllm_config(self.worker.vllm_config):
                setup_moe_comm_and_quant_method(module)
