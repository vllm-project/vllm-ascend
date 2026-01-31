from dataclasses import dataclass
from typing import Any, Optional

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                            AFDConnectorMetadata)


__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch
import pickle

from torch.distributed.distributed_c10d import _get_default_group
import re

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm_ascend.distributed.metadata import (CAMM2NAFDConnectorMetadata)
from vllm.config import VllmConfig,CUDAGraphMode,CompilationLevel
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher

from vllm.utils import direct_register_custom_op
from vllm.forward_context import ForwardContext, get_forward_context
from vllm_ascend.utils import npu_stream_switch_within_graph

logger = init_logger(__name__)

def _get_group_ep(ubatch_idx: int, hccl_comm_name: str, hccl_comm_name2: str, hccl_comm_name3: Optional[str]) -> str:
    groupEp = hccl_comm_name
    if ubatch_idx == 1:
        groupEp = hccl_comm_name2
    elif ubatch_idx == 2:
        assert hccl_comm_name3 is not None
        groupEp = hccl_comm_name3
    return groupEp

class CAMM2NAFDConnector(AFDConnectorBase):
    def __init__(self,
                 rank: int,
                 local_rank: int,
                 config: "VllmConfig"
                 ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self.hf_config = config.model_config.hf_config
        self.scheduler_config = config.scheduler_config
        decode_max_num_seqs = getattr(self.scheduler_config,
                                      'decode_max_num_seqs', 0)
        self.max_num_reqs = max(self.scheduler_config.max_num_seqs,
                                decode_max_num_seqs)
        self.attn_size = 0
        self.ffn_size = 0
        self.use_aclgraph = self._use_aclgraph()
        self.dst_list = []
        print(f'self.use_aclgraph in CAMM2NAFDConnector is {self.use_aclgraph}')

    def _use_aclgraph(self) -> bool:
        return self.config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and self.config.compilation_config.level == CompilationLevel.PIECEWISE and not self.config.model_config.enforce_eager

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass
    
    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())

        self.min_size = min(self.ffn_size, self.attn_size)
        world_rank = self.rank + self.ffn_size if role == "attention" else self.rank
        # p2p_rank: 所有FFN [0, ffn_size), 前min_size个Attention [ffn_size, ffn_size+min_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        self.rank = world_rank

        print(f"world_size = {self.ffn_size + self.attn_size}, world_rank = {self.rank}")
        logger.debug(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {self.rank}")
        # TODO(jcz) : 这里要根据实际的num_of_stages创建，需要改成list
        self.afd_pg_list = []
        self.hccl_comm_name_list = []
        num_ubatches = self.config.parallel_config.num_ubatches if self.config.parallel_config.num_ubatches else 1
        for i in range(num_ubatches):
            group_name = "afd" + str(i) if i > 0 else "afd"
            afd_pg = init_afd_process_group(
                backend="hccl",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.attn_size,
                rank=self.rank,
                group_name=group_name
            )
            self.afd_pg_list.append(afd_pg)
            self.hccl_comm_name_list.append(afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        self.hccl_comm_name = self.hccl_comm_name_list[0]
        self.hccl_comm_name2 = self.hccl_comm_name_list[1] if num_ubatches > 1 else self.hccl_comm_name
        self.hccl_comm_name3 = self.hccl_comm_name_list[2] if num_ubatches > 2 else None
        
        # 所有FFN和前min_size的Attention参与p2p通信
        # 所有FFN: world_rank in [0, ffn_size), 前min_size个Attention: world_rank in [ffn_size, ffn_size+min_size)
        import datetime
        timeout = datetime.timedelta(seconds=30000)
        if self.is_vaild_rank_for_inequal_AF(self.rank):
            self.p2p_pg = init_afd_process_group(
                backend="hccl",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timeout # TODO(yxj):use timeout set
            )

        # 前min_size的Attention向多个FFN发送metadata（1对多映射）
        # attn_i 向所有 ffn_j (其中 j % min_size == i) 发送
        if self.is_attn_top_min_size_rank(self.rank):
            local_attn_rank = self.rank - self.ffn_size
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size

        self.aiv_num = int(self.config.afd_config.multistream_info["core_num"]) if self.config.afd_config.is_multistream else 48

        logger.debug(f"[CAM] world_rank={self.rank}, p2p_rank={self.p2p_rank}, min_size={self.min_size}, "
                    f"dst_list={self.dst_list}, cam connector initialized")
        logger.info("m2n connector initialized")

        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def configure_metadata(self, metadata: "AFDConnectorMetadata", **kwargs) -> None:
        if metadata.connector_data is None:
            metadata.connector_data = CAMM2NAFDConnectorMetadata()

        config = kwargs.get('config')
        batch_size = kwargs.get('batch_size')
        if config:
            metadata.connector_data.moe_expert_num = config.n_routed_experts
            # TODO: quant_mode and aiv_num read from config
            metadata.connector_data.quant_mode = 0
            metadata.connector_data.aiv_num = self.aiv_num
            metadata.connector_data.scale = None
            metadata.connector_data.batch_size = batch_size
            metadata.connector_data.h = config.hidden_size
            metadata.connector_data.k = config.num_experts_per_tok

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Any] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from vllm_ascend.ops.moe.experts_selector import select_experts
        return select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            e_score_correction_bias=e_score_correction_bias
        )

    def compute_moe(self, experts, hidden_states, **kwargs):
        group_list = kwargs.get('group_list')
        dynamic_scales = kwargs.get('dynamic_scales')

        return experts.afd_m2n_ffn_compute(
                layer=experts,
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scale=dynamic_scales,
                connector_name="camm2nconnector"
                )

    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    # TODO(jcz): 这里ubatch_idx的入参需要优化
    def send_attn_output(self, 
                         hidden_states: torch.Tensor,
                         metadata: AFDConnectorMetadata,
                         **kwargs) -> Any:

        # Extract from kwargs
        topk_weights = kwargs.get('topk_weights')
        topk_idx = kwargs.get('topk_ids')

        if metadata.connector_data:
            get_forward_context().cam_afdconnector_data = metadata.connector_data

        multistream_enable = False if metadata.layer_idx == self.hf_config.first_k_dense_replace else self.config.afd_config.is_multistream # dense层的后一层不分流
        return torch.ops.vllm.cam_send_attn_output(hidden_states, topk_weights, topk_idx,
                                                self.hccl_comm_name,
                                                self.hccl_comm_name2,
                                                self.hccl_comm_name3,
                                                self.rank,
                                                self.ffn_size,
                                                self.attn_size,
                                                self.hf_config.n_routed_experts,
                                                self.max_num_reqs,
                                                self.hf_config.hidden_size,
                                                self.hf_config.num_experts_per_tok,
                                                multistream_enable,
                                                self.aiv_num), None


    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: Optional["AFDConnectorMetadata"] = None) -> torch.Tensor:
        return torch.ops.vllm.cam_recv_ffn_output(hidden_states,
                                                self.hccl_comm_name,
                                                self.hccl_comm_name2,
                                                self.hccl_comm_name3,
                                                self.rank,
                                                self.ffn_size,
                                                self.attn_size,
                                                self.config.afd_config.is_multistream)
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMM2NAFDConnectorMetadata, **kwargs):
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        aiv_num = metadata.aiv_num
        handle = metadata.handle
        
        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)

        torch_npu.cam_e2a(expandXOut = ffn_output, simulateExpertIds = handle[0],
                            simulateExpertScales = handle[1],
                            expandIdx = handle[2],
                            epRecvCounts = handle[3],
                            commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            attenBatchSize = handle[4],
                            commId = 0,
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num,
                            rank = self.rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0,
                            groupEp = groupEp,
                            aivNum = aiv_num)

        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: Optional[Any] = None, **kwargs) -> Any:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        afdmetadata = None
        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        quant_mode = metadata.quant_mode
        aiv_num = metadata.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)

        outputs = torch_npu.cam_a2e(expandX=torch.tensor([], dtype=torch.bfloat16, device='npu'),
                                    expertIds=torch.tensor([], dtype=torch.int32, device='npu'),
                                    scales=torch.tensor([], dtype=torch.float, device='npu'),
                                    commArgs=torch.tensor([], dtype=torch.float16, device='npu'),
                                    expandXOutDType=expandXOutDType,
                                    commId=0, batchSize=batch_size, hiddenSize=h, topk=k,
                                    expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                    sharedExpertNum=shared_expert_num,
                                    totalExpertNum=moe_expert_num + shared_expert_num, rank=self.rank,
                                    loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant=quant_mode,
                                    groupEp=groupEp,
                                    aivNum=aiv_num)
        
        # [hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize]
        expertTokenNums = outputs[3].to(torch.int64)  # expertTokenNums
        from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
        return AFDRecvOutput(
            hidden_states=outputs[0],
            metadata=afdmetadata,
            dynamic_scales=outputs[1],
            expand_idx=outputs[2],
            group_list=expertTokenNums,  # expertTokenNums
            ep_recv_counts=outputs[4],
            topk_ids=outputs[5],  # simulateExpertIds
            topk_weights=outputs[6],  # simulateExpertScales
            atten_batch_size=outputs[7]
        )
    
    def is_vaild_rank_for_inequal_AF(self,rank):
        # Only support ffn rank < attn rank
        return ((rank >= self.ffn_size and rank < self.ffn_size + self.min_size) or rank < self.ffn_size)
    
    def is_attn_top_min_size_rank(self,rank):
        # Only support ffn rank < attn rank
        return (rank >= self.ffn_size and rank < self.ffn_size + self.min_size)

    def send_is_ubatch(self, data):
        for dst in self.dst_list:
            object_bytes = pickle.dumps(data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            object_tensor_npu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="npu")
            object_tensor_npu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                        dtype=torch.long,
                                        device="npu")

            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_npu, dst=dst, group=self.p2p_pg)

    def recv_is_ubatch(self):
        src = self.p2p_rank % self.min_size + self.ffn_size

        size_tensor = torch.empty(1, dtype=torch.long, device="npu")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)
        object_tensor_npu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="npu")
        rank_object = torch.distributed.recv(object_tensor_npu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_npu.cpu()
        data = pickle.loads(object_tensor_cpu.numpy().tobytes())
        return data

    def create_recv_metadata(self, **kwargs):
        max_num_tokens = kwargs.get('max_num_tokens', 0)
        hf_config = self.config.model_config.hf_config

        return CAMM2NAFDConnectorMetadata(
            moe_expert_num = hf_config.n_routed_experts,
            shared_expert_num = 0,
            scale = None,
            handle = None,
            quant_mode = 0,
            aiv_num = self.aiv_num,
            batch_size = max_num_tokens,
            h = hf_config.hidden_size,
            k = hf_config.num_experts_per_tok
        )

    def update_metadata(self, metadata, recv_output):
        metadata.handle = [
            recv_output.topk_ids,
            recv_output.topk_weights,
            recv_output.expand_idx,
            recv_output.ep_recv_counts,
            recv_output.atten_batch_size
        ]


def cam_send_attn_output_impl(hidden_states: torch.Tensor,
                              topk_weights: torch.Tensor,
                              topk_idx: torch.Tensor,
                              hccl_comm_name: str,
                              hccl_comm_name2: str,
                              hccl_comm_name3: Optional[str],
                              rank: int,
                              ffn_size: int,
                              attn_size: int,
                              moe_expert_num:int,
                              batch_size:int,
                              h:int,
                              k:int,
                              multistream_enable: bool,
                              aiv_num: int) -> torch.Tensor:
    ubatch_idx = get_forward_context().ubatch_idx
    comm_stream = get_forward_context().afd_comm_stream
    comm_event = get_forward_context().afd_comm_event
    if get_forward_context().cam_afdconnector_data is None:
        cam_afdconnector_data = CAMM2NAFDConnectorMetadata(
                    moe_expert_num = moe_expert_num,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = aiv_num,
                    batch_size = batch_size,
                    h = h,
                    k = k
                )
        get_forward_context().cam_afdconnector_data = cam_afdconnector_data

    cam_metadata = get_forward_context().cam_afdconnector_data
    batch_size = cam_metadata.batch_size
    h = cam_metadata.h
    k = cam_metadata.k
    moe_expert_num = cam_metadata.moe_expert_num
    shared_expert_num = cam_metadata.shared_expert_num
    quant_mode = cam_metadata.quant_mode
    aiv_num = cam_metadata.aiv_num
    expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

    groupEp = _get_group_ep(ubatch_idx, hccl_comm_name, hccl_comm_name2, hccl_comm_name3)

    curr_stream = torch.npu.current_stream()
    with npu_stream_switch_within_graph(curr_stream, comm_stream, multistream_enable):
        handle_out = torch_npu.cam_a2e(expandX = hidden_states, expertIds = topk_idx,
                            scales = topk_weights, commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            expandXOutDType = expandXOutDType,
                            commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = ffn_size, attentionRankSize = attn_size,
                            sharedExpertNum = shared_expert_num,
                            totalExpertNum = moe_expert_num + shared_expert_num,
                            rank = rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
                            groupEp = groupEp,
                            aivNum = aiv_num)
        hidden_states1, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, \
            simulateExpertIds, simulateExpertScales, attenBatchSize = handle_out[0:8]
        handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
        cam_metadata.handle = handle
        get_forward_context().cam_afdconnector_data = cam_metadata
        if multistream_enable:
            comm_event.record(comm_stream)
    return hidden_states

def cam_send_attn_output_fake_impl(hidden_states: torch.Tensor,
                                    topk_weights: torch.Tensor,
                                    topk_idx: torch.Tensor,
                                    hccl_comm_name: str,
                                    hccl_comm_name2: str,
                                    hccl_comm_name3: Optional[str],
                                    rank: int,
                                    ffn_size: int,
                                    attn_size: int,
                                    moe_expert_num:int,
                                    batch_size:int,
                                    h:int,
                                    k:int,
                                    multistream_enable: bool,
                                    aiv_num: int) -> torch.Tensor:
    return hidden_states

def cam_recv_ffn_output_impl(hidden_states: torch.Tensor,
                              hccl_comm_name: str,
                              hccl_comm_name2: str,
                              hccl_comm_name3: Optional[str],
                              rank: int,
                              ffn_size: int,
                              attn_size: int,
                              multistream_enable: bool) -> torch.Tensor:
    cam_metadata = get_forward_context().cam_afdconnector_data
    assert cam_metadata is not None, "cam_metadata is None"
    ubatch_idx = get_forward_context().ubatch_idx
    comm_event = get_forward_context().afd_comm_event
    batch_size = cam_metadata.batch_size
    h = cam_metadata.h
    k = cam_metadata.k
    moe_expert_num = cam_metadata.moe_expert_num
    shared_expert_num = cam_metadata.shared_expert_num
    aiv_num = cam_metadata.aiv_num
    handle = cam_metadata.handle
    
    groupEp = _get_group_ep(ubatch_idx, hccl_comm_name, hccl_comm_name2, hccl_comm_name3)

    if multistream_enable:
        curr_stream = torch.npu.current_stream()
        comm_event.wait(curr_stream)
    output2 = torch_npu.cam_e2a(expandXOut = hidden_states, simulateExpertIds = handle[0],
                        simulateExpertScales = handle[1], expandIdx = handle[2],
                        epRecvCounts = handle[3],
                        commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                        attenBatchSize = handle[4],
                        commId = 0,
                        batchSize = batch_size, hiddenSize = h, topk = k,
                        expertRankSize = ffn_size, attentionRankSize = attn_size,
                        sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num,
                        rank = rank,
                        loadBalancingRankNum=1, loadBalancingThreshold=0,
                        groupEp = groupEp,
                        aivNum = aiv_num)

    return output2

    

def cam_recv_ffn_output_fake_impl(hidden_states: torch.Tensor,
                                  hccl_comm_name: str,
                                  hccl_comm_name2: str,
                                  hccl_comm_name3: Optional[str],
                                  rank: int,
                                  ffn_size: int,
                                  attn_size: int,
                                  multistream_enable: bool) -> torch.Tensor:
    return hidden_states

direct_register_custom_op(op_name="cam_send_attn_output",
                          op_func=cam_send_attn_output_impl,
                          fake_impl=cam_send_attn_output_fake_impl,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

direct_register_custom_op(op_name="cam_recv_ffn_output",
                          op_func=cam_recv_ffn_output_impl,
                          fake_impl=cam_recv_ffn_output_fake_impl,
                          mutates_args=[],

                          dispatch_key="PrivateUse1")
