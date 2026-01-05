from dataclasses import dataclass
from typing import Any

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                            AFDConnectorMetadata)


__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch

from torch.distributed.distributed_c10d import _get_default_group
import re

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.distributed.afd_transfer.afd_connector.metadata import (CAMM2NAFDConnectorMetadata)
from vllm.config import VllmConfig,CUDAGraphMode,CompilationLevel
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher

from vllm.utils import direct_register_custom_op
from vllm.forward_context import ForwardContext, get_forward_context

logger = init_logger(__name__)

# # TODO(yxj):move to ascend ,use kwargs 
# @dataclass
# class CAMM2NAFDConnectorMetadata:
#     def __init__(self):
#         self.topk_idx = None
#         self.topk_weights = None
#         self.moe_expert_num = 0
#         self.shared_expert_num = 0
#         self.scale = None
#         self.handle = None
#         self.quant_mode = 0
#         self.aiv_num = 0
#         self.batch_size = 0
#         self.h = 0
#         self.k = 0

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
        #ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        #attn_ranks = [i for i in range(attn_size)]
        # self rank atten:0 ffn:0
        self.rank = self.rank + self.ffn_size if role == "attention" else self.rank


        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {self.rank}")
        # TODO(jcz) : 这里要根据实际的num_of_stages创建，需要改成list
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:29888",
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd"
        )
        self.afd_pg2 = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:29888",
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd2"
        )
        self.hccl_comm_name = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        self.hccl_comm_name2 = self.afd_pg2._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        ffn_ranks = [i for i in range(0, self.ffn_size)]
        attn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), self.afd_pg)
        # TODO(yxj):m2n ae_group is different
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            self.process_group = init_model_parallel_group(sub_group_ranks,
                                                 self.rank,
                                                 backend="hccl",
                                                 group_name="ae")

        logger.info("m2n connector initialized")

        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized
                                  
    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    # TODO(jcz): 这里ubatch_idx的入参需要优化
    def send_attn_output(self, 
                         hidden_states: torch.Tensor,  
                         topk_weights: torch.Tensor, 
                         topk_idx:torch.Tensor, 
                         metadata: AFDConnectorMetadata,
                         ubatch_idx: int = 0) -> Any:
        if not self.use_aclgraph and self.ffn_size <= self.rank < self.ffn_size + self.min_size:
            for dst in self.dst_list:
                # Serialize object to tensor and get the size as well
                object_tensor = torch.frombuffer(pickle.dumps(metadata), dtype=torch.uint8)

                size_tensor = torch.tensor([object_tensor.numel()],
                                           dtype=torch.long,
                                           device="cpu")
                # Send object size
                torch.distributed.send(size_tensor,
                                       dst=dst,
                                       group=self.p2p_pg)

                # Send object
                torch.distributed.send(object_tensor,
                                       dst=dst,
                                       group=self.p2p_pg)
                print(f'attn_src_rank: {self.rank} send_attn_output metadata success')
        
        # batch_size = metadata.cam_afdconnector_data.batch_size
        # h = metadata.cam_afdconnector_data.h
        # k = metadata.cam_afdconnector_data.k
        # moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        # shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        # quant_mode = metadata.cam_afdconnector_data.quant_mode
        # aiv_num = metadata.cam_afdconnector_data.aiv_num
        # expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        # handle_out = torch_npu.cam_a2e(expandX = hidden_states, expertIds = topk_idx,
        #                     scales = topk_weights, commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
        #                     expandXOutDType = expandXOutDType,
        #                     commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
        #                     expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
        #                     sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num, rank = self.rank,
        #                     loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
        #                     groupEp = self.hccl_comm_name2 if ubatch_idx == 1 else self.hccl_comm_name,
        #                     aivNum = aiv_num)

        # return handle_out
        return torch.ops.vllm.cam_send_attn_output(hidden_states, topk_weights, topk_idx,
                                                self.hccl_comm_name,
                                                self.hccl_comm_name2,
                                                self.rank,
                                                self.ffn_size,
                                                self.attn_size)

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata, ubatch_idx: int = 0) -> torch.Tensor:
        # batch_size = metadata.cam_afdconnector_data.batch_size
        # h = metadata.cam_afdconnector_data.h
        # k = metadata.cam_afdconnector_data.k
        # moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        # shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        # aiv_num = metadata.cam_afdconnector_data.aiv_num
        # handle = metadata.cam_afdconnector_data.handle
        
        # output2 = torch_npu.cam_e2a(expandXOut = hidden_states, simulateExpertIds = handle[0],
        #                     simulateExpertScales = handle[1], expandIdx = handle[2],
        #                     epRecvCounts = handle[3],
        #                     commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
        #                     attenBatchSize = handle[4],
        #                     commId = 0,
        #                     batchSize = batch_size, hiddenSize = h, topk = k,
        #                     expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
        #                     sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num,
        #                     rank = self.rank,
        #                     loadBalancingRankNum=1, loadBalancingThreshold=0,
        #                     groupEp = self.hccl_comm_name2 if ubatch_idx == 1 else self.hccl_comm_name,
        #                     aivNum = aiv_num)

        # return output2
        return torch.ops.vllm.cam_recv_ffn_output(hidden_states,
                                                self.hccl_comm_name,
                                                self.hccl_comm_name2,
                                                self.rank,
                                                self.ffn_size,
                                                self.attn_size)
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMM2NAFDConnectorMetadata, ubatch_idx: int = 0):
        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        aiv_num = metadata.aiv_num
        handle = metadata.handle
        
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
                            groupEp = self.hccl_comm_name2 if ubatch_idx == 1 else self.hccl_comm_name,
                            aivNum = aiv_num)

        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: CAMM2NAFDConnectorMetadata, ubatch_idx: int = 0) -> Any: 
        afdmetadata = None
        if not self.use_aclgraph:
            src = (self.process_group.rank_in_group - 1) % self.process_group.world_size
            afdmetadata = self.process_group.recv_object(src)

            print(f'recv_attn_output start rank:{self.rank}')

        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        quant_mode = metadata.quant_mode
        aiv_num = metadata.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        output1 = torch_npu.cam_a2e(expandX = torch.tensor([], dtype=torch.bfloat16, device='npu'), expertIds = torch.tensor([], dtype=torch.int32, device='npu'),
                            scales = torch.tensor([], dtype=torch.float, device='npu'), commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            expandXOutDType = expandXOutDType,
                            commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num, rank = self.rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
                            groupEp = self.hccl_comm_name2 if ubatch_idx == 1 else self.hccl_comm_name,
                            aivNum = aiv_num)
        
        return output1, afdmetadata

def cam_send_attn_output_impl(hidden_states: torch.Tensor,
                              topk_weights: torch.Tensor,
                              topk_idx: torch.Tensor,
                              hccl_comm_name: str,
                              hccl_comm_name2: str,
                              rank: int,
                              ffn_size: int,
                              attn_size: int) -> torch.Tensor:
    ubatch_idx = get_forward_context().ubatch_idx
    if get_forward_context().cam_afdconnector_data is None:
        cam_afdconnector_data = CAMAFDConnectorMetadata(
            moe_expert_num = 64,
            shared_expert_num = 0,
            scale = None,
            handle = None,
            quant_mode = 0,
            aiv_num = 48,
            batch_size = 20,
            h = 2048,
            k = 8
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

    handle_out = torch_npu.cam_a2e(expandX = hidden_states, expertIds = topk_idx,
                        scales = topk_weights, commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                        expandXOutDType = expandXOutDType,
                        commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
                        expertRankSize = ffn_size, attentionRankSize = attn_size,
                        sharedExpertNum = shared_expert_num,
                        totalExpertNum = moe_expert_num + shared_expert_num,
                        rank = rank,
                        loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
                        groupEp = hccl_comm_name2 if ubatch_idx == 1 else hccl_comm_name,
                        aivNum = aiv_num)
    hidden_states1, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, \
        simulateExpertIds, simulateExpertScales, attenBatchSize = handle_out[0:8]
    handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
    cam_metadata.handle = handle
    get_forward_context().cam_afdconnector_data = cam_metadata
    return hidden_states

def cam_send_attn_output_fake_impl(hidden_states: torch.Tensor,
                                    topk_weights: torch.Tensor,
                                    topk_idx: torch.Tensor,
                                    hccl_comm_name: str,
                                    hccl_comm_name2: str,
                                    rank: int,
                                    ffn_size: int,
                                    attn_size: int) -> torch.Tensor:
    return hidden_states

def cam_recv_ffn_output_impl(hidden_states: torch.Tensor,
                              hccl_comm_name: str,
                              hccl_comm_name2: str,
                              rank: int,
                              ffn_size: int,
                              attn_size: int) -> torch.Tensor:
    cam_metadata = get_forward_context().cam_afdconnector_data
    assert cam_metadata is not None, "cam_metadata is None"
    ubatch_idx = get_forward_context().ubatch_idx
    batch_size = cam_metadata.batch_size
    h = cam_metadata.h
    k = cam_metadata.k
    moe_expert_num = cam_metadata.moe_expert_num
    shared_expert_num = cam_metadata.shared_expert_num
    aiv_num = cam_metadata.aiv_num
    handle = cam_metadata.handle
    
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
                        groupEp = hccl_comm_name2 if ubatch_idx == 1 else hccl_comm_name,
                        aivNum = aiv_num)

    return output2

def cam_recv_ffn_output_fake_impl(hidden_states: torch.Tensor,
                                  hccl_comm_name: str,
                                  hccl_comm_name2: str,
                                  rank: int,
                                  ffn_size: int,
                                  attn_size: int) -> torch.Tensor:
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