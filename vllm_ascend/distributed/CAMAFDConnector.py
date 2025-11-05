from dataclasses import dataclass
from typing import Any

from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                            AFDConnectorMetadata)


__all__ = ["AFDConnectorBase", "AFDConnectorMetadata", "AFDConnectorFactory"]

import torch_npu
import torch

from torch.distributed.distributed_c10d import _get_default_group
from vllm.distributed.parallel_state import init_afd_process_group, DefaultProcessGroupSwitcher
import re

import torch
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group
from vllm.logger import init_logger
from vllm.config import VllmConfig
logger = init_logger(__name__)


# # TODO(yxj):move to ascend ,use kwargs 
# @dataclass
# class CAMAFDConnectorMetadata:
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

class CAMAFDConnector(AFDConnectorBase):
    def __init__(self, 
        rank: int, 
        local_rank: int,
        config: "VllmConfig"
)-> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self.attn_size = 0
        self.ffn_size = 0
    
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
        # TODO : get backend to replace hardcode
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:29509",
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd"
        )
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
    def send_attn_output(self, 
                         hidden_states: torch.Tensor,  
                         topk_weights: torch.Tensor, 
                         topk_idx:torch.Tensor, 
                         metadata: AFDConnectorMetadata) -> Any:
        print(f'send_attn_output start rank:{self.rank}')
        dst = (self.process_group.rank_in_group + 1) % self.process_group.world_size
        print(f'send_attn_output dst is {dst}')
        self.process_group.send_object(metadata,dst)
        
        batch_size = metadata.cam_afdconnector_data.batch_size
        h = metadata.cam_afdconnector_data.h
        k = metadata.cam_afdconnector_data.k
        moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        quant_mode = metadata.cam_afdconnector_data.quant_mode
        aiv_num = metadata.cam_afdconnector_data.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        torch_npu.cam_a2e(expandX = hidden_states, expertIds = topk_idx,
                            scales = topk_weights, commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            expandXOutDType = expandXOutDType,
                            commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num, rank = self.rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
                            groupEp = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                            aivNum = aiv_num)
        
        print(f'send_attn_output end rank:{self.rank}')
        return

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, metadata: AFDConnectorMetadata) -> torch.Tensor:
        print(f'recv_ffn_output start rank:{self.rank}')
        batch_size = metadata.cam_afdconnector_data.batch_size
        h = metadata.cam_afdconnector_data.h
        k = metadata.cam_afdconnector_data.k
        moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        aiv_num = metadata.cam_afdconnector_data.aiv_num
        
        output2 = torch_npu.cam_e2a(expandXOut = torch.tensor([], dtype=torch.bfloat16, device='npu'), simulateExpertIds = torch.tensor([], dtype=torch.int32, device='npu'),
                            simulateExpertScales = torch.tensor([], dtype=torch.float, device='npu'), expandIdx = torch.tensor([], dtype=torch.int32, device='npu'),
                            epRecvCounts = torch.tensor([], dtype=torch.int32, device='npu'),
                            commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            attenBatchSize = torch.tensor([], dtype=torch.int32, device='npu'),
                            commId = 0,
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num,
                            rank = self.rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0,
                            groupEp = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                            aivNum = aiv_num)
        print(f'recv_ffn_output end rank:{self.rank}')
        return output2
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata):
        print(f'send_ffn_output start rank:{self.rank}')
        batch_size = metadata.cam_afdconnector_data.batch_size
        h = metadata.cam_afdconnector_data.h
        k = metadata.cam_afdconnector_data.k
        moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        aiv_num = metadata.cam_afdconnector_data.aiv_num
        handle = metadata.cam_afdconnector_data.handle
        
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
                            groupEp = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                            aivNum = aiv_num)
        print(f'send_ffn_output end rank:{self.rank}')
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self) -> Any: 
        src = (self.process_group.rank_in_group - 1) % self.process_group.world_size
        metadata = self.process_group.recv_object(src)

        print(f'recv_attn_output start rank:{self.rank}')

        batch_size = metadata.cam_afdconnector_data.batch_size
        h = metadata.cam_afdconnector_data.h
        k = metadata.cam_afdconnector_data.k
        moe_expert_num = metadata.cam_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_afdconnector_data.shared_expert_num
        quant_mode = metadata.cam_afdconnector_data.quant_mode
        aiv_num = metadata.cam_afdconnector_data.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        output1 = torch_npu.cam_a2e(expandX = torch.tensor([], dtype=torch.bfloat16, device='npu'), expertIds = torch.tensor([], dtype=torch.int32, device='npu'),
                            scales = torch.tensor([], dtype=torch.float, device='npu'), commArgs = torch.tensor([], dtype=torch.float16, device='npu'),
                            expandXOutDType = expandXOutDType,
                            commId = 0, batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            sharedExpertNum = shared_expert_num, totalExpertNum = moe_expert_num + shared_expert_num, rank = self.rank,
                            loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant = quant_mode,
                            groupEp = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                            aivNum = aiv_num)
        
        print(f'recv_attn_output end rank:{self.rank}')

        return output1, metadata