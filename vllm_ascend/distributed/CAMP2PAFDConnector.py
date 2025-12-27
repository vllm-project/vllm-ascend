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
from vllm.distributed.afd_transfer.afd_connector.metadata import (CAMP2PAFDConnectorMetadata)
from vllm.config import VllmConfig,CUDAGraphMode,CompilationLevel
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher
logger = init_logger(__name__)

# # TODO(yxj):move to ascend ,use kwargs 
# @dataclass
# class CAMP2PAFDConnectorMetadata:
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

class CAMP2PAFDConnector(AFDConnectorBase):
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
        self.use_aclgraph = self._use_aclgraph()
        self.hccl_comm_name1 = ""
        print(f'self.use_aclgraph in CAMP2PAFDConnector is {self.use_aclgraph}')
        
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
        # 多机需要改成master_ip
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:29509",
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd"
        )
        self.hccl_comm_name = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        
        if self.rank < self.ffn_size:
            # 多机需要改成master_ip
            self.afd_pg1 = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://127.0.0.1:29999",
                world_size=self.ffn_size,
                rank=self.rank,
                group_name="afd1"
            )
            self.hccl_comm_name1 = self.afd_pg1._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
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
        if not self.use_aclgraph:
            print(f'send_attn_output start rank:{self.rank}')
            dst = (self.process_group.rank_in_group + 1) % self.process_group.world_size
            print(f'send_attn_output dst is {dst}')
            self.process_group.send_object(metadata,dst)
        
        batch_size = metadata.cam_p2p_afdconnector_data.batch_size
        h = metadata.cam_p2p_afdconnector_data.h
        k = metadata.cam_p2p_afdconnector_data.k
        aiv_num = metadata.cam_p2p_afdconnector_data.aiv_num

        handle_out = torch_npu.cam_a2e(expandX = hidden_states, expertIds = topk_idx,
                            scales = topk_weights,
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            ank = self.rank, groupEp = self.hccl_comm_name,
                            aivNum = aiv_num)

        return handle_out

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> torch.Tensor:
        batch_size = metadata.cam_p2p_afdconnector_data.batch_size
        h = metadata.cam_p2p_afdconnector_data.h
        k = metadata.cam_p2p_afdconnector_data.k
        aiv_num = metadata.cam_p2p_afdconnector_data.aiv_num
        handle = metadata.cam_p2p_afdconnector_data.handle
        
        output2 = torch_npu.cam_e2a(expandXOut = hidden_states, attenBatchSize = handle[3],
                            commId = 0,
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            rank = self.rank, groupEp = self.hccl_comm_name,
                            aivNum = aiv_num)

        return output2
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMP2PAFDConnectorMetadata):
        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        aiv_num = metadata.aiv_num
        handle = metadata.handle
        
        torch_npu.cam_e2a(expandXOut = ffn_output, attenBatchSize = handle[0],
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            rank = self.rank, groupEp = self.hccl_comm_name,
                            aivNum = aiv_num)

        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: CAMP2PAFDConnectorMetadata) -> Any: 
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

        outputs = torch_npu.cam_a2e(expandX = torch.tensor([], dtype=torch.bfloat16, device='npu'),
                            expertIds = torch.tensor([], dtype=torch.int32, device='npu'),
                            scales = torch.tensor([], dtype=torch.float, device='npu'),
                            batchSize = batch_size, hiddenSize = h, topk = k,
                            expertRankSize = self.ffn_size, attentionRankSize = self.attn_size,
                            rank = self.rank, groupEp = self.hccl_comm_name,
                            aivNum = aiv_num)
        
        return outputs, afdmetadata, self.hccl_comm_name1