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
# class M2NAFDConnectorMetadata:
#     def __init__(self):
#         self.topk_idx = None
#         self.topk_weights = None
#         self.moe_expert_num = 0
#         self.scale = None
#         self.handle = None
#         self.quant_mode = 0
#         self.aiv_num = 0
#         self.batch_size = 0
#         self.h = 0
#         self.k = 0
#         self.expert_token_nums_type = 0
#         self.expand_x_type = torch.float16

class M2NAFDConnector(AFDConnectorBase):
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
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size

        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {world_rank}")
        # TODO : get backend to replace hardcode
        afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=f"tcp://127.0.0.1:29509",
            world_size=self.ffn_size + self.attn_size,
            rank=world_rank,
            group_name="afd"
        )
        ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]
        attn_ranks = [i for i in range(self.attn_size)]

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            self.process_group = init_model_parallel_group(sub_group_ranks,
                                                 self.rank,
                                                 backend="hccl",
                                                 group_name="ae")

        logger.info("p2p connector initialized")

        self._initialized = True
    
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.
        
        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized
                                  
    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> Any:
        dynamic_scales = metadata.extra_fields.custom_fields["scale"]
        topk_idx = metadata.extra_fields.custom_fields["topk_idx"]
        topk_weights = metadata.extra_fields.custom_fields["topk_weights"]
        moe_expert_num = metadata.extra_fields.custom_fields["moe_expert_num"]
        quant_mode = metadata.extra_fields.custom_fields["quant_mode"]
        aiv_num = metadata.extra_fields.custom_fields["aiv_num"]
        scale = metadata.extra_fields.custom_fields["scale"]
        if dynamic_scales is None:
            dynamic_scales = torch.tensor([], dtype=torch.float32, device='npu')
        recv_counts = torch_npu.npu_m2n_distribute_send(x=hidden_states,
                                                        expert_ids=topk_idx,
                                                        expert_scales=topk_weights,
                                                        group_ep=self.process_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                        world_size=self.attn_size + self.ffn_size,
                                                        moe_world_size=self.ffn_size,
                                                        ep_rank_id=self.rank,
                                                        moe_expert_num=moe_expert_num,
                                                        quant_mode=quant_mode,
                                                        aiv_num=aiv_num,
                                                        dynamic_scales=scale)
        return recv_counts

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> torch.Tensor:
        handle = metadata.extra_fields.custom_fields["handle"]
        moe_expert_num = metadata.extra_fields.custom_fields["moe_expert_num"]
        aiv_num = metadata.extra_fields.custom_fields["aiv_num"]
        xOut = torch_npu.npu_n2m_distribute_recv(x=hidden_states,
                                                ep_recv_counts=handle,
                                                group_ep=self.process_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                world_size=self.attn_size + self.ffn_size,
                                                moe_world_size=self.ffn_size,
                                                ep_rank_id=self.rank,
                                                moe_expert_num=moe_expert_num,
                                                aiv_num=aiv_num)
        return xOut
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata):
        batch_size = metadata.extra_fields.custom_fields["batch_size"]
        topk_weights = metadata.extra_fields.custom_fields["topk_weights"]
        moe_expert_num = metadata.extra_fields.custom_fields["moe_expert_num"]
        aiv_num = metadata.extra_fields.custom_fields["aiv_num"]
        k = metadata.extra_fields.custom_fields["k"]
        handle = metadata.extra_fields.custom_fields["handle"]
        torch_npu.npu_n2m_distribute_send(expandX=ffn_output,
                                        ep_send_counts=handle,
                                        expert_scales=topk_weights,
                                        group_ep=self.process_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                        world_size=self.attn_size + self.ffn_size,
                                        moe_world_size=self.ffn_size,
                                        ep_rank_id=self.rank,
                                        moe_expert_num=moe_expert_num,
                                        batch_size=batch_size,
                                        k=k,
                                        aiv_num=aiv_num)
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: AFDConnectorMetadata) -> Any: 
        x_type = torch.int8
        if metadata.quant_mode == 0 :
            x_type = metadata.expand_x_type
        batch_size = metadata.extra_fields.custom_fields["batch_size"]
        quant_mode = metadata.extra_fields.custom_fields["quant_mode"]
        moe_expert_num = metadata.extra_fields.custom_fields["moe_expert_num"]
        aiv_num = metadata.extra_fields.custom_fields["aiv_num"]
        k = metadata.extra_fields.custom_fields["k"]
        h = metadata.extra_fields.custom_fields["h"]
        expert_token_nums_type = metadata.extra_fields.custom_fields["expert_token_nums_type"]
        expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales = torch_npu.npu_m2n_distribute_recv(x = torch.tensor([], dtype=x_type, device='npu'),
                                                                                group_ep=self.process_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank),
                                                                                world_size=self.attn_size + self.ffn_size,
                                                                                moe_world_size=self.ffn_size,
                                                                                ep_rank_id=self.rank,
                                                                                moe_expert_num=moe_expert_num,
                                                                                quant_mode=quant_mode,
                                                                                batch_size=batch_size,
                                                                                h=h,
                                                                                k=k,
                                                                                expert_token_nums_type=expert_token_nums_type,
                                                                                aiv_num=aiv_num)
        return expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales