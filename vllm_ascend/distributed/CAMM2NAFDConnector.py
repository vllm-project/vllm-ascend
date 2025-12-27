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
import pickle
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.distributed.afd_transfer.afd_connector.metadata import (CAMM2NAFDConnectorMetadata)
from vllm.config import VllmConfig, CUDAGraphMode, CompilationLevel

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
        # ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        # attn_ranks = [i for i in range(attn_size)]
        self.min_size = min(self.ffn_size, self.attn_size)
        world_rank = self.rank + self.ffn_size if role == "attention" else self.rank
        # p2p_rank: 所有FFN [0, ffn_size), 前min_size个Attention [ffn_size, ffn_size+min_size)
        self.p2p_rank = self.rank + self.ffn_size if role == "attention" else self.rank
        self.rank = world_rank

        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {self.rank}, "
            f"afd_hots: {self.config.afd_config.afd_host}, afd_port: {self.config.afd_config.afd_port}")
        # TODO : get backend to replace hardcode
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=(
                f"tcp://{self.config.afd_config.afd_host}"
                f":{self.config.afd_config.afd_port}"
            ),
            world_size=self.ffn_size + self.attn_size,
            rank=self.rank,
            group_name="afd"
        )
        self.hccl_comm_name = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        # 所有FFN和前min_size的Attention参与p2p通信
        # 所有FFN: world_rank in [0, ffn_size), 前min_size个Attention: world_rank in [ffn_size, ffn_size+min_size)
        if self.rank < self.ffn_size or (self.rank >= self.ffn_size and self.rank < self.ffn_size + self.min_size):
            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=(
                    f"tcp://{self.config.afd_config.afd_host}"
                    f":{self.config.afd_config.afd_port}"
                ),
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p"
            )

        # 前min_size的Attention向多个FFN发送metadata（1对多映射）
        # attn_i 向所有 ffn_j (其中 j % min_size == i) 发送
        if self.rank >= self.ffn_size and self.rank < self.ffn_size + self.min_size:
            local_attn_rank = self.rank - self.ffn_size
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size

        logger.info(f"[CAM] world_rank={self.rank}, p2p_rank={self.p2p_rank}, min_size={self.min_size}, "
                    f"dst_list={self.dst_list}, cam connector initialized")

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
                         topk_idx: torch.Tensor,
                         metadata: AFDConnectorMetadata) -> Any:
        # 只有前min_size的Attention发送metadata
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

        batch_size = metadata.cam_m2n_afdconnector_data.batch_size
        h = metadata.cam_m2n_afdconnector_data.h
        k = metadata.cam_m2n_afdconnector_data.k
        moe_expert_num = metadata.cam_m2n_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_m2n_afdconnector_data.shared_expert_num
        quant_mode = metadata.cam_m2n_afdconnector_data.quant_mode
        aiv_num = metadata.cam_m2n_afdconnector_data.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        handle_out = torch_npu.cam_a2e(expandX=hidden_states, expertIds=topk_idx,
                                       scales=topk_weights,
                                       commArgs=torch.tensor([], dtype=torch.float16, device='npu'),
                                       expandXOutDType=expandXOutDType,
                                       commId=0, batchSize=batch_size, hiddenSize=h, topk=k,
                                       expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                       sharedExpertNum=shared_expert_num,
                                       totalExpertNum=moe_expert_num + shared_expert_num, rank=self.rank,
                                       loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant=quant_mode,
                                       groupEp=self.hccl_comm_name,
                                       aivNum=aiv_num)

        return handle_out

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> torch.Tensor:
        batch_size = metadata.cam_m2n_afdconnector_data.batch_size
        h = metadata.cam_m2n_afdconnector_data.h
        k = metadata.cam_m2n_afdconnector_data.k
        moe_expert_num = metadata.cam_m2n_afdconnector_data.moe_expert_num
        shared_expert_num = metadata.cam_m2n_afdconnector_data.shared_expert_num
        aiv_num = metadata.cam_m2n_afdconnector_data.aiv_num
        handle = metadata.cam_m2n_afdconnector_data.handle

        output2 = torch_npu.cam_e2a(expandXOut=hidden_states, simulateExpertIds=handle[0],
                                    simulateExpertScales=handle[1], expandIdx=handle[2],
                                    epRecvCounts=handle[3],
                                    commArgs=torch.tensor([], dtype=torch.float16, device='npu'),
                                    attenBatchSize=handle[4],
                                    commId=0,
                                    batchSize=batch_size, hiddenSize=h, topk=k,
                                    expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                    sharedExpertNum=shared_expert_num,
                                    totalExpertNum=moe_expert_num + shared_expert_num,
                                    rank=self.rank,
                                    loadBalancingRankNum=1, loadBalancingThreshold=0,
                                    groupEp=self.hccl_comm_name,
                                    aivNum=aiv_num)

        return output2

    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: CAMM2NAFDConnectorMetadata):
        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        aiv_num = metadata.aiv_num
        handle = metadata.handle

        torch_npu.cam_e2a(expandXOut=ffn_output, simulateExpertIds=handle[0],
                          simulateExpertScales=handle[1],
                          expandIdx=handle[2],
                          epRecvCounts=handle[3],
                          commArgs=torch.tensor([], dtype=torch.float16, device='npu'),
                          attenBatchSize=handle[4],
                          commId=0,
                          batchSize=batch_size, hiddenSize=h, topk=k,
                          expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                          sharedExpertNum=shared_expert_num, totalExpertNum=moe_expert_num + shared_expert_num,
                          rank=self.rank,
                          loadBalancingRankNum=1, loadBalancingThreshold=0,
                          groupEp=self.hccl_comm_name,
                          aivNum=aiv_num)

        return

    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: CAMM2NAFDConnectorMetadata) -> Any:
        afdmetadata = None
        if not self.use_aclgraph and self.rank < self.ffn_size:
            local_ffn_rank = self.rank
            src = (local_ffn_rank % self.min_size) + self.ffn_size  # 对应的Attention在p2p组中的rank
            print(f'rank: {self.rank}, recv_attn_output src is {src}')
            size_tensor = torch.empty(1, dtype=torch.long, device="cpu")

            # Receive object size
            rank_size = torch.distributed.recv(size_tensor,
                                               src=src,
                                               group=self.p2p_pg)

            # Tensor to receive serialized objects into.
            object_tensor = torch.empty(  # type: ignore[call-overload]
                size_tensor.item(),  # type: ignore[arg-type]
                dtype=torch.uint8,
                device="cpu")

            rank_object = torch.distributed.recv(object_tensor,
                                                 src=src,
                                                 group=self.p2p_pg)

            assert rank_object == rank_size, (
                "Received object sender rank does not match the size sender rank.")

            afdmetadata = pickle.loads(object_tensor.numpy().tobytes())
            print(f'recv_attn_output afdConnectorMetadata success')

        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        moe_expert_num = metadata.moe_expert_num
        shared_expert_num = metadata.shared_expert_num
        quant_mode = metadata.quant_mode
        aiv_num = metadata.aiv_num
        expandXOutDType = torch.tensor([], dtype=torch.bfloat16 if not quant_mode else torch.int8, device='npu')

        output1 = torch_npu.cam_a2e(expandX=torch.tensor([], dtype=torch.bfloat16, device='npu'),
                                    expertIds=torch.tensor([], dtype=torch.int32, device='npu'),
                                    scales=torch.tensor([], dtype=torch.float, device='npu'),
                                    commArgs=torch.tensor([], dtype=torch.float16, device='npu'),
                                    expandXOutDType=expandXOutDType,
                                    commId=0, batchSize=batch_size, hiddenSize=h, topk=k,
                                    expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                    sharedExpertNum=shared_expert_num,
                                    totalExpertNum=moe_expert_num + shared_expert_num, rank=self.rank,
                                    loadBalancingRankNum=1, loadBalancingThreshold=0, dynamicQuant=quant_mode,
                                    groupEp=self.hccl_comm_name,
                                    aivNum=aiv_num)

        return output1, afdmetadata
