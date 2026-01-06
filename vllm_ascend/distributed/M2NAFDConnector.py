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
from torch.distributed.distributed_c10d import  _update_default_pg, _get_default_group
from vllm.distributed.afd_transfer.afd_connector.metadata import M2NAFDConnectorMetadata

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group

from vllm.logger import init_logger
from vllm.config import VllmConfig,CUDAGraphMode,CompilationLevel
from vllm_ascend.utils import npu_stream_switch_within_graph
logger = init_logger(__name__)


class DefaultProcessGroupSwitcher:
    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)

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
        self.use_aclgraph = self._use_aclgraph()
        print(f'self.use_aclgraph in M2NAFDConnector is {self.use_aclgraph}')
        
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
        self.min_size = min(self.ffn_size, self.attn_size)
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size
        self.p2p_rank = self.rank if role == "attention" else self.rank + self.min_size
        self.rank = world_rank
        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {world_rank}")
        print(f"world_size = {self.ffn_size + self.attn_size}, world_rank = {world_rank}")
        # TODO : get backend to replace hardcode
        self.afd_pg = init_afd_process_group(
            backend="hccl",
            init_method=(
                f"tcp://{self.config.afd_config.afd_host}"
                f":{self.config.afd_config.afd_port}"
            ),
            world_size=self.ffn_size + self.attn_size,
            rank=world_rank,
            group_name="afd"
        )
        # print(f'hccl_comm_name is {self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)}')
        self.hccl_comm_name = self.afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank)
        if world_rank < self.ffn_size or world_rank >= self.attn_size:
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

        if world_rank < self.min_size:
            self.dst_list = []
            dst = world_rank + self.min_size
            while (dst < self.ffn_size + self.min_size):
                self.dst_list.append(dst)
                dst += self.min_size

        import math
        # 节点的卡数,最好取一个公约数,比如a侧8卡f侧4卡，那可以取2或者4
        self.server_rank_size = math.gcd(self.attn_size, self.ffn_size)
        logger.info("m2n connector initialized")

        self.comm_stream = None
        if self.config.afd_config.is_multistream:
            self.comm_stream = torch.npu.Stream()
            aic_num = int(self.config.afd_config.multistream_info["core_num"])
            aiv_num = 2 * aic_num
            torch.npu.set_stream_limit(self.comm_stream, aic_num, aiv_num)

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
                         topk_ids:torch.Tensor, 
                         metadata: AFDConnectorMetadata) -> Any:
        # TODO():move to support aclgraph
        # torch.npu.synchronize()
        if not self.use_aclgraph and self.rank < self.min_size:
            for dst in self.dst_list:
                print(f'send_attn_output dst is {dst}')
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
                print(f'send_attn_output metadata success')
        dynamic_scales = metadata.m2n_afdconnector_data.scale
        # moe_expert_num
        moe_expert_num = metadata.m2n_afdconnector_data.moe_expert_num
        quant_mode = metadata.m2n_afdconnector_data.quant_mode
        aiv_num = metadata.m2n_afdconnector_data.aiv_num
        
        if dynamic_scales is None:
            dynamic_scales = torch.tensor([], dtype=torch.float32, device='npu')
        curr_stream = torch.npu.current_stream()
        with npu_stream_switch_within_graph(curr_stream, self.comm_stream, self.config.afd_config.is_multistream):
            recv_counts = torch_npu.npu_m2n_distribute_send(x=hidden_states,
                                                            expert_ids=topk_ids,
                                                            expert_scales=topk_weights,
                                                            group_ep=self.hccl_comm_name,
                                                            world_size=self.attn_size + self.ffn_size,
                                                            moe_world_size=self.ffn_size,
                                                            ep_rank_id=self.rank,
                                                            moe_expert_num=moe_expert_num,
                                                            quant_mode=quant_mode,
                                                            aiv_num=aiv_num,
                                                            server_rank_size = self.server_rank_size,
                                                            dynamic_scales=dynamic_scales)
        return recv_counts

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self, hidden_states: torch.Tensor, metadata: AFDConnectorMetadata) -> torch.Tensor:
        # handle = send_attn_output（）recv_counts
        handle = metadata.m2n_afdconnector_data.handle
        moe_expert_num = metadata.m2n_afdconnector_data.moe_expert_num
        aiv_num = metadata.m2n_afdconnector_data.aiv_num
        curr_stream = torch.npu.current_stream()
        with npu_stream_switch_within_graph(curr_stream, self.comm_stream, self.config.afd_config.is_multistream):
            xOut = torch_npu.npu_n2m_distribute_recv(x=hidden_states,
                                                    ep_recv_counts=handle,
                                                    group_ep=self.hccl_comm_name,
                                                    world_size=self.attn_size + self.ffn_size,
                                                    moe_world_size=self.ffn_size,
                                                    ep_rank_id=self.rank,
                                                    moe_expert_num=moe_expert_num,
                                                    server_rank_size = self.server_rank_size,
                                                    aiv_num=aiv_num)
        return xOut
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: M2NAFDConnectorMetadata):
        # 配置
        batch_size = metadata.batch_size
        topk_weights = metadata.topk_weights
        moe_expert_num = metadata.moe_expert_num
        aiv_num = metadata.aiv_num
        k = metadata.k
        handle = metadata.handle
        curr_stream = torch.npu.current_stream()
        with npu_stream_switch_within_graph(curr_stream, self.comm_stream, self.config.afd_config.is_multistream):
            torch_npu.npu_n2m_distribute_send(expandX=ffn_output,
                                            ep_send_counts=handle,
                                            expert_scales=topk_weights,
                                            group_ep=self.hccl_comm_name,
                                            world_size=self.attn_size + self.ffn_size,
                                            moe_world_size=self.ffn_size,
                                            ep_rank_id=self.rank,
                                            moe_expert_num=moe_expert_num,# config
                                            batch_size=batch_size,# config
                                            k=k,# config
                                            server_rank_size = self.server_rank_size,
                                            aiv_num=aiv_num)# config 未分核48 
        return
    
    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: M2NAFDConnectorMetadata) -> Any: 
        afdConnectorMetadata = None
        if not self.use_aclgraph and self.rank >= self.attn_size:
            src = self.p2p_rank % self.min_size
            print(f'recv_attn_output src is {src}')
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

            afdConnectorMetadata = pickle.loads(object_tensor.numpy().tobytes())
            print(f'recv_attn_output afdConnectorMetadata success')
        # TODO(yxj): 对比
        x_type = torch.int8
        if metadata.quant_mode == 0 :
            x_type = metadata.expand_x_type
        batch_size = metadata.batch_size
        quant_mode = metadata.quant_mode
        moe_expert_num = metadata.moe_expert_num
        aiv_num = metadata.aiv_num
        k = metadata.k
        h = metadata.h
        expert_token_nums_type = metadata.expert_token_nums_type
        #npu::npu_m2n_distribute_recv(Tensor x, str group_ep, int world_size, int server_rank_size, int moe_world_size, int ep_rank_id, int moe_expert_num, int quant_mode, int batch_size, int h, int k, int expert_token_nums_type, int aiv_num) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
        curr_stream = torch.npu.current_stream()
        with npu_stream_switch_within_graph(curr_stream, self.comm_stream, self.config.afd_config.is_multistream):
                expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales = torch_npu.npu_m2n_distribute_recv(x = torch.tensor([], dtype=x_type, device='npu'),
                                                                                        group_ep=self.hccl_comm_name,
                                                                                        world_size=self.attn_size + self.ffn_size,
                                                                                        moe_world_size=self.ffn_size,
                                                                                        ep_rank_id=self.rank,
                                                                                        moe_expert_num=moe_expert_num,
                                                                                        quant_mode=quant_mode,
                                                                                        batch_size=batch_size,
                                                                                        h=h,
                                                                                        k=k,
                                                                                        expert_token_nums_type=expert_token_nums_type,
                                                                                        server_rank_size = self.server_rank_size,
                                                                                        aiv_num=aiv_num)
        
        # recv_counts 返程路由
        return expand_x, dynamic_scales, expert_token_nums, recv_counts, expand_scales,afdConnectorMetadata