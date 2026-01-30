from dataclasses import dataclass
from typing import Any, Optional

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
from vllm_ascend.distributed.metadata import M2NAFDConnectorMetadata

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

    def configure_metadata(self, metadata: "AFDConnectorMetadata", **kwargs) -> None:
        if metadata.connector_data is None:
            metadata.connector_data = M2NAFDConnectorMetadata()
        
        config = kwargs.get('config')
        if config:
            metadata.connector_data.moe_expert_num = config.n_routed_experts
            # TODO: quant_mode and aiv_num read from config
            metadata.connector_data.quant_mode = 0
            metadata.connector_data.aiv_num = 48
            metadata.connector_data.scale = None
                                  
    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self, 
                         hidden_states: torch.Tensor,  
                         metadata: AFDConnectorMetadata,
                         **kwargs) -> Any:
        # Get args from kwargs
        topk_weights = kwargs.get('topk_weights')
        topk_ids = kwargs.get('topk_ids')

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

        dynamic_scales = metadata.connector_data.scale
        moe_expert_num = metadata.connector_data.moe_expert_num
        quant_mode = metadata.connector_data.quant_mode
        aiv_num = metadata.connector_data.aiv_num

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

        return None, recv_counts

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None, 
                        metadata: Optional["AFDConnectorMetadata"] = None,
                        ) -> Optional[torch.Tensor]:
        moe_expert_num = metadata.connector_data.moe_expert_num
        aiv_num = metadata.connector_data.aiv_num
        handle = metadata.connector_data.handle

        xOut = torch_npu.npu_n2m_distribute_recv(x=hidden_states,
                                                 ep_recv_counts=handle,
                                                 group_ep=self.hccl_comm_name,
                                                 world_size=self.attn_size + self.ffn_size,
                                                 moe_world_size=self.ffn_size,
                                                 ep_rank_id=self.rank,
                                                 moe_expert_num=moe_expert_num,
                                                 server_rank_size=self.server_rank_size,
                                                 aiv_num=aiv_num)

        return xOut
    
    # MOE发给ATTN(MOE发送) 
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata, **kwargs):
        batch_size = metadata.connector_data.batch_size
        topk_weights = metadata.connector_data.topk_weights
        moe_expert_num = metadata.connector_data.moe_expert_num
        aiv_num = metadata.connector_data.aiv_num
        k = metadata.connector_data.k
        handle = metadata.connector_data.handle
        
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
    def recv_attn_output(self, metadata: Optional[Any] = None, **kwargs) -> Any:
        m2n_afdconnector_data = metadata
        
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
        
        # Use passed metadata or received one
        if m2n_afdconnector_data:
             quant_mode = m2n_afdconnector_data.quant_mode
             expand_x_type = m2n_afdconnector_data.expand_x_type
             moe_expert_num = m2n_afdconnector_data.moe_expert_num
             h = m2n_afdconnector_data.h
             k = m2n_afdconnector_data.k
             expert_token_nums_type = m2n_afdconnector_data.expert_token_nums_type
             aiv_num = m2n_afdconnector_data.aiv_num
             batch_size = m2n_afdconnector_data.batch_size
        else:
             # Fallback or error if not provided
             quant_mode = 0
             expand_x_type = torch.bfloat16
             moe_expert_num = 0
             h = 0
             k = 0
             expert_token_nums_type = 0
             aiv_num = 0
             batch_size = 0

        # ... logic for npu_m2n_distribute_recv ...
        recv_result = torch_npu.npu_m2n_distribute_recv(
                                                src_rank=self.p2p_rank % self.min_size,
                                                group_ep=self.hccl_comm_name,
                                                world_size=self.attn_size + self.ffn_size,
                                                moe_world_size=self.ffn_size,
                                                ep_rank_id=self.rank,
                                                moe_expert_num=moe_expert_num,
                                                quant_mode=quant_mode,
                                                expand_x_type=expand_x_type,
                                                h=h,
                                                k=k,
                                                expert_token_nums_type=expert_token_nums_type,
                                                aiv_num=aiv_num,
                                                batch_size=batch_size,
                                                server_rank_size = self.server_rank_size)
        
        hidden_states, dynamic_scales, group_list, handle, topk_weights = recv_result
        
        from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
        return AFDRecvOutput(
            hidden_states=hidden_states,
            metadata=afdConnectorMetadata,
            topk_weights=topk_weights,
            dynamic_scales=dynamic_scales,
            group_list=group_list,
            handle=handle
        )

    def compute_moe(self, experts, hidden_states, **kwargs):
        # Logic from DeepseekV2MoE.afd_forward for m2n
        group_list = kwargs.get('group_list')
        dynamic_scales = kwargs.get('dynamic_scales')
        
        return experts.afd_m2n_ffn_compute(
                layer=experts,  
                hidden_states=hidden_states,  
                group_list=group_list, 
                dynamic_scale=dynamic_scales,
                group_list_type=0,
                connector_name="m2nconnector"
                )

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

    def create_recv_metadata(self, **kwargs):
        max_num_tokens = kwargs.get('max_num_tokens', 0)
        hf_config = self.config.model_config.hf_config
        
        metadata = M2NAFDConnectorMetadata()
        metadata.quant_mode = 0
        metadata.expand_x_type = torch.bfloat16
        metadata.moe_expert_num = hf_config.n_routed_experts
        metadata.h = hf_config.hidden_size
        metadata.k = hf_config.num_experts_per_tok
        metadata.expert_token_nums_type = 0
        metadata.aiv_num = 48
        metadata.batch_size = max_num_tokens * metadata.k * self.attn_size
        return metadata

    def update_metadata(self, metadata, recv_output):
        metadata.handle = recv_output.handle
        metadata.topk_weights = recv_output.topk_weights
