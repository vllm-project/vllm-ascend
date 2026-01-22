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
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group

from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm_ascend.distributed.metadata import (CAMP2PAFDConnectorMetadata)
from vllm.config import VllmConfig, CUDAGraphMode, CompilationLevel
from vllm.distributed.afd_transfer.afd_connector.p2p_connector import DefaultProcessGroupSwitcher

logger = init_logger(__name__)


class CAMP2PAFDConnector(AFDConnectorBase):
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
        # ffn_ranks = [i for i in range(ffn_size, ffn_size + attn_size)]
        # attn_ranks = [i for i in range(attn_size)]
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

    def configure_metadata(self, metadata: "AFDConnectorMetadata", **kwargs) -> None:
        if metadata.connector_data is None:
            metadata.connector_data = CAMP2PAFDConnectorMetadata()

        config = kwargs.get('config')
        batch_size = kwargs.get('batch_size')
        if config:
            metadata.connector_data.moe_expert_num = config.n_routed_experts
            # TODO: quant_mode and aiv_num read from config
            metadata.connector_data.quant_mode = 0
            metadata.connector_data.aiv_num = 48
            metadata.connector_data.scale = None
            metadata.connector_data.batch_size = batch_size
            metadata.connector_data.h = config.hidden_size
            metadata.connector_data.k = config.num_experts_per_tok

    def compute_moe(self, experts, hidden_states, **kwargs):
        topk_ids = kwargs.get('topk_ids')
        topk_weights = kwargs.get('topk_weights')
        x_active_mask = kwargs.get('x_active_mask')
        cam_p2p_ep_name = kwargs.get('cam_p2p_ep_name')

        return experts.afd_m2n_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            connector_name="camp2pconnector",
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name
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

    # ATTN发给MOE（ATTN发送）
    # TODO:metadata的获取，最好从框架侧去拿
    def send_attn_output(self,
                         hidden_states: torch.Tensor,
                         metadata: AFDConnectorMetadata,
                         **kwargs) -> Any:
        # Get from kwargs
        topk_weights = kwargs.get('topk_weights')
        topk_idx = kwargs.get('topk_ids')

        if not self.use_aclgraph:
            print(f'send_attn_output start rank:{self.rank}')
            dst = (self.process_group.rank_in_group + 1) % self.process_group.world_size
            print(f'send_attn_output dst is {dst}')
            self.process_group.send_object(metadata, dst)

        # Access p2p data
        batch_size = metadata.connector_data.batch_size
        h = metadata.connector_data.h
        k = metadata.connector_data.k
        aiv_num = metadata.connector_data.aiv_num

        output_list = torch_npu.cam_a2e(expandX=hidden_states, expertIds=topk_idx,
                                        scales=topk_weights,
                                        batchSize=batch_size, hiddenSize=h, topk=k,
                                        expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                        ank=self.rank, groupEp=self.hccl_comm_name,
                                        aivNum=aiv_num)

        hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = output_list[0:5]
        handle_out = [hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize]

        return None, handle_out

    # MOE发给ATTN（ATTN接收）
    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: Optional["AFDConnectorMetadata"] = None,
                        ) -> Optional[torch.Tensor]:
        batch_size = metadata.connector_data.batch_size
        h = metadata.connector_data.h
        k = metadata.connector_data.k
        aiv_num = metadata.connector_data.aiv_num
        handle = metadata.connector_data.handle

        output2 = torch_npu.cam_e2a(expandXOut=hidden_states, attenBatchSize=handle[3],
                                    commId=0,
                                    batchSize=batch_size, hiddenSize=h, topk=k,
                                    expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                    rank=self.rank, groupEp=self.hccl_comm_name,
                                    aivNum=aiv_num)

        return output2

    # MOE发给ATTN(MOE发送)
    def send_ffn_output(self, ffn_output: torch.Tensor, metadata: AFDConnectorMetadata, **kwargs):
        batch_size = metadata.connector_data.batch_size
        h = metadata.connector_data.h
        k = metadata.connector_data.k
        aiv_num = metadata.connector_data.aiv_num
        handle = metadata.connector_data.handle

        torch_npu.cam_e2a(expandXOut=ffn_output, attenBatchSize=handle[0],
                          batchSize=batch_size, hiddenSize=h, topk=k,
                          expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                          rank=self.rank, groupEp=self.hccl_comm_name,
                          aivNum=aiv_num)

        return

    # ATTN发给MOE(MOE接收)
    def recv_attn_output(self, metadata: Optional[Any] = None, **kwargs) -> Any:
        afdmetadata = None
        if not self.use_aclgraph:
            src = (self.process_group.rank_in_group - 1) % self.process_group.world_size
            afdmetadata = self.process_group.recv_object(src)
            print(f'recv_attn_output start rank:{self.rank}')

        batch_size = metadata.batch_size
        h = metadata.h
        k = metadata.k
        aiv_num = metadata.aiv_num

        outputs = torch_npu.cam_a2e(expandX=torch.tensor([], dtype=torch.bfloat16, device='npu'),
                                    expertIds=torch.tensor([], dtype=torch.int32, device='npu'),
                                    scales=torch.tensor([], dtype=torch.float, device='npu'),
                                    batchSize=batch_size, hiddenSize=h, topk=k,
                                    expertRankSize=self.ffn_size, attentionRankSize=self.attn_size,
                                    rank=self.rank, groupEp=self.hccl_comm_name,
                                    aivNum=aiv_num)

        # outputs: [hidden_states1, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut]
        from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
        return AFDRecvOutput(
            hidden_states=outputs[0],
            metadata=afdmetadata,
            topk_ids=outputs[1],  # simulateExpertIds
            topk_weights=outputs[2],  # simulateExpertScales
            atten_batch_size=outputs[3],
            x_active_mask=outputs[4],
            cam_p2p_ep_name=self.hccl_comm_name1
        )

    def create_recv_metadata(self, **kwargs):
        max_num_tokens = kwargs.get('max_num_tokens', 0)
        hf_config = self.config.model_config.hf_config

        return CAMP2PAFDConnectorMetadata(
            moe_expert_num=hf_config.n_routed_experts,
            shared_expert_num=0,
            scale=None,
            handle=None,
            quant_mode=0,
            aiv_num=48,
            batch_size=max_num_tokens,
            h=hf_config.hidden_size,
            k=hf_config.num_experts_per_tok
        )

    def update_metadata(self, metadata, recv_output):
        metadata.handle = [recv_output.atten_batch_size]
