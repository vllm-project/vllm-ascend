# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
from typing import Any, Optional

import torch
import pickle
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group
from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase, AFDConnectorFactory,
                            AFDConnectorMetadata)
from vllm.distributed.parallel_state import init_afd_process_group, init_model_parallel_group, _split_tensor_dict, \
    TensorMetadata, GroupCoordinator, get_world_group
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
from vllm.config import VllmConfig,CUDAGraphMode, CompilationMode
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.distributed.metadata import NPUP2PAFDConnectorMetadata

logger = init_logger(__name__)


class DefaultProcessGroupSwitcher:

    def __init__(self, default_group, new_default_group):
        self.default_group = default_group
        self.new_default_group = new_default_group

    def __enter__(self):
        _update_default_pg(self.new_default_group)

    def __exit__(self, exc_type, exc_value, traceback):
        _update_default_pg(self.default_group)


class NPUP2PAFDConnector(AFDConnectorBase):
    def __init__(self,
                 rank: int,
                 local_rank: int,
                 config: "VllmConfig",
                 ) -> None:
        self.rank = rank
        self.local_rank = local_rank
        self._initialized = False
        self.config = config
        self.backend = "hccl"
        self.attn_size = 0
        self.ffn_size = 0
        self.use_aclgraph = self._use_aclgraph()
        self.dst_list = []

    def _use_aclgraph(self) -> bool:
        return self.config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and \
               self.config.compilation_config.mode == CompilationMode.VLLM_COMPILE and \
               not self.config.model_config.enforce_eager

    def close(self) -> None:
        """Close the connector and release resources."""
        # destroy process group
        pass

    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        assert self.config.afd_config is not None, "AFD config is not set"
        self.backend = torch.distributed.get_backend(get_world_group().device_group)
        if self.backend != "hccl":
            assert not self.config.afd_config.compute_gate_on_attention, \
                "Compute gate on attention is not supported"
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        assert self.attn_size == self.ffn_size, "Attention size and FFN size must be the same"
        self.min_size = self.attn_size
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size
        # p2p_rank: 所有FFN [0, ffn_size), 前min_size个Attention [ffn_size, ffn_size+min_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        self.rank = world_rank

        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, world_rank = {world_rank},backend = {self.backend}")
        afd_pg = init_afd_process_group(
            backend=self.backend,
            init_method=f"tcp://127.0.0.1:29500",
            world_size=self.ffn_size + self.attn_size,
            rank=world_rank,
            group_name="afd",
            timeout=timedelta(minutes=2),
        )
        ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]
        attn_ranks = [i for i in range(self.attn_size)]

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
                timeout=timeout  # TODO(yxj):use timeout set
            )

        # 前min_size的Attention向多个FFN发送metadata（1对多映射）
        # attn_i 向所有 ffn_j (其中 j % min_size == i) 发送
        if self.is_attn_top_min_size_rank(self.rank):
            local_attn_rank = self.rank
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size
        print(f"[P2p] world_rank={self.rank}, p2p_rank={self.p2p_rank}, min_size={self.min_size}, "
              f"dst_list={self.dst_list}, cam connector initialized")
        logger.debug(f"[p2p] world_rank={self.rank}, p2p_rank={self.p2p_rank}, min_size={self.min_size}, "
                     f"dst_list={self.dst_list}, cam connector initialized")

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            # Create two independent groups:
            # a2e_group: for attention -> expert/ffn communication (send_attn, recv_attn)
            # e2a_group: for expert/ffn -> attention communication (send_ffn, recv_ffn)
            # The communication domain (rank range) is the same, but different group_name
            # creates independent groups
            self.a2e_group = init_model_parallel_group(sub_group_ranks,
                                                       self.local_rank,
                                                       backend=self.backend,
                                                       group_name="a2e")
            self.e2a_group = init_model_parallel_group(sub_group_ranks,
                                                       self.local_rank,
                                                       backend=self.backend,
                                                       group_name="e2a")

        logger.info("p2p connector initialized")

        self._initialized = True

    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        return self._initialized

    def _send_tensor_dict_async(
            self,
            tensor_dict: dict[str, torch.Tensor],
            dst: int,
            process_group: GroupCoordinator,
    ) -> list:
        """Asynchronously send a tensor dictionary.

        Args:
            tensor_dict: The tensor dictionary to send
            dst: Destination rank (local rank)
            process_group: The process group to use for communication

        Returns:
            List of work objects that can be used to wait for operation completion
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return []

        assert dst < process_group.world_size, f"Invalid dst rank ({dst})"

        # Split tensor dictionary into metadata and tensor list
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)

        # Send metadata first (synchronously, as metadata is small and on CPU)
        process_group.send_object(metadata_list, dst=dst)

        # Asynchronously send each tensor
        work_list = []
        group = process_group.device_group
        metadata_group = process_group.cpu_group

        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip empty tensors
                continue

            if tensor.is_cpu:
                # CPU tensor uses metadata_group
                work = torch.distributed.isend(
                    tensor, dst=process_group.ranks[dst], group=metadata_group
                )
            else:
                # GPU tensor uses device_group
                work = torch.distributed.isend(
                    tensor, dst=process_group.ranks[dst], group=group
                )
            work_list.append(work)

        return work_list

    def _recv_tensor_dict_async(
            self,
            src: int,
            process_group: GroupCoordinator,
            all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> tuple[dict[str, torch.Tensor | Any], list]:
        """Asynchronously receive a tensor dictionary.

        Args:
            src: Source rank (local rank)
            process_group: The process group to use for communication
            all_gather_group: Group for all-gather optimization

        Returns:
            tuple: (tensor_dict, work_list) - tensor dictionary and work object list
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []

        assert src < process_group.world_size, f"Invalid src rank ({src})"

        # Receive metadata first synchronously (need to know tensor shape and type)
        recv_metadata_list = process_group.recv_object(src=src)

        # Create empty tensor dictionary and work list
        tensor_dict: dict[str, Any] = {}
        work_list = []
        group = process_group.device_group
        metadata_group = process_group.cpu_group

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                # Create empty tensor from metadata
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)

                if tensor.numel() == 0:
                    # Skip empty tensors
                    tensor_dict[key] = tensor
                    continue

                # Asynchronously receive tensor
                if tensor.is_cpu:
                    # CPU tensor uses metadata_group
                    work = torch.distributed.irecv(
                        tensor, src=process_group.ranks[src], group=metadata_group
                    )
                else:
                    # GPU tensor uses device_group
                    work = torch.distributed.irecv(
                        tensor, src=process_group.ranks[src], group=group
                    )
                work_list.append(work)
                tensor_dict[key] = tensor
            else:
                # Non-tensor values are added directly
                tensor_dict[key] = value

        return tensor_dict, work_list

    def configure_metadata(self, metadata: "AFDConnectorMetadata", **kwargs) -> None:
        if metadata.connector_data is None:
            metadata.connector_data = NPUP2PAFDConnectorMetadata()

    def send_attn_output(self,
                         hidden_states: torch.Tensor,
                         metadata: AFDConnectorMetadata,
                         **kwargs) -> Any:
        """
        This method will be called by the ATTN side.


        * To send the intermediate tensors generated by ATTN instances to FFN.
        """
        # Usage
        intermediate_tensors = self.create_intermediate_tensors(
            backend=self.backend,
            hidden_states=hidden_states,
            **kwargs
        )
        try:
            # Use async send instead of sync send
            # Use a2e_group for attention -> expert/ffn communication
            self.current_stream_synchronize(self.backend)
            dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size
            work_list = self._send_tensor_dict_async(
                intermediate_tensors.tensors,
                dst=dst,
                process_group=self.a2e_group,
            )
            # work_list can be used for waiting later if we need to ensure send completion
            # Here we don't wait, letting the send proceed asynchronously in the background
            self.a2e_group.send_object(metadata, dst)
            if metadata is not None:
                metadata.send_handle_list = work_list

            return hidden_states, work_list
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_attn_output(
            self,
            metadata: AFDConnectorMetadata = None,
            **kwargs
    ) -> Any:
        """
        This method will be called by the FFN side.


        * To receive the intermediate tensors from ATTN.
        * And (Maybe) dispatch them from the receiver to other GPUs.
        """
        # Use a2e_group for attention -> expert/ffn communication
        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        # Use async receive for tensor_dict
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.a2e_group,
            all_gather_group=None,
        )
        # Asynchronously receive independent metadata
        metadata = self.a2e_group.recv_object(src)
        metadata.recv_handle_list = work_list

        if self.backend == "hccl":
            return AFDRecvOutput(
                hidden_states=intermediate_tensors["hidden_states"],
                metadata=metadata,
                router_logits=intermediate_tensors["router_logits"],
                topk_weights=intermediate_tensors["topk_weights"],
                topk_ids=intermediate_tensors["topk_ids"],
                row_idx=intermediate_tensors["row_idx"]
            )
        else:
            return AFDRecvOutput(
                hidden_states=intermediate_tensors["hidden_states"],
                metadata=metadata
            )

    def create_recv_metadata(self, **kwargs):
        return None

    def update_metadata(self, metadata, recv_output):
        pass

    # -------------------------------------------------------------------------
    #                                attn <- ffn
    # -------------------------------------------------------------------------
    def send_ffn_output(
            self,
            hidden_states: torch.Tensor,
            metadata: AFDConnectorMetadata,
            **kwargs
    ) -> None:
        """
        This method will be called by the FFN side.


        * To send the intermediate tensors generated by FFN instances back to
            the sender (this should be the same GPU as it comes from)
        """
        intermediate_tensors = IntermediateTensors(
            {
                "hidden_states": hidden_states,
            }
        )
        # Use async send instead of sync send
        # Use e2a_group for expert/ffn -> attention communication
        self.current_stream_synchronize(self.backend)
        dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        work_list = self._send_tensor_dict_async(
            intermediate_tensors.tensors,
            dst=dst,
            process_group=self.e2a_group,
        )
        # work_list can be used for waiting later if we need to ensure send completion
        # Here we don't wait, letting the send proceed asynchronously in the background
        self.e2a_group.send_object(metadata, dst)
        if metadata is not None:
            metadata.send_handle_list = work_list

    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: AFDConnectorMetadata = None) -> torch.Tensor:
        """
        This method will be called by the ATTN side.


        * To receive the MOE output intermediate tensors.
        * And (Maybe) dispatch them from the receiver to other GPUs.
            (this should be the same GPU as it comes from)
        """
        # Use e2a_group for expert/ffn -> attention communication
        src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size
        # Use async receive for tensor_dict
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.e2a_group,
            all_gather_group=None,
        )
        # Asynchronously receive independent metadata
        metadata = self.e2a_group.recv_object(src)
        # Wait for tensor receive completion (because we need to use data immediately)
        if metadata is not None:
            metadata.recv_handle_list = work_list
        else:
            for work in work_list:
                work.wait()
        return intermediate_tensors["hidden_states"]

    def create_intermediate_tensors(self, backend, hidden_states, **kwargs):
        """Factory method for creating intermediate tensor objects"""
        base_tensors = {"hidden_states": hidden_states}

        if backend == "hccl":
            hccl_tensors = {
                "router_logits": kwargs.get('router_logits'),
                "topk_weights": kwargs.get('topk_weights'),
                "topk_ids": kwargs.get('topk_ids'),
                "row_idx": kwargs.get('row_idx'),
            }
            base_tensors.update(hccl_tensors)

        return IntermediateTensors(base_tensors)

    def current_stream_synchronize(self, backend):
        if backend == "hccl":
            torch.npu.current_stream().synchronize()
        else:
            torch.cuda.current_stream().synchronize()

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        router_logits = kwargs.get('router_logits')
        topk_weights = kwargs.get('topk_weights')
        topk_ids = kwargs.get('topk_ids')
        row_idx = kwargs.get('row_idx')

        return experts.afd_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx)

    def is_vaild_rank_for_inequal_AF(self, rank):
        # Only support ffn rank < attn rank
        return (self.ffn_size <= rank < self.ffn_size + self.min_size) or rank < self.ffn_size

    def is_attn_top_min_size_rank(self, rank):
        # Only support ffn rank < attn rank
        return rank < self.min_size

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
