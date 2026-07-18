# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from datetime import timedelta
from typing import Any, Optional

import torch
import pickle
from torch.distributed.distributed_c10d import _update_default_pg, _get_default_group
from vllm.distributed.afd_transfer.afd_connector import (AFDConnectorBase,
                                                         AFDConnectorFactory,
                                                         AFDConnectorMetadata)
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDRecvOutput
from vllm.distributed.parallel_state import (
    GroupCoordinator, TensorMetadata, _split_tensor_dict,
    get_world_group, init_afd_process_group, init_model_parallel_group)
from vllm.config import VllmConfig, CUDAGraphMode, CompilationMode
from vllm.sequence import IntermediateTensors
from vllm.logger import logger
from vllm.forward_context import get_forward_context
from vllm_ascend.distributed.metadata import NPUP2PAFDConnectorMetadata

def _get_group_ep(ubatch_idx: int, hccl_comm_name: str, hccl_comm_name2: str, hccl_comm_name3: Optional[str]) -> str:
    groupEp = hccl_comm_name
    if ubatch_idx == 1:
        groupEp = hccl_comm_name2
    elif ubatch_idx == 2:
        assert hccl_comm_name3 is not None
        groupEp = hccl_comm_name3
    return groupEp

class DefaultProcessGroupSwitcher:
    """Context manager that temporarily swaps the default process group.

    Used so that ``init_model_parallel_group`` creates the a2e / e2a
    sub-groups on top of the AFD process group instead of the world group.
    """

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
        # Cache afd_config for use in init_afd_connector and downstream
        # consumers (e.g. quant_mode exposure for the FFN runner).
        self.afd_config = config.afd_config
        self.backend = "hccl"
        self.attn_size = 0
        self.ffn_size = 0
        self.use_aclgraph = self._use_aclgraph()
        self.dst_list = []
        # dp_metadata_list cache populated by update_state_from_dp_metadata
        self.dp_metadata_list: dict = {}
        # Expose quant_mode from AFDConfig so the FFN runner can decide
        # whether to pass dynamic_scales into compute_ffn_output.
        self.quant_mode = config.afd_config.quant_mode if config.afd_config else 0

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
        # Idempotent guard: NPUFFNModelRunner.__init__ already invokes this
        # once to populate attn_size / ffn_size, and worker.py
        # start_ffn_server_loop invokes initialize_afd_connector() again.
        # Without this guard the second call hits
        # _patched_new_process_group_helper with a duplicate group_name
        # ("afd" / "p2p") and raises ValueError.
        if self._initialized:
            logger.info("NPUP2PAFDConnector already initialized, skipping")
            return
        assert self.config.afd_config is not None, "AFD config is not set"
        self.backend = torch.distributed.get_backend(get_world_group().device_group)
        afd_size = self.config.afd_config.afd_extra_config.get("afd_size")
        role = self.config.afd_config.afd_role
        self.attn_size, self.ffn_size = map(
            int,
            re.match(r"(\d+)\D+(\d+)", afd_size).groups())
        assert self.attn_size == self.ffn_size, "Attention size and FFN size must be the same"
        self.min_size = self.attn_size
        world_rank = self.rank if role == "attention" else self.rank + self.attn_size
        # p2p_rank: all FFN [0, ffn_size), the first min_size Attention ranks
        # use [ffn_size, ffn_size + min_size)
        self.p2p_rank = self.rank + self.min_size if role == "attention" else self.rank
        self.rank = world_rank

        logger.info(
            f"world_size = {self.ffn_size + self.attn_size}, "
            f"world_rank = {world_rank}, backend = {self.backend}")

        afd_host = self.config.afd_config.afd_host
        afd_port = self.config.afd_config.afd_port

        self.afd_pg_list = []
        self.hccl_comm_name_list = []
        num_ubatches = 1
        for i in range(num_ubatches):
            group_name = "afd" + str(i) if i > 0 else "afd"
            afd_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{afd_host}:{afd_port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.rank,
                group_name=group_name
            )
            self.afd_pg_list.append(afd_pg)
            self.hccl_comm_name_list.append(afd_pg._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank))
        self.hccl_comm_name = self.hccl_comm_name_list[0]
        self.hccl_comm_name2 = self.hccl_comm_name_list[1] if num_ubatches > 1 else self.hccl_comm_name
        self.hccl_comm_name3 = self.hccl_comm_name_list[2] if num_ubatches > 2 else None

        ffn_ranks = [i for i in range(self.ffn_size, self.ffn_size + self.attn_size)]
        attn_ranks = [i for i in range(self.attn_size)]

        # All FFN ranks and the first min_size Attention ranks participate in
        # p2p communication.
        # All FFN:        world_rank in [0, ffn_size)
        # First min_size Attention: world_rank in [ffn_size, ffn_size + min_size)
        import datetime
        timeout = datetime.timedelta(seconds=30000)
        if self.is_vaild_rank_for_inequal_AF(self.rank):
            self.p2p_pg = init_afd_process_group(
                backend="hccl",
                init_method=f"tcp://{afd_host}:{afd_port + 1}",
                world_size=self.ffn_size + self.min_size,
                rank=self.p2p_rank,
                group_name="p2p",
                timeout=timeout  # TODO(yxj):use timeout set
            )

        if self.is_attn_top_min_size_rank(self.rank):
            local_attn_rank = self.rank
            dst = local_attn_rank
            while dst < self.ffn_size:
                self.dst_list.append(dst)
                dst += self.min_size

        default_pg_switcher = DefaultProcessGroupSwitcher(
            _get_default_group(), afd_pg)
        with default_pg_switcher:
            sub_group_ranks = []
            for i in range(len(ffn_ranks)):
                ranks = list([attn_ranks[i], ffn_ranks[i]])
                sub_group_ranks.append(ranks)
            self.a2e_group = init_model_parallel_group(sub_group_ranks,
                                                       self.local_rank,
                                                       backend=self.backend,
                                                       group_name="a2e")
            self.e2a_group = init_model_parallel_group(sub_group_ranks,
                                                       self.local_rank,
                                                       backend=self.backend,
                                                       group_name="e2a")
        logger.info(f"a2e_group ranks: {self.a2e_group.ranks}")
        logger.info(f"e2a_group ranks: {self.e2a_group.ranks}")
        logger.info("p2p connector initialized")

        self._initialized = True

    @property
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
            p2p_group: GroupCoordinator,
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

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        process_group.send_object(metadata_list, dst=dst)
        work_list = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip empty tensors
                continue
            num = torch.distributed.send(tensor, dst=process_group.ranks[dst], group=process_group)
            work_list.append(num)
        return work_list

    def _recv_tensor_dict_async(
            self,
            src: int,
            process_group: GroupCoordinator,
            p2p_group: GroupCoordinator,
    ) -> tuple[dict[str, torch.Tensor | Any], list]:
        """Asynchronously receive a tensor dictionary.

        Args:
            src: Source rank (local rank)
            process_group: The process group to use for communication
            p2p_group: The process group to use for communication

        Returns:
            tuple: (tensor_dict, work_list) - tensor dictionary and work object list
        """
        if not torch.distributed.is_initialized() or process_group.world_size == 1:
            return {}, []

        assert src < process_group.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = process_group.recv_object(src=src)

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
                work = torch.distributed.recv(tensor, src=process_group.ranks[src], group=process_group)
                work_list.append(work)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict, work_list

    def configure_metadata(self, metadata: Optional["AFDConnectorMetadata"],
                           **kwargs) -> None:
        if metadata is not None and metadata.connector_data is None:
            metadata.connector_data = NPUP2PAFDConnectorMetadata()

    def send_attn_output(self,
                         hidden_states: torch.Tensor,
                         metadata: Optional[AFDConnectorMetadata] = None,
                         **kwargs) -> Any:
        """
        This method will be called by the ATTN side.


        * To send the intermediate tensors generated by ATTN instances to FFN.
        """
        ubatch_idx = get_forward_context().ubatch_idx
        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)
        intermediate_tensors = self.create_intermediate_tensors(
            backend=self.backend,
            hidden_states=hidden_states,
            **kwargs
        )
        try:
            dst = (self.a2e_group.rank_in_group + 1) % self.a2e_group.world_size
            logger.info(f"send_attn_output: dst={dst}")
            self._send_tensor_dict_async(
                intermediate_tensors.tensors,
                dst=dst,
                process_group=self.a2e_group,
                p2p_group=groupEp,
            )
            return hidden_states
        except Exception as e:
            raise RuntimeError(f"Communication error: {e}")

    def recv_attn_output(
            self,
            metadata: Optional[AFDConnectorMetadata] = None,
            **kwargs
    ) -> Any:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)

        src = (self.a2e_group.rank_in_group - 1) % self.a2e_group.world_size
        logger.info(f"recv_attn_output: src={src}")
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.a2e_group,
            p2p_group=groupEp,
        )

        if self.backend == "hccl":
            recv_input_ids = intermediate_tensors["input_ids"]
            if recv_input_ids is not None:
                get_forward_context().input_ids = recv_input_ids
            return AFDRecvOutput(
                hidden_states=intermediate_tensors["hidden_states"],
                metadata=metadata,
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
            metadata: Optional[AFDConnectorMetadata] = None,
            **kwargs
    ) -> None:
        ubatch_idx = kwargs.get('ubatch_idx', 0)
        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)
        intermediate_tensors = IntermediateTensors(
            {
                "hidden_states": hidden_states,
            }
        )
        dst = (self.e2a_group.rank_in_group + 1) % self.e2a_group.world_size
        logger.info(f"send_ffn_output: dst={dst}")
        self._send_tensor_dict_async(
            intermediate_tensors.tensors,
            dst=dst,
            process_group=self.e2a_group,
            p2p_group=groupEp,
        )

    def recv_ffn_output(self,
                        hidden_states: Optional[torch.Tensor] = None,
                        metadata: Optional[AFDConnectorMetadata] = None) -> torch.Tensor:
        ubatch_idx = get_forward_context().ubatch_idx
        groupEp = _get_group_ep(ubatch_idx, self.hccl_comm_name, self.hccl_comm_name2, self.hccl_comm_name3)
        # Use e2a_group for expert/ffn -> attention communication
        src = (self.e2a_group.rank_in_group - 1) % self.e2a_group.world_size
        logger.info(f"recv_ffn_output: src={src}")
        intermediate_tensors, work_list = self._recv_tensor_dict_async(
            src=src,
            process_group=self.e2a_group,
            p2p_group=groupEp,
        )
        recv_hs = intermediate_tensors["hidden_states"]
        return recv_hs

    def create_intermediate_tensors(self, backend, hidden_states, **kwargs):
        """Factory method for creating intermediate tensor objects.

        Gate computation is always on the FFN side, so only ``hidden_states``
        and ``input_ids`` (needed for tid2eid mapping) are transferred.
        """
        base_tensors = {"hidden_states": hidden_states}

        if backend == "hccl":
            base_tensors["input_ids"] = kwargs.get('input_ids')

        return IntermediateTensors(base_tensors)

    def current_stream_synchronize(self, backend):
        if backend == "hccl":
            torch.npu.current_stream().synchronize()
        else:
            torch.cuda.current_stream().synchronize()

    def compute_moe(self, experts, hidden_states, **kwargs):
        """Delegate to ``afd_ffn_compute`` on the experts module.

        Gate computation is always on the FFN side, so routing tensors
        (``router_logits`` / ``topk_weights`` / ``topk_ids``) are not
        transferred from the attention side and are computed locally by
        ``afd_ffn_compute`` which reuses the non-AFD MoE path.
        """
        return experts.afd_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            router_logits=kwargs.get('router_logits'),
            topk_weights=kwargs.get('topk_weights'),
            topk_ids=kwargs.get('topk_ids'),
            row_idx=kwargs.get('row_idx'))

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

    # -------------------------------------------------------------------------
    #                       dp_metadata_list exchange
    #    (ported from v0.13 CAMP2PAFDConnector, used by start_ffn_server_loop)
    # -------------------------------------------------------------------------
    def send_dp_metadata_list(
        self,
        data,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ):
        """Send dp_metadata_list to the corresponding FFN ranks.

        Args:
            data: dp_metadata_list dict
            is_graph_capturing: whether in graph capture stage
            is_warmup: whether in warmup stage
            cudagraph_mode: cudagraph mode
        """
        send_data = (data, is_graph_capturing, is_warmup, cudagraph_mode)

        for dst in self.dst_list:
            object_bytes = pickle.dumps(send_data)
            object_tensor_cpu = torch.frombuffer(bytearray(object_bytes), dtype=torch.uint8)

            # p2p_pg is created with the hccl backend, which only supports
            # NPU tensors. Metadata tensors must therefore live on NPU
            # (same as send_is_ubatch). The v0.13 CAMP2PAFDConnector kept
            # them on CPU because it used a gloo-backed group; that does
            # not apply to NPUP2PAFDConnector on Ascend.
            object_tensor_npu = torch.empty(object_tensor_cpu.shape,
                                            dtype=torch.uint8,
                                            device="npu")
            object_tensor_npu.copy_(object_tensor_cpu)

            size_tensor = torch.tensor([object_tensor_cpu.numel()],
                                       dtype=torch.long,
                                       device="npu")

            # logger.info(
            #     "send_dp_metadata_list dst:%s is_graph_capturing:%s is_warmup:%s cudagraph_mode:%s",
            #     dst, is_graph_capturing, is_warmup, cudagraph_mode)

            torch.distributed.send(size_tensor, dst=dst, group=self.p2p_pg)
            torch.distributed.send(object_tensor_npu, dst=dst, group=self.p2p_pg)

    def recv_dp_metadata_list(self):
        """Receive dp_metadata_list.

        Returns:
            tuple: (data, is_graph_capturing, is_warmup, cudagraph_mode)
        """
        src = self.p2p_rank % self.min_size + self.ffn_size

        size_tensor = torch.empty(1, dtype=torch.long, device="npu")
        rank_size = torch.distributed.recv(size_tensor, src=src, group=self.p2p_pg)

        object_tensor_npu = torch.empty(size_tensor.item(), dtype=torch.uint8, device="npu")
        rank_object = torch.distributed.recv(object_tensor_npu, src=src, group=self.p2p_pg)

        assert rank_object == rank_size, \
            "Received object sender rank does not match the size sender rank."

        object_tensor_cpu = object_tensor_npu.cpu()
        obj = pickle.loads(object_tensor_cpu.numpy().tobytes())

        if len(obj) == 4:
            data, is_graph_capturing, is_warmup, cudagraph_mode = obj
        else:
            # Backward compatibility with old format
            data, is_graph_capturing = obj
            is_warmup = False
            cudagraph_mode = CUDAGraphMode.NONE

        # logger.info("recv_dp_metadata_list is_graph_capturing:%s is_warmup:%s cudagraph_mode:%s",
        #              is_graph_capturing, is_warmup, cudagraph_mode)

        return data, is_graph_capturing, is_warmup, cudagraph_mode

    def update_state_from_dp_metadata(
        self,
        dp_metadata_list: dict,
        is_graph_capturing: bool = False,
    ):
        """Update connector state from received dp_metadata_list.

        Args:
            dp_metadata_list: dp_metadata_list dict
            is_graph_capturing: whether in graph capture stage
        """
        self.dp_metadata_list = dp_metadata_list
