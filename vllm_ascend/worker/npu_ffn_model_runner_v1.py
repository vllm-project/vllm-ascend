# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
import re
from vllm.config import VllmConfig
from vllm.distributed.afd_transfer.afd_connector.factory import (
    AFDConnectorFactory)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
                                             get_world_group,is_global_first_rank)
from vllm.forward_context import set_forward_context,BatchDescriptor
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.utils import DeviceMemoryProfiler, GiB_bytes
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner,graph_capture
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm.distributed.afd_transfer.afd_connector.metadata import (
    AFDConnectorMetadata,FFNNeedForwardData,M2NAFDConnectorMetadata)
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.v1.worker.gpu_ffn_model_runner import GPUFFNModelRunner
from vllm.platforms import current_platform


if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

logger = init_logger(__name__)


class NPUFFNModelRunner(NPUModelRunner,GPUFFNModelRunner):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config=vllm_config,
                         device=device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.load_config = vllm_config.load_config
        self.first_k_dense_replace = self.model_config.hf_config.first_k_dense_replace
        self._forword_cnt = 0

        self.afd_config = vllm_config.afd_config
        if not self.afd_config or not self.afd_config.is_ffn_server:
            raise ValueError(
                "AFD config must be provided with afd_role='ffn' for FFN server"
            )
        self.connector_name = self.afd_config.afd_connector
        
        self._counter = 0

        # Initialize ACL graph support
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))
        # self.aclgraph_batch_sizes = list(
        #     reversed(self.compilation_config.cudagraph_capture_sizes))
        
        # Storage for captured graphs
        self._acl_graphs: dict[tuple[int, int], torch.npu.NPUGraph] = {
        }  # {(layer_idx, num_tokens): ACLGraph}
        self.graph_pool = None

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank,
            get_world_group().local_rank, self.vllm_config)
        
        self.connector.init_afd_connector()
        if getattr(self.model_config.hf_config, "text_config",
                   None) is not None:
            self.num_layers = (
                self.model_config.hf_config.text_config.num_hidden_layers)
        else:
            self.num_layers = self.model_config.hf_config.num_hidden_layers

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def _get_current_layer_idx(self) -> int:
        return (self._counter //
                self.afd_config.num_afd_stages) % self.num_layers
    
    def profile_run(self):
        self._dummy_run(self.max_num_tokens)

        
    @torch.inference_mode()
    def execute_model(self, scheduler_output=None, intermediate_tensors=None):
        """Execute FFN computation for a single request"""
        # scheduler_output and intermediate_tensors are unused in FFN server
        # mode
        current_layer_idx = self._get_current_layer_idx()
        try:
            # AFDConnectorMetadata
            if current_layer_idx < 1:
                return
            if self.connector_name == "m2nconnector":
                # TODO metadata
                m2n_afdconnector_data = M2NAFDConnectorMetadata()
                m2n_afdconnector_data.quant_mode = 0
                m2n_afdconnector_data.expand_x_type = torch.bfloat16
                m2n_afdconnector_data.moe_expert_num = 64
                m2n_afdconnector_data.h = 2048
                m2n_afdconnector_data.k = 8
                m2n_afdconnector_data.expert_token_nums_type = 0
                m2n_afdconnector_data.aiv_num = 48
                m2n_afdconnector_data.batch_size = self.max_num_tokens * m2n_afdconnector_data.k * 2
                
                hidden_states, dynamic_scales, group_list, handle, topk_weights,afdConnectorMetadata = self.connector.recv_attn_output(m2n_afdconnector_data)
                print(f'recv_attn_output success ,layer id is {current_layer_idx}')
                m2n_afdconnector_data.handle = handle
                m2n_afdconnector_data.topk_weights = topk_weights
                print(f'dynamic_scales shape is {dynamic_scales.shape},dtype is {dynamic_scales.dtype}')
                print(f'group_list shape is {group_list.shape},dtype is {group_list.dtype}')
                print(f'topk_weights shape is {topk_weights.shape},dtype is {topk_weights.dtype}')
            elif self.connector_name == "camconnector":
                output1,afdConnectorMetadata = self.connector.recv_attn_output()
                current_layer_idx = afdConnectorMetadata.layer_idx
                hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize = output1[0:8]
                group_list = expertTokenNums.to(torch.int64)
                topk_weights = simulateExpertScales
            else:
                hidden_states,router_logits,topk_weights, topk_ids, row_idx, afdConnectorMetadata = self.connector.recv_attn_output()
                print(f'recv_attn_output success ,layer id is {current_layer_idx}')
            logger.info("*"*50)
            logger.info(f"layer {current_layer_idx} moe recv hidden states type:{type(hidden_states)}, shape:{hidden_states.shape}")
            num_tokens = hidden_states.shape[0]

            # Try to use ACL graph if available
            acl_graph_info = self._find_cuda_graph(current_layer_idx,
                                                    num_tokens)
            # print(f'acl_graph_info is {acl_graph_info}')
            # print(f'current_layer_idx is {current_layer_idx},num_tokens is {num_tokens}')
            print(f'self._forword_cnt is {self._forword_cnt},num_tokens is {num_tokens}')
            # if cuda_graph_info is not None:
            # uniform_decode = (max_query_len == self.uniform_decode_query_len) and (
            # scheduler_output.total_num_scheduled_tokens
            # == self.input_batch.num_reqs * max_query_len)
            # batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
            #                                 uniform_decode=uniform_decode)
            # aclgraph_runtime_mode, batch_descriptor = \
            # self.aclgraph_dispatcher.dispatch(batch_descriptor)
            if acl_graph_info is not None:
                # Use captured ACL graph for computation
                with set_ascend_forward_context(
                        attn_metadata=None,
                        vllm_config=self.vllm_config,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        # batch_descriptor=batch_descriptor,
                        # aclgraph_runtime_mode=aclgraph_runtime_mode,
                        prefetch_stream=self.prefetch_stream,
                        model_instance=self.model):
                    if self.connector_name == "m2nconnector":
                        # 未combine hidden
                        rank_ffn_output = self._execute_with_acl_graph(
                            acl_graph_info = acl_graph_info,
                            hidden_states=hidden_states,
                            group_list=group_list,
                            dynamic_scales=dynamic_scales,
                            topk_weights=topk_weights,
                            current_layer_idx=current_layer_idx)
                    elif self.connector_name == "camconnector":
                        # 未combine hidden
                        rank_ffn_output = self._execute_with_acl_graph(
                            acl_graph_info = acl_graph_info,
                            hidden_states=hidden_states,
                            group_list=group_list,
                            dynamic_scales=dynamic_scales,
                            topk_weights=topk_weights,
                            current_layer_idx=current_layer_idx)
                    else:
                        rank_ffn_output = self._execute_with_acl_graph(
                            acl_graph_info = acl_graph_info,
                            hidden_states = hidden_states,
                            router_logits = router_logits,
                            current_layer_idx = current_layer_idx,
                            topk_weights = topk_weights, 
                            topk_ids = topk_ids,
                            row_idx = row_idx,
                            )
            else:
                # Fallback to eager mode
                # TODO(yxj):
                if self.connector_name == "p2pconnector":
                    ffn_need_forward_data = afdConnectorMetadata.ffn_need_forward_data
                    with_prefill = ffn_need_forward_data.with_prefill
                    moe_comm_type = ffn_need_forward_data.moe_comm_type
                    num_input_tokens = ffn_need_forward_data.num_input_tokens
                    total_num_scheduled_tokens = ffn_need_forward_data.total_num_scheduled_tokens
                
                with set_ascend_forward_context(
                        attn_metadata=None,
                        vllm_config=self.vllm_config,
                        num_tokens=num_input_tokens if self.connector_name == "p2pconnector" else None,
                        with_prefill=with_prefill if self.connector_name == "p2pconnector" else None,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        moe_comm_type=moe_comm_type if self.connector_name == "p2pconnector" else None,
                        prefetch_stream=self.prefetch_stream,
                        num_actual_tokens=total_num_scheduled_tokens if self.connector_name == "p2pconnector" else None,
                        model_instance=self.model):
                    if self.connector_name == "m2nconnector":
                        # 未combine hidden
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states=hidden_states,
                            group_list=group_list,
                            dynamic_scales=dynamic_scales,
                            topk_weights=topk_weights,
                            current_layer_idx=current_layer_idx)
                    elif self.connector_name == "camconnector":
                        # 未combine hidden
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states=hidden_states,
                            group_list=group_list,
                            dynamic_scales=dynamic_scales,
                            topk_weights=topk_weights,
                            current_layer_idx=current_layer_idx)
                    else:
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states = hidden_states,
                            router_logits = router_logits,
                            current_layer_idx = current_layer_idx,
                            topk_weights = topk_weights, 
                            topk_ids = topk_ids,
                            row_idx = row_idx,
                            )

            if self.connector_name == "camconnector":
                handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
                afdConnectorMetadata.cam_afdconnector_data.handle = handle
                self.connector.send_ffn_output(rank_ffn_output, afdConnectorMetadata)
            elif self.connector_name == "m2nconnector":
                self.connector.send_ffn_output(rank_ffn_output, m2n_afdconnector_data)
                print(f'send_ffn_output success ,layer id is {current_layer_idx}')
            else :
                self.connector.send_ffn_output(rank_ffn_output, afdConnectorMetadata)
                print(f'send_ffn_output success ,layer id is {current_layer_idx}')
        except Exception as e:
            raise ValueError(
                f"Error computing FFN for layer {current_layer_idx}: {e}"
            ) from e
        finally:
            self._counter += 1
            if (self._counter == self.num_layers *
                    self.afd_config.num_afd_stages):
                self._counter = 0
                self._forword_cnt += 1
        return None  # FFN server doesn't return ModelRunnerOutput

    def _execute_with_cuda_graph(self, hidden_states: torch.Tensor,
                                 cuda_graph_info: dict):
        """Execute FFN computation using captured ACL graph."""
        graph = cuda_graph_info['graph']
        input_tensor = cuda_graph_info['input_hidden_states']
        output_tensor = cuda_graph_info['output']

        # Copy input data to graph's input tensor
        # Handle padding if necessary
        actual_tokens = hidden_states.shape[0]
        graph_tokens = input_tensor.shape[0]

        if actual_tokens <= graph_tokens:
            # Copy actual data and pad with zeros if needed
            input_tensor[:actual_tokens].copy_(hidden_states)
            if actual_tokens < graph_tokens:
                input_tensor[actual_tokens:].zero_()
        else:
            raise ValueError(
                f"Input size {actual_tokens} exceeds graph capacity "
                f"{graph_tokens}")

        # Replay the captured graph
        graph.replay()

        # Return only the actual output (without padding)
        return output_tensor[:actual_tokens].clone()
    
    def _execute_with_acl_graph(self, 
                                hidden_states: torch.Tensor,
                                current_layer_idx: int,
                                acl_graph_info: Optional[dict] = None,
                                router_logits: Optional[torch.Tensor] = None,
                                group_list: Optional[torch.Tensor] = None,
                                dynamic_scales: Optional[torch.Tensor] = None,
                                topk_weights: Optional[torch.Tensor] = None,
                                topk_ids: Optional[torch.Tensor] = None,
                                row_idx: Optional[torch.Tensor] = None,
                                ):
        """Execute FFN computation using captured ACL graph."""
        graph = acl_graph_info['graph']
        input_tensor = acl_graph_info['input_hidden_states']
        output_tensor = acl_graph_info['output']

        # Copy input data to graph's input tensor
        # Handle padding if necessary
        actual_tokens = hidden_states.shape[0]
        graph_tokens = input_tensor.shape[0]

        if actual_tokens <= graph_tokens:
            # Copy actual data and pad with zeros if needed
            input_tensor[:actual_tokens].copy_(hidden_states)
            if actual_tokens < graph_tokens:
                input_tensor[actual_tokens:].zero_()
        else:
            raise ValueError(
                f"Input size {actual_tokens} exceeds graph capacity "
                f"{graph_tokens}")

        # Replay the captured graph
        graph.replay()
        print("FFN Replay graphs")
        # Return only the actual output (without padding)
        return output_tensor[:actual_tokens].clone()
        

    def _execute_eager_mode(self, 
                            hidden_states: torch.Tensor,
                            current_layer_idx: int,
                            router_logits: Optional[torch.Tensor] = None,
                            group_list: Optional[torch.Tensor] = None,
                            dynamic_scales: Optional[torch.Tensor] = None,
                            topk_weights: Optional[torch.Tensor] = None,
                            topk_ids: Optional[torch.Tensor] = None,
                            row_idx: Optional[torch.Tensor] = None,):
        """Execute FFN computation in eager mode (fallback)."""
        # Handle TP case: all-gather tensors from all TP ranks
        # print('_execute_eager_mode')
        print(f'current_layer_idx in _execute_eager_mode is {current_layer_idx}')
        tp_world_size = get_tensor_model_parallel_world_size()
        # if tp_world_size > 1:
        #     # All-gather hidden states from all TP ranks
        #     gathered_hidden_states = tensor_model_parallel_all_gather(
        #         hidden_states, dim=0)
        #     ffn_output = self.model.compute_ffn_output(current_layer_idx,
        #                                                gathered_hidden_states)
        #     # Extract the output corresponding to current rank
        #     start_idx = hidden_states.shape[
        #         0] * get_tensor_model_parallel_rank()
        #     end_idx = start_idx + hidden_states.shape[0]
        #     rank_ffn_output = ffn_output[start_idx:end_idx, :]
        # else:
        # Single TP case
        if self.connector_name == "m2nconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids)
        elif self.connector_name == "camconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)
        else:
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)

        return rank_ffn_output

    

    
    def capture_model(self) -> int:
        """Capture ACL graphs for FFN operations."""
        
        logger.debug("Starting ACL graph capture for FFN operations...")
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        
        self._capture_model()
        
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

        return npu_graph_size

    def _capture_model(self):
        if self.graph_pool is None:
            self.graph_pool = current_platform.get_global_graph_pool()
        if not self.use_aclgraph:
            logger.warning(
                "Skipping ACL graph capture. To turn on ACL graph capture, "
                "ensure `aclraph_mode` was not manually set to `NONE`")
            return
        else:
            self.initialize_aclgraph_capture()
            
        set_cudagraph_capturing_enabled(True)
        # Trigger ACL graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            # pass
            max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
            decode_cudagraph_batch_sizes = [
                x for x in self.aclgraph_batch_sizes if x <= max_num_tokens
                and x >= self.uniform_decode_query_len
            ]
            compilation_cases_decode = list(
                reversed(decode_cudagraph_batch_sizes))
            self._capture_aclgraphs(
                    compilation_cases=compilation_cases_decode,
                    aclgraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True)
           
        set_cudagraph_capturing_enabled(False)
        
    def _capture_aclgraphs(self, compilation_cases: list[int],
                           aclgraph_runtime_mode: CUDAGraphMode,
                           uniform_decode: bool):
        assert aclgraph_runtime_mode != CUDAGraphMode.NONE and \
            aclgraph_runtime_mode in [CUDAGraphMode.FULL,
                                      CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            logger.info(
                "Starting to capture ACL graphs for cases: %s, "
                "mode: %s, uniform_decode: %s", compilation_cases,
                aclgraph_runtime_mode.name, uniform_decode)
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing ACL graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    aclgraph_runtime_mode.name))
        for num_tokens in compilation_cases:
            # Warm up the operations for this specific layer
            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                force_attention = (aclgraph_runtime_mode == CUDAGraphMode.FULL)
                self._dummy_run(num_tokens,
                                aclgraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode)
            self._dummy_run(num_tokens,
                            aclgraph_runtime_mode=CUDAGraphMode.FULL,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode)
    
    def _dummy_run(self, 
                   num_tokens: int = 1, 
                   aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                   force_attention: bool = False,
                   uniform_decode: bool = False,**kwargs):
        
        # only support eager mode and piecewise graph now
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        # filter out the valid batch descriptor
        _ag_mode, batch_descriptor = \
            self.aclgraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens,
                                uniform_decode=uniform_decode))
        if aclgraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for aclgraph capture
            assert aclgraph_runtime_mode == CUDAGraphMode.NONE or \
                aclgraph_runtime_mode == _ag_mode, (
                f"Aclgraph runtime mode mismatch at dummy_run. "
                f"Expected {_ag_mode}, but got {aclgraph_runtime_mode}.")
        else:
            aclgraph_runtime_mode = _ag_mode
            
        # mock self.model
        for layer_idx in range(self.first_k_dense_replace,self.num_layers):
            with set_ascend_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                    reserved_mc2_mask=self.reserved_mc2_mask,
                    batch_descriptor=batch_descriptor,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    prefetch_stream=self.prefetch_stream,
                    model_instance=self.model):
                if self.connector_name == "m2nconnector":
                    self._capture_graph_for_layer_and_size(
                        layer_idx, num_tokens,aclgraph_runtime_mode)
                elif self.connector_name == "camconnector":

                    self._capture_graph_for_layer_and_size(
                        layer_idx, num_tokens,aclgraph_runtime_mode)
                else:
                    self._capture_graph_for_layer_and_size(
                        layer_idx, num_tokens,aclgraph_runtime_mode)
        
                       
    def _capture_graph_for_layer_and_size(self, layer_idx: int,
                                          num_tokens: int,
                                          aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,):
        """Capture ACL graph for specific layer and number of tokens."""
        # Create dummy hidden states
        dummy_hidden_states = torch.randn(
            num_tokens,
            self.model_config.hf_config.hidden_size,
            dtype=self.dtype,
            device=self.device)
        # for m2n
        dummy_group_list = torch.randn(
            32,
            dtype=torch.int64,
            device=self.device)
        dummy_topk_weights = torch.randn(
            num_tokens,
            dtype=torch.float32,
            device=self.device)
        dummy_dynamic_scales = torch.randn(
            num_tokens,
            dtype=torch.float32,
            device=self.device)
        
        if aclgraph_runtime_mode == CUDAGraphMode.NONE:
            # warm up
            if self.connector_name == "m2nconnector":
                # TODO metadata
                m2n_afdconnector_data = M2NAFDConnectorMetadata()
                m2n_afdconnector_data.quant_mode = 0
                m2n_afdconnector_data.expand_x_type = torch.bfloat16
                m2n_afdconnector_data.moe_expert_num = 64
                m2n_afdconnector_data.h = 2048
                m2n_afdconnector_data.k = 8
                m2n_afdconnector_data.expert_token_nums_type = 0
                m2n_afdconnector_data.aiv_num = 48
                m2n_afdconnector_data.batch_size = num_tokens * m2n_afdconnector_data.k * 2
                
                hidden_states, dynamic_scales, group_list, handle, topk_weights,afdConnectorMetadata = self.connector.recv_attn_output(m2n_afdconnector_data)
                # print(f'recv_attn_output success ,layer id is {current_layer_idx}')
                m2n_afdconnector_data.handle = handle
                m2n_afdconnector_data.topk_weights = topk_weights
            # compute_ffn_output ,未combine hidden
            output = self._run_ffn_computation(hidden_states = hidden_states,
                                               layer_idx=layer_idx,
                                               capture_mode=True,
                                               group_list=group_list,
                                                dynamic_scales=dynamic_scales,
                                                topk_weights=topk_weights
                                               )
            # send_ffn_output
            if self.connector_name == "camconnector":
                pass
            elif self.connector_name == "m2nconnector":
                self.connector.send_ffn_output(output, m2n_afdconnector_data)
            else :
                self.connector.send_ffn_output(output, afdConnectorMetadata)
            return 

        # Create and capture the graph
        aclgraph = torch.npu.NPUGraph()

        # Start graph capture
        with torch.npu.graph(aclgraph, pool=self.graph_pool):
            # recv_attn_output
            if self.connector_name == "m2nconnector":
                # TODO metadata
                m2n_afdconnector_data = M2NAFDConnectorMetadata()
                m2n_afdconnector_data.quant_mode = 0
                m2n_afdconnector_data.expand_x_type = torch.bfloat16
                m2n_afdconnector_data.moe_expert_num = 64
                m2n_afdconnector_data.h = 2048
                m2n_afdconnector_data.k = 8
                m2n_afdconnector_data.expert_token_nums_type = 0
                m2n_afdconnector_data.aiv_num = 48
                m2n_afdconnector_data.batch_size = num_tokens * m2n_afdconnector_data.k * 2
                
                hidden_states, dynamic_scales, group_list, handle, topk_weights,afdConnectorMetadata = self.connector.recv_attn_output(m2n_afdconnector_data)
                # print(f'recv_attn_output success ,layer id is {current_layer_idx}')
                m2n_afdconnector_data.handle = handle
                m2n_afdconnector_data.topk_weights = topk_weights
            # compute_ffn_output
            output = self._run_ffn_computation(hidden_states = hidden_states,
                                               layer_idx=layer_idx,
                                               capture_mode=True,
                                               group_list=group_list,
                                                dynamic_scales=dynamic_scales,
                                                topk_weights=topk_weights
                                               )
            # send_ffn_output
            if self.connector_name == "camconnector":
                pass
            elif self.connector_name == "m2nconnector":
                self.connector.send_ffn_output(output, m2n_afdconnector_data)
            else :
                self.connector.send_ffn_output(output, afdConnectorMetadata)
        print(f'dummy_hidden_states shape is {dummy_hidden_states.shape}')
        print(f'output shape is {output.shape}')
        # Store the captured graph with layer and token count as key
        self._acl_graphs[(layer_idx, num_tokens)] = {
            'graph': aclgraph,
            'input_hidden_states': dummy_hidden_states,
            'output': output
        }
        print(f'self._acl_graphs is {self._acl_graphs}')

        logger.debug("Captured ACL graph for layer %s with %s tokens",
                     layer_idx, num_tokens)

    def _run_ffn_computation(self,
                             hidden_states: torch.Tensor,
                             layer_idx: Optional[int] = None,
                             capture_mode: bool = False,
                             router_logits: Optional[torch.Tensor] = None,
                             group_list: Optional[torch.Tensor] = None,
                             dynamic_scales: Optional[torch.Tensor] = None,
                             topk_weights: Optional[torch.Tensor] = None,
                             topk_ids: Optional[torch.Tensor] = None,
                             row_idx: Optional[torch.Tensor] = None,):
        """Run FFN computation for graph capture or replay."""
        if layer_idx is None:
            current_layer_idx = self._get_current_layer_idx(
            ) if not capture_mode else 0
        else:
            current_layer_idx = layer_idx

        # tp_world_size = get_tensor_model_parallel_world_size()
        # if tp_world_size > 1:
        #     # Handle TP case: all-gather tensors from all TP ranks
        #     gathered_hidden_states = tensor_model_parallel_all_gather(
        #         hidden_states, dim=0)
        #     ffn_output = self.model.compute_ffn_output(current_layer_idx,
        #                                                gathered_hidden_states)

        #     # Extract the output corresponding to current rank
        #     start_idx = hidden_states.shape[
        #         0] * get_tensor_model_parallel_rank()
        #     end_idx = start_idx + hidden_states.shape[0]
        #     rank_ffn_output = ffn_output[start_idx:end_idx, :]
        # else:
        # Single TP case
        
        # TODO(yxj):support m2n\cam\p2p
        
        if self.connector_name == "m2nconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids)
        elif self.connector_name == "camconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)
        else:
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)
        
        # rank_ffn_output = self.model.compute_ffn_output(
        #     current_layer_idx, hidden_states)

        return rank_ffn_output

    def _find_cuda_graph(self, layer_idx: int, num_tokens: int):
        """Find the smallest graph that can handle the given layer and
        number of tokens."""
        if not self.use_aclgraph:
            return None

        # Find the minimum capture size that can handle num_tokens for this
        # layer
        for capture_size in self.cudagraph_batch_sizes:
            if num_tokens <= capture_size:
                return self._acl_graphs.get((layer_idx, capture_size))
        return None

    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        """FFN servers don't use samplers."""
        pass

    def update_config(self, overrides: dict[str, Any]) -> None:
        """Update configuration for FFN model runner."""
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            from vllm.config import update_config
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def reload_weights(self) -> None:
        """Reload model weights for FFN model runner."""
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model = self.get_model()
        model_loader.load_weights(model, model_config=self.model_config)


    def lora_config(self):
        """FFN servers don't support LoRA."""
        return None


    def is_pooling_model(self) -> bool:
        """FFN servers are not pooling models."""
        return False

    def _dummy_pooler_run(self, hidden_states: torch.Tensor):
        """FFN servers don't have poolers."""
        pass

    def get_supported_tasks(self):
        """Get supported tasks for FFN model runner."""
        return []

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        """Get number of input tokens for FFN model runner."""
        return num_scheduled_tokens

    def take_draft_token_ids(self, **kwargs):
        """FFN servers don't support draft tokens."""
        pass


    def eplb_state(self):
        """FFN servers don't have EPLB state."""
        return None

    def ensure_kv_transfer_shutdown(self):
        """FFN servers don't need KV transfer shutdown."""
        pass

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        """FFN servers don't support tensorized model saving."""
        raise NotImplementedError(
            "FFN servers don't support tensorized model saving")
