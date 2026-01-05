# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch_npu
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
    AFDConnectorMetadata, FFNNeedForwardData, M2NAFDConnectorMetadata, CAMM2NAFDConnectorMetadata, CAMP2PAFDConnectorMetadata)
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.v1.worker.gpu_ffn_model_runner import GPUFFNModelRunner
from vllm.platforms import current_platform
import vllm.envs as envs_vllm


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
        self.num_hidden_layers = self.model_config.hf_config.num_hidden_layers
        self._forword_cnt = 0

        self.afd_config = vllm_config.afd_config
        if not self.afd_config or not self.afd_config.is_ffn_server:
            raise ValueError(
                "AFD config must be provided with afd_role='ffn' for FFN server"
            )
        self.connector_name = self.afd_config.afd_connector
        
        self._counter = 0

        # Initialize ACL graph support
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))
        
        # Storage for captured graphs
        self._acl_graphs_full: dict[int, torch.npu.NPUGraph] = {
        }  # {num_tokens: ACLGraph}
        self._acl_graphs_ubatch_full: dict[int, torch.npu.NPUGraph] = {
        }  # {num_tokens: ACLGraph}
        self.graph_pool = None

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank,
            get_world_group().local_rank, self.vllm_config)
        
        self.connector.init_afd_connector()
        self.attn_size = self.connector.attn_size
        self.ffn_size = self.connector.ffn_size
        print(f'attn_size = {self.attn_size},ffn_size = {self.ffn_size}')
        if getattr(self.model_config.hf_config, "text_config",
                   None) is not None:
            self.num_layers = (
                self.model_config.hf_config.text_config.num_hidden_layers)
        else:
            self.num_layers = self.model_config.hf_config.num_hidden_layers
        self.dummy_run_call_cnt = 0
        self.replay_cnt = 0
        self.topk = self.model_config.hf_config.num_experts_per_tok
        self.n_routed_experts = self.model_config.hf_config.n_routed_experts
        self.hidden_size = self.model_config.hf_config.hidden_size
        print(f'self.topk is {self.topk}')

        # TODO(jcz): 对于ffn eager模式下_current_num_ubatches逻辑和layer_idx的计算逻辑需要优化
        #            这里的2是hardcode，需要根据实际情况调整，初始化的时候目前暂时需要初始化成dummy run跑的batch_size
        self._current_num_ubatches = self.afd_config.num_afd_stages
        self.is_ubatch = False
        
        # self.profiler
        # import os
        # experimental_config = torch_npu.profiler._ExperimentalConfig(
        #     export_type=torch_npu.profiler.ExportType.Text,
        #     profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        #     aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        # )
        # self.prof = torch_npu.profiler.profile(
        #     activities=[
        #         torch_npu.profiler.ProfilerActivity.CPU,
        #         torch_npu.profiler.ProfilerActivity.NPU
        #     ],
        #     schedule=torch_npu.profiler.schedule(wait=5, warmup=2, active=20, repeat=1, skip_first=20),
        #     # 初步采集最好不要使用下面两个选项， with_stack 会大幅增加采集时间及采集的数据大小，深入分析CPU测瓶颈时再打开
        #     experimental_config=experimental_config,
        #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("/dl/y00889327/profile/prof_ffn")
        # )
        # self.prof.start()

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    def _get_current_layer_idx(self) -> int:
        return (self._counter // self._current_num_ubatches) % self.num_layers
    
    def profile_run(self,is_ubatch:bool=False):
        print(f'yxj is_ubatch in profile_run is {is_ubatch}')
        self._dummy_run(self.max_num_tokens,is_ubatch=is_ubatch)

        
    @torch.inference_mode()
    def execute_model(self, scheduler_output=None, intermediate_tensors=None, is_ubatch:bool=False):
        """Execute FFN computation for a single request"""
        # self.prof.step()
        current_layer_idx = self._get_current_layer_idx()
        try:
            # skip dense layer
            # if current_layer_idx < self.first_k_dense_replace:
            #     return
            
            if self.use_aclgraph and not is_ubatch:
                # mock
                self._ffn_forward(aclgraph_runtime_mode=CUDAGraphMode.NONE,is_ubatch=is_ubatch)
                print(f"ffn replay,replay_cnt is {self.replay_cnt}",flush=True)
                return
            elif self.use_aclgraph and is_ubatch:
                # TODO(yxj):ffn图模式会直接replay，应该设计成ffn收到attn消息才开始replay
                # replay
                if self.connector_name == "camconnector":
                    max_num_tokens = self.max_num_tokens * self.attn_size * (self.n_routed_experts // self.ffn_size) * (self.attn_size // self.ffn_size)
                else:
                    max_num_tokens = self.max_num_tokens * self.topk * self.attn_size
                acl_graph_info = self._acl_graphs_ubatch_full.get(max_num_tokens)
                graph = acl_graph_info['graph']
                graph.replay()
                self.replay_cnt += 1
                print(f"ffn replay,replay_cnt is {self.replay_cnt}",flush=True)
                return
                
            if self.connector_name == "m2nconnector":
                # TODO metadata
                m2n_afdconnector_data = M2NAFDConnectorMetadata()

                hidden_states, dynamic_scales, group_list, topk_weights, afdConnectorMetadata = \
                self._build_and_recv_m2n_afdconnector(
                    m2n_afdconnector_data=m2n_afdconnector_data,
                    quant_mode= 0,
                    expand_x_type = torch.bfloat16,
                    n_routed_experts=self.n_routed_experts,
                    hidden_size=self.hidden_size,
                    topk=self.topk,
                    expert_token_nums_type=0,
                    attn_size=self.attn_size,
                    max_num_tokens=self.max_num_tokens,
                    )
            elif self.connector_name == "camm2nconnector":
                cam_afdconnector_data = CAMM2NAFDConnectorMetadata(
                    moe_expert_num = self.n_routed_experts,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = 48,
                    batch_size = self.max_num_tokens,
                    h = self.hidden_size,
                    k = self.topk
                )
                output1,afdConnectorMetadata = self.connector.recv_attn_output(cam_afdconnector_data, self._counter % self._current_num_ubatches)
                self._current_num_ubatches = afdConnectorMetadata.num_ubatches
                hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize = output1[0:8]
                group_list = expertTokenNums.to(torch.int64)
                topk_weights = simulateExpertScales
            elif self.connector_name == "camp2pconnector":
                cam_afdconnector_data = CAMP2PAFDConnectorMetadata(
                    moe_expert_num = self.n_routed_experts,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = 48,
                    batch_size = self.max_num_tokens,
                    h = self.hidden_size,
                    k = self.topk
                )
                a2eOutput, afdConnectorMetadata, cam_p2p_ep_name = self.connector.recv_attn_output(cam_afdconnector_data)
                hidden_states, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = a2eOutput[0:5]
                topk_weights = simulateExpertScales
                topk_ids = simulateExpertIds
                x_active_mask = xActiveMaskOut
                print(f'recv_attn_output success ,layer id is {current_layer_idx}')
                print(f'hidden_states shape is {hidden_states.shape},dtype is {hidden_states.dtype}')
            elif self.connector_name == "p2pconnector":
                hidden_states,router_logits,topk_weights, topk_ids, row_idx, afdConnectorMetadata = self.connector.recv_attn_output()
                if afdConnectorMetadata is not None and afdConnectorMetadata.recv_handle_list is not None:
                    for work in afdConnectorMetadata.recv_handle_list:
                        work.wait()
                print(f'router_logits shape is {router_logits.shape},dtype is {router_logits.dtype}')
                print(f'topk_weights shape is {topk_weights.shape},dtype is {topk_weights.dtype}')
                print(f'topk_ids shape is {topk_ids.shape},dtype is {topk_ids.dtype}')
                print(f'row_idx shape is {row_idx.shape},dtype is {row_idx.dtype}')
                print(f'recv_attn_output success ,layer id is {current_layer_idx}')

                
            # Try to use ACL graph if available
            # TODO(yxj):move layer
            # 先做成通信不如图的版本
            if self.connector_name == "camm2nconnector":
                max_num_tokens = self.max_num_tokens * self.attn_size * (cam_afdconnector_data.moe_expert_num // self.ffn_size) * (self.attn_size // self.ffn_size)
            else:
                max_num_tokens = self.max_num_tokens * self.topk * self.attn_size # 64
            self.aclgraph_batch_sizes.append(max_num_tokens)
            num_tokens = hidden_states.shape[0]
            print(f'num_tokens  is {num_tokens}')
            
            
            # Fallback to eager mode
            if afdConnectorMetadata is not None:
                ffn_need_forward_data = afdConnectorMetadata.ffn_need_forward_data
                with_prefill = ffn_need_forward_data.with_prefill
                moe_comm_type = ffn_need_forward_data.moe_comm_type
                num_input_tokens = ffn_need_forward_data.num_input_tokens
                total_num_scheduled_tokens = ffn_need_forward_data.total_num_scheduled_tokens
                current_layer_idx = afdConnectorMetadata.layer_idx
            # test
            with set_ascend_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                    num_tokens=num_input_tokens,
                    with_prefill=with_prefill,
                    reserved_mc2_mask=self.reserved_mc2_mask,
                    moe_comm_type=moe_comm_type,
                    prefetch_stream=self.prefetch_stream,
                    num_actual_tokens=total_num_scheduled_tokens,
                    model_instance=self.model):
                if self.connector_name == "m2nconnector":
                    # 未combine hidden
                    rank_ffn_output = self._execute_eager_mode(
                        hidden_states=hidden_states,
                        group_list=group_list,
                        dynamic_scales=dynamic_scales,
                        topk_weights=topk_weights,
                        current_layer_idx=current_layer_idx)
                elif self.connector_name == "camm2nconnector":
                    # 未combine hidden
                    rank_ffn_output = self._execute_eager_mode(
                        hidden_states=hidden_states,
                        group_list=group_list,
                        dynamic_scales=dynamic_scales,
                        topk_weights=topk_weights,
                        current_layer_idx=current_layer_idx)
                elif self.connector_name == "camp2pconnector":
                        # 未combine hidden
                        rank_ffn_output = self._execute_eager_mode(
                            hidden_states=hidden_states,
                            topk_ids=topk_ids,
                            x_active_mask=x_active_mask,
                            topk_weights=topk_weights,
                            current_layer_idx=current_layer_idx,
                            cam_p2p_ep_name=cam_p2p_ep_name)
                else:
                    rank_ffn_output = self._execute_eager_mode(
                        hidden_states = hidden_states,
                        router_logits = router_logits,
                        current_layer_idx = current_layer_idx,
                        topk_weights = topk_weights, 
                        topk_ids = topk_ids,
                        row_idx = row_idx,
                        )

            if self.connector_name == "camm2nconnector":
                handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
                cam_afdconnector_data.handle = handle
                self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data, self._counter % self._current_num_ubatches)
            elif self.connector_name == "camp2pconnector":
                handle = [attenBatchSize]
                cam_afdconnector_data.handle = handle
                self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data)
            elif self.connector_name == "m2nconnector":
                self.connector.send_ffn_output(rank_ffn_output, m2n_afdconnector_data)
                print(f'send_ffn_output success ,layer id is {current_layer_idx}',flush=True)
            else :
                afdConnectorMetadata.recv_handle_list = None
                self.connector.send_ffn_output(rank_ffn_output, afdConnectorMetadata)
                print(f'send_ffn_output success ,layer id is {current_layer_idx}')
        except Exception as e:
            raise ValueError(
                f"Error computing FFN for layer {current_layer_idx}: {e}"
            ) from e
        finally:
            self._counter += 1
            # if (self._counter == self.num_layers *
            #         self.afd_config.num_afd_stages):
            if (self._counter == self.num_layers *
                    self._current_num_ubatches):
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
                                x_active_mask: Optional[torch.Tensor] = None,
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
                            row_idx: Optional[torch.Tensor] = None,
                            x_active_mask: Optional[torch.Tensor] = None,
                            cam_p2p_ep_name: Optional[str] = ""):
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
        elif self.connector_name == "camm2nconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)
        elif self.connector_name == "camp2pconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx,
                x_active_mask=x_active_mask,
                cam_p2p_ep_name=cam_p2p_ep_name)
        else:
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)

        return rank_ffn_output

    

    
    def capture_model(self,is_ubatch:bool=False) -> int:
        """Capture ACL graphs for FFN operations."""
        
        logger.debug("Starting ACL graph capture for FFN operations...")
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        
        self._capture_model(is_ubatch)
        
        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

        return npu_graph_size

    def _capture_model(self,is_ubatch:bool=False):
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
                    uniform_decode=True,
                    is_ubatch=is_ubatch)
           
        set_cudagraph_capturing_enabled(False)
        
    def _capture_aclgraphs(self, compilation_cases: list[int],
                           aclgraph_runtime_mode: CUDAGraphMode,
                           uniform_decode: bool,
                           is_ubatch:bool=False):
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
                                uniform_decode=uniform_decode,
                                is_ubatch=is_ubatch)
            self._dummy_run(num_tokens,
                            aclgraph_runtime_mode=CUDAGraphMode.FULL,
                            force_attention=force_attention,
                            uniform_decode=uniform_decode,
                            is_ubatch=is_ubatch)
    
    def _dummy_run(self, 
                   num_tokens: int = 1, 
                   aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                   force_attention: bool = False,
                   uniform_decode: bool = False,
                   is_ubatch:bool=False,
                   **kwargs):
        
        # recv self.is_ubatch form attn side
        src = (self.connector.process_group.rank_in_group - 1) % self.connector.process_group.world_size
        is_ubatch = self.connector.process_group.recv_object(src)
        print(f'yxj src in _dummy_run is {src}')
        print(f'yxj is_ubatch in _dummy_run is {is_ubatch}')
        
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
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            # capture
            # Create and capture the graph
            aclgraph = torch.npu.NPUGraph()
            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                # compute_ffn_output
                output = self._ffn_forward(batch_descriptor=batch_descriptor,
                                  aclgraph_runtime_mode=aclgraph_runtime_mode,
                                  is_ubatch=is_ubatch)
            print(f'output shape is {output.shape}')
            # Store the captured graph with token count as key
            if is_ubatch:
                self._acl_graphs_ubatch_full[output.shape[0]] = {
                    'graph': aclgraph,
                    'input_hidden_states': output,
                    'output': output
                }
                print(f'self._acl_graphs_ubatch_full is {self._acl_graphs_ubatch_full}',flush=True)
            else:
                self._acl_graphs_full[output.shape[0]] = {
                    'graph': aclgraph,
                    'input_hidden_states': output,
                    'output': output
                }
                print(f'self._acl_graphs_full is {self._acl_graphs_full}',flush=True)
        else:
            self._ffn_forward(batch_descriptor=batch_descriptor,
                                  aclgraph_runtime_mode=aclgraph_runtime_mode,
                                  is_ubatch=is_ubatch) 
            print("finsh capture warm_up or prefile run",flush=True)
        print(f'self.dummy_run_call_cnt is {self.dummy_run_call_cnt}')
        self.dummy_run_call_cnt += 1

    def _build_and_recv_m2n_afdconnector(
        self,
        m2n_afdconnector_data: M2NAFDConnectorMetadata,
        n_routed_experts: int,
        hidden_size: int,
        topk: int,
        expert_token_nums_type:int,
        attn_size: int,
        max_num_tokens: int,
        quant_mode: int = 0,
        expand_x_type: torch.dtype = torch.bfloat16,
    ):
        m2n_afdconnector_data.quant_mode = quant_mode
        m2n_afdconnector_data.expand_x_type = expand_x_type
        m2n_afdconnector_data.moe_expert_num = n_routed_experts
        m2n_afdconnector_data.h = hidden_size
        m2n_afdconnector_data.k = topk
        m2n_afdconnector_data.expert_token_nums_type = expert_token_nums_type
        m2n_afdconnector_data.aiv_num = 48
        m2n_afdconnector_data.batch_size = max_num_tokens * m2n_afdconnector_data.k * attn_size
        
        hidden_states, dynamic_scales, group_list, handle, topk_weights,afdConnectorMetadata = self.connector.recv_attn_output(m2n_afdconnector_data)
        m2n_afdconnector_data.handle = handle
        m2n_afdconnector_data.topk_weights = topk_weights
        
        return hidden_states, dynamic_scales, group_list, topk_weights, afdConnectorMetadata


    def _ffn_forward(self,
                     batch_descriptor=None,
                     aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                     is_ubatch:bool=False):
        print(f'yxj is_ubatch in _ffn_forward is {is_ubatch}',flush=True)
        if is_ubatch:
            ubatch_nums = 2
        else:
            ubatch_nums = 1
        cur_num_stages = 0
        for layer_idx in range(self.first_k_dense_replace,self.num_layers):
            for ubatch_idx in range(ubatch_nums):
                # recv
                if self.connector_name == "m2nconnector":
                    hidden_states, dynamic_scales, group_list, topk_weights, afdConnectorMetadata = \
                        self._build_and_recv_m2n_afdconnector(
                        m2n_afdconnector_data=m2n_afdconnector_data,
                        quant_mode= 0,
                        expand_x_type = torch.bfloat16,
                        n_routed_experts=self.n_routed_experts,
                        hidden_size=self.hidden_size,
                        topk=self.topk,
                        expert_token_nums_type=0,
                        attn_size=self.attn_size,
                        max_num_tokens=self.max_num_tokens,
                        )
                    # [64,2048]
                    hidden_states, dynamic_scales, group_list, handle, topk_weights, afdConnectorMetadata = self.connector.recv_attn_output(m2n_afdconnector_data)
                    print(f'recv_attn_output success ,layer id is {layer_idx},ubatch_idx is {ubatch_idx}',flush=True)
                    m2n_afdconnector_data.handle = handle
                    m2n_afdconnector_data.topk_weights = topk_weights
                elif self.connector_name == "camm2nconnector":
                    cam_afdconnector_data = CAMM2NAFDConnectorMetadata(
                        moe_expert_num = self.n_routed_experts,
                        shared_expert_num = 0,
                        scale = None,
                        handle = None,
                        quant_mode = 0,
                        aiv_num = 48,
                        batch_size = self.max_num_tokens,
                        h = self.hidden_size,
                        k = self.topk
                    )
                    output1, afdConnectorMetadata = self.connector.recv_attn_output(cam_afdconnector_data, cur_num_stages)
                    hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize = output1[0:8]
                    group_list = expertTokenNums.to(torch.int64)
                    topk_weights = simulateExpertScales
                    print(f'cam recv_attn_output success ,layer id is {layer_idx},ubatch_idx is {ubatch_idx}',flush=True)
                elif self.connector_name == "camp2pconnector":
                cam_afdconnector_data = CAMP2PAFDConnectorMetadata(
                    moe_expert_num = self.n_routed_experts,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = 48,
                    batch_size = self.max_num_tokens,
                    h = self.hidden_size,
                    k = self.topk
                )
                a2eOutput, afdConnectorMetadata, cam_p2p_ep_name = self.connector.recv_attn_output(cam_afdconnector_data)
                hidden_states, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = a2eOutput[0:5]
                topk_weights = simulateExpertScales
                topk_ids = simulateExpertIds
                x_active_mask = xActiveMaskOut
            with set_ascend_forward_context(
                        attn_metadata=None,
                        vllm_config=self.vllm_config,
                        reserved_mc2_mask=self.reserved_mc2_mask,
                        batch_descriptor=batch_descriptor,
                        aclgraph_runtime_mode=aclgraph_runtime_mode,
                        prefetch_stream=self.prefetch_stream,
                        model_instance=self.model):
                    if self.connector_name == "camp2pconnector":
                    rank_ffn_output = self._run_ffn_computation(hidden_states = hidden_states,
                                                layer_idx=layer_idx,
                                                capture_mode=True,
                                                topk_ids=topk_ids,
                                                topk_weights=topk_weights,
                                                x_active_mask=x_active_mask,
                                                cam_p2p_ep_name=cam_p2p_ep_name
                                                )
                else:
                    rank_ffn_output = self._run_ffn_computation(hidden_states = hidden_states,
                                                     layer_idx=layer_idx,
                                                     capture_mode=True,
                                                     group_list=group_list,
                                                    dynamic_scales=dynamic_scales,
                                                    topk_weights=topk_weights
                                                 )
                # send
                if self.connector_name == "m2nconnector":
                    self.connector.send_ffn_output(rank_ffn_output, m2n_afdconnector_data)
                    print(f'send_ffn_output success ,layer id is {layer_idx},ubatch_idx is {ubatch_idx}',flush=True)
                elif self.connector_name == "camm2nconnector":
                    handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
                    cam_afdconnector_data.handle = handle
                    self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data, cur_num_stages)
                elif self.connector_name == "camp2pconnector":
                    handle = [attenBatchSize]
                    cam_afdconnector_data.handle = handle
                    self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data)
                    print(f'cam send_ffn_output success ,layer id is {layer_idx},ubatch_idx is {ubatch_idx}',flush=True)
                # 如果切分，则更新cur_num_stages
                if is_ubatch:
                    cur_num_stages ^= 1
        return rank_ffn_output
  
        
                       
    def _capture_graph_for_layer_and_size(self, layer_idx: int,
                                          num_tokens: int,
                                          aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                                          **kargs):
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
                hidden_states, dynamic_scales, group_list, topk_weights, afdConnectorMetadata = \
                self._build_and_recv_m2n_afdconnector(
                    m2n_afdconnector_data=m2n_afdconnector_data,
                    quant_mode= 0,
                    expand_x_type = torch.bfloat16,
                    n_routed_experts=self.n_routed_experts,
                    hidden_size=self.hidden_size,
                    topk=self.topk,
                    expert_token_nums_type=0,
                    attn_size=self.attn_size,
                    max_num_tokens=self.max_num_tokens,
                    )
            elif self.connector_name == "camm2nconnector":
                cam_afdconnector_data = CAMM2NAFDConnectorMetadata(
                    moe_expert_num = self.n_routed_experts,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = 48,
                    batch_size = self.max_num_tokens,
                    h = self.hidden_size,
                    k = self.topk
                )
                output1,afdConnectorMetadata = self.connector.recv_attn_output(cam_afdconnector_data)
                hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize = output1[0:8]
                group_list = expertTokenNums.to(torch.int64)
                topk_weights = simulateExpertScales
            elif self.connector_name == "camp2pconnector":
                cam_afdconnector_data = CAMP2PAFDConnectorMetadata(
                    moe_expert_num = self.n_routed_experts,
                    shared_expert_num = 0,
                    scale = None,
                    handle = None,
                    quant_mode = 0,
                    aiv_num = 48,
                    batch_size = self.max_num_tokens,
                    h = self.hidden_size,
                    k = self.topk
                )
                a2eOutput, afdConnectorMetadata, cam_p2p_ep_name = self.connector.recv_attn_output(cam_afdconnector_data)
                hidden_states, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = a2eOutput[0:5]
                topk_weights = simulateExpertScales
                topk_ids = simulateExpertIds
                x_active_mask = xActiveMaskOut
            # compute_ffn_output ,未combine hidden
            if self.connector_name == "camp2pconnector":
                output = self._run_ffn_computation(hidden_states = hidden_states,
                                                layer_idx=layer_idx,
                                                capture_mode=True,
                                                topk_ids=topk_ids,
                                                topk_weights=topk_weights,
                                                x_active_mask=x_active_mask,
                                                cam_p2p_ep_name=cam_p2p_ep_name
                                               )
            else:
                output = self._run_ffn_computation(hidden_states = hidden_states,
                                                layer_idx=layer_idx,
                                                capture_mode=True,
                                                group_list=group_list,
                                                dynamic_scales=dynamic_scales,
                                                topk_weights=topk_weights
                                                )
            # send_ffn_output
            if self.connector_name == "camm2nconnector":
                handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
                cam_afdconnector_data.handle = handle
                self.connector.send_ffn_output(output, cam_afdconnector_data)
            elif self.connector_name == "m2nconnector":
                self.connector.send_ffn_output(output, m2n_afdconnector_data)
            elif self.connector_name == "camp2pconnector":
                handle = [attenBatchSize]
                cam_afdconnector_data.handle = handle
                self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data)
            else :
                self.connector.send_ffn_output(output, afdConnectorMetadata)
            return 

        # Create and capture the graph
        aclgraph = torch.npu.NPUGraph()

        # Start graph capture
        # recv_attn_output
        if self.connector_name == "m2nconnector":
            # TODO metadata
            m2n_afdconnector_data = M2NAFDConnectorMetadata()
            hidden_states, dynamic_scales, group_list, topk_weights, afdConnectorMetadata = \
            self._build_and_recv_m2n_afdconnector(
                m2n_afdconnector_data=m2n_afdconnector_data,
                quant_mode= 0,
                expand_x_type = torch.bfloat16,
                n_routed_experts=self.n_routed_experts,
                hidden_size=self.hidden_size,
                topk=self.topk,
                expert_token_nums_type=0,
                attn_size=self.attn_size,
                max_num_tokens=self.max_num_tokens,
                )
        elif self.connector_name == "camm2nconnector":
            cam_afdconnector_data = CAMM2NAFDConnectorMetadata(
                moe_expert_num = self.n_routed_experts,
                shared_expert_num = 0,
                scale = None,
                handle = None,
                quant_mode = 0,
                aiv_num = 48,
                batch_size = self.max_num_tokens,
                h = self.hidden_size,
                k = self.topk
            )
            output1,afdConnectorMetadata = self.connector.recv_attn_output(cam_afdconnector_data)
            hidden_states, dynamic_scales, expandIdx, expertTokenNums, epRecvCounts, simulateExpertIds, simulateExpertScales, attenBatchSize = output1[0:8]
            group_list = expertTokenNums.to(torch.int64)
            topk_weights = simulateExpertScales
        elif self.connector_name == "camp2pconnector":
            cam_afdconnector_data = CAMP2PAFDConnectorMetadata(
                moe_expert_num = self.n_routed_experts,
                shared_expert_num = 0,
                scale = None,
                handle = None,
                quant_mode = 0,
                aiv_num = 48,
                batch_size = self.max_num_tokens,
                h = self.hidden_size,
                k = self.topk
            )
            a2eOutput, afdConnectorMetadata, cam_p2p_ep_name = self.connector.recv_attn_output(cam_afdconnector_data)
            hidden_states, simulateExpertIds, simulateExpertScales, attenBatchSize, xActiveMaskOut = a2eOutput[0:5]
            topk_weights = simulateExpertScales
            topk_ids = simulateExpertIds
            x_active_mask = xActiveMaskOut
        with torch.npu.graph(aclgraph, pool=self.graph_pool):
            # compute_ffn_output
            if self.connector_name == "camp2pconnector":
                output = self._run_ffn_computation(hidden_states = hidden_states,
                                                    layer_idx=layer_idx,
                                                    capture_mode=True,
                                                    topk_ids=topk_ids,
                                                    topk_weights=topk_weights,
                                                    x_active_mask=x_active_mask,
                                                    cam_p2p_ep_name=cam_p2p_ep_name
                                                )
            else:
                output = self._run_ffn_computation(hidden_states = hidden_states,
                                                    layer_idx=layer_idx,
                                                    capture_mode=True,
                                                    group_list=group_list,
                                                    dynamic_scales=dynamic_scales,
                                                    topk_weights=topk_weights
                                                )
        # send_ffn_output 暂时不入图
        if self.connector_name == "camm2nconnector":
            handle = [simulateExpertIds, simulateExpertScales, expandIdx, epRecvCounts, attenBatchSize]
            cam_afdconnector_data.handle = handle
            self.connector.send_ffn_output(output, cam_afdconnector_data)
        elif self.connector_name == "m2nconnector":
            self.connector.send_ffn_output(output, m2n_afdconnector_data)
        elif self.connector_name == "camp2pconnector":
            handle = [attenBatchSize]
            cam_afdconnector_data.handle = handle
            self.connector.send_ffn_output(rank_ffn_output, cam_afdconnector_data)
        else :
            self.connector.send_ffn_output(output, afdConnectorMetadata)
        print(f'dummy_hidden_states shape is {dummy_hidden_states.shape}')
        print(f'output shape is {output.shape}')
        # Store the captured graph with layer and token count as key
        # dummy_hidden_states ->[4,2048];output ->[4,2048];
        self._acl_graphs[(layer_idx, output.shape[0])] = {
            'graph': aclgraph,
            'input_hidden_states': output,
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
                             row_idx: Optional[torch.Tensor] = None,
                             x_active_mask: Optional[torch.Tensor] = None,
                             cam_p2p_ep_name: Optional[str] = ""):
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
        elif self.connector_name == "camm2nconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx)
        elif self.connector_name == "camp2pconnector":
            rank_ffn_output = self.model.compute_ffn_output(
                layer_idx=current_layer_idx, 
                hidden_states=hidden_states,
                group_list=group_list,
                dynamic_scales=dynamic_scales,
                topk_weights=topk_weights, 
                topk_ids=topk_ids,
                row_idx=row_idx,
                x_active_mask=x_active_mask,
                cam_p2p_ep_name=cam_p2p_ep_name)
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
        for capture_size in self.aclgraph_batch_sizes:
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
