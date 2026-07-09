# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch_npu
import torch.nn as nn
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.distributed.afd_transfer.afd_connector.factory import (
    AFDConnectorFactory)
from vllm.distributed.parallel_state import get_world_group
from vllm.forward_context import AFDMetadata
from vllm.logger import logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.platforms import current_platform
from vllm.v1.worker.gpu_ffn_model_runner import GPUFFNModelRunner
import vllm.envs as envs

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.worker.model_runner_v1 import (
    NPUModelRunner, _replace_gpu_model_runner_function_wrapper,
    _torch_cuda_wrapper, graph_capture)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec



class NPUFFNModelRunner(NPUModelRunner, GPUFFNModelRunner):
    """NPU FFN model runner for AFD (Attention-FFN Disaggregation).

    Multiple inheritance combines NPUModelRunner (Ascend attention / NPU
    graph support) with GPUFFNModelRunner (AFD FFN server loop). The MRO
    ensures GPUModelRunner.__init__ runs first (setting up scheduler /
    parallel state), then GPUFFNModelRunner.__init__ sets up the AFD
    connector, and finally NPUFFNModelRunner adds aclgraph / multistream
    state on top.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config=vllm_config, device=device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.load_config = vllm_config.load_config
        self.first_k_dense_replace = getattr(
            self.model_config.hf_config, "first_k_dense_replace", 0)
        self.num_hidden_layers = self.model_config.hf_config.num_hidden_layers

        self.afd_config = vllm_config.afd_config
        if not self.afd_config or not self.afd_config.is_ffn_server:
            raise ValueError(
                "AFD config must be provided with afd_role='ffn' for FFN server"
            )
        self.connector_name = self.afd_config.afd_connector

        # Initialize ACL graph support
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # Storage for captured graphs - keyed by dp_metadata_key.
        # Key format: ((stage_idx, tuple(num_tokens_across_dp_cpu)), ...)
        self._acl_graphs: dict[tuple, dict] = {}
        self.graph_pool = None
        if self.use_aclgraph:
            self.graph_pool = current_platform.get_global_graph_pool()

        assert self.afd_config.is_ffn_server
        self.connector = AFDConnectorFactory.create_connector(
            get_world_group().rank,
            get_world_group().local_rank, self.vllm_config)

        self.connector.init_afd_connector()
        self.attn_size = self.connector.attn_size
        self.ffn_size = self.connector.ffn_size

        self.ffn_multistream_capable = self.afd_config.is_ffn_multistream
        num_ubatches_cfg = self.parallel_config.num_ubatches if self.parallel_config.num_ubatches else 1
        self.ffn_comm_stream = torch.npu.Stream() if self.ffn_multistream_capable else None
        self.ffn_comm_events = [torch.npu.Event() for _ in range(num_ubatches_cfg)] if self.ffn_multistream_capable else []
        logger.info("attn_size = %s, ffn_size = %s", self.attn_size,
                    self.ffn_size)
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
        logger.info("self.topk is %s", self.topk)
        self.decode_max_num_token = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len

        # Initialize cudagraph keys for FFN server as initialize_kv_cache
        # is not called.
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.vllm_config.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len
        )

        self.prof = None

    def get_model(self) -> nn.Module:
        return self.model

    def initialize_afd_connector(self) -> None:
        self.connector.init_afd_connector()

    @torch.inference_mode()
    def execute_model(self, scheduler_output=None, intermediate_tensors=None,
                     dp_metadata_list: dict | None = None):
        """Execute FFN computation for a single request

        Args:
            scheduler_output: 调度器输出（FFN侧通常为None）
            intermediate_tensors: 中间张量（FFN侧通常为None）
            dp_metadata_list: dp_metadata列表，包含每个stage的token数量信息
        """
        try:
            logger.info("execute_model pre, dp_metadata_list is %s", dp_metadata_list)
            if dp_metadata_list is None and self.connector is not None:
                dp_metadata_list, _, _ = (
                    self.connector.recv_dp_metadata_list()
                )
            is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1

            if self.use_aclgraph:
                logger.info("execute_model aclgraph pre")
                # Look up the captured graph by dp_metadata_key.
                dp_metadata_key = self._get_dp_metadata_key(dp_metadata_list)
                acl_graph_info = self._acl_graphs.get(dp_metadata_key)
                if acl_graph_info is not None:
                    graph = acl_graph_info['graph']
                    graph.replay()
                    self.replay_cnt += 1
                    logger.info(
                        "ffn replay, replay_cnt is %s, dp_metadata_key=%s",
                        self.replay_cnt, dp_metadata_key)
                else:
                    # Fallback to eager mode when no matching graph found.
                    logger.warning(
                        "No acl graph found for dp_metadata_key=%s, "
                        "fallback to eager", dp_metadata_key)
                    self._ffn_forward(
                        aclgraph_runtime_mode=CUDAGraphMode.NONE,
                        dp_metadata_list=dp_metadata_list)
            else:
                # Eager mode for non-ubatch or no aclgraph.
                logger.info("execute_model eager pre")
                self._ffn_forward(
                    aclgraph_runtime_mode=CUDAGraphMode.NONE,
                    dp_metadata_list=dp_metadata_list)

        except Exception as e:
            raise ValueError(
                f"Error computing FFN: {e}"
            ) from e
        return None  # FFN server doesn't return ModelRunnerOutput

    def capture_model(self,
                      dp_metadata_list: Optional[dict] = None,
                      is_warmup: bool = False,
                      is_attn_graph_capturing: bool = True) -> int:
        """Capture ACL graphs for FFN operations.

        Args:
            dp_metadata_list: 从Attention侧接收的dp_metadata列表
            is_warmup: 是否为warmup模式（只执行forward，不capture graph）
            is_attn_graph_capturing: Attention侧是否正在capture（用于同步）
        """
        if not self.use_aclgraph:
            return 0
        logger.debug("Starting ACL graph capture for FFN operations, "
                     "is_warmup=%s", is_warmup)
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        set_cudagraph_capturing_enabled(True)
        if is_warmup:
            # Warmup模式：只执行forward，不capture graph
            self._warmup_model(dp_metadata_list=dp_metadata_list)
            logger.info("FFN warmup completed, dp_metadata_list=%s", dp_metadata_list)
        else:
            # 正式Capture模式：根据dp_metadata_list捕获单个graph
            self._capture_model(dp_metadata_list=dp_metadata_list)
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

        return npu_graph_size

    def _get_dp_metadata_key(self, dp_metadata_list: dict) -> tuple:
        """Extract a hashable key from dp_metadata_list for CUDA graph lookup.

        Consistent with the GPU version's _make_graph_key.
        The key is a tuple of (stage_idx, tuple(num_tokens_across_dp_cpu))
        for each stage, sorted by stage_idx.

        Args:
            dp_metadata_list: {stage_idx: DPMetadata}

        Returns:
            tuple: ((stage_idx, tuple(num_tokens_across_dp_cpu)), ...)
        """
        if dp_metadata_list is None:
            return ()

        return tuple(
            (stage_idx, tuple(meta.num_tokens_across_dp_cpu.tolist()))
            for stage_idx, meta in sorted(dp_metadata_list.items())
        )

    def _warmup_model(self, dp_metadata_list: dict = None) -> None:
        """执行warmup，只运行forward不capture graph

        Args:
            is_ubatch: 是否为ubatch模式
            dp_metadata_list: 从Attention侧接收的dp_metadata列表
        """
        # Warmup只执行eager模式的forward，根据dp_metadata_list确定num_tokens
        dp_metadata_key = self._get_dp_metadata_key(dp_metadata_list)

        self._dummy_run(aclgraph_runtime_mode=CUDAGraphMode.NONE,
                        uniform_decode=True,
                        dp_metadata_list=dp_metadata_list,
                        dp_metadata_key=dp_metadata_key)
        logger.debug("FFN warmup for dp_metadata_key=%s", dp_metadata_key)

    def _capture_model(self, dp_metadata_list: dict = None):
        """Internal capture implementation - capture a single graph based on
        dp_metadata_list.

        Args:
            dp_metadata_list: dp_metadata list received from the attention side
        """
        if dp_metadata_list is None:
            logger.warning("dp_metadata_list is None, skip capture")
            return

        dp_metadata_key = self._get_dp_metadata_key(dp_metadata_list)

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        with freeze_gc(), graph_capture(device=self.device):
            # Capture a single graph directly (warmup is synchronized by the
            # attention side).
            self._capture_single_aclgraph(
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                uniform_decode=True,
                dp_metadata_key=dp_metadata_key,
                dp_metadata_list=dp_metadata_list
            )

    def _capture_single_aclgraph(self,
                                  cudagraph_runtime_mode: CUDAGraphMode,
                                  uniform_decode: bool,
                                  dp_metadata_key: tuple = None,
                                  dp_metadata_list: dict = None):
        """捕获单个 ACL graph（无 warmup 循环）

        Args:
            num_tokens: token数量
            cudagraph_runtime_mode: graph模式
            uniform_decode: 是否为uniform decode
            is_ubatch: 是否为ubatch模式
            dp_metadata_key: dp_metadata的key，用于存储graph
        """
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE
        logger.info("Capturing ACL graph for dp_metadata_key=%s",
                    dp_metadata_key)

        # 直接capture，不做warmup（warmup由Attention侧控制）
        self._dummy_run(aclgraph_runtime_mode=cudagraph_runtime_mode,
                        uniform_decode=uniform_decode,
                        dp_metadata_list=dp_metadata_list,
                        dp_metadata_key=dp_metadata_key)

    def _dummy_run(self,
                   aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                   uniform_decode: bool = False,
                   dp_metadata_list: dict | None = None,
                   dp_metadata_key: tuple = None,
                   **kwargs):
        """执行dummy run用于warmup或capture

        Args:
            num_tokens: token数量
            aclgraph_runtime_mode: ACL graph运行模式
            force_attention: 是否强制attention
            uniform_decode: 是否为uniform decode
            dp_metadata_list: dp_metadata列表，包含每个stage的token数量信息
            dp_metadata_key: dp_metadata的key，用于存储graph（替代num_tokens作为key）
        """

        is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1
        logger.info("_dummy_run pre")

        # Only support eager mode and piecewise graph now.
        assert aclgraph_runtime_mode is None or aclgraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }
        logger.info("_dummy_run aclgraph_runtime_mode is %s", aclgraph_runtime_mode)
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            # Create and capture the graph.
            aclgraph = torch.npu.NPUGraph()
            with torch.npu.graph(aclgraph, pool=self.graph_pool):
                # compute_ffn_output
                output = self._ffn_forward(
                                  aclgraph_runtime_mode=aclgraph_runtime_mode,
                                  dp_metadata_list=dp_metadata_list)
            # Store the captured graph with dp_metadata_key as key.
            self._acl_graphs[dp_metadata_key] = {
                'graph': aclgraph,
                'input_hidden_states': output,
                'output': output
            }
            logger.info("_dummy_run graph post, dp_metadata_key=%s", dp_metadata_key)
        else:
            self._ffn_forward(aclgraph_runtime_mode=aclgraph_runtime_mode,
                              dp_metadata_list=dp_metadata_list)
            logger.info("_dummy_run eager post")
        logger.info("self.dummy_run_call_cnt is %s", self.dummy_run_call_cnt)
        self.dummy_run_call_cnt += 1

    # TODO: to adapt m2nAFDConnector for deepseek w9a8 quantization.
    # NPUP2P does not use this method, but it is retained for compatibility
    # with the AFD FFN server loop that may call it for other connectors.
    def _build_and_recv_m2n_afdconnector(
        self,
        m2n_afdconnector_data: Any,
        n_routed_experts: int,
        hidden_size: int,
        topk: int,
        expert_token_nums_type: int,
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

        recv_output = self.connector.recv_attn_output(metadata=m2n_afdconnector_data)
        m2n_afdconnector_data.handle = recv_output.handle
        m2n_afdconnector_data.topk_weights = recv_output.topk_weights

        return recv_output

    def _build_ffn_num_tokens_across_dp(self, dp_metadata_list: dict) -> Optional[torch.Tensor]:
        """Build the num_tokens_across_dp tensor for the FFN side.

        For asymmetric A/F scenarios (A > F), multiple A token counts need to
        be merged into the corresponding F.
        """
        dp_metadata = dp_metadata_list.get(0, None) if dp_metadata_list else None

        if dp_metadata is None:
            return None

        attn_num_tokens = dp_metadata.num_tokens_across_dp_cpu

        # For asymmetric A/F, scale token counts by ratio because FFN
        # receives concatenated tokens from multiple A's.
        if hasattr(self.connector, 'ratio') and self.connector.ratio > 1:
            ffn_size = self.connector.ffn_size
            ratio = self.connector.ratio

            ffn_num_tokens_list = []
            for f_idx in range(ffn_size):
                start_a_idx = f_idx * ratio
                end_a_idx = start_a_idx + ratio
                ffn_num_tokens_list.append(
                    attn_num_tokens[start_a_idx:end_a_idx].sum().item()
                )

            ffn_num_tokens_across_dp_cpu = torch.tensor(
                ffn_num_tokens_list,
                dtype=attn_num_tokens.dtype,
                device=attn_num_tokens.device,
            )
            return ffn_num_tokens_across_dp_cpu
        else:
            return attn_num_tokens

    def _ffn_forward(self,
                     aclgraph_runtime_mode: Optional[CUDAGraphMode] = None,
                     dp_metadata_list: dict | None = None):
        """Run FFN computation for graph capture or replay."""
        is_ubatch = dp_metadata_list is not None and len(dp_metadata_list) > 1
        num_ubatches = self.parallel_config.num_ubatches if is_ubatch else 1
        rank_ffn_output = None
        logger.info("_ffn_forward pre max_num_tokens:%s", self.max_num_tokens)

        ffn_multistream_enable = self.ffn_multistream_capable and num_ubatches > 1

        afd_metadata = AFDMetadata(
            afd_tokens_start_loc=[],
            afd_reqs_start_loc=[],
            afd_stage_idx=0,
            afd_connector=self.connector,
            afd_tokens_lens=[],
            num_of_stages=num_ubatches
        )
        num_tokens_across_dp = self._build_ffn_num_tokens_across_dp(dp_metadata_list)
        ffn_event_recorded = [False] * num_ubatches
        with set_ascend_forward_context(
                    attn_metadata=None,
                    vllm_config=self.vllm_config,
                    batch_descriptor=None,
                    aclgraph_runtime_mode=aclgraph_runtime_mode,
                    model_instance=self.model,
                    afd_metadata=afd_metadata,
                    afd_comm_stream=self.ffn_comm_stream,
                    num_tokens=num_tokens_across_dp[self.parallel_config.data_parallel_rank].item() if num_tokens_across_dp is not None else 0,
                    num_tokens_across_dp=num_tokens_across_dp):
            for layer_idx in range(0, self.num_layers):
                layer_multistream = ffn_multistream_enable and (layer_idx > 0)
                for ubatch_idx in range(num_ubatches):
                    if ffn_multistream_enable and ffn_event_recorded[ubatch_idx]:
                        self.ffn_comm_events[ubatch_idx].wait(torch.npu.current_stream())
                    # recv (a2f): runs on default stream
                    afd_connector_data = self.connector.create_recv_metadata(
                        dp_metadata_list=dp_metadata_list,
                        ubatch_idx=ubatch_idx,
                        layer_idx=layer_idx,
                        max_num_tokens=self.max_num_tokens)
                    recv_output = self.connector.recv_attn_output(metadata=afd_connector_data, ubatch_idx=ubatch_idx)
                    if hasattr(self.connector, "update_metadata") and afd_connector_data is not None:
                        self.connector.update_metadata(afd_connector_data, recv_output)
                    logger.info('_ffn_forward recv_attn_output success, layer id is %s, recv_output:%s', layer_idx, recv_output.hidden_states.shape)

                    hidden_states = recv_output.hidden_states
                    dynamic_scales = recv_output.dynamic_scales
                    group_list = recv_output.group_list
                    topk_weights = recv_output.topk_weights
                    topk_ids = recv_output.topk_ids
                    router_logits = recv_output.router_logits
                    row_idx = recv_output.row_idx
                    x_active_mask = recv_output.x_active_mask

                    # FFN compute: runs on default stream
                    rank_ffn_output = self._run_ffn_computation(
                        hidden_states=hidden_states,
                        layer_idx=layer_idx,
                        group_list=group_list,
                        dynamic_scales=dynamic_scales if self.connector.quant_mode == 1 else None,
                        topk_weights=topk_weights,
                        topk_ids=topk_ids,
                        router_logits=router_logits,
                        row_idx=row_idx,
                        x_active_mask=x_active_mask,
                        cam_p2p_ep_name=recv_output.cam_p2p_ep_name or ""
                    )
                    # send (f2a): when multistream enabled, dispatched to comm_stream
                    self.connector.send_ffn_output(
                        rank_ffn_output, afd_connector_data,
                        ubatch_idx=ubatch_idx,
                        multistream_enable=layer_multistream,
                        comm_stream=self.ffn_comm_stream if layer_multistream else None,
                        comm_event=self.ffn_comm_events[ubatch_idx] if layer_multistream else None)
                    if layer_multistream:
                        ffn_event_recorded[ubatch_idx] = True
                    logger.info('_ffn_forward send_ffn_output success, layer id is %s', layer_idx)

            if ffn_multistream_enable:
                curr_stream = torch.npu.current_stream()
                for i, ev in enumerate(self.ffn_comm_events):
                    if ffn_event_recorded[i]:
                        ev.wait(curr_stream)
        return rank_ffn_output

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
        rank_ffn_output = self.model.compute_ffn_output(
            layer_idx=layer_idx,
            hidden_states=hidden_states,
            router_logits=router_logits,
            group_list=group_list,
            dynamic_scales=dynamic_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            row_idx=row_idx,
            x_active_mask=x_active_mask,
            cam_p2p_ep_name=cam_p2p_ep_name)
        return rank_ffn_output

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
