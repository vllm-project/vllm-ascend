from vllm.config.scheduler import SchedulerConfig
from vllm.config.utils import config
from pydantic.dataclasses import dataclass
import vllm
import copy
from itertools import permutations
import argparse
from dataclasses import dataclass
from vllm.logger import init_logger
import vllm.envs as envs
import json
import sys
from vllm.config.observability import DetailedTraceModules
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    CompilationConfig,
    LoRAConfig,
    LoadConfig,
    ModelConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    StructuredOutputsConfig,
    MultiModalConfig,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    get_args,
)
from vllm.config.device import Device
from vllm.platforms import current_platform
from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized
from vllm.transformers_utils.config import (
    is_interleaved,
    maybe_override_with_speculators,
)
from vllm.transformers_utils.utils import is_cloud_storage
from vllm.utils.network_utils import get_ip
if TYPE_CHECKING:
    from vllm.usage.usage_lib import UsageContext
else:
    UsageContext = Any
from vllm.engine.arg_utils import get_kwargs, optional_type
logger = init_logger(__name__)

@config
@dataclass
class BalanceSchedulerConfig(SchedulerConfig):

    balance_scheduling: bool = True
    """EXPERIMENTAL: If set to True, perform balance scheduling. This may help
    increase output throughput and reduce TPOT in v1 sheduler. However, TTFT 
    may degrade in some scenarios. Furthermore, enabling this feature is not 
    recommended in scenarios where PD is separated.
    """

def create_engine_config(
    self,
    usage_context: UsageContext | None = None,
    headless: bool = False,
) -> VllmConfig:
    """
    Create the VllmConfig.

    NOTE: If VllmConfig is incompatible, we raise an error.
    """
    current_platform.pre_register_and_update()

    device_config = DeviceConfig(device=cast(Device, current_platform.device_type))

    # Check if the model is a speculator and override model/tokenizer/config
    # BEFORE creating ModelConfig, so the config is created with the target model
    # Skip speculator detection for cloud storage models (eg: S3, GCS) since
    # HuggingFace cannot load configs directly from S3 URLs. S3 models can still
    # use speculators with explicit --speculative-config.
    if not is_cloud_storage(self.model):
        (self.model, self.tokenizer, self.speculative_config) = (
            maybe_override_with_speculators(
                model=self.model,
                tokenizer=self.tokenizer,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                vllm_speculative_config=self.speculative_config,
            )
        )

    model_config = self.create_model_config()
    self.model = model_config.model
    self.tokenizer = model_config.tokenizer

    self._check_feature_supported(model_config)
    self._set_default_chunked_prefill_and_prefix_caching_args(model_config)
    self._set_default_max_num_seqs_and_batched_tokens_args(
        usage_context, model_config
    )

    sliding_window: int | None = None
    if not is_interleaved(model_config.hf_text_config):
        # Only set CacheConfig.sliding_window if the model is all sliding
        # window. Otherwise CacheConfig.sliding_window will override the
        # global layers in interleaved sliding window models.
        sliding_window = model_config.get_sliding_window()

    # Note(hc): In the current implementation of decode context
    # parallel(DCP), tp_size needs to be divisible by dcp_size,
    # because the world size does not change by dcp, it simply
    # reuses the GPUs of TP group, and split one TP group into
    # tp_size//dcp_size DCP groups.
    assert self.tensor_parallel_size % self.decode_context_parallel_size == 0, (
        f"tp_size={self.tensor_parallel_size} must be divisible by"
        f"dcp_size={self.decode_context_parallel_size}."
    )

    cache_config = CacheConfig(
        block_size=self.block_size,
        gpu_memory_utilization=self.gpu_memory_utilization,
        kv_cache_memory_bytes=self.kv_cache_memory_bytes,
        swap_space=self.swap_space,
        cache_dtype=self.kv_cache_dtype,
        is_attention_free=model_config.is_attention_free,
        num_gpu_blocks_override=self.num_gpu_blocks_override,
        sliding_window=sliding_window,
        enable_prefix_caching=self.enable_prefix_caching,
        prefix_caching_hash_algo=self.prefix_caching_hash_algo,
        cpu_offload_gb=self.cpu_offload_gb,
        calculate_kv_scales=self.calculate_kv_scales,
        kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,
        mamba_cache_dtype=self.mamba_cache_dtype,
        mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,
        mamba_block_size=self.mamba_block_size,
        kv_offloading_size=self.kv_offloading_size,
        kv_offloading_backend=self.kv_offloading_backend,
    )

    ray_runtime_env = None
    if is_ray_initialized():
        # Ray Serve LLM calls `create_engine_config` in the context
        # of a Ray task, therefore we check is_ray_initialized()
        # as opposed to is_in_ray_actor().
        import ray

        ray_runtime_env = ray.get_runtime_context().runtime_env
        # Avoid logging sensitive environment variables
        sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}
        if "env_vars" in sanitized_env:
            sanitized_env["env_vars"] = {
                k: "***" for k in sanitized_env["env_vars"]
            }
        logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)

    # Get the current placement group if Ray is initialized and
    # we are in a Ray actor. If so, then the placement group will be
    # passed to spawned processes.
    placement_group = None
    if is_in_ray_actor():
        import ray

        # This call initializes Ray automatically if it is not initialized,
        # but we should not do this here.
        placement_group = ray.util.get_current_placement_group()

    assert not headless or not self.data_parallel_hybrid_lb, (
        "data_parallel_hybrid_lb is not applicable in headless mode"
    )
    assert not (self.data_parallel_hybrid_lb and self.data_parallel_external_lb), (
        "data_parallel_hybrid_lb and data_parallel_external_lb cannot both be True."
    )
    assert self.data_parallel_backend == "mp" or self.nnodes == 1, (
        "nnodes > 1 is only supported with data_parallel_backend=mp"
    )
    inferred_data_parallel_rank = 0
    if self.nnodes > 1:
        world_size = (
            self.data_parallel_size
            * self.pipeline_parallel_size
            * self.tensor_parallel_size
        )
        world_size_within_dp = (
            self.pipeline_parallel_size * self.tensor_parallel_size
        )
        local_world_size = world_size // self.nnodes
        assert world_size % self.nnodes == 0, (
            f"world_size={world_size} must be divisible by nnodes={self.nnodes}."
        )
        assert self.node_rank < self.nnodes, (
            f"node_rank={self.node_rank} must be less than nnodes={self.nnodes}."
        )
        inferred_data_parallel_rank = (
            self.node_rank * local_world_size
        ) // world_size_within_dp
        if self.data_parallel_size > 1 and self.data_parallel_external_lb:
            self.data_parallel_rank = inferred_data_parallel_rank
            logger.info(
                "Inferred data_parallel_rank %d from node_rank %d for external lb",
                self.data_parallel_rank,
                self.node_rank,
            )
        elif self.data_parallel_size_local is None:
            # Infer data parallel size local for internal dplb:
            self.data_parallel_size_local = max(
                local_world_size // world_size_within_dp, 1
            )
    data_parallel_external_lb = (
        self.data_parallel_external_lb or self.data_parallel_rank is not None
    )
    # Local DP rank = 1, use pure-external LB.
    if data_parallel_external_lb:
        assert self.data_parallel_rank is not None, (
            "data_parallel_rank or node_rank must be specified if "
            "data_parallel_external_lb is enable."
        )
        assert self.data_parallel_size_local in (1, None), (
            "data_parallel_size_local must be 1 or None when data_parallel_rank "
            "is set"
        )
        data_parallel_size_local = 1
        # Use full external lb if we have local_size of 1.
        self.data_parallel_hybrid_lb = False
    elif self.data_parallel_size_local is not None:
        data_parallel_size_local = self.data_parallel_size_local

        if self.data_parallel_start_rank and not headless:
            # Infer hybrid LB mode.
            self.data_parallel_hybrid_lb = True

        if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
            # Use full external lb if we have local_size of 1.
            logger.warning(
                "data_parallel_hybrid_lb is not eligible when "
                "data_parallel_size_local = 1, autoswitch to "
                "data_parallel_external_lb."
            )
            data_parallel_external_lb = True
            self.data_parallel_hybrid_lb = False

        if data_parallel_size_local == self.data_parallel_size:
            # Disable hybrid LB mode if set for a single node
            self.data_parallel_hybrid_lb = False

        self.data_parallel_rank = (
            self.data_parallel_start_rank or inferred_data_parallel_rank
        )
        if self.nnodes > 1:
            logger.info(
                "Inferred data_parallel_rank %d from node_rank %d",
                self.data_parallel_rank,
                self.node_rank,
            )
    else:
        assert not self.data_parallel_hybrid_lb, (
            "data_parallel_size_local must be set to use data_parallel_hybrid_lb."
        )

        if self.data_parallel_backend == "ray" and (
            envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
        ):
            # Data parallel size defaults to 1 if DP ranks are spanning
            # multiple nodes
            data_parallel_size_local = 1
        else:
            # Otherwise local DP size defaults to global DP size if not set
            data_parallel_size_local = self.data_parallel_size

    # DP address, used in multi-node case for torch distributed group
    # and ZMQ sockets.
    if self.data_parallel_address is None:
        if self.data_parallel_backend == "ray":
            host_ip = get_ip()
            logger.info(
                "Using host IP %s as ray-based data parallel address", host_ip
            )
            data_parallel_address = host_ip
        else:
            assert self.data_parallel_backend == "mp", (
                "data_parallel_backend can only be ray or mp, got %s",
                self.data_parallel_backend,
            )
            data_parallel_address = (
                self.master_addr or ParallelConfig.data_parallel_master_ip
            )
    else:
        data_parallel_address = self.data_parallel_address

    # This port is only used when there are remote data parallel engines,
    # otherwise the local IPC transport is used.
    data_parallel_rpc_port = (
        self.data_parallel_rpc_port
        if (self.data_parallel_rpc_port is not None)
        else ParallelConfig.data_parallel_rpc_port
    )

    if self.tokens_only and not model_config.skip_tokenizer_init:
        model_config.skip_tokenizer_init = True
        logger.info("Skipping tokenizer initialization for tokens-only mode.")

    if self.async_scheduling and not self.disable_nccl_for_dp_synchronization:
        logger.info(
            "Disabling NCCL for DP synchronization when using async scheduling."
        )
        self.disable_nccl_for_dp_synchronization = True

    parallel_config = ParallelConfig(
        pipeline_parallel_size=self.pipeline_parallel_size,
        tensor_parallel_size=self.tensor_parallel_size,
        prefill_context_parallel_size=self.prefill_context_parallel_size,
        data_parallel_size=self.data_parallel_size,
        data_parallel_rank=self.data_parallel_rank or 0,
        data_parallel_external_lb=data_parallel_external_lb,
        data_parallel_size_local=data_parallel_size_local,
        master_addr=self.master_addr,
        master_port=self.master_port,
        nnodes=self.nnodes,
        node_rank=self.node_rank,
        data_parallel_master_ip=data_parallel_address,
        data_parallel_rpc_port=data_parallel_rpc_port,
        data_parallel_backend=self.data_parallel_backend,
        data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,
        enable_expert_parallel=self.enable_expert_parallel,
        all2all_backend=self.all2all_backend,
        enable_dbo=self.enable_dbo,
        dbo_decode_token_threshold=self.dbo_decode_token_threshold,
        dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,
        disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,
        enable_eplb=self.enable_eplb,
        eplb_config=self.eplb_config,
        expert_placement_strategy=self.expert_placement_strategy,
        max_parallel_loading_workers=self.max_parallel_loading_workers,
        disable_custom_all_reduce=self.disable_custom_all_reduce,
        ray_workers_use_nsight=self.ray_workers_use_nsight,
        ray_runtime_env=ray_runtime_env,
        placement_group=placement_group,
        distributed_executor_backend=self.distributed_executor_backend,
        worker_cls=self.worker_cls,
        worker_extension_cls=self.worker_extension_cls,
        decode_context_parallel_size=self.decode_context_parallel_size,
        dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,
        cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        _api_process_count=self._api_process_count,
        _api_process_rank=self._api_process_rank,
    )

    speculative_config = self.create_speculative_config(
        target_model_config=model_config,
        target_parallel_config=parallel_config,
    )

    scheduler_config = BalanceSchedulerConfig(
        runner_type=model_config.runner_type,
        max_num_batched_tokens=self.max_num_batched_tokens,
        max_num_seqs=self.max_num_seqs,
        max_model_len=model_config.max_model_len,
        enable_chunked_prefill=self.enable_chunked_prefill,
        disable_chunked_mm_input=self.disable_chunked_mm_input,
        is_multimodal_model=model_config.is_multimodal_model,
        is_encoder_decoder=model_config.is_encoder_decoder,
        policy=self.scheduling_policy,
        scheduler_cls=self.scheduler_cls,
        max_num_partial_prefills=self.max_num_partial_prefills,
        max_long_partial_prefills=self.max_long_partial_prefills,
        long_prefill_token_threshold=self.long_prefill_token_threshold,
        disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
        async_scheduling=self.async_scheduling,
        stream_interval=self.stream_interval,
    )

    if not model_config.is_multimodal_model and self.default_mm_loras:
        raise ValueError(
            "Default modality-specific LoRA(s) were provided for a "
            "non multimodal model"
        )

    lora_config = (
        LoRAConfig(
            max_lora_rank=self.max_lora_rank,
            max_loras=self.max_loras,
            default_mm_loras=self.default_mm_loras,
            fully_sharded_loras=self.fully_sharded_loras,
            lora_dtype=self.lora_dtype,
            max_cpu_loras=self.max_cpu_loras
            if self.max_cpu_loras and self.max_cpu_loras > 0
            else None,
        )
        if self.enable_lora
        else None
    )

    if (
        lora_config is not None
        and speculative_config is not None
        and scheduler_config.max_num_batched_tokens
        < (
            scheduler_config.max_num_seqs
            * (speculative_config.num_speculative_tokens + 1)
        )
    ):
        raise ValueError(
            "Consider increasing max_num_batched_tokens or "
            "decreasing num_speculative_tokens"
        )

    # bitsandbytes pre-quantized model need a specific model loader
    if model_config.quantization == "bitsandbytes":
        self.quantization = self.load_format = "bitsandbytes"

    load_config = self.create_load_config()

    # Pass reasoning_parser into StructuredOutputsConfig
    if self.reasoning_parser:
        self.structured_outputs_config.reasoning_parser = self.reasoning_parser

    if self.reasoning_parser_plugin:
        self.structured_outputs_config.reasoning_parser_plugin = (
            self.reasoning_parser_plugin
        )

    observability_config = ObservabilityConfig(
        show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
        otlp_traces_endpoint=self.otlp_traces_endpoint,
        collect_detailed_traces=self.collect_detailed_traces,
        kv_cache_metrics=self.kv_cache_metrics,
        kv_cache_metrics_sample=self.kv_cache_metrics_sample,
    )

    # Compilation config overrides
    compilation_config = copy.deepcopy(self.compilation_config)
    if self.cuda_graph_sizes is not None:
        logger.warning(
            "--cuda-graph-sizes is deprecated and will be removed in v0.13.0 or "
            "v1.0.0, whichever is soonest. Please use --cudagraph-capture-sizes "
            "instead."
        )
        if compilation_config.cudagraph_capture_sizes is not None:
            raise ValueError(
                "cuda_graph_sizes and compilation_config."
                "cudagraph_capture_sizes are mutually exclusive"
            )
        compilation_config.cudagraph_capture_sizes = self.cuda_graph_sizes
    if self.cudagraph_capture_sizes is not None:
        if compilation_config.cudagraph_capture_sizes is not None:
            raise ValueError(
                "cudagraph_capture_sizes and compilation_config."
                "cudagraph_capture_sizes are mutually exclusive"
            )
        compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes
    if self.max_cudagraph_capture_size is not None:
        if compilation_config.max_cudagraph_capture_size is not None:
            raise ValueError(
                "max_cudagraph_capture_size and compilation_config."
                "max_cudagraph_capture_size are mutually exclusive"
            )
        compilation_config.max_cudagraph_capture_size = (
            self.max_cudagraph_capture_size
        )
    config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        lora_config=lora_config,
        speculative_config=speculative_config,
        load_config=load_config,
        structured_outputs_config=self.structured_outputs_config,
        observability_config=observability_config,
        compilation_config=compilation_config,
        kv_transfer_config=self.kv_transfer_config,
        kv_events_config=self.kv_events_config,
        ec_transfer_config=self.ec_transfer_config,
        additional_config=self.additional_config,
        optimization_level=self.optimization_level,
    )

    return config


vllm.vllm.engine.arg_utils.EngineArgs.create_engine_config = create_engine_config
