from collections.abc import Callable
from typing import get_args

import torch
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.distributed import (
    get_dp_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe import GroupedTopk
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    aux_stream,
)

logger = init_logger(__name__)

from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoERouterImpl,
    determine_expert_placement_strategy,
    get_compressed_expert_map,
    maybe_roundup_hidden_size,
)


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    ### Replicating experts
    import os

    EXPERT_PARTITION_SPLIT = os.getenv("EXPERT_PARTITION_SPLIT", "")
    if EXPERT_PARTITION_SPLIT:
        EXPERT_PARTITION_SPLIT = int(EXPERT_PARTITION_SPLIT)

        LOCAL_NUM_EXPERTS = os.getenv("LOCAL_NUM_EXPERTS", "0")
        if LOCAL_NUM_EXPERTS:
            local_num_experts = int(LOCAL_NUM_EXPERTS)
        else:
            local_num_experts = global_num_experts // EXPERT_PARTITION_SPLIT

        # Determine which partition this rank represents
        representative_ep_rank = ep_rank % EXPERT_PARTITION_SPLIT

        # Create expert map
        expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)

        start = representative_ep_rank * local_num_experts
        end = start + local_num_experts

        print(f"{start=} {end=} {representative_ep_rank=} {local_num_experts=}")

        expert_map[start:end] = torch.arange(0, local_num_experts, dtype=torch.int32)

        expert_mask = None
        if return_expert_mask:
            expert_mask = torch.ones(
                (global_num_experts + num_fused_shared_experts + 1,),
                dtype=torch.int32,
            )
            expert_mask[-1] = 0
            expert_mask[:global_num_experts] = expert_map > -1

            if num_fused_shared_experts > 0:
                expert_map = torch.cat(
                    (
                        expert_map,
                        torch.tensor(
                            [local_num_experts + i for i in range(num_fused_shared_experts)],
                            dtype=torch.int32,
                        ),
                    ),
                    dim=0,
                )

        return (local_num_experts, expert_map, expert_mask)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(ep_rank, global_num_experts, ep_size, dtype=torch.int32)

        expert_map[local_log_experts] = torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones((global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32)
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def __init__(
    self,
    num_experts: int,  # Global number of experts
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    params_dtype: torch.dtype | None = None,
    reduce_results: bool = False,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    quant_config: QuantizationConfig | None = None,
    tp_size: int | None = None,
    ep_size: int | None = None,
    dp_size: int | None = None,
    pcp_size: int | None = None,
    prefix: str = "",
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    is_act_and_mul: bool = True,
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    is_sequence_parallel=False,
    expert_mapping: list[tuple[str, str, int, str]] | None = None,
    n_shared_experts: int | None = None,
    routing_method_type: RoutingMethodType | None = None,
    router_logits_dtype: torch.dtype | None = None,
):
    super(FusedMoE, self).__init__()
    import os

    num_experts = int(os.getenv("GLOBAL_NUM_EXPERTS", "0"))

    # Allow disabling of the separate shared experts stream for
    # debug purposes.
    # TODO: Remove this after more extensive testings with TP/DP
    # and other execution modes
    if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
        logger.debug_once("Disabling MoE shared_experts cuda stream", scope="local")
        self.shared_experts_stream = None
    else:
        # TODO(rob): enable shared expert overlap with non-cuda-alike.
        # aux_stream() returns None on non-cuda-alike platforms.
        self.shared_experts_stream = aux_stream()
        if self.shared_experts_stream is not None:
            logger.debug_once("Enabled separate cuda stream for MoE shared_experts", scope="local")

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()
    self.params_dtype = params_dtype

    vllm_config = get_current_vllm_config()
    self.vllm_config = vllm_config

    # FIXME (varun): We should have a better way of inferring the activation
    # datatype. This works for now as the tensor datatype entering the MoE
    # operation is typically unquantized (i.e. float16/bfloat16).
    if vllm_config.model_config is not None:
        moe_in_dtype = vllm_config.model_config.dtype
    else:
        # TODO (bnell): This is a hack to get test_mixtral_moe to work
        # since model_config is not set in the pytest test.
        moe_in_dtype = params_dtype

    tp_size_ = tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
    dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
    pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

    self.is_sequence_parallel = is_sequence_parallel
    self.sp_size = tp_size_ if is_sequence_parallel else 1

    self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
        tp_size_=tp_size_,
        pcp_size_=pcp_size_,
        dp_size_=dp_size_,
        vllm_parallel_config=vllm_config.parallel_config,
    )

    self.global_num_experts = num_experts + num_redundant_experts
    self.logical_num_experts = num_experts

    # Expert mapping used in self.load_weights
    self.expert_mapping = expert_mapping

    # Round up hidden size if needed.
    hidden_size = maybe_roundup_hidden_size(
        hidden_size,
        moe_in_dtype,
        quant_config,
        self.moe_parallel_config,
        is_lora_enabled=self.vllm_config.lora_config is not None,
    )

    # For smuggling this layer into the fused moe custom op
    compilation_config = vllm_config.compilation_config
    if prefix in compilation_config.static_forward_context:
        raise ValueError("Duplicate layer name: {}".format(prefix))
    compilation_config.static_forward_context[prefix] = self
    self.layer_name = prefix

    self.enable_eplb = enable_eplb
    self.expert_load_view: torch.Tensor | None = None
    self.logical_to_physical_map: torch.Tensor | None = None
    self.logical_replica_count: torch.Tensor | None = None
    self.expert_placement_strategy: ExpertPlacementStrategy = vllm_config.parallel_config.expert_placement_strategy

    # ROCm aiter shared experts fusion
    self.rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
    self.aiter_fmoe_shared_expert_enabled = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()

    self.num_fused_shared_experts = (
        n_shared_experts if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled else 0
    )
    if not self.aiter_fmoe_shared_expert_enabled and self.num_fused_shared_experts != 0:
        raise ValueError(
            "n_shared_experts is only supported on ROCm aiter when VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
        )

    # Determine expert maps
    if self.use_ep:
        if self.enable_eplb:
            assert self.global_num_experts % self.ep_size == 0, (
                "EPLB currently only supports even distribution of experts across ranks."
            )
        else:
            assert num_redundant_experts == 0, "Redundant experts are only supported with EPLB."

        self.expert_placement_strategy = determine_expert_placement_strategy(
            expert_placement_strategy=self.expert_placement_strategy,
            moe_parallel_config=self.moe_parallel_config,
            num_expert_group=num_expert_group,
            num_redundant_experts=num_redundant_experts,
            enable_eplb=self.enable_eplb,
        )

        self._expert_map: torch.Tensor | None
        local_num_experts, expert_map, expert_mask = determine_expert_map(
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            global_num_experts=self.global_num_experts,
            expert_placement_strategy=self.expert_placement_strategy,
            num_fused_shared_experts=self.num_fused_shared_experts,
            return_expert_mask=self.rocm_aiter_fmoe_enabled,
        )
        self.local_num_experts = local_num_experts
        self.register_buffer("_expert_map", expert_map)
        self.register_buffer("expert_mask", expert_mask)
        self._maybe_init_expert_routing_tables()
        logger.info_once(
            "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
            "placement strategy: %s. Local/global"
            " number of experts: %s/%s. Experts local to global index map:"
            " %s.",
            self.ep_rank,
            self.ep_size,
            self.expert_placement_strategy,
            self.local_num_experts,
            self.global_num_experts,
            get_compressed_expert_map(self._expert_map),
        )
    else:
        self.local_num_experts, self._expert_map, self.expert_mask = (
            self.global_num_experts,
            None,
            None,
        )

    self.top_k = top_k

    self._init_aiter_shared_experts_topK_buffer(vllm_config=vllm_config, dp_size=dp_size_)
    if self.use_ep and self.rocm_aiter_fmoe_enabled:
        assert self.expert_mask is None or torch.all((expert_mask == 0) | (expert_mask == 1)), (
            "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."
        )

    assert intermediate_size % self.tp_size == 0
    self.hidden_size = hidden_size
    self.intermediate_size_per_partition = intermediate_size // self.tp_size
    self.reduce_results = reduce_results
    self.renormalize = renormalize
    self.use_grouped_topk = use_grouped_topk
    if self.use_grouped_topk:
        assert num_expert_group is not None and topk_group is not None
    self.num_expert_group = num_expert_group
    self.topk_group = topk_group
    self.custom_routing_function = custom_routing_function
    self.scoring_func = scoring_func
    self.routed_scaling_factor = routed_scaling_factor
    self.e_score_correction_bias = e_score_correction_bias
    self.apply_router_weight_on_input = apply_router_weight_on_input
    self.activation = activation

    self._grouped_topk_impl: GroupedTopk | None = None
    if self.use_grouped_topk:
        assert self.num_expert_group is not None
        assert self.topk_group is not None
        self._grouped_topk_impl = GroupedTopk(
            topk=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

    if self.scoring_func != "softmax" and not self.use_grouped_topk:
        raise ValueError("Only softmax scoring function is supported for non-grouped topk.")

    # ToDo: Better logic to determine the routing method type
    if routing_method_type is not None:
        self.routing_method_type: RoutingMethodType = routing_method_type
    else:
        if scoring_func == "sigmoid":
            if self.use_grouped_topk:
                self.routing_method_type = RoutingMethodType.DeepSeekV3
            elif self.top_k == 1:
                self.routing_method_type = RoutingMethodType.Llama4
        elif self.scoring_func == "softmax":
            self.routing_method_type = (
                RoutingMethodType.Renormalize if not self.renormalize else RoutingMethodType.RenormalizeNaive
            )
        else:
            self.routing_method_type = RoutingMethodType.TopK

    self.moe_config: FusedMoEConfig = FusedMoEConfig(
        num_experts=self.global_num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        num_local_experts=self.local_num_experts,
        moe_parallel_config=self.moe_parallel_config,
        in_dtype=moe_in_dtype,
        router_logits_dtype=router_logits_dtype,
        max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
        has_bias=has_bias,
        is_act_and_mul=is_act_and_mul,
        is_lora_enabled=vllm_config.lora_config is not None,
    )
    self.moe_config_use_flashinfer_cutlass_kernels = self.moe_config.use_flashinfer_cutlass_kernels

    self.quant_config = quant_config

    def _get_quant_method() -> FusedMoEMethodBase:
        """
        Helper method to ensure self.quant_method is never None and
        of the proper type.
        """
        quant_method = None
        if self.quant_config is not None:
            quant_method = self.quant_config.get_quant_method(self, prefix)
        if quant_method is None:
            quant_method = UnquantizedFusedMoEMethod(self.moe_config)
        assert isinstance(quant_method, FusedMoEMethodBase)
        return quant_method

    # Note: get_quant_method will look at the layer's local_num_experts
    # for heuristic purposes, so it must be initialized first.
    self.quant_method: FusedMoEMethodBase = _get_quant_method()

    if not self.moe_config.is_act_and_mul:
        # Avoid circular import
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptFp8MoEMethod,
            ModelOptNvFp4FusedMoE,
        )

        if not isinstance(
            self.quant_method,
            (
                UnquantizedFusedMoEMethod,
                ModelOptFp8MoEMethod,
                ModelOptNvFp4FusedMoE,
            ),
        ):
            raise NotImplementedError(
                "is_act_and_mul=False is supported only for unquantized , ModelOpt FP8, and ModelOpt NvFp4 checkpoints"
            )
        if not current_platform.is_cuda():
            raise NotImplementedError("is_act_and_mul=False is supported only for CUDA for now")

    if self.enable_eplb and not self.quant_method.supports_eplb:
        # TODO: Add support for additional quantization methods.
        # The implementation for other quantization methods does not
        # contain essential differences, but the current quant API
        # design causes duplicated work when extending to new
        # quantization methods, so I'm leaving it for now.
        # If you plan to add support for more quantization methods,
        # please refer to the implementation in `Fp8MoEMethod`.
        raise NotImplementedError(
            f"EPLB is not supported {self.quant_method.__class__.__name__}. "
            "EPLB is only supported for FP8 quantization for now."
        )

    moe_quant_params = {
        "num_experts": self.local_num_experts,
        "hidden_size": hidden_size,
        "intermediate_size_per_partition": self.intermediate_size_per_partition,
        "params_dtype": params_dtype,
        "weight_loader": self.weight_loader,
        "global_num_experts": self.global_num_experts,
    }
    # need full intermediate size pre-sharding for WNA16 act order
    if self.quant_method.__class__.__name__ in (
        "GPTQMarlinMoEMethod",
        "CompressedTensorsWNA16MarlinMoEMethod",
        "CompressedTensorsWNA16MoEMethod",
    ):
        moe_quant_params["intermediate_size_full"] = intermediate_size

    self.quant_method.create_weights(layer=self, **moe_quant_params)

    # Chunked all2all staging tensor
    self.batched_hidden_states: torch.Tensor | None = None
    self.batched_router_logits: torch.Tensor | None = None

    self.router = FusedMoERouterImpl(self)
