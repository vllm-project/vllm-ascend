import math
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config, is_mega_moe_supported
from vllm_ascend.device.hardware_profile import (
    HardwareCapability,
    MoECommPolicy,
    get_current_hardware_profile,
)
from vllm_ascend.utils import (
    has_layer_idx,
    is_moe_model,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


_MRV2_IN_PROFILE_RUN: ContextVar[bool] = ContextVar("_MRV2_IN_PROFILE_RUN", default=False)


_MEGA_MOE_TOKENS_PER_RANK_LIMIT = 4096
_DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT = 512
_MC2_TOKENS_PER_RANK_LIMIT = 512


def _is_decode_only_node(vllm_config: VllmConfig) -> bool:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if kv_transfer_config is None:
        return False

    is_decode_bench = getattr(kv_transfer_config, "kv_connector", None) == "DecodeBenchConnector"
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    is_kv_consumer = (
        kv_role == "kv_consumer"
        if kv_role is not None
        else bool(
            getattr(kv_transfer_config, "is_kv_consumer", False)
            and not getattr(kv_transfer_config, "is_kv_producer", False)
        )
    )
    if not (is_decode_bench or is_kv_consumer):
        return False

    scheduler_config = getattr(get_ascend_config(), "scheduler_config", None)
    # Actual semantics of `recompute_scheduler_enable`:
    # - Enabled: when preemption occurs on the decode node, the request is sent back
    #     to the P node to redo prefill, so the decode node only ever decodes;
    # - Disabled: prefill is executed locally on the decode node.
    return bool(getattr(scheduler_config, "recompute_scheduler_enable", False))


@contextmanager
def override_mrv2_in_profile_run(enabled: bool):
    """Override MRv2's extra profile-run marker for one forward path.

    MRv2 builds the base forward context inside upstream vLLM, so Ascend's
    platform hook cannot tell whether the current forward is the extra MC2
    profile dummy run. A ContextVar keeps this MRv2-only state scoped to the
    current forward path without adding default fallback behavior.
    """
    token = _MRV2_IN_PROFILE_RUN.set(enabled)
    try:
        yield
    finally:
        _MRV2_IN_PROFILE_RUN.reset(token)


def get_mrv2_in_profile_run() -> bool:
    return _MRV2_IN_PROFILE_RUN.get()


def use_cann_megamoe(vllm_config: VllmConfig) -> bool:
    # TODO: drop the EP-size guard when MegaMoe supports larger EP sizes.
    return (
        is_mega_moe_supported()
        and get_current_hardware_profile().supports(HardwareCapability.CANN_MEGAMOE)
        and get_ascend_config().enable_fused_mc2 == 1
        and is_moe_model(vllm_config)
        and vllm_config.parallel_config.enable_expert_parallel
        and 1 < get_ep_group().world_size <= 32
        and getattr(vllm_config, "lora_config", None) is None
    )


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    num_tokens: int = 0,
    num_tokens_across_dp: torch.Tensor | None = None,
    in_profile_run: bool = False,
    num_actual_tokens: int | None = None,
    aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    model_instance: torch.nn.Module = None,
    is_draft_model=False,
    skip_compiled: bool = False,
    max_tokens_across_pcp: int = 0,
    draft_attn_metadatas=None,
    has_sinks=False,
    eplb_heat_collection_status: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.

    Also publish the process-global current vLLM config for this forward so
    CustomOps (RMSNorm, rotary, Linear, MoE) can call get_current_vllm_config()
    from ``__init__`` when they are first created during eager prefill.
    ``set_current_vllm_config`` is a context manager and restores ``None`` on
    exit, so wrapping only ``load_model`` is not enough; pin it here instead of
    in the Worker.
    """
    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }
    with set_current_vllm_config(vllm_config), set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
        moe_comm_type = select_moe_comm_method(
            max_num_tokens,
            vllm_config,
        )

        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)
        forward_context.is_decode_only_node = _is_decode_only_node(vllm_config)
        forward_context.use_mega_moe = use_cann_megamoe(vllm_config)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when fia merge in fiav2
        forward_context.sinks = has_sinks

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        forward_context.prefetch_mlp_gate_up_proj = False
        forward_context.prefetch_mlp_down_proj = False
        forward_context.model_instance = model_instance
        forward_context.is_draft_model = is_draft_model
        forward_context.is_draft_model_prefill = False

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            dp_meta = forward_context.dp_metadata
            max_tokens_across_dp = dp_meta.num_tokens_across_dp_cpu.max().item()
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp
        forward_context.padded_length = (
            math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            if max_tokens_across_dp is not None
            else None
        )

        forward_context.eplb_heat_collection_status = eplb_heat_collection_status

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[: forward_context.padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask
        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: int | None = None
_reserved_mc2_mask: torch.Tensor | None = None


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return

    ascend_config = get_ascend_config()
    use_mega_moe = use_cann_megamoe(vllm_config)

    # Cap for fused MC2 / MegaMoe: regular MC2 (gated by enable_prefill_mc2) uses
    # HCCL comm buffer (HCCL_BUFFSIZE); MegaMoe (use_mega_moe, non-decode-only)
    # uses the symm buffer (separate torch alloc, not HCCL_BUFFSIZE).
    if ascend_config.enable_prefill_mc2 or (use_mega_moe and not _is_decode_only_node(vllm_config)):
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
    tp_size = vllm_config.parallel_config.tensor_parallel_size

    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    # keep the num_tokens_per_tp_rank less than fused_mc2 (mega_moe) tokens per rank limit
    if ascend_config.enable_fused_mc2:
        if use_mega_moe:
            num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _MEGA_MOE_TOKENS_PER_RANK_LIMIT)
        else:
            num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT)

    # keep the num_tokens_per_tp_rank less than mc2 tokens per rank limit
    else:
        num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, _MC2_TOKENS_PER_RANK_LIMIT)
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens, dtype=torch.bool, device=device
        )
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def _select_capacity_and_expert_density_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
) -> MoECommType:
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = (
        vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
    )
    num_experts_per_device = num_experts // ep_world_size
    if (
        num_experts_per_device <= 24
        and ep_world_size >= 16
        and (num_tokens is None or num_tokens <= mc2_tokens_capacity)
    ):
        return MoECommType.MC2
    return MoECommType.ALLGATHER


def _mega_moe_worst_case_recv_tokens(num_tokens: int, vllm_config: VllmConfig) -> int:
    """Worst-case per-rank received token count for the current forward.

    Mirrors ``absolute_safe_max_recv_token_num`` in ``FusedMC2CommImpl`` but
    uses the actual forward's token count instead of the compile-time capacity.
    Collective-safe: mega_moe forces the DP all-reduce
    (``should_skip_allreduce_across_dp_group`` returns False when
    ``use_cann_megamoe`` is True), so ``num_tokens`` is uniform across all EP
    ranks.
    """
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    per_tp_rank = (num_tokens + tp_size - 1) // tp_size
    ep_world_size = get_ep_group().world_size
    hf = vllm_config.model_config.hf_text_config
    num_topk = getattr(hf, "num_experts_per_tok", getattr(hf, "top_k_experts", 1))
    num_experts = vllm_config.model_config.get_num_experts()
    expert_per_rank = max(1, num_experts // ep_world_size)
    return max(1, per_tp_rank * ep_world_size * min(num_topk, expert_per_rank))


def _select_mega_moe_or_all2all(num_tokens: int, vllm_config: VllmConfig) -> MoECommType:
    """Pick FUSED_MC2 (CANN mega_moe) or ALLTOALL fallback for the mega_moe path.

    On non-decode-only nodes the symm buffer is sized by ``mega_moe_max_tokens``
    (a reference value). If the worst-case per-rank received tokens, scaled by
    ``mega_moe_threshold_ratio``, exceed the buffer, fall back to all2all to
    avoid silent precision degradation. Decode-only nodes size the buffer to
    the absolute safe bound and are exempt.
    """
    if _is_decode_only_node(vllm_config):
        return MoECommType.FUSED_MC2
    cfg = get_ascend_config()
    worst_case = _mega_moe_worst_case_recv_tokens(num_tokens, vllm_config)
    if worst_case * cfg.mega_moe_threshold_ratio > cfg.mega_moe_max_tokens:
        logger.warning_once(
            "MegaMoe falls back to all2all: worst-case per-rank received "
            "tokens (%d) * ratio (%.2f) > mega_moe_max_tokens (%d). "
            "Raise mega_moe_max_tokens or lower mega_moe_threshold_ratio "
            "to keep using mega_moe.",
            worst_case,
            cfg.mega_moe_threshold_ratio,
            cfg.mega_moe_max_tokens,
        )
        return MoECommType.ALLTOALL
    return MoECommType.FUSED_MC2


def _hccl_buffsize_would_overflow(num_tokens: int, vllm_config: VllmConfig) -> bool:
    """Framework-side replica of the HCCL_BUFFSIZE check in
    ``dispatch_ffn_combine_tiling.cpp`` (``M*topK*K*3+10MB > HCCL_BUFFSIZE``).

    ``M`` is the per-TP-rank input token count the op actually sees: the MC2
    prepare pads the DP-uniform batch to ``ceil(max/tp)*tp`` then splits by TP,
    so ``M = ceil(num_tokens / tp_size)`` where ``num_tokens`` is the DP-max
    value passed to the selector. ``K`` is ``hidden_size``, ``topK`` is
    ``num_experts_per_tok``. Returns True when the dispatch_ffn_combine comm
    buffer would exceed ``HCCL_BUFFSIZE`` so the caller can fall back to
    all2all before the C++ op errors out.
    """
    if num_tokens is None:
        return False
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    m = (num_tokens + tp_size - 1) // tp_size
    hf = vllm_config.model_config.hf_text_config
    topk = getattr(hf, "num_experts_per_tok", getattr(hf, "top_k_experts", 1))
    k = hf.hidden_size
    mb = 1 << 20
    actual = m * topk * k * 3 + 10 * mb
    hccl_mb = envs.HCCL_BUFFSIZE
    return actual > hccl_mb * mb


def _select_fused_or_capacity_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
) -> MoECommType:
    if get_ascend_config().enable_fused_mc2 == 1 and get_ep_group().world_size <= 32:
        # dispatch_ffn_combine and mega_moe both select FUSED_MC2 here (both
        # have enable_fused_mc2 == 1 after the =2->=1 remap, both support
        # ep <= 32). They overflow different buffers, so the sub-branch picks
        # the matching guard; the execution distinction stays in
        # FusedMC2CommImpl.fused_experts via _EXTRA_CTX.use_mega_moe.
        if use_cann_megamoe(vllm_config):
            return _select_mega_moe_or_all2all(num_tokens, vllm_config)
        if _hccl_buffsize_would_overflow(num_tokens, vllm_config):
            logger.warning_once(
                "dispatch_ffn_combine falls back to all2all: the per-TP-rank "
                "comm buffer (M*topK*K*3+10MB, M=ceil(num_tokens/tp)) would "
                "exceed HCCL_BUFFSIZE(%dMB) for num_tokens=%d. "
                "Raise HCCL_BUFFSIZE to keep the fused path.",
                envs.HCCL_BUFFSIZE,
                num_tokens,
            )
            return MoECommType.ALLTOALL
        return MoECommType.FUSED_MC2

    if num_tokens is None or num_tokens <= mc2_tokens_capacity:
        return MoECommType.MC2

    return MoECommType.ALLTOALL


def _select_capacity_and_world_size_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
) -> MoECommType:
    num_experts_per_tok = getattr(
        vllm_config.model_config.hf_text_config,
        "num_experts_per_tok",
        getattr(vllm_config.model_config.hf_text_config, "top_k_experts", 1),
    )
    world_size = vllm_config.parallel_config.world_size_across_dp
    if (num_tokens is None or num_tokens <= mc2_tokens_capacity) and world_size > 1:
        return MoECommType.MC2
    if world_size <= num_experts_per_tok:
        return MoECommType.ALLGATHER
    return MoECommType.ALLTOALL


_MOE_COMM_SELECTORS = {
    MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY: _select_capacity_and_expert_density_moe_comm_method,
    MoECommPolicy.FUSED_OR_CAPACITY: _select_fused_or_capacity_moe_comm_method,
    MoECommPolicy.CAPACITY_AND_WORLD_SIZE: _select_capacity_and_world_size_moe_comm_method,
}


def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig) -> MoECommType | None:
    """Select the MoE communication method from the active hardware policy,
    parallel settings, and token count.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None

    mc2_tokens_capacity = get_mc2_tokens_capacity()
    moe_comm_policy = get_current_hardware_profile().moe_comm_policy
    lora_config = getattr(vllm_config, "lora_config", None)
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif lora_config is not None and vllm_config.parallel_config.enable_expert_parallel:
        # LoRA + EP requires AlltoAll because the MC2/FusedMC2 paths
        # Ascend MoE LoRA cannot patch FusedMC2 path for dispatch_ffn_combine/mega_moe
        # is a single fused C++ op. This covers both normal model
        # forward and _dummy_run during profile_run.
        moe_comm_type = MoECommType.ALLTOALL
    elif moe_comm_policy is MoECommPolicy.ALLGATHER:
        moe_comm_type = MoECommType.ALLGATHER
    else:
        moe_comm_type = _MOE_COMM_SELECTORS[moe_comm_policy](
            num_tokens,
            vllm_config,
            mc2_tokens_capacity,
        )
    logger.debug(
        "MoE comm method selected: policy=%s, method=%s, num_tokens=%d, mc2_capacity=%s",
        moe_comm_policy,
        moe_comm_type,
        num_tokens,
        mc2_tokens_capacity,
    )
    return moe_comm_type


class _ExtraForwardContextProxy:
    """Unified forward-context access for v1/v2 model runners."""

    extra_attrs = (
        "capturing",
        "moe_comm_type",
        "moe_comm_method",
        "is_decode_only_node",
        "use_mega_moe",
        "mmrs_fusion",
        "num_tokens",
        "padded_length",
        "num_tokens_across_dp",
        "mc2_mask",
        "is_draft_model",
        "is_draft_model_prefill",
        "prefetch_mlp_gate_up_proj",
        "prefetch_mlp_down_proj",
        "model_instance",
        "layer_idx",
        "max_tokens_across_dp",
        "max_tokens_across_pcp",
        "num_accept_tokens",
        "in_profile_run",
        "padded_num_tokens",
        "sinks",
        "eplb_heat_collection_status",
    )

    def check_extra_attr(self, name: str):
        if name not in self.extra_attrs:
            raise AttributeError(
                f"{name} is not extra forward context attribute, "
                "please get/set it from vllm's _forward_context directly."
            )

    @staticmethod
    def _ctx():
        return get_forward_context()

    def __getattr__(self, name: str) -> Any:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            # Unset known extras default to None so optional flags (e.g. `sinks`)
            # can be read with truthiness checks before the V2 path populates them.
            return ctx.additional_kwargs.get(name)
        return getattr(ctx, name, None)

    def __setattr__(self, name: str, value: Any) -> None:
        self.check_extra_attr(name)
        ctx = self._ctx()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            ctx.additional_kwargs[name] = value
        else:
            setattr(ctx, name, value)


# usage: from vllm_ascend.ascend_forward_context import _EXTRA_CTX
_EXTRA_CTX = _ExtraForwardContextProxy()
