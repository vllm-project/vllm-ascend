import math
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import (
    AscendDeviceType,
    enable_sp,
    flashcomm2_enable,
    get_ascend_device_type,
    has_layer_idx,
    is_drafter_moe_model,
    is_moe_model,
    speculative_enable_dispatch_gmm_combine_decode,
)


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


_MRV2_IN_PROFILE_RUN: ContextVar[bool] = ContextVar("_MRV2_IN_PROFILE_RUN", default=False)


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
    input_ids=None,
    eplb_heat_collection_status: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
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
    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas

        forward_context.input_ids = input_ids

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        max_num_tokens = int(num_tokens_across_dp.max().item()) if num_tokens_across_dp is not None else num_tokens
        moe_comm_type = select_moe_comm_method(max_num_tokens, vllm_config, is_draft_model)

        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # TODO: remove it when fia merge in fiav2
        forward_context.sinks = has_sinks

        # TODO: remove it when torch_npu.npu_mm_reduce_scatter_base supports tp_size >= 16.
        mmrs_fusion = tp_world_size <= 8

        # set for sequence parallelism, 1000 is the batch size concurrency threshold
        # for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency
        # exceeds this threshold, the performance benefits can be maximized.
        # Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.

        # main model and drafter model may have different architecture
        is_context_moe_model = is_drafter_moe_model(vllm_config) if is_draft_model else is_moe_model(vllm_config)
        if is_context_moe_model:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        elif is_draft_model:
            # TODO: for dense drafter, `sp` is redundant and is not compatible with `dp` and `graph`.
            # Disable it to avoid more problems.
            flash_comm_v1_enabled = False
        else:
            flash_comm_v1_enabled = enable_sp(vllm_config) and num_tokens is not None and num_tokens > 1000
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.flash_comm_v1_enabled = flash_comm_v1_enabled
        # TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        forward_context.flashcomm_v2_enabled = flashcomm2_enable() and tp_world_size > 1 and num_tokens is not None

        forward_context.pad_size = 0
        if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
            pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

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
            if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp

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
# dispatch_gmm_combine_decode currently rejects batch sizes above this limit
# in its C++ tiling check (MAX_BATCH_SIZE).
_DISPATCH_GMM_COMBINE_DECODE_MAX_TOKENS = 256


def set_mc2_tokens_capacity(vllm_config, max_num_reqs, uniform_decode_query_len):
    global _mc2_tokens_capacity
    if _mc2_tokens_capacity is not None:
        return
    if get_ascend_config().enable_prefill_mc2:
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    elif vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        max_num_tokens = max_num_reqs * uniform_decode_query_len
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    # NOTE: To save memory, we cap the max number of tokens to 512.
    num_tokens_per_tp_rank = min(num_tokens_per_tp_rank, 512)
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


def _select_a2_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    mc2_tokens_capacity: int,
) -> MoECommType:
    num_experts = vllm_config.model_config.get_num_experts()
    ep_world_size = (
        vllm_config.parallel_config.world_size_across_dp // vllm_config.parallel_config.pipeline_parallel_size
    )
    num_experts_per_device = num_experts // ep_world_size
    if num_experts_per_device <= 24 and ep_world_size >= 16 and num_tokens <= mc2_tokens_capacity:
        return MoECommType.MC2
    return MoECommType.ALLGATHER


def _fits_mc2_capacity(num_tokens: int, mc2_tokens_capacity: int | None) -> bool:
    return mc2_tokens_capacity is not None and num_tokens <= mc2_tokens_capacity


def _can_use_dispatch_ffn_combine() -> bool:
    # Legacy dispatch_ffn_combine path. Keep its EP-size limit isolated from
    # dispatch_gmm_combine_decode and regular MC2.
    return get_ep_group().world_size <= 32


def _fits_dispatch_ffn_combine_capacity(num_tokens: int) -> bool:
    return num_tokens <= get_ascend_config().mega_moe_max_tokens


def _can_use_dispatch_gmm_combine_decode(
    vllm_config: VllmConfig,
    quant_type: str | None,
) -> bool:
    # Keep speculative/MTP quantization checks local to the mode 2 fused path.
    # TODO: drop this guard when dispatch_gmm_combine_decode supports w16a16.
    return speculative_enable_dispatch_gmm_combine_decode(vllm_config) and quant_type == "w8a8_dynamic"


def _fits_dispatch_gmm_combine_decode_capacity(num_tokens: int) -> bool:
    return num_tokens <= _DISPATCH_GMM_COMBINE_DECODE_MAX_TOKENS


def _select_a3_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    quant_type: str | None,
    mc2_tokens_capacity: int,
    enable_fused_mc2: int,
) -> MoECommType:
    if enable_fused_mc2 == 1 and _can_use_dispatch_ffn_combine() and _fits_dispatch_ffn_combine_capacity(num_tokens):
        return MoECommType.FUSED_MC2

    if (
        enable_fused_mc2 == 2
        and _can_use_dispatch_gmm_combine_decode(vllm_config, quant_type)
        and _fits_dispatch_gmm_combine_decode_capacity(num_tokens)
    ):
        return MoECommType.FUSED_MC2

    if _fits_mc2_capacity(num_tokens, mc2_tokens_capacity):
        return MoECommType.MC2

    return MoECommType.ALLTOALL


def _select_a5_moe_comm_method(
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
    if num_tokens <= mc2_tokens_capacity and world_size > 1:
        return MoECommType.MC2
    if world_size <= num_experts_per_tok:
        return MoECommType.ALLGATHER
    return MoECommType.ALLTOALL


def _is_effective_expert_parallel(vllm_config: VllmConfig) -> bool:
    return vllm_config.parallel_config.enable_expert_parallel and get_ep_group().world_size > 1


def _get_moe_quant_type(vllm_config: VllmConfig) -> str | None:
    hf_text_config = vllm_config.model_config.hf_text_config
    return getattr(
        hf_text_config,
        "moe_quantize",
        getattr(hf_text_config, "quantize", None),
    )


def _select_device_moe_comm_method(
    num_tokens: int,
    vllm_config: VllmConfig,
    soc_version: AscendDeviceType,
    quant_type: str | None,
    mc2_tokens_capacity: int,
) -> MoECommType:
    if soc_version == AscendDeviceType.A2:
        # A2 MC2 is limited by expert sharding, EP size, and regular MC2
        # token capacity; otherwise all-gather is the safe fallback.
        return _select_a2_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity)

    if soc_version == AscendDeviceType.A3:
        # A3 has multiple MC2-like paths. Keep fused operator capability and
        # capacity checks isolated inside the A3 selector.
        return _select_a3_moe_comm_method(
            num_tokens,
            vllm_config,
            quant_type,
            mc2_tokens_capacity,
            get_ascend_config().enable_fused_mc2,
        )

    if soc_version == AscendDeviceType.A5:
        # A5 prefers MC2 within regular capacity, then falls back according to
        # top-k and parallel world size.
        return _select_a5_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity)

    if soc_version == AscendDeviceType._310P:
        # 310P currently uses the all-gather MoE path.
        return MoECommType.ALLGATHER

    raise ValueError(f"Unsupported soc_version: {soc_version}")


def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> MoECommType | None:
    """Select the MoE communication method through a small public entry point.

    The public selector handles common gating and unified logging. Hardware-
    specific policy stays in internal helpers so new device generations can be
    added without growing this function.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Retained for caller compatibility. The
            selector does not branch on draft-model or prefill/decode stage.

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None

    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = _get_moe_quant_type(vllm_config)

    if not _is_effective_expert_parallel(vllm_config):
        moe_comm_type = MoECommType.ALLGATHER
    else:
        moe_comm_type = _select_device_moe_comm_method(
            num_tokens,
            vllm_config,
            soc_version,
            quant_type,
            mc2_tokens_capacity,
        )

    logger.debug(
        "MoE comm method selected: soc=%s, method=%s, num_tokens=%d, mc2_capacity=%s",
        soc_version,
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
        "mmrs_fusion",
        "num_tokens",
        "flash_comm_v1_enabled",
        "flashcomm_v2_enabled",
        "pad_size",
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
