import math
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from typing import Any

import torch
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_dp_group, get_ep_group, get_tensor_model_parallel_world_size
from vllm.forward_context import BatchDescriptor, get_forward_context, set_forward_context

import vllm_ascend.envs as envs_ascend
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


@contextmanager
def set_ascend_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
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
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    forward_context_kwargs = {
        "attn_metadata": attn_metadata,
        "vllm_config": vllm_config,
        "virtual_engine": virtual_engine,
        "num_tokens": num_tokens,
        "num_tokens_across_dp": num_tokens_across_dp,
        "cudagraph_runtime_mode": aclgraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "skip_compiled": skip_compiled,
    }

    with set_forward_context(**forward_context_kwargs):
        forward_context = get_forward_context()
        forward_context.draft_attn_metadatas = draft_attn_metadatas

        from vllm_ascend.ops.fused_moe.moe_comm_method import get_moe_comm_method

        moe_comm_type = select_moe_comm_method(num_tokens, vllm_config, is_draft_model)
        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

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

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = forward_context.dp_metadata.max_tokens_across_dp_cpu.item()
            if forward_context.flash_comm_v1_enabled or forward_context.flashcomm_v2_enabled:
                padded_length = (max_tokens_across_dp + tp_world_size - 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp
        forward_context.max_tokens_across_pcp = max_tokens_across_pcp

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
    if vllm_config.compilation_config.cudagraph_capture_sizes:
        max_num_tokens = vllm_config.compilation_config.max_cudagraph_capture_size
    else:
        # NOTE: To save memory, we cap the max number of tokens to 512.
        max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    # Use integer arithmetic for ceiling division.
    num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
    _mc2_tokens_capacity = num_tokens_per_tp_rank * tp_size


def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


def set_mc2_mask(vllm_config, device):
    global _reserved_mc2_mask
    if _reserved_mc2_mask is not None:
        return
    if is_moe_model(vllm_config):
        _reserved_mc2_mask = torch.zeros(get_mc2_tokens_capacity(), dtype=torch.bool, device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> MoECommType | None:
    """Select the MoE communication method according to parallel settings,
    device generation, token count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2 with expert parallel, pick MC2 when tokens fit the MC2 capacity
       and the DP size is large enough; otherwise use all-gather.
    4. On A3 with expert parallel, prefer fused MC2 when using w8a8_dynamic
       quantization with small EP size, no dynamic_eplb, and not in MTP
       mode; otherwise use MC2 within capacity or all-to-all.
    5. On 310P, always use all-gather.

    Args:
        num_tokens (int): The number of tokens in the current batch.
        vllm_config (VllmConfig): Runtime configuration for the model.
        is_draft_model (bool): Whether the model runs in MTP mode (disables fused MC2).

    Raises:
        ValueError: If the soc version is unsupported.

    Returns:
        MoECommType | None: The selected MoE communication method.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = getattr(
        vllm_config.model_config.hf_text_config,
        "moe_quantize",
        getattr(vllm_config.model_config.hf_text_config, "quantize", None),
    )

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        if (
            num_tokens <= mc2_tokens_capacity
            and vllm_config.parallel_config.world_size_across_dp / vllm_config.parallel_config.pipeline_parallel_size
            >= 16
        ):
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLGATHER

    elif soc_version in {AscendDeviceType.A3}:
        # TODO: drop the EP-size guard when dispatch_ffn_combine supports larger EP sizes
        # TODO: drop speculative method guard when dispatch_gmm_combine_decode supports w16a16
        fused_mc2_enable = envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 and quant_type == "w8a8_dynamic"
        dispatch_ffn_combine_enable = get_ep_group().world_size <= 32 and (not is_draft_model)
        if num_tokens <= mc2_tokens_capacity:
            fused_decode_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_decode_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_decode_enable = fused_mc2_enable and speculative_enable_dispatch_gmm_combine_decode(vllm_config)
            moe_comm_type = MoECommType.FUSED_MC2 if fused_decode_enable else MoECommType.MC2
        else:
            fused_prefill_enable = fused_mc2_enable
            if envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 1:
                fused_prefill_enable = fused_mc2_enable and dispatch_ffn_combine_enable
            elif envs_ascend.VLLM_ASCEND_ENABLE_FUSED_MC2 == 2:
                fused_prefill_enable = False
            moe_comm_type = MoECommType.FUSED_MC2 if fused_prefill_enable else MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        if num_tokens <= mc2_tokens_capacity and vllm_config.parallel_config.world_size_across_dp > 1:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


class ExtraForwardContext:
    """Extra forward context for ascend.
    model runner v1 still have some additional forward context that can't
    be set by set_additional_forward_context,
    we can't unify the extra forward context usage for model runner v1 and v1 now.
    """

    @staticmethod
    def get_attr(name: str, default: Any = None):
        forward_context = get_forward_context()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            return forward_context.additional_kwargs.get(name, default)
        return getattr(forward_context, name, default)

    @staticmethod
    def set_attr(name: str, val: Any):
        forward_context = get_forward_context()
        if envs_vllm.VLLM_USE_V2_MODEL_RUNNER:
            forward_context.additional_kwargs[name] = val
        else:
            setattr(forward_context, name, val)

    @staticmethod
    def capturing() -> bool:
        """A flag to indicate whether to capture graph."""
        return ExtraForwardContext.get_attr("capturing", False)

    @staticmethod
    def set_capturing(val: bool):
        """Set the capturing flag."""
        ExtraForwardContext.set_attr("capturing", val)

    @staticmethod
    def moe_comm_type() -> MoECommType | None:
        """The MoE communication type."""
        return ExtraForwardContext.get_attr("moe_comm_type", None)

    @staticmethod
    def set_moe_comm_type(val: MoECommType | None):
        """Set the MoE communication type."""
        return ExtraForwardContext.set_attr("moe_comm_type", val)

    @staticmethod
    def moe_comm_method():
        """The MoE communication method."""
        return ExtraForwardContext.get_attr("moe_comm_method", None)

    @staticmethod
    def set_moe_comm_method(val):
        """Set the MoE communication method."""
        return ExtraForwardContext.set_attr("moe_comm_method", val)

    @staticmethod
    def mmrs_fusion() -> bool:
        """A flag to indicate whether to use mmrs fusion."""
        return ExtraForwardContext.get_attr("mmrs_fusion", False)

    @staticmethod
    def set_mmrs_fusion(val: bool):
        """Set the mmrs fusion flag."""
        return ExtraForwardContext.set_attr("mmrs_fusion", val)

    @staticmethod
    def num_tokens() -> int:
        """The number of tokens in the current batch."""
        return ExtraForwardContext.get_attr("num_tokens", 0)

    @staticmethod
    def set_num_tokens(val: int):
        """Set the number of tokens in the current batch."""
        return ExtraForwardContext.set_attr("num_tokens", val)

    @staticmethod
    def flash_comm_v1_enabled() -> bool:
        """A flag to indicate whether to use flashcomm v1."""
        return ExtraForwardContext.get_attr("flash_comm_v1_enabled", False)

    @staticmethod
    def set_flash_comm_v1_enabled(val: bool):
        """Set the flashcomm v1 enabled flag."""
        return ExtraForwardContext.set_attr("flash_comm_v1_enabled", val)

    @staticmethod
    def flashcomm_v2_enabled() -> bool:
        """A flag to indicate whether to use flashcomm v2."""
        return ExtraForwardContext.get_attr("flashcomm_v2_enabled", False)

    @staticmethod
    def set_flashcomm_v2_enabled(val: bool):
        """Set the flashcomm v2 enabled flag."""
        return ExtraForwardContext.set_attr("flashcomm_v2_enabled", val)

    @staticmethod
    def pad_size() -> int:
        """The pad size in the current batch."""
        return ExtraForwardContext.get_attr("pad_size", 0)

    @staticmethod
    def set_pad_size(val: int):
        """Set the pad size in the current batch."""
        return ExtraForwardContext.set_attr("pad_size", val)

    @staticmethod
    def padded_length() -> int:
        """The padded length in the current batch."""
        return ExtraForwardContext.get_attr("padded_length", 0)

    @staticmethod
    def set_padded_length(val: int):
        """Set the padded length in the current batch."""
        return ExtraForwardContext.set_attr("padded_length", val)

    @staticmethod
    def max_tokens_across_dp() -> int:
        """The max tokens across dp in the current batch."""
        return ExtraForwardContext.get_attr("max_tokens_across_dp", 0)

    @staticmethod
    def set_max_tokens_across_dp(val: int):
        """Set the max tokens across dp in the current batch."""
        return ExtraForwardContext.set_attr("max_tokens_across_dp", val)

    @staticmethod
    def mc2_mask() -> torch.Tensor | None:
        """The mc2 mask in the current batch."""
        return ExtraForwardContext.get_attr("mc2_mask", None)

    @staticmethod
    def set_mc2_mask(val: torch.Tensor | None):
        """Set the mc2 mask in the current batch."""
        return ExtraForwardContext.set_attr("mc2_mask", val)

    @staticmethod
    def is_draft_model() -> bool:
        """A flag to indicate whether to use draft model."""
        return ExtraForwardContext.get_attr("is_draft_model", False)

    @staticmethod
    def set_is_draft_model(val: bool):
        """Set the is_draft_model flag."""
        return ExtraForwardContext.set_attr("is_draft_model", val)

    @staticmethod
    def moe_layer_index() -> int | None:
        """The MoE layer index in the current batch."""
        return ExtraForwardContext.get_attr("moe_layer_index", None)

    @staticmethod
    def set_moe_layer_index(val: int | None):
        """Set the MoE layer index in the current batch."""
        return ExtraForwardContext.set_attr("moe_layer_index", val)

    @staticmethod
    def prefetch_mlp_gate_up_proj() -> bool:
        """A flag to indicate whether to prefetch mlp gate up proj."""
        return ExtraForwardContext.get_attr("prefetch_mlp_gate_up_proj", False)

    @staticmethod
    def set_prefetch_mlp_gate_up_proj(val: bool):
        """Set the prefetch mlp gate up proj flag."""
        return ExtraForwardContext.set_attr("prefetch_mlp_gate_up_proj", val)

    @staticmethod
    def prefetch_mlp_down_proj() -> bool:
        """A flag to indicate whether to prefetch mlp down proj."""
        return ExtraForwardContext.get_attr("prefetch_mlp_down_proj", False)

    @staticmethod
    def set_prefetch_mlp_down_proj(val: bool):
        """Set the prefetch mlp down proj flag."""
        return ExtraForwardContext.set_attr("prefetch_mlp_down_proj", val)

    @staticmethod
    def model_instance() -> torch.nn.Module | None:
        """The model instance."""
        return ExtraForwardContext.get_attr("model_instance", None)

    @staticmethod
    def set_model_instance(val: torch.nn.Module | None):
        """Set the model instance."""
        return ExtraForwardContext.set_attr("model_instance", val)

    @staticmethod
    def layer_idx() -> int | None:
        """The layer index in the current batch."""
        return ExtraForwardContext.get_attr("layer_idx", None)

    @staticmethod
    def set_layer_idx(val: int | None):
        """Set the layer index in the current batch."""
        return ExtraForwardContext.set_attr("layer_idx", val)

    @staticmethod
    def max_tokens_across_pcp() -> int:
        """The max tokens across pcp in the current batch."""
        return ExtraForwardContext.get_attr("max_tokens_across_pcp", 0)

    @staticmethod
    def set_max_tokens_across_pcp(val: int):
        """Set the max tokens across pcp in the current batch."""
        return ExtraForwardContext.set_attr("max_tokens_across_pcp", val)

    @staticmethod
    def num_accept_tokens() -> int:
        """The number of accept tokens in the current batch."""
        return ExtraForwardContext.get_attr("num_accept_tokens", 0)

    @staticmethod
    def set_num_accept_tokens(val: int):
        """Set the number of accept tokens in the current batch."""
        return ExtraForwardContext.set_attr("num_accept_tokens", val)

    @staticmethod
    def in_profile_run() -> bool:
        """A flag to indicate whether to run in profile mode."""
        return ExtraForwardContext.get_attr("in_profile_run", False)

    @staticmethod
    def set_in_profile_run(val: bool):
        """Set the in_profile_run flag."""
        return ExtraForwardContext.set_attr("in_profile_run", val)

    @staticmethod
    def padded_num_tokens() -> int:
        """The padded number of tokens in the current batch."""
        return ExtraForwardContext.get_attr("padded_num_tokens", 0)

    @staticmethod
    def set_padded_num_tokens(val: int):
        """Set the padded number of tokens in the current batch."""
        return ExtraForwardContext.set_attr("padded_num_tokens", val)
