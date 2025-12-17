import math
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import (get_dp_group, get_ep_group,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import (BatchDescriptor, get_forward_context,
                                  set_forward_context)

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import (AscendDeviceType, enable_sp, flashcomm2_enable,
                               get_ascend_device_type, has_layer_idx,
                               is_moe_model)

if TYPE_CHECKING:
    from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
else:
    WeightPrefetchMethod = None


class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_ALLTOALL = 3


@contextmanager
def set_ascend_forward_context(
        attn_metadata: Any,
        vllm_config: VllmConfig,
        virtual_engine: int = 0,
        num_tokens: int = 0,
        num_tokens_across_dp: Optional[torch.Tensor] = None,
        with_prefill: bool = True,
        in_profile_run: bool = False,
        num_actual_tokens: Optional[int] = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: Optional[BatchDescriptor] = None,
        prefetch_stream: torch.npu.Stream = None,
        model_instance: torch.nn.Module = None,
        weight_prefetch_method: Optional[WeightPrefetchMethod] = None,
        is_mtp_model=False):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    We add some additional param into forward_context.
    """
    with set_forward_context(
            attn_metadata,
            vllm_config,
            virtual_engine=virtual_engine,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=aclgraph_runtime_mode,
            batch_descriptor=batch_descriptor,
    ):
        forward_context = get_forward_context()

        from vllm_ascend.ops.fused_moe.moe_comm_method import \
            get_moe_comm_method
        moe_comm_type = select_moe_comm_method(num_tokens, vllm_config)
        # TODO: remove this after moe_comm_type selection logic is finalized
        if is_mtp_model:
            moe_comm_type = (MoECommType.ALLTOALL if moe_comm_type
                             == MoECommType.FUSED_ALLTOALL else moe_comm_type)
        forward_context.moe_comm_type = moe_comm_type
        forward_context.moe_comm_method = get_moe_comm_method(moe_comm_type)

        forward_context.with_prefill = with_prefill
        tp_world_size = get_tensor_model_parallel_world_size()

        forward_context.in_profile_run = in_profile_run

        # NOTE: This cannot be set using set_forward_context
        # due to multiple warmups before actual capturing
        forward_context.capturing = False

        # set for sequence parallelism, 1000 is the batch size concurrency threshold for enabling the flashcomm_v1 or sequence_parallelism feature.
        # Currently, it is an empirical value. In normal scenarios, if the concurrency exceeds this threshold,
        # the performance benefits can be maximized. Conversely, if the concurrency is below the threshold,
        # the performance may degrade due to the switching of communication methods.
        mmrs_fusion = True
        if is_moe_model(vllm_config):
            sp_enabled = enable_sp(vllm_config) and num_tokens is not None
            mmrs_fusion = False
        else:
            sp_enabled = enable_sp(vllm_config) and \
                num_tokens is not None and num_tokens > 1000
        forward_context.mmrs_fusion = mmrs_fusion
        forward_context.num_tokens = num_tokens
        forward_context.sp_enabled = sp_enabled
        #TODO(Levi-JQ): another PR to normalize the enabling logic for sp/fc2
        forward_context.flashcomm_v2_enabled = flashcomm2_enable(
        ) and tp_world_size > 1 and num_tokens is not None

        if (forward_context.sp_enabled
                or forward_context.flashcomm_v2_enabled):
            pad_size = (tp_world_size -
                        (num_tokens % tp_world_size)) % tp_world_size
            forward_context.pad_size = pad_size

        # set this for rope forward_oot using
        forward_context.is_first_layer = True

        # set layer_idx to enable optimization features that depend on this information.
        # This is only applicable to models that contain these necessary attributes.
        forward_context.layer_idx = None
        if has_layer_idx(model_instance):
            forward_context.layer_idx = model_instance.model.start_layer

        # TODO(rjg-lyh): refactor mlp weight prefetch method
        # set for mlp weight prefetch
        prefetch_mlp_enabled = envs_ascend.VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE and \
            envs_ascend.VLLM_ASCEND_ENABLE_PREFETCH_MLP and \
            forward_context.layer_idx is not None and \
            num_tokens is not None and num_tokens < 500
        if prefetch_mlp_enabled:
            forward_context.prefetch_stream = prefetch_stream
            forward_context.model_instance = model_instance
            forward_context.prefetch_mlp_gate_up_proj = False
            forward_context.prefetch_mlp_down_proj = False
        forward_context.prefetch_mlp_enabled = prefetch_mlp_enabled
        forward_context.model_instance = model_instance
        forward_context.weight_prefetch_method = weight_prefetch_method
        forward_context.is_mtp_model = is_mtp_model

        if num_tokens is None and attn_metadata is not None:
            num_tokens = attn_metadata.num_actual_tokens

        dp_world_size = get_dp_group().world_size
        if dp_world_size > 1 and forward_context.dp_metadata is not None:
            max_tokens_across_dp = \
                forward_context.dp_metadata.max_tokens_across_dp_cpu.item()
            if (forward_context.sp_enabled
                    or forward_context.flashcomm_v2_enabled):
                padded_length = (max_tokens_across_dp + tp_world_size -
                                 1) // tp_world_size * tp_world_size
                pad_size = padded_length - num_tokens
                forward_context.padded_length = padded_length
                forward_context.pad_size = pad_size
        else:
            max_tokens_across_dp = num_tokens

        forward_context.max_tokens_across_dp = max_tokens_across_dp

        if num_tokens is not None:
            if num_actual_tokens is None:
                num_actual_tokens = num_tokens
            # NOTE: token num which need to pad to when mc2
            forward_context.padded_num_tokens = math.ceil(
                max_tokens_across_dp / tp_world_size) * tp_world_size
            reserved_mc2_mask = get_mc2_mask()
            if reserved_mc2_mask is not None:
                mc2_mask = reserved_mc2_mask[:forward_context.
                                             padded_num_tokens]
                mc2_mask[:num_actual_tokens] = True
                mc2_mask[num_actual_tokens:] = False
                forward_context.mc2_mask = mc2_mask

        try:
            yield
        finally:
            pass


_mc2_tokens_capacity: Optional[int] = None
_reserved_mc2_mask: Optional[torch.Tensor] = None
_sin: Optional[torch.Tensor] = None
_cos: Optional[torch.Tensor] = None


def set_mc2_tokens_capacity(vllm_config, max_num_reqs,
                            uniform_decode_query_len):
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
        _reserved_mc2_mask = torch.zeros(get_mc2_tokens_capacity(),
                                         dtype=torch.bool,
                                         device=device)
    else:
        _reserved_mc2_mask = None


def get_mc2_mask():
    return _reserved_mc2_mask


def select_moe_comm_method(num_tokens: int,
                           vllm_config: VllmConfig) -> Optional[MoECommType]:
    """1. If expert parallel is not enabled, we use all-gather since MC2 and all-to-all
        are designed for expert parallelism.
        2. If expert parallel is enabled, we need to consider the soc version and the
        number of tokens. This is based on the observation that all-gather is more
        efficient than all-to-all when running on A2.

            a. For A2, we choose from MC2 and all-gather.

            b. For A3, we choose from MC2 and all-to-all.

            In both cases, we use MC2 when the number of tokens is smaller than
            a its capacity threshold.

        Args:
            num_tokens (int): The number of tokens in the current batch.

        Raises:
            ValueError: If the soc version is unsupported.

        Returns:
            MoECommType: The selected MoE communication method.
        """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    quant_type = getattr(
        vllm_config.model_config.hf_config, 'moe_quantize',
        getattr(vllm_config.model_config.hf_config, 'quantize', None))

    if not vllm_config.parallel_config.enable_expert_parallel:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        if (num_tokens <= mc2_tokens_capacity
                and vllm_config.parallel_config.world_size_across_dp /
                vllm_config.parallel_config.pipeline_parallel_size >= 16):
            moe_comm_type = MoECommType.MC2
        else:
            # Currently, w4a8_dynamic does not support allgatherep
            if quant_type == "w4a8_dynamic":
                moe_comm_type = MoECommType.ALLTOALL
            else:
                moe_comm_type = MoECommType.ALLGATHER

    elif soc_version in {AscendDeviceType.A3}:
        ascend_config = get_ascend_config()
        dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path
        # TODO: drop the EP-size guard when dispatch_ffn_combine supports larger EP sizes
        fused_all2all_enable = quant_type == "w8a8_dynamic" and get_ep_group(
        ).world_size <= 16 and (not dynamic_eplb)
        moe_comm_type = (MoECommType.MC2 if num_tokens <= mc2_tokens_capacity
                         else MoECommType.FUSED_ALLTOALL
                         if fused_all2all_enable else MoECommType.ALLTOALL)
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type
