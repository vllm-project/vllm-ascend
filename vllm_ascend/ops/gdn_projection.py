# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Optional overlap of independent GDN input projections during ACL capture."""

import torch
from vllm.config import CompilationMode, CUDAGraphMode, VllmConfig
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod


def initialize_gdn_projection_overlap(layer, config: VllmConfig) -> None:
    limit = get_ascend_config().gdn_projection_overlap_max_tokens
    compilation_config = config.compilation_config
    # LoRA can replace these modules, and quantized GEMMs may share scratch
    # storage. Do not overlap either projection unless both are plain GEMMs.
    if (
        not limit
        or compilation_config.mode != CompilationMode.NONE
        or compilation_config.cudagraph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY)
        or hasattr(layer, "in_proj_qkv")
        or config.lora_config is not None
        or config.parallel_config.use_ubatching
        or config.parallel_config.prefill_context_parallel_size != 1
        or config.model_config.dtype != torch.bfloat16
    ):
        return
    for projection in (layer.in_proj_qkvz, layer.in_proj_ba):
        if (
            type(projection.quant_method) not in (UnquantizedLinearMethod, AscendUnquantizedLinearMethod)
            or getattr(projection, "gather_output", False)
            or getattr(projection, "custom_op", None) is not None
        ):
            return
    # Model-owned, device-local sharing avoids one extra stream per GDN layer.
    # Events stay per layer so each captured fork/join has distinct ownership.
    device = torch.npu.current_device()
    for candidate in compilation_config.static_forward_context.values():
        if getattr(candidate, "_ascend_ba_device", None) == device:
            layer._ascend_ba_stream = candidate._ascend_ba_stream
            break
    else:
        layer._ascend_ba_stream = torch.npu.Stream()
    layer._ascend_ba_device = device
    layer._ascend_ba_max_tokens = limit
    layer._ascend_ba_events = (torch.npu.Event(), torch.npu.Event())


def project_gdn_inputs(layer, hidden_states):
    stream = getattr(layer, "_ascend_ba_stream", None)
    if (
        stream is None
        or torch.compiler.is_compiling()
        or hidden_states.dtype != torch.bfloat16
        or not 0 < hidden_states.shape[0] <= layer._ascend_ba_max_tokens
        or not torch.npu.is_current_stream_capturing()
    ):
        return layer.in_proj_qkvz(hidden_states)[0], layer.in_proj_ba(hidden_states)[0]

    main_stream = torch.npu.current_stream()
    ready, done = layer._ascend_ba_events
    ready.record(main_stream)
    hidden_states.record_stream(stream)
    with torch.npu.stream(stream):
        stream.wait_event(ready)
        ba = layer.in_proj_ba(hidden_states)[0]
        done.record(stream)
    qkvz = layer.in_proj_qkvz(hidden_states)[0]
    main_stream.wait_event(done)
    # BA was allocated on the side stream and is consumed on the main stream.
    ba.record_stream(main_stream)
    return qkvz, ba
