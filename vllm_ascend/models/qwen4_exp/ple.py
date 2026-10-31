# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ascend runtime adaptation for Qwen4Exp PLE short convolution."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ops.ple import ple_conv as upstream_ple_conv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm_ascend.attention.utils import wait_for_kv_layer_from_connector
from vllm_ascend.ops.triton_ple import ple_conv_prefill


def _ple_conv_ascend(*args: Any, mode: Literal["decode", "spec", "prefill"], **kwargs: Any) -> None:
    if mode == "prefill":
        ple_conv_prefill(*args, **kwargs)
    else:
        upstream_ple_conv(*args, mode=mode, **kwargs)


def _get_allow_internal_format() -> bool:
    raw_value = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    # Ascend worker initialization enables this option before model construction. Preserve
    # the torch-npu default when the model is instantiated in isolation.
    return raw_value is None or raw_value.decode() == "enable"


def _set_allow_internal_format(enabled: bool) -> None:
    # Prefer the high-level config sugar where torch_npu provides it.
    config = getattr(torch.npu, "config", None)
    if config is not None and hasattr(config, "allow_internal_format"):
        config.allow_internal_format = enabled
        return
    # torch_npu 2.10 (910B) ships an empty torch.npu.config module: assigning
    # the attribute would succeed silently without toggling anything. Fall
    # back to the low-level option API, which is present there.
    set_option = getattr(torch_npu._C, "_npu_setOption", None)  # noqa: SLF001
    if set_option is not None:
        set_option({"ALLOW_INTERNAL_FORMAT": "enable" if enabled else "disable"})
        return
    logger.warning_once(
        "No ALLOW_INTERNAL_FORMAT setter available on this torch_npu version; "
        "the PLE short convolution follows the process-wide setting."
    )


_PLE_PREFILL_CONV_CHUNK_SIZE = 4


def _accumulate_compact_prefill_output(
    conv_output: torch.Tensor,
    output: torch.Tensor,
    req_indices: torch.Tensor,
    col_indices: torch.Tensor,
    valid_state: torch.Tensor,
    request_start: int,
) -> None:
    """Activate and accumulate real tokens from one prefill request chunk.

    Requests outside this chunk gather from a clamped row and are masked to
    zero.  Each token therefore contributes exactly once across all chunks.
    """
    F.silu(conv_output, inplace=True)
    local_req_indices = req_indices - request_start
    in_chunk = (local_req_indices >= 0) & (local_req_indices < conv_output.shape[0])
    safe_req_indices = local_req_indices.clamp(0, conv_output.shape[0] - 1)
    compact_output = conv_output[safe_req_indices, :, col_indices]
    valid_tokens = in_chunk & valid_state.to(device=output.device)[req_indices]
    output.add_(compact_output * valid_tokens.view(-1, 1).to(output.dtype))


def _short_conv_dilated_prefill_batched(
    self: Any,
    x_p: torch.Tensor,
    metadata: Any,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    state_indices_tensor_p: torch.Tensor,
    num_prefills: int,
    num_decode_tokens: int,
    num_prefill_tokens: int,
) -> torch.Tensor:
    """Ascend PLE prefill path with compact post-convolution gathering.

    Keep the upstream state/cache semantics, but bound peak memory by running
    the padded depthwise convolution over a few requests at a time.  Each
    chunk is activated in place and gathered directly into the compact output.
    """
    query_start_loc_p = getattr(metadata, "query_start_loc_p", None)
    if query_start_loc_p is None:
        non_spec_query_start_loc = getattr(metadata, "non_spec_query_start_loc", None)
        if non_spec_query_start_loc is not None:
            query_start_loc_p = non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
    if query_start_loc_p is None:
        raise ValueError("query_start_loc is required for prefill short-conv")
    has_initial_states_p = metadata.has_initial_states_p
    if has_initial_states_p is None:
        raise ValueError("has_initial_states_p is required for prefill short-conv")

    output = torch.empty_like(x_p)
    q_starts = query_start_loc_p.to(torch.int64)
    if state_indices_tensor_p.numel() < num_prefills:
        raise ValueError(
            f"state_indices_tensor_p size mismatch: got {state_indices_tensor_p.numel()}, need >= {num_prefills}."
        )
    if has_initial_states_p.numel() < num_prefills:
        raise ValueError(
            f"has_initial_states_p size mismatch: got {has_initial_states_p.numel()}, need >= {num_prefills}."
        )
    if num_prefills == 0 or x_p.numel() == 0:
        return output
    lengths = q_starts[1:] - q_starts[:-1]
    max_len = metadata.max_prefill_query_len
    if max_len <= 0:
        return output

    hidden_size = x_p.shape[1]
    positions = torch.arange(num_prefill_tokens, device=x_p.device, dtype=torch.int64)
    req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
    col_indices = positions - q_starts[req_indices]

    packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
    packed_tokens[req_indices, col_indices] = x_p
    packed_tokens = packed_tokens.transpose(1, 2).contiguous()

    state_indices = state_indices_tensor_p[:num_prefills].to(device=conv_state.device, dtype=torch.int64)
    valid_state = state_indices != NULL_BLOCK_ID
    state_indices = torch.where(valid_state, state_indices, torch.zeros_like(state_indices))
    has_initial = has_initial_states_p[:num_prefills].to(device=conv_state.device, dtype=torch.bool)
    if self.conv_state_len > 0:
        if conv_state.shape[0] == 0:
            state = conv_state.new_zeros(
                (num_prefills, hidden_size, self.conv_state_len),
                dtype=x_p.dtype,
            )
        else:
            state = conv_state.index_select(0, state_indices)[..., : self.conv_state_len].to(x_p.dtype)
        use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
        initial_state = torch.where(
            use_initial_mask,
            state,
            torch.zeros_like(state),
        )
        history = torch.cat((initial_state, packed_tokens), dim=-1)
    else:
        history = packed_tokens

    output.zero_()
    conv_filter = conv_weights.unsqueeze(1).contiguous()
    for request_start in range(0, num_prefills, _PLE_PREFILL_CONV_CHUNK_SIZE):
        request_end = min(
            request_start + _PLE_PREFILL_CONV_CHUNK_SIZE,
            num_prefills,
        )
        conv_output = F.conv1d(
            history[request_start:request_end],
            conv_filter,
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        _accumulate_compact_prefill_output(
            conv_output,
            output,
            req_indices,
            col_indices,
            valid_state,
            request_start,
        )

    if self.conv_state_len > 0 and conv_state.shape[0] > 0:
        state_starts = lengths.to(device=history.device, dtype=torch.int64).view(num_prefills, 1, 1)
        state_offsets = torch.arange(self.conv_state_len, device=history.device, dtype=torch.int64).view(
            1, 1, self.conv_state_len
        )
        next_state = history.gather(
            dim=2,
            index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
        )
        existing_state = conv_state.index_select(0, state_indices)
        existing_base_state = existing_state[..., : self.conv_state_len]
        update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
        safe_next_state = torch.where(
            update_mask.view(num_prefills, 1, 1),
            next_state.to(conv_state.dtype),
            existing_base_state,
        )
        existing_state[..., : self.conv_state_len] = safe_next_state
        conv_state.index_copy_(0, state_indices, existing_state)
    return output


def _wrap_ple_short_conv(original: Callable[..., Any]) -> Callable[..., Any]:
    """Run only the PLE depthwise convolution with internal formats disabled.

    With internal formats enabled, torch-npu lowers PLE's dilated depthwise
    ``F.conv1d`` to the legacy ACLop Conv2D path. ACLop operators cannot run
    during NPU graph capture. Disabling internal formats selects the graph-safe
    path, but changing the process-wide option permanently would regress other
    Ascend kernels. Keep the override scoped to the upstream PLE custom op and
    restore the previous value even when convolution raises.
    """

    @wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        # PLE runs before attention in the same decoder layer.
        wait_for_kv_layer_from_connector(self.prefix)
        previous = _get_allow_internal_format()
        _set_allow_internal_format(False)
        try:
            return original(self, *args, **kwargs)
        finally:
            _set_allow_internal_format(previous)

    return wrapped


def _patch_legacy_prefill(ple_layer_cls: type[Any]) -> None:
    """Install the bounded legacy prefill path when upstream still exposes it.

    Newer vLLM revisions route prefill through the fused ``ple_conv``
    dispatcher and no longer define ``_short_conv_dilated_prefill_batched``.
    Keep the override for older supported revisions without making a removed
    private method an import-time requirement.
    """
    method_name = "_short_conv_dilated_prefill_batched"
    original_attr = "_ascend_original_short_conv_dilated_prefill_batched"
    original_prefill = getattr(ple_layer_cls, method_name, None)
    if original_prefill is None or hasattr(ple_layer_cls, original_attr):
        return
    setattr(ple_layer_cls, original_attr, original_prefill)
    setattr(ple_layer_cls, method_name, _short_conv_dilated_prefill_batched)


def patch_upstream_ple_short_conv() -> None:
    """Install Ascend PLE runtime adaptations on the upstream classes."""
    from vllm.models.qwen4_exp.amd import ple_layer as upstream_ple_layer
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer
    from vllm.models.qwen4_exp.common.ngram_embedding import (
        Qwen4ExpPLEEmbeddingMethod,
        Qwen4ExpPLEUnquantizedEmbeddingMethod,
    )

    from vllm_ascend.quantization.configs.modelslim_config import (
        AscendModelSlimConfig,
    )

    # The AMD PLE class binds this function at module import time.  Replace
    # only that model-local binding; other platforms keep the upstream kernel.
    upstream_ple_layer.ple_conv = _ple_conv_ascend

    from .ple_embedding import AscendPLEHostEmbedding

    upstream_ple_layer.Qwen4ExpPLEPinnedHostEmbedding = AscendPLEHostEmbedding

    original_attr = "_ascend_original_short_conv"
    if not hasattr(Qwen4ExpPLELayer, original_attr):
        original = Qwen4ExpPLELayer._short_conv
        setattr(Qwen4ExpPLELayer, original_attr, original)
        Qwen4ExpPLELayer._short_conv = _wrap_ple_short_conv(original)

    quant_attr = "_ascend_original_from_quant_config"
    if not hasattr(Qwen4ExpPLEEmbeddingMethod, quant_attr):
        original_from_quant_config = Qwen4ExpPLEEmbeddingMethod.from_quant_config
        setattr(
            Qwen4ExpPLEEmbeddingMethod,
            quant_attr,
            original_from_quant_config,
        )

        def from_quant_config(
            quant_config: Any,
            prefix: str,
            embedding_dtype: str | None = None,
        ) -> Qwen4ExpPLEEmbeddingMethod:
            if isinstance(quant_config, AscendModelSlimConfig):
                return Qwen4ExpPLEUnquantizedEmbeddingMethod()
            return original_from_quant_config(
                quant_config,
                prefix,
                embedding_dtype,
            )

        Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)

    _patch_legacy_prefill(Qwen4ExpPLELayer)


__all__ = ["patch_upstream_ple_short_conv"]
