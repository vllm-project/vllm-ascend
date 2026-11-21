# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANNBot FlashKDA TND prefill ABI used by Kimi K3's fused prefill path."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

import torch
from vllm.logger import logger


@dataclass(frozen=True)
class PrefillFusedKDAOperators:
    prefill: Callable
    metadata: Callable


class PrefillKDAMetadataWorkspace:
    """Persistent allocation/stream; schedule contents belong to one forward.

    Ported from recipe PR #29. The metadata kernel reads cu_seqlens values;
    Q/V/state provide descriptors, so their payloads need not be shared.
    """

    def __init__(self, device, *, capacity: Callable | None = None, max_batch=0, max_tokens=0):
        if not callable(capacity):
            raise ImportError("FlashKDA metadata reuse requires ops.flash_kda_metadata.metadata_capacity_upper_bound")
        self.capacity = capacity
        self.stream = torch.npu.Stream(device=device)
        self.inputs_ready = torch.npu.Event()
        self.metadata_ready = torch.npu.Event()
        self.consumed = torch.npu.Event()
        words = capacity(max_batch, max_tokens, is_packed=True)
        with torch.npu.stream(self.stream):
            self.buffer = torch.empty(words, dtype=torch.int32, device=device)

    def new_context(self, query_start_loc, keep_meta=None):
        # Empty segments must be removed once, so all layers share both the
        # same boundaries and their device address, including mixed batches.
        if keep_meta is not None:
            query_start_loc = torch.cat((query_start_loc[:1], query_start_loc[1:][keep_meta]))
        query_start_loc = query_start_loc.to(dtype=torch.int32).contiguous()
        return PrefillKDAMetadataContext(self, query_start_loc)


class PrefillKDAMetadataContext:
    """Fresh per-build context shared by compatible layers in an attention group."""

    def __init__(self, workspace, query_start_loc):
        self.workspace = workspace
        self.query_start_loc = query_start_loc
        self.metadata = None
        self.signature = None

    def record_inputs_ready(self):
        if self.metadata is None:
            self.workspace.inputs_ready.record()

    def build(self, operator, q, v, initial_state):
        workspace = self.workspace
        signature = (
            tuple((tensor.shape, tensor.stride(), tensor.dtype, tensor.device) for tensor in (q, v, initial_state)),
            self.query_start_loc.data_ptr(),
            operator.metadata,
        )
        if self.metadata is not None:
            if signature != self.signature:
                raise RuntimeError("FlashKDA metadata cannot be shared across different layouts or operators")
            return self.metadata
        capacity_upper_bound = workspace.capacity(initial_state.shape[0], q.shape[0], is_packed=True)
        if capacity_upper_bound > workspace.buffer.numel():
            raise RuntimeError(
                f"FlashKDA metadata capacity upper bound is {capacity_upper_bound} words "
                f"but its fixed buffer has {workspace.buffer.numel()}; "
                "check max_num_seqs/max_num_batched_tokens"
            )
        with torch.npu.stream(workspace.stream):
            workspace.stream.wait_event(workspace.inputs_ready)
            self.query_start_loc.record_stream(workspace.stream)
            metadata = operator.metadata(q, v, initial_state, layout_qkv="TND", cu_seqlens=self.query_start_loc)
            metadata_words = metadata.numel()
            if (
                metadata_words > capacity_upper_bound
                or metadata.dtype != workspace.buffer.dtype
                or metadata.device != workspace.buffer.device
            ):
                raise RuntimeError(
                    "FlashKDA metadata does not match its capacity/dtype/device contract: "
                    f"actual(shape={tuple(metadata.shape)}, numel={metadata_words}, "
                    f"dtype={metadata.dtype}, device={metadata.device}); "
                    f"expected(numel<={capacity_upper_bound}, dtype={workspace.buffer.dtype}, "
                    f"device={workspace.buffer.device}); "
                    f"buffer_capacity={workspace.buffer.numel()}, q_shape={tuple(q.shape)}, "
                    f"v_shape={tuple(v.shape)}, initial_state_shape={tuple(initial_state.shape)}, "
                    f"cu_seqlens_shape={tuple(self.query_start_loc.shape)}"
                )
            # The capacity helper returns a bound, not the operator's exact
            # output length. Preserve that length for both copy and consumption.
            view = workspace.buffer[:metadata_words]
            view.copy_(metadata)
            del metadata
            metadata = view
            workspace.metadata_ready.record()
        torch.npu.current_stream().wait_event(workspace.metadata_ready)
        self.signature = signature
        self.metadata = metadata
        return metadata

    def finish(self):
        # Join every main-stream consumer back to the allocation stream. The
        # next forward cannot overwrite the buffer before the last layer ends.
        self.workspace.consumed.record()
        self.workspace.stream.wait_event(self.workspace.consumed)


def create_prefill_kda_metadata_workspace(vllm_config, layer_names, device):
    """Initialize FlashKDA groups automatically, outside inference and graph capture."""
    layers = vllm_config.compilation_config.static_forward_context
    if not any(getattr(layers.get(name), "_prefill_fused_kda_operator", None) is not None for name in layer_names):
        return None
    capacity = getattr(import_module("ops.flash_kda_metadata"), "metadata_capacity_upper_bound", None)
    scheduler = vllm_config.scheduler_config
    return PrefillKDAMetadataWorkspace(
        device,
        capacity=capacity,
        max_batch=scheduler.max_num_seqs,
        max_tokens=scheduler.max_num_batched_tokens,
    )


def load_prefill_fused_kda_operator() -> PrefillFusedKDAOperators | None:
    """Resolve both exports at worker construction, without trial execution."""
    try:
        prefill = import_module("ops.flash_kda").flash_kda
        metadata = import_module("ops.flash_kda_metadata").flash_kda_metadata
    except (ImportError, AttributeError) as error:
        logger.warning_once("CANNBot KDA prefill is unavailable; using AscendC prefill: %s", error)
        return None
    if not callable(prefill) or not callable(metadata):
        raise TypeError("ops.flash_kda.flash_kda and ops.flash_kda_metadata.flash_kda_metadata must be callable")
    return PrefillFusedKDAOperators(prefill, metadata)


def run_prefill_fused_kda(
    operator: PrefillFusedKDAOperators,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    metadata_context: PrefillKDAMetadataContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume raw Q/K/gate/beta and return output plus final VK state.

    This is the ``ops.flash_kda`` ABI from the reference ``27.0_kda`` branch.
    The kernel performs Q/K normalization and gate/beta activation internally.
    The caller gathers initial states and scatters final states around this
    call; runner-owned page-strided caches never reach the external operator.
    """
    if metadata_context is not None:
        metadata_context.record_inputs_ready()
    q, k, v, gate, beta = (tensor.contiguous() for tensor in (q, k, v, gate, beta))
    metadata = (
        metadata_context.build(operator, q, v, initial_state)
        if metadata_context is not None
        else operator.metadata(q, v, initial_state, layout_qkv="TND", cu_seqlens=cu_seqlens)
    )
    try:
        output, final_state = operator.prefill(
            q,
            k,
            v,
            g=gate,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            A_log=a_log.reshape(-1).contiguous(),
            dt_bias=dt_bias.contiguous(),
            lower_bound=lower_bound,
            layout_qkv="TND",
            metadata=metadata,
        )
    finally:
        if metadata_context is not None:
            metadata_context.finish()
    if output.shape != v.shape or output.dtype != v.dtype or output.device != v.device:
        raise RuntimeError("FlashKDA returned an unexpected output shape, dtype or device")
    if (
        final_state.shape != initial_state.shape
        or final_state.dtype != initial_state.dtype
        or final_state.device != initial_state.device
    ):
        raise RuntimeError("FlashKDA returned an unexpected final state shape, dtype or device")
    return output, final_state
