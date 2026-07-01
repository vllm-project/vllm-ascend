# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# NPU adaptation of gpu_ubatch_wrapper.py for Ascend NPU DBO support.
# Replaces torch.cuda.Stream/Event with torch.npu equivalents.

import threading
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import (
    create_forward_context,
    get_forward_context,
    override_forward_context,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.ubatching import UBatchContext


@dataclass
class UbatchMetadata:
    context: UBatchContext
    input_ids: torch.Tensor
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: IntermediateTensors | None
    num_tokens: int


class NPUUBatchContext(UBatchContext):
    """NPU version of UBatchContext that uses torch.npu Stream/Event APIs."""

    def update_stream(self, stream):
        self.current_stream = stream
        torch.npu.set_stream(self.current_stream)

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)


def _make_npu_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.npu.Stream,
    comm_stream: torch.npu.Stream,
    forward_contexts: list,
    ready_barrier: threading.Barrier,
    schedule: str = "default",
) -> list[NPUUBatchContext]:
    """Create NPUUBatchContext instances mirroring make_ubatch_contexts."""
    import vllm.v1.worker.ubatching as _ubatching_mod

    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"

    _ubatching_mod._NUM_UBATCHES = num_micro_batches
    if len(_ubatching_mod._CURRENT_CONTEXTS) < num_micro_batches:
        _ubatching_mod._CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_ubatching_mod._CURRENT_CONTEXTS)))

    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.npu.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = NPUUBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
        )
        ctxs.append(ctx)

    return ctxs


class NPUUBatchWrapper:
    """NPU version of UBatchWrapper for Ascend DBO support.

    Supports eager mode only (no ACL Graph capture in phase 1).
    """

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.comm_stream = torch.npu.Stream(device=device)
        self.ready_barrier = threading.Barrier(self.vllm_config.parallel_config.num_ubatches + 1)
        self.device = device

    def __getattr__(self, key: str):
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not found in NPUUBatchWrapper or its runnable")

    def unwrap(self) -> Callable:
        return self.runnable

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        ubatch_slices = getattr(forward_context, "ubatch_slices", None)

        # If there's no ubatching, just run the runnable directly
        if ubatch_slices is None:
            return self.runnable(*args, **kwargs)

        input_ids = kwargs.get("input_ids") if "input_ids" in kwargs else (args[0] if len(args) > 0 else None)
        positions = kwargs.get("positions") if "positions" in kwargs else (args[1] if len(args) > 1 else None)
        intermediate_tensors = (
            kwargs.get("intermediate_tensors")
            if "intermediate_tensors" in kwargs
            else (args[2] if len(args) > 2 else None)
        )
        inputs_embeds = (
            kwargs.get("inputs_embeds") if "inputs_embeds" in kwargs else (args[3] if len(args) > 3 else None)
        )
        compute_stream = torch.npu.current_stream()

        attn_metadata = forward_context.attn_metadata
        slot_mapping = getattr(forward_context, "slot_mapping", None)
        batch_descriptor = forward_context.batch_descriptor

        # Build per-ubatch forward contexts
        has_slot_mapping = slot_mapping and isinstance(slot_mapping, list)
        forward_contexts = []
        for i, _ubatch_slice in enumerate(ubatch_slices):
            fc = create_forward_context(
                attn_metadata[i] if attn_metadata is not None else None,
                self.vllm_config,
                batch_descriptor=batch_descriptor,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                slot_mapping=(slot_mapping[i] if has_slot_mapping and slot_mapping is not None else None),
            )
            # Copy Ascend-specific attributes from the parent forward context
            # so _EXTRA_CTX proxy works correctly inside ubatch threads.
            # Under V2 model runner these live in `additional_kwargs` rather
            # than directly on the forward_context object.
            from vllm_ascend.ascend_forward_context import _ExtraForwardContextProxy

            additional_kwargs = getattr(forward_context, "additional_kwargs", None)
            for attr in _ExtraForwardContextProxy.extra_attrs:
                if hasattr(forward_context, attr):
                    setattr(fc, attr, getattr(forward_context, attr))
                elif additional_kwargs is not None and attr in additional_kwargs:
                    setattr(fc, attr, additional_kwargs[attr])
            forward_contexts.append(fc)

        ubatch_ctxs = _make_npu_ubatch_contexts(
            num_micro_batches=len(ubatch_slices),
            comm_stream=self.comm_stream,
            compute_stream=compute_stream,
            forward_contexts=forward_contexts,
            ready_barrier=self.ready_barrier,
        )

        ubatch_metadata: list[UbatchMetadata] = []
        for i, ubatch_slice in enumerate(ubatch_slices):
            (
                sliced_input_ids,
                sliced_positions,
                sliced_inputs_embeds,
                sliced_intermediate_tensors,
            ) = self._slice_model_inputs(
                ubatch_slice.token_slice,
                input_ids,
                positions,
                inputs_embeds,
                intermediate_tensors,
            )
            ubatch_metadata.append(
                UbatchMetadata(
                    context=ubatch_ctxs[i],
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=ubatch_slice.token_slice.stop - ubatch_slice.token_slice.start,
                )
            )

        return self._run_ubatches(ubatch_metadata, self.runnable)

    def _run_ubatches(self, ubatch_metadata: list[UbatchMetadata], model: Callable) -> torch.Tensor:
        @torch.inference_mode()
        def _ubatch_thread(results, model, ubatch_metadata):
            with ubatch_metadata.context:
                model_output = model(
                    input_ids=ubatch_metadata.input_ids,
                    positions=ubatch_metadata.positions,
                    intermediate_tensors=ubatch_metadata.intermediate_tensors,
                    inputs_embeds=ubatch_metadata.inputs_embeds,
                )
            results.append((ubatch_metadata.context.id, model_output))

        results: list[tuple[int, torch.Tensor]] = []

        with override_forward_context(None):
            ubatch_threads = []
            for metadata in ubatch_metadata:
                thread = threading.Thread(
                    target=_ubatch_thread,
                    args=(results, model, metadata),
                )
                ubatch_threads.append(thread)
                thread.start()
            self.ready_barrier.wait()
            ubatch_metadata[0].context.cpu_wait_event.set()
            for thread in ubatch_threads:
                thread.join()

        sorted_results = [value for position, value in sorted(results)]
        result = torch.cat(sorted_results, dim=0)
        return result

    def _slice_model_inputs(
        self,
        tokens_slice: slice,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
    ):
        sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
        if positions is not None:
            if positions.ndim == 2:
                sliced_positions = positions[:, tokens_slice]
            else:
                sliced_positions = positions[tokens_slice]
        else:
            sliced_positions = None
        sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds is not None else None
        sliced_intermediate_tensors = intermediate_tensors[tokens_slice] if intermediate_tensors is not None else None

        return (
            sliced_input_ids,
            sliced_positions,
            sliced_inputs_embeds,
            sliced_intermediate_tensors,
        )
