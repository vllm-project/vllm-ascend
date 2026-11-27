# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from itertools import takewhile
from typing import TYPE_CHECKING, cast

import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from .expert_offload_config import validate_expert_offload_layer
from .expert_offload_utils import ExpertMapBuffer, ExpertWeightView, PreparedExpertTensor
from .expert_weight_transfer import TorchExpertTransfer

if TYPE_CHECKING:
    from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

ACL_MEMCPY_DEVICE_TO_HOST = 2
EXPERTS_PER_HOST_CHUNK = 32


class MXFPExpertAdapter:
    """Reuse native MXFP post-processing for bounded expert chunks."""

    tensor_names: tuple[str, ...] = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    output_names: tuple[str, ...] = tensor_names

    def __init__(self, layer, quant_scheme):
        self.quant_scheme = quant_scheme
        self.activation = layer.activation

    def load_experts(self, host_store, expert_ids, device) -> dict[str, torch.Tensor]:
        expert_indices = torch.tensor(expert_ids, dtype=torch.long, device="cpu")
        temporary_layer = torch.nn.Module()
        temporary_layer.activation = self.activation
        for name in self.tensor_names:
            # One bounded chunk, never all layers' experts on the NPU.
            checkpoint_tensor = host_store.tensors[name].index_select(0, expert_indices).to(device)
            setattr(temporary_layer, name, torch.nn.Parameter(checkpoint_tensor, requires_grad=False))
        self.quant_scheme.process_weights_after_loading(temporary_layer)
        return {name: getattr(temporary_layer, name).detach() for name in self.output_names}

    def allocate_weights(self, host_store, num_experts, device) -> dict[str, torch.Tensor]:
        """Allocate once through the native scheme using a bounded raw prototype."""
        if not host_store.prepared:
            return self.load_experts(host_store, range(num_experts), device)
        temporary_layer = torch.nn.Module()
        temporary_layer.activation = self.activation
        for name in self.tensor_names:
            prototype = host_store.prototypes[name]
            checkpoint_tensor = prototype.expand(num_experts, *prototype.shape[1:]).contiguous().to(device)
            setattr(temporary_layer, name, torch.nn.Parameter(checkpoint_tensor, requires_grad=False))
        self.quant_scheme.process_weights_after_loading(temporary_layer)
        return {name: getattr(temporary_layer, name).detach() for name in self.output_names}


class HostExpertStore:
    """Owned checkpoint tensors and prepared physical bytes."""

    def __init__(self, layer, names: tuple[str, ...]):
        self.names = names
        self.num_experts = layer.local_num_experts
        self.tensors: dict[str, torch.Tensor] = {}
        self.prepared: dict[str, PreparedExpertTensor] = {}
        self.prototypes: dict[str, torch.Tensor] = {}

    def copy_from_layer(self, layer, *, pin_memory: bool = True) -> None:
        for name in self.names:
            tensor = getattr(layer, name).detach()
            # Own the CPU bytes independently of the checkpoint mmap.
            self.tensors[name] = tensor.pin_memory() if pin_memory else tensor.clone()

    def convert_to_native_layout(self, adapter, device) -> None:
        """Convert each expert once, retaining its complete physical storage on CPU.

        Only one conversion bundle is live on the NPU. Physical copies bypass
        ND interpretation of NZ padding and preserve packed scale strides.
        Native tensors must start at their storage base and have a consistent
        physical size across experts; zero-offset layout views are supported.
        Raw checkpoint tensors are retired after the complete layer succeeds.
        """
        import acl  # type: ignore[import-not-found]
        import torch_npu

        prepared: dict[str, PreparedExpertTensor] = {}
        for expert_id in range(self.num_experts):
            expert_weights = adapter.load_experts(self, (expert_id,), device)
            # Loading only: complete format conversion before reading its storage.
            torch.npu.current_stream(device).synchronize()
            for name, tensor in expert_weights.items():
                # Include native padding in weights and packed scale storage.
                # A shifted pointer cannot copy the complete physical allocation.
                if tensor.storage_offset() != 0:
                    raise ValueError(
                        f"expert offload requires zero-offset native tensors; {name} for expert {expert_id} "
                        f"has storage offset {tensor.storage_offset()}"
                    )
                expert_bytes = int(torch_npu.get_storage_size(tensor)) * tensor.element_size()
                if not 0 < expert_bytes <= tensor.untyped_storage().nbytes():
                    raise ValueError(f"invalid native storage size for {name} of expert {expert_id}: {expert_bytes}")
                if expert_id > 0 and expert_bytes != prepared[name].expert_bytes:
                    raise ValueError(f"inconsistent native expert storage size for {name} of expert {expert_id}")
                if expert_id == 0:
                    # The host allocator rounds large pinned allocations up to a
                    # power of two. Splitting 288 experts into 32-expert chunks
                    # avoids reserving space for 512 experts in every GLM layer.
                    host_chunks = tuple(
                        torch.empty(
                            (min(EXPERTS_PER_HOST_CHUNK, self.num_experts - start), expert_bytes),
                            dtype=torch.uint8,
                            device="cpu",
                            pin_memory=True,
                        )
                        for start in range(0, self.num_experts, EXPERTS_PER_HOST_CHUNK)
                    )
                    prepared[name] = PreparedExpertTensor(
                        data=host_chunks[0],
                        expert_bytes=expert_bytes,
                        extra_data=host_chunks[1:],
                    )
                # ACL reads the storage as bytes, never interpreting NZ as ND.
                chunk, row = prepared[name].source_chunk(expert_id)
                error = acl.rt.memcpy(
                    chunk[row].data_ptr(),
                    expert_bytes,
                    tensor.data_ptr(),
                    expert_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if error:
                    raise RuntimeError(f"expert preparation device-to-host copy failed: ACL error {error}")
            del tensor, expert_weights
        # A single raw expert is sufficient to allocate a full pool through the
        # exact quantization scheme, including native semantic dtype metadata.
        self.prototypes = {name: tensor[:1].clone() for name, tensor in self.tensors.items()}
        self.prepared = prepared
        self.tensors.clear()


class LayerOffloadBinding:
    """Constructed with its target layer; the runner adopts it after loading."""

    def __init__(self, layer, quant_scheme, expert_offload_config):
        validate_expert_offload_layer(expert_offload_config, layer.local_num_experts, layer.top_k)
        if layer.moe_config.is_sequence_parallel or layer.moe_config.has_bias:
            raise ValueError(
                "expert offload has no slot transfer/dispatch support for sequence-parallel or biased experts"
            )
        self.layer_name = layer.layer_name
        self.num_experts = layer.local_num_experts
        self.num_slots = expert_offload_config.num_device_experts
        self.top_k = layer.top_k
        self.device = torch.device(layer.moe_config.device)
        self.adapter = MXFPExpertAdapter(layer, quant_scheme)
        self.host_store = HostExpertStore(layer, self.adapter.tensor_names)
        self.slot_to_expert: tuple[int, ...]
        self.manager: ExpertOffloadManager | None = None
        self._uses_dspark_expert_pool = False
        self.resident: dict[str, torch.Tensor] = {}
        self.mapping: ExpertMapBuffer | None = None
        self.topk_ids_cpu: torch.Tensor

    @property
    def dspark_expert_pool(self) -> DSparkExpertPool | None:
        if self._uses_dspark_expert_pool and self.manager is not None:
            return self.manager.dspark_expert_pool
        return None

    def initialize_storage(self, layer) -> None:
        self.host_store.copy_from_layer(layer, pin_memory=self.device.type == "npu")
        self.resident = self.adapter.load_experts(self.host_store, range(self.num_slots), self.device)
        if self.device.type == "npu":
            torch.npu.current_stream(self.device).synchronize()
            self.host_store.convert_to_native_layout(self.adapter, self.device)
        self.mapping = ExpertMapBuffer(self.num_experts, self.device)
        # Capturable descriptors bound route entries by the resident capacity.
        self.topk_ids_cpu = torch.empty(
            self.num_slots, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "npu"
        )
        # Preserve Parameter objects and their loader metadata.
        for name, tensor in self.resident.items():
            getattr(layer, name).data = tensor
        self.slot_to_expert = tuple(range(self.num_slots))

    def prepare(self, layer, topk_ids, topk_weights):
        # Loading attaches the manager and creates the map before any forward.
        manager = cast(ExpertOffloadManager, self.manager)
        mapping = cast(ExpertMapBuffer, self.mapping)
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            padding = get_forward_context().is_padding
            if padding is not None:
                padding = padding[: topk_ids.shape[0], None]
                # Real rows form the prefix. Reuse the first row's IDs so padding
                # needs no extra expert loads and still maps to valid slots.
                topk_ids = torch.where(padding, topk_ids[:1], topk_ids)
                topk_weights = topk_weights.masked_fill(padding, 0)
            return (
                ExpertWeightView(layer, self.resident),
                self.prepare_graph(topk_ids),
                topk_weights,
                manager.get_comm(self, layer.moe_config, self.num_slots),
            )
        if not manager.has_prefill:
            # Only resident selection needs routes on the CPU; prefill loads all experts.
            selected_expert_ids = set(topk_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist())
            if len(selected_expert_ids) <= self.num_slots:
                # The blocking route D2H above also fences the previous use of this host map.
                slot_to_expert, expert_loads = self.plan_expert_loads(selected_expert_ids)
                self.load_resident(slot_to_expert, expert_loads)
                return (
                    ExpertWeightView(layer, self.resident),
                    mapping.remap(topk_ids),
                    topk_weights,
                    manager.get_comm(self, layer.moe_config, self.num_slots),
                )
        # Prefill and decode batches exceeding resident capacity use all experts.
        return (
            ExpertWeightView(layer, manager.full_pool.load_layer(self)),
            topk_ids,
            topk_weights,
            manager.get_comm(self, layer.moe_config, self.num_experts),
        )

    def plan_expert_loads(self, selected_expert_ids: set[int]) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        slot_to_expert = list(self.slot_to_expert)
        missing_expert_ids = sorted(selected_expert_ids - set(slot_to_expert))
        # Keep hits and replace unneeded experts in slot order.
        replaceable_slots = [
            slot for slot, expert_id in enumerate(slot_to_expert) if expert_id not in selected_expert_ids
        ]
        expert_loads = tuple(zip(missing_expert_ids, replaceable_slots))
        for expert_id, slot in expert_loads:
            slot_to_expert[slot] = expert_id
        return tuple(slot_to_expert), expert_loads

    def load_resident(self, slot_to_expert, expert_loads) -> None:
        """Update slots and their map after every weight/scale copy completes."""
        TorchExpertTransfer.copy_experts(self.adapter, self.host_store, self.resident, expert_loads, self.device)
        self.slot_to_expert = slot_to_expert
        cast(ExpertMapBuffer, self.mapping).update(slot_to_expert)

    @torch.inference_mode()
    def update_graph(self, num_topk_ids) -> None:
        try:
            manager = cast(ExpertOffloadManager, self.manager)
            # Each graph supplies its own fixed length; unused tail entries are stale.
            selected_expert_ids = set(self.topk_ids_cpu[:num_topk_ids].tolist())
            slot_to_expert, expert_loads = self.plan_expert_loads(selected_expert_ids)
            # Report threads have their own current device/stream state.
            torch.npu.set_device(self.device)
            with torch.npu.stream(manager.graph_load_stream):
                self.load_resident(slot_to_expert, expert_loads)
        except Exception:
            # torch_npu reports and clears callback exceptions. Do not let the
            # graph continue with partially overwritten weights and an old map.
            logger.exception("Expert offload graph callback failed for %s; terminating worker", self.layer_name)
            os._exit(1)

    def prepare_graph(self, topk_ids):
        num_topk_ids = topk_ids.numel()
        # Convert on the device so D2H writes directly into pinned storage.
        self.topk_ids_cpu[:num_topk_ids].copy_(topk_ids.reshape(-1).to(self.topk_ids_cpu.dtype), non_blocking=True)
        torch.npu._launch_host_func(torch.npu.current_stream(self.device), self.update_graph, num_topk_ids)
        return cast(ExpertMapBuffer, self.mapping).remap(topk_ids)


class FullExpertPool:
    """One reusable full layer shared by the manager's homogeneous expert layers."""

    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def allocate(self, bindings) -> None:
        first_binding, *remaining_bindings = bindings
        expert_layout = self._expert_layout(first_binding)
        for binding in remaining_bindings:
            if self._expert_layout(binding) != expert_layout:
                raise ValueError(
                    "expert offload requires identical expert layouts for the shared full pool; "
                    f"{binding.layer_name} differs from {first_binding.layer_name}"
                )
        # Allocate in the scheme's native format, including scale strides.
        self.tensors = first_binding.adapter.allocate_weights(
            first_binding.host_store, first_binding.num_experts, first_binding.device
        )

    @staticmethod
    def _expert_layout(binding):
        if binding.device.type == "npu":
            import torch_npu

        return (
            binding.num_experts,
            binding.adapter.quant_scheme.quant_type,
            {
                name: (
                    tuple(tensor.shape[1:]),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    torch_npu.get_npu_format(tensor) if binding.device.type == "npu" else None,
                    binding.host_store.prepared[name].expert_bytes if binding.host_store.prepared else None,
                )
                for name, tensor in binding.resident.items()
            },
        )

    def load_layer(self, binding):
        TorchExpertTransfer.copy_experts(
            binding.adapter,
            binding.host_store,
            self.tensors,
            tuple((expert_id, expert_id) for expert_id in range(binding.num_experts)),
            binding.device,
        )
        return self.tensors


class ExpertOffloadManager:
    """Own expert storage and transfers for serialized MRV2 eager/graph execution."""

    def __init__(self):
        self.bindings: dict[str, LayerOffloadBinding] = {}
        self.dspark_expert_pool: DSparkExpertPool | None = None
        self.full_pool = FullExpertPool()
        self.comm_methods = {}
        self.has_prefill = True
        self.graph_load_stream = None
        self.graph_streams = set()

    def before_execute(self) -> None:
        if self.dspark_expert_pool is not None:
            self.dspark_expert_pool.active = False

    @staticmethod
    def _verify_graph_eligible(batch_state, *, num_speculative_tokens: int, structured: bool) -> bool:
        rows = batch_state.num_scheduled_tokens
        drafts = batch_state.num_draft_tokens_np
        verify_width = num_speculative_tokens + 1
        return (
            not structured
            and drafts is not None
            and len(rows) == len(drafts) > 0
            and not any(batch_state.is_prefilling_np)
            and all(count == num_speculative_tokens for count in drafts)
            and all(count == verify_width for count in rows)
        )

    def prepare_batch(self, batch_state, uniform_token_count: int | None, scheduler_output) -> int | None:
        # Dummy batches have no request state and use the full expert pool.
        self.has_prefill = batch_state is None or batch_state.has_prefill
        pool = self.dspark_expert_pool
        if (
            pool is not None
            and batch_state is not None
            and uniform_token_count not in (None, 1)
            and not self._verify_graph_eligible(
                batch_state,
                num_speculative_tokens=pool.num_speculative_tokens,
                structured=scheduler_output.has_structured_output_requests
                or bool(scheduler_output.num_invalid_spec_tokens),
            )
        ):
            # Mixed and short verification retain the eager routing path.
            return None
        return uniform_token_count

    def prepare_inputs(self, input_batch, scheduler_output) -> None:
        if self.dspark_expert_pool is not None:
            self.dspark_expert_pool.prepare(input_batch, invalid_drafts=bool(scheduler_output.num_invalid_spec_tokens))

    def prepare_capture(self, input_batch) -> None:
        if self.dspark_expert_pool is not None:
            self.dspark_expert_pool.prepare_capture(input_batch)

    def shutdown(self) -> None:
        self.unsubscribe_graph_streams()
        self.dspark_expert_pool = None

    def capture_stream(self, _desc):
        # Use the actual NPU stream rather than torch_cuda_wrapper's CUDA cache.
        stream = torch.npu.current_stream(self.device)
        if stream not in self.graph_streams:
            torch.npu._subscribe_report(stream)
            self.graph_streams.add(stream)
        return stream

    def unsubscribe_graph_streams(self) -> None:
        torch.npu.synchronize()
        for stream in self.graph_streams:
            torch.npu._unsubscribe_report(stream)
        self.graph_streams.clear()

    def _supports_graph_descriptor(self, desc) -> bool:
        """Bound distinct routes by the padded size, including dummy/padding rows."""
        return (
            bool(self.bindings)
            and desc.uniform_token_count is not None
            and desc.uniform_token_count > 0
            and desc.num_tokens % desc.uniform_token_count == 0
            and all(0 < desc.num_tokens * binding.top_k <= binding.num_slots for binding in self.bindings.values())
        )

    def prepare_graph_capture(self, graph_manager, needs_capture) -> None:
        # Capture and dispatch must use the same padded descriptor capacity.
        graph_manager._capture_descs = {
            mode: [desc for desc in descs if self._supports_graph_descriptor(desc)]
            for mode, descs in graph_manager._capture_descs.items()
        }
        graph_manager._candidates = {
            key: [desc for desc in descs if self._supports_graph_descriptor(desc)]
            for key, descs in graph_manager._candidates.items()
        }
        # Main also dispatches uniform decode through a request-indexed table.
        # Capacity excludes a suffix; keep the surviving indices unchanged.
        if hasattr(graph_manager, "_decode_descs"):
            graph_manager._decode_descs = {
                key: list(takewhile(self._supports_graph_descriptor, descs))
                for key, descs in graph_manager._decode_descs.items()
            }
        if needs_capture():
            self.device = graph_manager.device
            self.graph_load_stream = torch.npu.Stream(device=graph_manager.device)
            graph_manager._capture_stream = self.capture_stream
            graph_manager.capture = partial(self.capture, graph_manager.capture)

    def capture(self, capture_fn, *args, **kwargs):
        self.has_prefill = False
        return capture_fn(*args, **kwargs)

    def initialize(self, model) -> None:
        self.bindings = {
            binding.layer_name: binding
            for layer in model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        }
        if not self.bindings:
            raise ValueError("expert offload is enabled but found no layers with expert offload bindings")
        for binding in self.bindings.values():
            binding.manager = self
        self.full_pool.allocate(self.bindings.values())

    def initialize_dspark_expert_pool(self, runner: NPUModelRunner) -> None:
        """Allocate one optional pool inside the offload loading-memory window."""
        self.dspark_expert_pool = None
        for binding in self.bindings.values():
            binding._uses_dspark_expert_pool = False
        offload_config = runner.ascend_config.expert_offload_config
        protected_positions = offload_config.dspark_protected_positions
        suffix_experts = offload_config.dspark_suffix_experts
        if protected_positions is None and suffix_experts is None:
            return

        # Import only for enabled pools; router compatibility imports MoE code.
        from vllm_ascend.distributed.expert_transfer.dspark_expert_pool import DSparkExpertPool, supports_router

        layers = [
            layer
            for layer in runner.model.modules()
            if getattr(layer, "_expert_offload_binding", None) is not None and supports_router(layer)
        ]
        if not layers:
            raise ValueError("dspark_expert_pool found no supported target routed expert layer")
        assert protected_positions is not None and suffix_experts is not None
        assert runner.speculative_config is not None and runner.speculative_config.num_speculative_tokens is not None
        assert offload_config.num_device_experts is not None
        self.dspark_expert_pool = DSparkExpertPool(
            runner.device,
            runner.max_num_tokens,
            protected_positions,
            suffix_experts,
            num_speculative_tokens=runner.speculative_config.num_speculative_tokens,
            num_device_experts=offload_config.num_device_experts,
        )
        for layer in layers:
            binding = layer._expert_offload_binding
            binding._uses_dspark_expert_pool = True
            layer.router._expert_pool_binding = binding

    def get_comm(self, binding, moe_config, num_physical_experts):
        # Lazy import avoids an initialization cycle with quantization and routed experts.
        from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

        comm_key = (binding.layer_name, num_physical_experts)
        if comm_key not in self.comm_methods:
            execution_moe_config = replace(
                moe_config, num_experts=num_physical_experts, num_local_experts=num_physical_experts
            )
            self.comm_methods[comm_key] = AllGatherCommImpl(execution_moe_config)
        return self.comm_methods[comm_key]


def initialize_expert_offload(runner: NPUModelRunner) -> None:
    """Attach the manager and account for the shared pool after native loading."""
    expert_offload_manager = ExpertOffloadManager()
    memory_allocated_before = torch.npu.memory_allocated(runner.device)
    expert_offload_manager.initialize(runner.model)
    expert_offload_manager.initialize_dspark_expert_pool(runner)
    runner.model._expert_offload_manager = expert_offload_manager
    # The shared pool is allocated after the parent's loading profiler.
    runner.model_memory_usage += torch.npu.memory_allocated(runner.device) - memory_allocated_before
