# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from typing import cast

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts

from .transfer import TorchExpertTransfer
from .utils import ExpertMapBuffer, ExpertWeightView, PreparedExpertTensor

ACL_MEMCPY_DEVICE_TO_HOST = 2
EXPERTS_PER_HOST_CHUNK = 32


def expert_offload_quant_method(layer, native_method, expert_offload_config):
    from vllm.compilation import backends

    # Native draft loaders switch away from the target's default backbone tag.
    if backends.model_tag != "backbone":
        return native_method

    from .method import ExpertOffloadMoEMethod

    return ExpertOffloadMoEMethod(native_method, layer.moe_config, expert_offload_config)


class ExpertOffloadRoutedExperts(AscendRoutedExperts):
    """Resolve the native quantization method before adding target offload."""

    def _get_quant_method(self, prefix, quant_config, moe_config):
        native_method = super()._get_quant_method(prefix, quant_config, moe_config)
        expert_offload_config = get_ascend_config().expert_offload_config
        return expert_offload_quant_method(self, native_method, expert_offload_config)


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
        Raw checkpoint tensors are retired after the complete layer succeeds.
        """
        import acl  # type: ignore[import-not-found]
        import torch_npu

        prepared = {}
        for expert_id in range(self.num_experts):
            expert_weights = adapter.load_experts(self, (expert_id,), device)
            # Loading only: complete format conversion before reading its storage.
            torch.npu.current_stream(device).synchronize()
            for name, tensor in expert_weights.items():
                # Include native padding in weights and packed scale storage.
                expert_bytes = int(torch_npu.get_storage_size(tensor)) * tensor.element_size()
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
                    tensor.untyped_storage().data_ptr(),
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
        expert_offload_config.validate_layer(layer.local_num_experts, layer.top_k)
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
        self.resident: dict[str, torch.Tensor] = {}
        self.mapping: ExpertMapBuffer | None = None
        self.topk_ids_cpu: torch.Tensor

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

    def prepare(self, layer, topk_ids):
        # Loading attaches the manager and creates the map before any forward.
        manager = cast(ExpertOffloadManager, self.manager)
        mapping = cast(ExpertMapBuffer, self.mapping)
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            return (
                ExpertWeightView(layer, self.resident),
                self.prepare_graph(topk_ids),
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
                    manager.get_comm(self, layer.moe_config, self.num_slots),
                )
        # Prefill and decode batches exceeding resident capacity use all experts.
        return (
            ExpertWeightView(layer, manager.full_pool.load_layer(self)),
            topk_ids,
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
        self.full_pool = FullExpertPool()
        self.comm_methods = {}
        self.has_prefill = True
        self.graph_load_stream = None
        self.graph_streams = set()

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
        for binding in self.bindings.values():
            binding.manager = self
        self.full_pool.allocate(self.bindings.values())

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
