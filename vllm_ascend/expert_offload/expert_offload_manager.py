# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import cast

import torch

from .transfer import TorchExpertTransfer
from .utils import ExpertMapBuffer, ExpertWeightView, PreparedExpertTensor

ACL_MEMCPY_DEVICE_TO_HOST = 2
EXPERTS_PER_HOST_CHUNK = 32


def expert_offload_quant_method(layer, method, config):
    from vllm.compilation import backends

    # Native draft loaders switch away from the target's default backbone tag.
    if backends.model_tag != "backbone":
        return method

    from .method import ExpertOffloadMoEMethod

    return ExpertOffloadMoEMethod(method, layer.moe_config, config)


class MXFPExpertAdapter:
    """Reuse native MXFP post-processing for bounded expert chunks."""

    tensor_names: tuple[str, ...] = ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale")
    output_names: tuple[str, ...] = tensor_names

    def __init__(self, layer, scheme):
        self.scheme = scheme
        self.activation = layer.activation

    def materialize(self, host_store, expert_ids, device) -> dict[str, torch.Tensor]:
        indices = torch.tensor(expert_ids, dtype=torch.long, device="cpu")
        view = torch.nn.Module()
        view.activation = self.activation
        for name in self.tensor_names:
            # One bounded chunk, never all layers' experts on the NPU.
            raw = host_store.tensors[name].index_select(0, indices).to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.output_names}

    def allocate_pool(self, host_store, count, device) -> dict[str, torch.Tensor]:
        """Allocate once through the native scheme using a bounded raw prototype."""
        if not host_store.prepared:
            return self.materialize(host_store, range(count), device)
        view = torch.nn.Module()
        view.activation = self.activation
        for name in self.tensor_names:
            prototype = host_store.prototypes[name]
            raw = prototype.expand(count, *prototype.shape[1:]).contiguous().to(device)
            setattr(view, name, torch.nn.Parameter(raw, requires_grad=False))
        self.scheme.process_weights_after_loading(view)
        return {name: getattr(view, name).detach() for name in self.output_names}


class HostExpertStore:
    """Owned checkpoint tensors and prepared physical bytes."""

    def __init__(self, layer, names: tuple[str, ...]):
        self.names = names
        self.num_experts = layer.local_num_experts
        self.tensors: dict[str, torch.Tensor] = {}
        self.prepared: dict[str, PreparedExpertTensor] = {}
        self.prototypes: dict[str, torch.Tensor] = {}

    def finalize(self, layer, *, pin_memory: bool = True) -> None:
        for name in self.names:
            tensor = getattr(layer, name).detach()
            # Own the CPU bytes independently of the checkpoint mmap.
            self.tensors[name] = tensor.pin_memory() if pin_memory else tensor.clone()

    def prepare_native(self, adapter, destination, device) -> None:
        """Convert each expert once, retaining its complete physical storage on CPU.

        Only one conversion bundle is live on the NPU. Physical copies bypass
        ND interpretation of NZ padding and preserve packed scale strides.
        Raw checkpoint tensors are retired after the complete layer succeeds.
        """
        import acl  # type: ignore[import-not-found]
        import torch_npu

        prepared = {}
        for expert_id in range(self.num_experts):
            bundle = adapter.materialize(self, (expert_id,), device)
            # Loading only: complete format conversion before reading its storage.
            torch.npu.current_stream(device).synchronize()
            for name, tensor in bundle.items():
                _, _, size = TorchExpertTransfer._physical_copy_parameters(destination[name], tensor, 0, torch_npu)
                if expert_id == 0:
                    # The host allocator rounds large pinned allocations up to a
                    # power of two. Splitting 288 experts into 32-expert chunks
                    # avoids reserving space for 512 experts in every GLM layer.
                    chunks = tuple(
                        torch.empty(
                            (min(EXPERTS_PER_HOST_CHUNK, self.num_experts - start), size),
                            dtype=torch.uint8,
                            device="cpu",
                            pin_memory=True,
                        )
                        for start in range(0, self.num_experts, EXPERTS_PER_HOST_CHUNK)
                    )
                    prepared[name] = PreparedExpertTensor(
                        data=chunks[0],
                        expert_bytes=size,
                        extra_data=chunks[1:],
                    )
                # ACL reads the storage as bytes, never interpreting NZ as ND.
                chunk, row = prepared[name].source_chunk(expert_id)
                error = acl.rt.memcpy(
                    chunk[row].data_ptr(),
                    size,
                    tensor.untyped_storage().data_ptr(),
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if error:
                    raise RuntimeError(f"expert preparation device-to-host copy failed: ACL error {error}")
            del tensor, bundle
        # A single raw expert is sufficient to allocate a full pool through the
        # exact quantization scheme, including native semantic dtype metadata.
        self.prototypes = {name: tensor[:1].clone() for name, tensor in self.tensors.items()}
        self.prepared = prepared
        self.tensors.clear()


class LayerOffloadBinding:
    """Constructed with its target layer; the runner adopts it after loading."""

    def __init__(self, layer, scheme, config):
        config.validate_layer(layer.local_num_experts, layer.top_k)
        if layer.moe_config.is_sequence_parallel or layer.moe_config.has_bias:
            raise ValueError(
                "expert offload has no slot transfer/dispatch support for sequence-parallel or biased experts"
            )
        self.layer_name = layer.layer_name
        self.num_experts = layer.local_num_experts
        self.num_slots = config.num_device_experts
        self.top_k = layer.top_k
        self.device = torch.device(layer.moe_config.device)
        self.adapter = MXFPExpertAdapter(layer, scheme)
        self.host_store = HostExpertStore(layer, self.adapter.tensor_names)
        self.slot_to_expert: tuple[int, ...]
        self.manager: ExpertOffloadManager | None = None
        self.resident: dict[str, torch.Tensor] = {}
        self.mapping: ExpertMapBuffer | None = None
        self.ids_cpu: torch.Tensor

    def process_weights_after_loading(self, layer) -> None:
        self.host_store.finalize(layer, pin_memory=self.device.type == "npu")
        self.resident = self.adapter.materialize(self.host_store, range(self.num_slots), self.device)
        if self.device.type == "npu":
            torch.npu.current_stream(self.device).synchronize()
            self.host_store.prepare_native(self.adapter, self.resident, self.device)
        self.mapping = ExpertMapBuffer(self.num_experts, self.device)
        # Capturable descriptors bound route entries by the resident capacity.
        self.ids_cpu = torch.empty(
            self.num_slots, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "npu"
        )
        # Preserve Parameter objects and their loader metadata.
        for name, tensor in self.resident.items():
            getattr(layer, name).data = tensor
        self.slot_to_expert = tuple(range(self.num_slots))

    def prepare(self, layer, selected_ids):
        # Loading attaches the manager and creates the map before any forward.
        manager = cast(ExpertOffloadManager, self.manager)
        mapping = cast(ExpertMapBuffer, self.mapping)
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            return (
                ExpertWeightView(layer, self.resident),
                self.prepare_graph(selected_ids),
                manager.get_comm(self, layer.moe_config, self.num_slots),
            )
        # One batched synchronization. No scalar device .item() loop.
        selected = set(selected_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist())
        full_pool = manager.has_prefill or len(selected) > self.num_slots
        if full_pool:
            tensors = manager.prefill_pool.fill(self)
            physical_ids = selected_ids
            count = self.num_experts
        else:
            # The blocking route D2H above also fences the previous use of this host map.
            slots, loads = self.plan_expert_loads(selected)
            self.load_resident(slots, loads)
            physical_ids = mapping.remap(selected_ids)
            tensors = self.resident
            count = self.num_slots
        # A private comm instance avoids mutating the cached full-layer dispatcher.
        return ExpertWeightView(layer, tensors), physical_ids, manager.get_comm(self, layer.moe_config, count)

    def plan_expert_loads(self, selected: set[int]) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        slots = list(self.slot_to_expert)
        missing = sorted(selected - set(slots))
        # Keep hits and replace unneeded experts in slot order.
        victims = [slot for slot, expert_id in enumerate(slots) if expert_id not in selected]
        loads = tuple(zip(missing, victims))
        for expert_id, slot in loads:
            slots[slot] = expert_id
        return tuple(slots), loads

    def load_resident(self, slots, loads) -> None:
        """Update slots and their map after every weight/scale copy completes."""
        TorchExpertTransfer.copy_experts(self.adapter, self.host_store, self.resident, loads, self.device)
        self.slot_to_expert = slots
        cast(ExpertMapBuffer, self.mapping).update(slots)

    @torch.inference_mode()
    def update_graph(self, n) -> None:
        manager = cast(ExpertOffloadManager, self.manager)
        # Each graph supplies its own fixed length; unused tail entries are stale.
        slots, loads = self.plan_expert_loads(set(self.ids_cpu[:n].tolist()))
        # Report threads have their own current device/stream state.
        torch.npu.set_device(self.device)
        with torch.npu.stream(manager.graph_load_stream):
            self.load_resident(slots, loads)

    def prepare_graph(self, selected_ids):
        n = selected_ids.numel()
        # Convert on the device so D2H writes directly into pinned storage.
        self.ids_cpu[:n].copy_(selected_ids.reshape(-1).to(self.ids_cpu.dtype), non_blocking=True)
        torch.npu._launch_host_func(torch.npu.current_stream(self.device), self.update_graph, n)
        return cast(ExpertMapBuffer, self.mapping).remap(selected_ids)


class PrefillExpertPool:
    """One reusable full layer shared by the manager's homogeneous expert layers."""

    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def initialize(self, binding) -> None:
        # Allocate in the scheme's native format, including scale strides.
        self.tensors = binding.adapter.allocate_pool(binding.host_store, binding.num_experts, binding.device)

    def fill(self, binding):
        TorchExpertTransfer.copy_experts(
            binding.adapter,
            binding.host_store,
            self.tensors,
            tuple((eid, eid) for eid in range(binding.num_experts)),
            binding.device,
        )
        return self.tensors


class ExpertOffloadManager:
    """Own expert storage and transfers for serialized MRV2 eager/graph execution."""

    def __init__(self):
        self.bindings: dict[str, LayerOffloadBinding] = {}
        self.prefill_pool = PrefillExpertPool()
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

    def capture(self, capture, *args, **kwargs):
        self.has_prefill = False
        return capture(*args, **kwargs)

    def finalize_loading(self, model) -> None:
        self.bindings = {
            binding.layer_name: binding
            for layer in model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        }
        for binding in self.bindings.values():
            binding.manager = self
        self.prefill_pool.initialize(next(iter(self.bindings.values())))

    def get_comm(self, binding, moe_config, count):
        # Lazy import avoids an initialization cycle with quantization and routed experts.
        from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

        key = (binding.layer_name, count)
        if key not in self.comm_methods:
            config = replace(moe_config, num_experts=count, num_local_experts=count)
            self.comm_methods[key] = AllGatherCommImpl(config)
        return self.comm_methods[key]
