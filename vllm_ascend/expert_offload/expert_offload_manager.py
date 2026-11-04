# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import cast

import torch

from .quantization import expert_adapter_type
from .residency import ExpertResidency
from .transfer import TorchExpertTransfer
from .utils import ExpertMapBuffer, ExpertWeightView, PreparedExpertTensor

ACL_MEMCPY_DEVICE_TO_HOST = 2
EXPERTS_PER_HOST_CHUNK = 32


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
            if set(bundle) != set(adapter.output_names) or set(destination) != set(adapter.output_names):
                raise RuntimeError("incomplete expert preparation bundle")
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
                        shape=tuple(tensor.shape[1:]),
                        stride=tuple(tensor.stride()),
                        dtype=tensor.dtype,
                        device=tensor.device,
                        npu_format=torch_npu.get_npu_format(tensor),
                        expert_bytes=size,
                        extra_data=chunks[1:],
                    )
                elif size != prepared[name].expert_bytes:
                    raise ValueError("expert physical layout changed during preparation")
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

    def close(self) -> None:
        self.tensors.clear()
        self.prepared.clear()
        self.prototypes.clear()


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
        self.adapter = expert_adapter_type(scheme)(layer, scheme)
        self.adapter.validate_layer(layer)
        self.host_store = HostExpertStore(layer, self.adapter.tensor_names)
        self.residency = ExpertResidency(self.num_experts, self.num_slots)
        self.manager: ExpertOffloadManager | None = None
        self.resident: dict[str, torch.Tensor] = {}
        self.mapping: ExpertMapBuffer | None = None
        self.graph_staging = {}

    def process_weights_after_loading(self, layer) -> None:
        try:
            self.host_store.finalize(layer, pin_memory=self.device.type == "npu")
            self.resident = self.adapter.materialize(self.host_store, range(self.num_slots), self.device)
            if self.device.type == "npu":
                torch.npu.current_stream(self.device).synchronize()
                self.host_store.prepare_native(self.adapter, self.resident, self.device)
            self.mapping = ExpertMapBuffer(self.num_experts, self.device)
            # Preserve Parameter metadata/loaders, and allocate no full device layer here.
            for name in self.adapter.tensor_names:
                if name not in self.resident:
                    delattr(layer, name)
            for name, tensor in self.resident.items():
                if isinstance(getattr(layer, name, None), torch.nn.Parameter):
                    getattr(layer, name).data = tensor
                else:
                    setattr(layer, name, tensor)
            self.residency.commit(self.residency.plan(range(self.num_slots)))
        except Exception:
            self.close()
            raise

    def prepare(self, layer, selected_ids):
        # Loading attaches the manager and creates the map before any forward.
        manager = cast(ExpertOffloadManager, self.manager)
        mapping = cast(ExpertMapBuffer, self.mapping)
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            key = (tuple(selected_ids.shape), selected_ids.dtype)
            staging: GraphExpertStaging = self.graph_staging[key]
            physical_ids = staging.prepare(selected_ids)
            return (
                ExpertWeightView(layer, self.resident),
                physical_ids,
                manager.get_comm(self, layer.moe_config, self.num_slots),
            )
        if manager.capturing_graphs:
            if selected_ids.numel() > self.num_slots:
                raise RuntimeError("expert graph shape exceeds resident capacity")
            key = (tuple(selected_ids.shape), selected_ids.dtype)
            if key not in self.graph_staging:
                self.graph_staging[key] = GraphExpertStaging(self, selected_ids.shape, selected_ids.dtype)
        # One batched synchronization. No scalar device .item() loop.
        logical_ids = selected_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
        plan = self.residency.plan(logical_ids)
        full_pool = manager.has_prefill or manager.is_profile or plan.use_full_pool
        if full_pool:
            tensors = manager.prefill_pool.fill(self)
            physical_ids = selected_ids
            count = self.num_experts
        else:
            # The blocking route D2H above also fences the previous use of this host map.
            self.load_resident(plan, mapping)
            physical_ids = mapping.remap(selected_ids)
            tensors = self.resident
            count = self.num_slots
        # A private comm instance avoids mutating the cached full-layer dispatcher.
        return ExpertWeightView(layer, tensors), physical_ids, manager.get_comm(self, layer.moe_config, count)

    def load_resident(self, plan, mapping) -> None:
        """Publish residency and its map after every weight/scale copy completes."""
        TorchExpertTransfer.copy_experts(self.adapter, self.host_store, self.resident, plan.loads, self.device)
        self.residency.commit(plan)
        mapping.update(self.residency)

    def close(self) -> None:
        self.host_store.close()
        self.resident.clear()
        self.mapping = None
        self.graph_staging.clear()
        self.manager = None


class GraphExpertStaging:
    """Stable, layer-owned route/map storage for one serialized graph shape.

    The captured stream orders D2H -> callback -> H2D mapping -> GEMMs.
    The callback only synchronizes the independent load stream, never the
    executing graph stream. The runner fences replay and checks failures before
    consuming hidden states. Owner lifetimes extend past every graph consumer.
    """

    def __init__(self, binding, shape, dtype):
        self.binding = binding
        self.ids = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
        self.mapping = ExpertMapBuffer(binding.num_experts, binding.device)

    @torch.inference_mode()
    def update(self, _unused=None) -> None:
        manager = self.binding.manager
        try:
            if manager.graph_error is not None:
                return
            plan = self.binding.residency.plan(self.ids.reshape(-1).tolist())
            if plan.use_full_pool:
                raise RuntimeError("expert graph exceeded its prevalidated resident capacity")
            # Report threads have their own current device/stream state.
            torch.npu.set_device(self.binding.device)
            with torch.npu.stream(manager.graph_load_stream):
                self.binding.load_resident(plan, self.mapping)
        except Exception as error:
            # Exceptions on report threads cannot be propagated by raising here.
            # Replay may finish, but its output must be rejected by the runner.
            if manager.graph_error is None:
                manager.graph_error = error
            self.mapping.host_map.zero_()

    def prepare(self, selected_ids):
        stream = torch.npu.current_stream(self.binding.device)
        self.ids.copy_(selected_ids, non_blocking=True)
        torch.npu._launch_host_func(stream, self.update, None)
        return self.mapping.remap(selected_ids)


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

    def close(self) -> None:
        self.tensors.clear()


class ExpertOffloadManager:
    """Own expert storage and transfers for serialized MRV2 eager/graph execution."""

    def __init__(self):
        self.bindings: dict[str, LayerOffloadBinding] = {}
        self.prefill_pool = PrefillExpertPool()
        self.comm_methods = {}
        self.has_prefill = True
        self.is_profile = False
        self.capturing_graphs = False
        self.graph_error: Exception | None = None
        self.graph_load_stream = None
        self.graph_streams = set()

    def subscribe_graph_stream(self, stream) -> None:
        if stream not in self.graph_streams:
            torch.npu._subscribe_report(stream)
            self.graph_streams.add(stream)

    def check_graph_errors(self) -> None:
        if self.graph_error is not None:
            raise RuntimeError("expert graph callback failed; rejecting execution output") from self.graph_error

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
        if self.graph_load_stream is None and needs_capture():
            self.graph_load_stream = torch.npu.Stream(device=graph_manager.device)

    @contextmanager
    def graph_capture(self):
        self.has_prefill = False
        self.is_profile = False
        self.capturing_graphs = True
        try:
            yield
            for device in {binding.device for binding in self.bindings.values()}:
                torch.npu.current_stream(device).synchronize()
            self.check_graph_errors()
        finally:
            self.capturing_graphs = False

    def finalize_loading(self, model) -> None:
        self.bindings = {
            binding.layer_name: binding
            for layer in model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        }
        try:
            for binding in self.bindings.values():
                binding.manager = self
            self.prefill_pool.initialize(next(iter(self.bindings.values())))
        except Exception:
            self.close()
            raise

    def get_comm(self, binding, moe_config, count):
        # Lazy import avoids an initialization cycle with quantization and routed experts.
        from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

        key = (binding.layer_name, count)
        if key not in self.comm_methods:
            config = replace(moe_config, num_experts=count, num_local_experts=count)
            self.comm_methods[key] = AllGatherCommImpl(config)
        return self.comm_methods[key]

    def close(self) -> None:
        try:
            for device in {binding.device for binding in self.bindings.values()}:
                if device.type == "npu":
                    torch.npu.current_stream(device).synchronize()
            if self.graph_load_stream is not None:
                self.graph_load_stream.synchronize()
            for stream in self.graph_streams:
                torch.npu._unsubscribe_report(stream)
        finally:
            self.graph_streams.clear()
            self.graph_load_stream = None
            for binding in self.bindings.values():
                binding.close()
            self.bindings.clear()
            self.comm_methods.clear()
            self.prefill_pool.close()
