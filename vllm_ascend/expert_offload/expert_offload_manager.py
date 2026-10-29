# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from enum import Enum, auto

import torch

from .host_store import HostExpertStore
from .prefill import PrefillExpertPool
from .quantization import expert_adapter_type
from .residency import ExpertResidency
from .transfer import TorchExpertTransfer
from .utils import ExpertMapBuffer, ExpertWeightView


class OffloadState(Enum):
    """Runner-owned resource lifetime; executions do not create a new state.

    Registration succeeds once into READY. Loading, transfer or execution
    failures enter FAILED, which cannot execute again. close() drains owned
    work and releases resources into CLOSED; recovery needs a new manager.
    """

    # Layers are loading; no forward or graph capture may consume their slots.
    REGISTERING = auto()
    # All layers and the shared pool are complete; `executing` prevents reentry.
    READY = auto()
    # Outputs/residency are no longer trustworthy; only cleanup is allowed.
    FAILED = auto()
    # Resources have been released. Repeated close() is safe.
    CLOSED = auto()


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
        if self.resident:
            raise RuntimeError("expert offload does not support weight reload or repeated post-processing")
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
        manager = self.manager
        if manager is None or manager.state is not OffloadState.READY or not manager.executing or self.mapping is None:
            raise RuntimeError("expert offload layer requires a ready runner-owned execution")
        if self.device.type == "npu" and torch.npu.is_current_stream_capturing():
            from .graph import GraphExpertStaging

            key = (tuple(selected_ids.shape), selected_ids.dtype)
            if key not in self.graph_staging:
                raise RuntimeError("expert graph staging must be initialized before capture")
            staging: GraphExpertStaging = self.graph_staging[key]
            physical_ids = staging.prepare(selected_ids)
            return (
                ExpertWeightView(layer, self.resident),
                physical_ids,
                manager.get_comm(self, layer.moe_config, self.num_slots),
            )
        try:
            if manager.capturing_graphs:
                from .graph import GraphExpertStaging

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
                TorchExpertTransfer.copy_experts(self.adapter, self.host_store, self.resident, plan.loads, self.device)
                self.residency.commit(plan)
                # The blocking route D2H above also fences the previous use of this host map.
                self.mapping.update(self.residency)
                physical_ids = self.mapping.remap(selected_ids)
                tensors = self.resident
                count = self.num_slots
            # A private comm instance avoids mutating the cached full-layer dispatcher.
            return ExpertWeightView(layer, tensors), physical_ids, manager.get_comm(self, layer.moe_config, count)
        except Exception:
            manager.state = OffloadState.FAILED
            raise

    def close(self) -> None:
        self.host_store.close()
        self.resident.clear()
        self.mapping = None
        self.graph_staging.clear()
        self.manager = None


class ExpertOffloadManager:
    """MRV2 runner-owned serialized eager/graph lifecycle without global state."""

    def __init__(self):
        self.state = OffloadState.REGISTERING
        self.bindings: dict[str, LayerOffloadBinding] = {}
        self.prefill_pool = PrefillExpertPool()
        self.comm_methods = {}
        self.executing = False
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
            self.state = OffloadState.FAILED
            raise RuntimeError("expert graph callback failed; rejecting execution output") from self.graph_error

    def prepare_graph_capture(self, graph_manager, needs_capture) -> None:
        from .graph import supports_descriptor

        bindings = tuple(self.bindings.values())
        # Capture and dispatch must use the same padded descriptor capacity.
        graph_manager._capture_descs = {
            mode: [desc for desc in descs if supports_descriptor(bindings, desc)]
            for mode, descs in graph_manager._capture_descs.items()
        }
        graph_manager._candidates = {
            key: [desc for desc in descs if supports_descriptor(bindings, desc)]
            for key, descs in graph_manager._candidates.items()
        }
        if self.graph_load_stream is None and needs_capture():
            self.graph_load_stream = torch.npu.Stream(device=graph_manager.device)

    @contextmanager
    def graph_capture(self):
        self.begin_execution()
        self.has_prefill = False
        self.capturing_graphs = True
        failed = True
        try:
            yield
            for device in {binding.device for binding in self.bindings.values()}:
                torch.npu.current_stream(device).synchronize()
            self.check_graph_errors()
            failed = False
        finally:
            self.capturing_graphs = False
            self.finish_execution(failed=failed)

    def finalize_loading(self, model) -> None:
        if self.state is not OffloadState.REGISTERING:
            raise RuntimeError("expert offload can only finalize once")
        discovered = [
            binding
            for layer in model.modules()
            if (binding := getattr(layer, "_expert_offload_binding", None)) is not None
        ]
        try:
            for binding in discovered:
                if not binding.resident or binding.layer_name in self.bindings:
                    raise RuntimeError("incomplete or duplicate expert offload layer registration")
                self.bindings[binding.layer_name] = binding
                binding.manager = self
                self.prefill_pool.initialize(binding)
            if not self.bindings:
                raise RuntimeError("expert offload found no supported target-model expert layers")
            self.state = OffloadState.READY
        except Exception:
            try:
                self.close()
            finally:
                # Also release layers not yet adopted when registration failed.
                for binding in discovered:
                    binding.close()
                self.state = OffloadState.FAILED
            raise

    def get_comm(self, binding, moe_config, count):
        # Lazy import avoids an initialization cycle with quantization and routed experts.
        from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

        key = (binding.layer_name, count)
        if key not in self.comm_methods:
            config = replace(moe_config, num_experts=count, num_local_experts=count)
            self.comm_methods[key] = AllGatherCommImpl(config)
        return self.comm_methods[key]

    def begin_execution(self, *, is_profile: bool = False) -> None:
        if self.state is not OffloadState.READY or self.executing:
            raise RuntimeError("expert offload requires ready, serialized executions")
        self.executing = True
        # Conservative until V2 batch metadata proves this is decode-only.
        self.has_prefill = True
        self.is_profile = is_profile

    def finish_execution(self, *, failed: bool) -> None:
        self.executing = False
        self.has_prefill = True
        self.is_profile = False
        if failed:
            self.state = OffloadState.FAILED

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
            self.executing = False
            self.state = OffloadState.CLOSED
