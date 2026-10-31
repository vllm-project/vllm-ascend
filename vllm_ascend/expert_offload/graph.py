# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from .transfer import TorchExpertTransfer
from .utils import ExpertMapBuffer


def supports_descriptor(bindings, desc) -> bool:
    """Bound distinct routes by the padded size, including dummy/padding rows."""
    return (
        bool(bindings)
        and desc.uniform_token_count is not None
        and desc.uniform_token_count > 0
        and desc.num_tokens % desc.uniform_token_count == 0
        and all(0 < desc.num_tokens * binding.top_k <= binding.num_slots for binding in bindings)
    )


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
            if not manager.executing:
                raise RuntimeError("expert graph callback outside runner-owned execution")
            plan = self.binding.residency.plan(self.ids.reshape(-1).tolist())
            if plan.use_full_pool:
                raise RuntimeError("expert graph exceeded its prevalidated resident capacity")
            # Report threads have their own current device/stream state.
            torch.npu.set_device(self.binding.device)
            with torch.npu.stream(manager.graph_load_stream):
                TorchExpertTransfer.copy_experts(
                    self.binding.adapter,
                    self.binding.host_store,
                    self.binding.resident,
                    plan.loads,
                    self.binding.device,
                )
            # Publish only after every weight/scale copy completed successfully.
            self.binding.residency.commit(plan)
            self.mapping.update(self.binding.residency)
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
