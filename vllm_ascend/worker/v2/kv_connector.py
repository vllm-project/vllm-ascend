# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

import torch
from vllm.v1.worker.gpu.kv_connector import KVConnector

from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput


def kv_connector_tree(connector: Any) -> Iterator[Any]:
    """Include connectors nested in a vLLM MultiConnector."""
    yield connector
    for child in getattr(connector, "sub_connectors", ()):
        yield from kv_connector_tree(child)


class MambaCopyKVConnector(KVConnector):
    """Keep Mamba bulk copies between synchronous KV loads and forward.

    Wrap only this runner's connector. In particular, do not mutate the
    upstream class or the shared no-op connector used during profiling.
    """

    def __init__(self, connector: KVConnector, copy_mamba_state: Callable[[], None]) -> None:
        self._connector = connector
        self._copy_mamba_state = copy_mamba_state

    def pre_forward(self, scheduler_output: "SchedulerOutput", *args: Any, **kwargs: Any) -> None:
        self._connector.pre_forward(scheduler_output, *args, **kwargs)
        self._copy_mamba_state()

    def finish_forward(self) -> None:
        self._connector.finish_forward()

    def finalize_step(self) -> None:
        self._connector.finalize_step()

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> "KVConnectorOutput | None":
        return self._connector.post_forward(finished_req_ids, *args, **kwargs)

    def no_forward(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        # No state preprocessing or model execution on this path.
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._connector.reset_capture_state()


class C8VScaleKVConnector(KVConnector):
    """Restore static C8 V scales after Pull admission and Mamba copies.

    Async Pull requests reach this hook only after their receive has completed.
    Wrapping the runner's connector keeps the restore before eager execution
    and FULL graph replay without modifying vLLM's shared connector class.
    """

    def __init__(
        self,
        connector: KVConnector,
        kv_cache_config: "KVCacheConfig",
        static_forward_context: dict[str, Any],
    ) -> None:
        self._connector = connector
        self._entries: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
        num_blocks = kv_cache_config.num_blocks
        if num_blocks <= 0:
            return
        group_by_layer = {
            layer_name: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        for layer_name, layer in static_forward_context.items():
            if not isinstance(getattr(layer, "impl", None), AscendC8MXFPAttentionBackendImpl):
                continue
            group_id = group_by_layer.get(layer_name)
            if group_id is None:
                continue
            cache = layer.kv_cache[3]
            assert cache.shape[0] % num_blocks == 0
            self._entries.append((layer.v_cache_scale, cache, cache.shape[0] // num_blocks, group_id))

    def pre_forward(self, scheduler_output: "SchedulerOutput", *args: Any, **kwargs: Any) -> None:
        self._connector.pre_forward(scheduler_output, *args, **kwargs)
        self._restore(scheduler_output)

    def _restore(self, scheduler_output: "SchedulerOutput") -> None:
        """Restore untransferred V-scale tails on newly admitted Pull pages."""
        if scheduler_output.has_sync_kv_loads or not self._entries:
            return
        blocks_by_group: dict[int, set[int]] = {}
        for request in scheduler_output.scheduled_new_reqs:
            for group_id, blocks in enumerate(getattr(request, "block_ids", None) or ()):
                blocks_by_group.setdefault(group_id, set()).update(
                    block_id for block_id in (blocks or ()) if block_id is not None
                )
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, block_ids in zip(cached.req_ids, cached.new_block_ids or ()):
            if req_id in cached.resumed_req_ids and block_ids is not None:
                for group_id, blocks in enumerate(block_ids):
                    blocks_by_group.setdefault(group_id, set()).update(
                        block_id for block_id in (blocks or ()) if block_id is not None
                    )
        if not blocks_by_group:
            return
        zeroed = set(scheduler_output.new_block_ids_to_zero or ())
        page_indices_by_layout: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        for value_scale, cache, ratio, group_id in self._entries:
            block_ids = sorted(block_id for block_id in blocks_by_group.get(group_id, set()) - zeroed if block_id > 0)
            if not block_ids:
                continue
            if (block_ids[-1] + 1) * ratio > cache.shape[0]:
                raise IndexError(f"C8 physical block ID out of range: {block_ids[-1]}")
            layout = (group_id, ratio, cache.device)
            page_indices = page_indices_by_layout.get(layout)
            if page_indices is None:
                page_indices = torch.tensor(
                    [block_id * ratio + offset for block_id in block_ids for offset in range(ratio)],
                    dtype=torch.long,
                    device=cache.device,
                )
                page_indices_by_layout[layout] = page_indices
            source = value_scale.view(1, cache.shape[1], cache.shape[2], 1, cache.shape[4], 1)
            cache[page_indices] = source.expand(
                page_indices.numel(), cache.shape[1], cache.shape[2], cache.shape[3], cache.shape[4], cache.shape[5]
            )

    def finish_forward(self) -> None:
        self._connector.finish_forward()

    def finalize_step(self) -> None:
        self._connector.finalize_step()

    def post_forward(self, finished_req_ids: set[str], *args: Any, **kwargs: Any) -> "KVConnectorOutput | None":
        return self._connector.post_forward(finished_req_ids, *args, **kwargs)

    def no_forward(self, scheduler_output: "SchedulerOutput") -> "ModelRunnerOutput":
        return self._connector.no_forward(scheduler_output)

    def set_disabled(self, disabled: bool) -> None:
        self._connector.set_disabled(disabled)

    def reset_capture_state(self) -> None:
        self._connector.reset_capture_state()

    def release_cache_views(self) -> KVConnector:
        self._entries.clear()
        return self._connector
