from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from vllm.compilation import breakable_cudagraph
from vllm.logger import logger
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    weak_ref_workspaces,
)
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.models.deepseek_v41.cache_config import (
    LayerPackedKVCacheConfig,
    dummy_ring_pages_per_group,
)
from vllm_ascend.utils import super_kernel_scope, weak_ref_tensor, weak_ref_tensors
from vllm_ascend.worker.utils import copy_kv_cache_blocks_inplace
from vllm_ascend.worker.v2.attn_utils import ring_state_update_skipped


class AscendV2KVBlockZeroer(KVBlockZeroer):
    """Zero V2 caches whose forward-context binding is a tensor tuple/list.

    Upstream's V2 zeroer only walks ``torch.Tensor`` bindings. Ascend attention
    caches can expose ``(key_cache, value_cache)`` (and component-major MLA)
    as multiple logical views. Build one upstream zeroer per component; this
    also keeps views sharing one backing storage from being skipped.
    """

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable[AttentionGroup],
        kernel_block_sizes: list[int],
        static_forward_context: dict[str, Any],
        num_blocks: int,
        runner_only_attn_layers: set[str] | None = None,
        cache_dtype: str | None = None,
        excluded_cache_components: set[tuple[str, int]] | None = None,
    ) -> None:
        # The upstream constructor changed between v0.28 and main. Set the two
        # fields consumed by this subclass directly, then delegate each logical
        # component to the version-specific upstream constructor below.
        self.device = device
        self._meta = None
        runner_only_attn_layers = runner_only_attn_layers or set()
        excluded_cache_components = excluded_cache_components or set()
        groups = list(attn_groups_iter)
        # The 4th component of an mxfp8 tuple is the static checkpoint V
        # scale: clearing a recycled block must preserve it.
        layer_caches: dict[str, Any] = {}
        for group in groups:
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_cache = static_forward_context[layer_name].kv_cache
                if cache_dtype == "mxfp8" and isinstance(kv_cache, tuple) and len(kv_cache) == 4:
                    kv_cache = kv_cache[:3]
                layer_caches[layer_name] = kv_cache
        component_count = 1
        for kv_cache in layer_caches.values():
            if isinstance(kv_cache, (tuple, list)):
                component_count = max(component_count, len(kv_cache))

        self._zeroers: list[KVBlockZeroer] = []
        for component_id in range(component_count):
            component_groups: list[AttentionGroup] = []
            component_context: dict[str, Any] = {}
            for group in groups:
                layer_names: list[str] = []
                for layer_name in group.layer_names:
                    if layer_name in runner_only_attn_layers:
                        continue
                    # Offload TopK views are query-row scratch, not paged KV.
                    if (layer_name, component_id) in excluded_cache_components:
                        continue
                    kv_cache = layer_caches[layer_name]
                    component = None
                    if isinstance(kv_cache, torch.Tensor) and component_id == 0:
                        component = kv_cache
                    elif isinstance(kv_cache, (tuple, list)) and component_id < len(kv_cache):
                        component = kv_cache[component_id]
                    # Ascend Block Zeroer unpacks tuple/list KV caches; skip empty views to avoid zero-sized launches.
                    if isinstance(component, torch.Tensor) and component.numel() > 0:
                        layer_names.append(layer_name)
                        component_context[layer_name] = SimpleNamespace(kv_cache=component)
                if layer_names:
                    component_groups.append(replace(group, layer_names=layer_names))

            if not component_groups:
                continue
            self._zeroers.append(
                KVBlockZeroer(
                    device,
                    attn_groups_iter=component_groups,
                    kernel_block_sizes=kernel_block_sizes,
                    static_forward_context=component_context,
                    num_blocks=num_blocks,
                    runner_only_attn_layers=runner_only_attn_layers,
                )
            )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        for zeroer in self._zeroers:
            zeroer.zero_block_ids(block_ids)

    def warmup(self, num_kv_blocks: int) -> None:
        for zeroer in self._zeroers:
            zeroer.warmup(num_kv_blocks)


class LayerPackedKVBlockOps(KVBlockZeroer):
    """Clear and copy both physical slots for each shared block ID."""

    def __init__(
        self, config: LayerPackedKVCacheConfig, attn_groups_iter, kernel_block_sizes, static_forward_context, device
    ):
        self.num_blocks = config.num_blocks
        self.pages: list[torch.Tensor] = []
        groups = []
        context = {}
        group_by_name = {name: group for group in attn_groups_iter for name in group.layer_names}
        placements = {desc.layers[0]: desc for desc in config.kv_cache_tensors}
        for name in config.page_owners:
            # The owner starts at byte zero of its slot, including all aliased components.
            owner = static_forward_context[name].kv_cache
            width = placements[name].block_stride // owner.element_size()
            page = owner.as_strided((config.num_blocks, width), (width, 1)).view(torch.uint8)
            self.pages.append(page)
            context[name] = SimpleNamespace(kv_cache=page)
            groups.append(replace(group_by_name[name], layer_names=[name], kernel_block_stride=0))
        super().__init__(device, groups, kernel_block_sizes, context, config.num_blocks)

    def copy_blocks(self, copies: list[KVCacheBlockCopy]) -> None:
        copy_kv_cache_blocks_inplace(self.pages, self.num_blocks, copies)


class C8KVBlockZeroer:
    """Zero recycled blocks via a delegate, then restore static C8 V scales.

    A decorator around any KVBlockZeroer (or None): the delegate clears the
    dynamic components, this class restores the checkpoint V-scale constants
    that hybrid page reuse may have overwritten. Composition instead of
    inheritance keeps the delegate swappable (re-wrapped on reinitialization,
    unwrapped on shutdown) and optional (restore-only when no zeroer exists).
    """

    def __init__(self, delegate: Any, entries: list[tuple[Any, Any, int]], fill: Callable) -> None:
        self.delegate = delegate
        self.entries = entries
        self.fill = fill

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def warmup(self, num_kv_blocks: int) -> None:
        # The upstream warmup path forwards to the delegate when present.
        warmup = getattr(self.delegate, "warmup", None)
        if warmup is not None:
            warmup(num_kv_blocks)

    def zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        # Validate the whole batch before any device write.
        for _, cache, ratio in self.entries:
            num_blocks = cache.shape[0] // ratio
            for block_id in block_ids:
                if block_id < 0 or block_id >= num_blocks:
                    raise IndexError(f"C8 physical block ID out of range: {block_id}")
        if self.delegate is not None:
            self.delegate.zero_block_ids(block_ids)
        for value_scale, value_scale_cache, ratio in self.entries:
            for block_id in block_ids:
                start = block_id * ratio
                self.fill(value_scale, value_scale_cache[start : start + ratio])


if TYPE_CHECKING:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@contextmanager
def torch_cuda_wrapper():
    try:
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle
        torch.cuda.CUDAGraph = UpdatableGraph
        torch.cuda.graph = torch_npu_graph_wrapper
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.set_stream = torch.npu.set_stream
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.mem_get_info = torch.npu.mem_get_info
        # GPU V2 prefetch/offload calls this CUDA API during load_model.
        torch.cuda.is_current_stream_capturing = torch.npu.is_current_stream_capturing
        breakable_cudagraph.weak_ref_tensor = weak_ref_tensor
        breakable_cudagraph.weak_ref_tensors = weak_ref_tensors
        logger.info_once("Wrapping torch.cuda with torch.npu.")
        yield
    finally:
        pass


@contextmanager
def communicator_switch():
    import vllm.distributed.device_communicators.cuda_communicator

    from vllm_ascend.distributed.device_communicators.npu_communicator import NPUCommunicator

    CudaCommunicator = vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator
    vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = NPUCommunicator
    logger.debug("Switched CudaCommunicator -> NPUCommunicator for graph capture.")

    try:
        yield
    finally:
        vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator = CudaCommunicator
        logger.debug("Restored CudaCommunicator after graph capture.")


@contextmanager
def torch_npu_graph_wrapper(*args, **kwargs):
    # MRV2-specific cleanup hook: intentionally reuse the graph context
    # manager's exit to weak-ref graph workspaces after each capture,
    # without adding another upstream monkey patch.
    enable_super_kernel = get_ascend_config().ascend_compilation_config.enable_super_kernel
    graph = args[0] if args else (kwargs.get("cuda_graph") or kwargs.get("graph"))
    try:
        with torch.npu.graph(*args, **kwargs), super_kernel_scope("full_model", enable_super_kernel):
            yield
        if enable_super_kernel and graph is not None:
            graph.super_kernel_optimize(
                optimize_options={
                    "dcci_after_kernel_end": [".*"],
                },
            )
            logger.info_once("Super kernel optimization is enabled for ACL graph capture.")
    finally:
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())


def prepare_v41_source_rope(runner: "NPUModelRunner") -> None:
    """Prepare V4.1 RoPE and stable PCP buffer owners before capture."""
    # Lazy imports avoid the model/cache registration cycle.
    from vllm_ascend.attention.context_parallel.dsa_v41_cp import AscendDSAV41PCPMetadataBuilder
    from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder

    pcp_builders = []
    for groups in runner.attn_groups:
        for attn_group in groups:
            for builder in attn_group.metadata_builders:
                if isinstance(builder, AscendDSAV41MetadataBuilder):
                    builder.prepare_source_rope()
                if isinstance(builder, AscendDSAV41PCPMetadataBuilder):
                    pcp_builders.append(builder)
    AscendDSAV41PCPMetadataBuilder.prepare_shared_buffers(pcp_builders)


def prepare_v41_dummy_ring_state(runner: "NPUModelRunner", num_reqs: int) -> None:
    """Assign live ring pages to dummy requests for V4.1 graph runs."""
    if ring_state_update_skipped():
        return
    config = runner.kv_cache_config
    shared_groups = getattr(runner.block_tables, "v41_shared_ring_groups", ())
    for gid, group in enumerate(config.kv_cache_groups):
        if not is_circular_kv_cache_spec(group.kv_cache_spec):
            continue
        if shared_groups:
            live = dummy_ring_pages_per_group(config.num_blocks, num_reqs, len(shared_groups))
            stop = config.num_blocks - shared_groups.index(gid) * live
            start = stop - live
        else:
            live = num_reqs
            start, stop = 1, 1 + live
            if stop > config.num_blocks:
                raise ValueError("Insufficient ring pages for dummy graph requests")
        table = runner.block_tables.input_block_tables[gid]
        table[:live, 0] = torch.arange(start, stop, dtype=table.dtype, device=table.device)
        # C2 masks reads and writes to block zero. Clear stale IDs for excess rows.
        if live < num_reqs:
            table[live:num_reqs, 0].zero_()
        for name in group.layer_names:
            cache = runner.compilation_config.static_forward_context[name].kv_cache
            # V4.1 binds the ring tensor directly; GLM-Next KPool binds a list.
            ring = cache if isinstance(cache, torch.Tensor) else cache[0]
            ring[start:stop].zero_()
