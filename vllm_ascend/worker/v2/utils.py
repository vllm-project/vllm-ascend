from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch
from vllm.compilation import breakable_cudagraph
from vllm.logger import logger
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.compilation.acl_graph import get_draft_graph_params, get_graph_params, weak_ref_workspaces
from vllm_ascend.utils import vllm_version_is, weak_ref_tensor, weak_ref_tensors


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
    ) -> None:
        # The upstream constructor changed between v0.28 and main. Set the two
        # fields consumed by this subclass directly, then delegate each logical
        # component to the version-specific upstream constructor below.
        self.device = device
        self._meta = None
        runner_only_attn_layers = runner_only_attn_layers or set()
        groups = list(attn_groups_iter)
        component_count = 1
        for group in groups:
            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_cache = static_forward_context[layer_name].kv_cache
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
                    kv_cache = static_forward_context[layer_name].kv_cache
                    component = None
                    if isinstance(kv_cache, torch.Tensor) and component_id == 0:
                        component = kv_cache
                    elif isinstance(kv_cache, (tuple, list)) and component_id < len(kv_cache):
                        component = kv_cache[component_id]
                    if component is not None:
                        layer_names.append(layer_name)
                        component_context[layer_name] = SimpleNamespace(kv_cache=component)
                if layer_names:
                    component_groups.append(replace(group, layer_names=layer_names))

            if not component_groups:
                continue
            zeroer_kwargs: dict[str, Any]
            if vllm_version_is("0.28.0"):
                zeroer_kwargs = {
                    "cache_dtype": cache_dtype,
                    "static_forward_context": component_context,
                    "runner_only_attn_layers": runner_only_attn_layers,
                }
            else:
                zeroer_kwargs = {
                    "static_forward_context": component_context,
                    "num_blocks": num_blocks,
                    "runner_only_attn_layers": runner_only_attn_layers,
                }
            self._zeroers.append(
                KVBlockZeroer(
                    device,
                    attn_groups_iter=component_groups,
                    kernel_block_sizes=kernel_block_sizes,
                    **zeroer_kwargs,
                )
            )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        for zeroer in self._zeroers:
            zeroer.zero_block_ids(block_ids)

    def warmup(self, num_kv_blocks: int) -> None:
        for zeroer in self._zeroers:
            zeroer.warmup(num_kv_blocks)


@contextmanager
def torch_cuda_wrapper():
    try:
        torch.cuda.Event = torch.npu.Event
        torch.cuda.Stream = torch.npu.Stream
        torch.cuda.stream = torch.npu.stream
        torch.cuda.default_stream = torch.npu.default_stream
        torch.cuda.current_stream = torch.npu.current_stream
        torch.cuda.graph_pool_handle = torch.npu.graph_pool_handle
        torch.cuda.CUDAGraph = torch.npu.NPUGraph
        torch.cuda.graph = torch_npu_graph_wrapper
        torch.cuda.synchronize = torch.npu.synchronize
        torch.cuda.set_stream = torch.npu.set_stream
        torch.cuda.current_device = torch.npu.current_device
        torch.cuda.mem_get_info = torch.npu.mem_get_info
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
    try:
        with torch.npu.graph(*args, **kwargs):
            yield
    finally:
        weak_ref_workspaces(get_graph_params())
        weak_ref_workspaces(get_draft_graph_params())
