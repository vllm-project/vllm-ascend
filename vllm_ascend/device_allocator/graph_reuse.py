# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Conservative, per-worker certificates for sleepable MRV2 ACL graphs."""

from collections.abc import Callable
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_world_group
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.compilation import acl_graph
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device_allocator.camem import ACL_GRAPH_TAG, CaMemAllocator


def reuse_unsupported_reason(config: VllmConfig, runner: Any) -> str | None:
    """A support boundary, rather than a claim about other model combinations."""
    if not get_current_hardware_profile().supports(HardwareCapability.SLEEP_GRAPH_REUSE):
        return "hardware_not_validated"
    if not config.use_v2_model_runner:
        return "requires_mrv2"
    if config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        return "requires_full_decode_only"
    if any(size not in (1, 2, 4, 8) for size in config.compilation_config.cudagraph_capture_sizes or ()):
        return "capture_sizes_not_validated"
    model = config.model_config
    if model.dtype != torch.bfloat16:
        return "requires_bfloat16"
    if getattr(model, "quantization", None) is not None:
        return "quantization_not_validated"
    if model.hf_text_config.model_type not in ("qwen2", "qwen3", "qwen3_moe"):
        return "model_not_validated"
    if model.is_multimodal_model or model.use_mla:
        return "attention_layout_not_validated"
    if config.speculative_config is not None or config.lora_config is not None:
        return "speculation_or_lora_not_validated"
    parallel = config.parallel_config
    if parallel.enable_expert_parallel:
        return "expert_parallel_not_validated"
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.tensor_parallel_size not in (1, 2)
    ):
        return "parallel_configuration_not_validated"
    if config.kv_transfer_config is not None:
        return "kv_transfer_not_validated"
    if getattr(config, "weight_transfer_config", None) is not None:
        return "weight_transfer_not_validated"
    cache = getattr(config, "cache_config", None)
    if cache is not None and cache.cache_dtype not in ("auto", "bfloat16"):
        return "kv_dtype_not_validated"
    offload = getattr(config, "offload_config", None)
    if offload is not None and (offload.uva.cpu_offload_gb or offload.prefetch.offload_group_size):
        return "weight_offload_not_validated"
    ascend = get_ascend_config()
    if ascend.eplb_config.dynamic_eplb or ascend.ascend_compilation_config.enable_super_kernel:
        return "dynamic_eplb_or_super_kernel_not_validated"
    if runner is not None:
        backends = {group.backend.__module__ for groups in runner.attn_groups for group in groups}
        if backends != {"vllm_ascend.attention.attention_v1"}:
            return "attention_backend_not_validated"
    return None


def _tensor_records(root: Any, prefix: str) -> tuple:
    """Read layouts/ownership only. Weight values and tensor versions may change."""
    records = []
    seen: set[int] = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if value is None or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type == "npu":
                records.append(
                    (
                        path,
                        id(value),
                        value.data_ptr(),
                        value.storage_offset(),
                        tuple(value.shape),
                        tuple(value.stride()),
                        str(value.dtype),
                    )
                )
            return
        if depth == 0 or isinstance(value, (str, bytes, int, float, type, torch.nn.Module)) or callable(value):
            return
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda item: repr(item[0]))
        elif isinstance(value, (list, tuple)):
            items = enumerate(value)
        elif hasattr(value, "__dict__"):
            items = sorted((k, v) for k, v in vars(value).items() if not k.startswith("_"))
        else:
            return
        for key, child in items:
            visit(child, f"{path}.{key}", depth - 1)

    visit(root, prefix, 6)
    return tuple(records)


def graph_resource_signature(config: VllmConfig, runner: Any, groups: tuple, owners: list | None = None) -> tuple:
    manager = runner.cudagraph_manager
    if owners is not None:
        # Prevent a destroyed/recreated Python communicator from recycling its
        # old id and passing the compatibility comparison by coincidence.
        owners.extend(groups)
        owners.extend(group.device_group for group in groups)
    graphs = []
    for desc, graph in sorted(manager.graphs.items(), key=lambda item: repr(item[0])):
        if owners is not None:
            owners.extend(item for task in getattr(graph, "tasks", ()) for item in (task.handle, task.event))
        tasks = tuple(
            (id(task.handle), id(task.event), _tensor_records(getattr(task, "kwargs", None), "task_kwargs"))
            for task in getattr(graph, "tasks", ())
        )
        graphs.append(
            (repr(desc), id(graph), tasks, _tensor_records(getattr(graph, "capture_resources", {}), "resources"))
        )
    tensors = _tensor_records(dict(runner.model.named_parameters()), "parameters")
    tensors += _tensor_records(dict(runner.model.named_buffers()), "buffers")
    for name in ("input_buffers", "block_tables", "kv_caches", "model_state"):
        tensors += _tensor_records(getattr(runner, name, None), name)
    tensors += _tensor_records((manager.hidden_states, manager.aux_hidden_states), "outputs")
    communicators = tuple(sorted((id(group), id(group.device_group)) for group in groups))
    allocator = CaMemAllocator.get_instance()
    pool = allocator.allocator_and_pools.get(ACL_GRAPH_TAG)
    pool_id = pool[0].id if pool is not None else None
    allocations = tuple(
        sorted((ptr, data.tag, data.handle[1], data.handle[3]) for ptr, data in allocator.pointer_to_data.items())
    )
    backends = tuple(
        group.backend.__module__ + "." + group.backend.__qualname__ for groups in runner.attn_groups for group in groups
    )
    params = acl_graph.get_graph_params()
    legacy_tensors = _tensor_records(params, "attention_graph_params")
    legacy_handles: tuple = ()
    if params is not None:
        if owners is not None:
            owners.extend(
                item
                for name in ("events", "handles")
                for values in getattr(params, name, {}).values()
                for item in values
            )
        legacy_handles = tuple(
            (name, repr(size), tuple(id(item) for item in values))
            for name, mapping in sorted(vars(params).items())
            if isinstance(mapping, dict)
            for size, values in sorted(mapping.items(), key=lambda item: repr(item[0]))
            if isinstance(values, list)
        )
    return (
        id(runner),
        id(runner.model),
        id(manager),
        str(config.compilation_config.cudagraph_mode),
        str(getattr(config.compilation_config, "backend", None)),
        getattr(get_ascend_config().ascend_compilation_config, "enable_npugraph_ex", None),
        tuple(config.compilation_config.cudagraph_capture_sizes or ()),
        tuple(manager.capture_sizes),
        str(config.model_config.dtype),
        config.parallel_config.tensor_parallel_size,
        config.parallel_config.enable_expert_parallel,
        tuple(graphs),
        tensors,
        communicators,
        pool_id,
        allocations,
        backends,
        legacy_tensors,
        legacy_handles,
    )


class SleepGraphReuse:
    """Own the sleep transaction; never expose a partly restored graph for replay."""

    def __init__(self, config: VllmConfig, runner_getter: Callable[[], Any]):
        self.config = config
        self.runner_getter = runner_getter
        self.state = "READY"
        self.reusing = False
        self.pending_tags: set[str] = set()
        self.signature: tuple | None = None
        self.resource_owners: tuple = ()
        self.level: int | None = None
        self.reason: str | None = "not_slept"
        self.cycles = 0
        self.reuses = 0
        self.recaptures = 0

    @staticmethod
    def unanimous(local: bool) -> bool:
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return local
        world = get_world_group()
        votes = [None] * torch.distributed.get_world_size(group=world.cpu_group)
        torch.distributed.all_gather_object(votes, local, group=world.cpu_group)
        return all(vote is True for vote in votes)

    def allow_sleep(self, level: int) -> bool:
        self.require_healthy()
        if self.state == "ASLEEP" and self.level == level:
            return False
        if self.state != "READY":
            raise RuntimeError(f"Cannot sleep from graph-reuse state {self.state}")
        return True

    def require_healthy(self) -> None:
        if self.state == "CLOSED":
            raise RuntimeError("Sleep graph worker is closed; restart the worker before inference")
        if self.state == "ERROR":
            raise RuntimeError("Sleep graph recovery failed; restart the worker before inference")

    def prepare_sleep(self, level: int, groups: tuple) -> bool:
        runner = self.runner_getter()
        signature = None
        owners: list = []
        try:
            self.reason = reuse_unsupported_reason(self.config, runner)
            manager = getattr(runner, "cudagraph_manager", None)
            if self.reason is None and not getattr(manager, "sleep_reuse_capture_ready", False):
                self.reason = "graphs_not_captured_in_sleep_pool"
            if self.reason is None:
                signature = graph_resource_signature(self.config, runner, groups, owners)
        except Exception:
            logger.exception("Could not certify ACL graph resources; using recapture.")
            self.reason = "resource_inventory_failed"
        self.reusing = self.unanimous(self.reason is None)
        if self.reusing:
            self.reason = "eligible"
            self.signature = signature
            self.resource_owners = tuple(owners)
        else:
            self.reason = self.reason or "peer_not_reusable"
            self.signature = None
            self.resource_owners = ()
        self.level = level
        self.pending_tags = {"weights", "kv_cache"}
        self.state = "ASLEEP"
        self.cycles += 1
        logger.info(
            "Sleep graph decision=%s reason=%s cycle=%d",
            "reuse" if self.reusing else "recapture",
            self.reason,
            self.cycles,
        )
        return self.reusing

    def complete_after(self, tags: list[str] | None) -> bool:
        return tags is None or not self.pending_tags.difference(tags)

    def note_restored(self, tags: list[str] | None) -> None:
        if tags is None:
            self.pending_tags.clear()
        else:
            self.pending_tags.difference_update(tags)
        self.state = "PARTIAL" if self.pending_tags else "RESTORING"

    def can_replay(self, groups: tuple) -> bool:
        valid = self.reusing
        try:
            valid = valid and reuse_unsupported_reason(self.config, self.runner_getter()) is None
            valid = valid and self.signature == graph_resource_signature(self.config, self.runner_getter(), groups)
        except Exception:
            valid = False
        valid = self.unanimous(valid)
        if not valid and self.reason == "eligible":
            self.reason = "resource_signature_changed"
        return valid

    def finish(self, reused: bool) -> None:
        if reused:
            self.reuses += 1
        else:
            self.recaptures += 1
        self.state = "READY"
        self.signature = None
        self.resource_owners = ()
        logger.info(
            "Sleep graph wake action=%s reason=%s reuses=%d recaptures=%d",
            "reuse" if reused else "recapture",
            self.reason,
            self.reuses,
            self.recaptures,
        )
