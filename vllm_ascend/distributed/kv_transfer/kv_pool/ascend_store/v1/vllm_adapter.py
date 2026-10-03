"""Translate vLLM-owned state into AscendStore v1 domain facts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import vllm.v1.core.kv_cache_utils as kv_cache_utils
from vllm.distributed import get_dcp_group, get_pcp_group, get_pp_group, get_tp_group
from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_cache_spec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    KeyMetadata,
    infer_group_cache_families,
    infer_tp_mismatch_info,
    uses_hybrid_kv_cache,
)

from ..backend import get_layerwise_data_plane, get_layerwise_protocol
from .backend import resolve_backend_spec
from .planning.availability import RemoteAvailabilityProbe
from .planning.planner import TransferPlanner
from .planning.progress import AllocationLoadPublication, LoadPublication, ScheduledLoadPublication
from .planning.spec import TransferPlanningSpec
from .planning.step import (
    ScheduledRequest,
    ScheduledRequestKind,
    StateCheckpointHandoff,
    TransferPlanningStep,
)
from .program.compiler import compile_kv_pool_program
from .program.spec.compilation import KVPoolCompilationSpec
from .program.spec.schedule import KVPoolSchedule, LoadScheduleKind, StoreScheduleKind
from .program.spec.topology import (
    KVPoolGroupTopology,
    KVPoolTopology,
    TPPartitionSpec,
    kv_cache_spec_uses_align_state,
    resolve_group_layers,
)
from .protocol.transfer import StateCheckpointSource
from .rules import RuleBinder, compile_kv_pool_rules
from .runtime.resources import KVPoolResources
from .runtime.runtime import KVPoolRuntime

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


def create_transfer_planner(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
    lookup_address: str,
) -> TransferPlanner:
    """Bind vLLM configuration to the Scheduler-side transfer planner."""

    spec = resolve_transfer_planning_spec(vllm_config, kv_cache_config)
    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    load_publication: LoadPublication
    if extra_config.get("load_async", False) and not extra_config.get("use_layerwise", False):
        load_publication = AllocationLoadPublication()
    else:
        load_publication = ScheduledLoadPublication()
    availability_probe = RemoteAvailabilityProbe(
        lookup_address,
        group_ids=spec.transfer_group_ids,
        transfer_granularity=spec.cache_transfer_granularity,
        discard_partial_chunks=spec.discard_partial_chunks,
        enabled=transfer_config.kv_role != "kv_consumer" or extra_config.get("consumer_is_to_load", False),
    )
    store_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or extra_config.get(
        "consumer_is_to_put", False
    )
    return TransferPlanner(
        spec,
        availability_probe,
        load_publication,
        store_enabled=store_enabled,
        save_decode_cache=extra_config.get("save_decode_cache", False),
    )


def resolve_transfer_planning_spec(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> TransferPlanningSpec:
    """Capture the static vLLM facts used by Scheduler-side planning."""

    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    return TransferPlanningSpec(
        cache_transfer_granularity,
        hash_block_size,
        tuple(kv_cache_config.transfer_group_ids),
        extra_config.get("discard_partial_chunks", True),
    )


def adapt_scheduler_output(
    scheduler_output: SchedulerOutput,
    requests: Mapping[str, Request],
    *,
    store_enabled: bool,
) -> TransferPlanningStep:
    """Snapshot one vLLM Scheduler output into planner-owned request facts."""

    scheduled_requests = [
        _adapt_scheduled_request(
            requests,
            scheduled.req_id,
            ScheduledRequestKind.NEW,
            scheduled.block_ids,
            scheduled.num_computed_tokens,
            scheduler_output.num_scheduled_tokens[scheduled.req_id],
        )
        for scheduled in scheduler_output.scheduled_new_reqs
    ]
    if store_enabled:
        cached = scheduler_output.scheduled_cached_reqs
        scheduled_requests.extend(
            _adapt_scheduled_request(
                requests,
                request_id,
                ScheduledRequestKind.RESUMED if request_id in cached.resumed_req_ids else ScheduledRequestKind.RUNNING,
                cached.new_block_ids[index],
                cached.num_computed_tokens[index],
                scheduler_output.num_scheduled_tokens[request_id],
            )
            for index, request_id in enumerate(cached.req_ids)
        )
    return TransferPlanningStep(
        tuple(scheduled_requests),
        frozenset(scheduler_output.finished_req_ids),
        frozenset(scheduler_output.preempted_req_ids or ()),
        _adapt_checkpoint_handoffs(scheduler_output, requests) if store_enabled else (),
    )


def _adapt_scheduled_request(
    requests: Mapping[str, Request],
    request_id: str,
    kind: ScheduledRequestKind,
    block_ids_by_group: tuple[list[int], ...] | None,
    num_computed_tokens: int,
    num_scheduled_tokens: int,
) -> ScheduledRequest:
    request = requests.get(request_id)
    if request is None:
        raise ValueError(f"Scheduled request {request_id} has not passed allocation confirmation")
    return ScheduledRequest(
        request_id,
        kind,
        None if block_ids_by_group is None else tuple(tuple(block_ids) for block_ids in block_ids_by_group),
        tuple(request.block_hashes),
        request.num_prompt_tokens,
        request.num_tokens,
        num_computed_tokens,
        num_scheduled_tokens,
    )


def _adapt_checkpoint_handoffs(
    scheduler_output: SchedulerOutput,
    requests: Mapping[str, Request],
) -> tuple[StateCheckpointHandoff, ...]:
    block_state = scheduler_output.kv_connector_block_state
    if block_state is None:
        return ()
    handoffs = []
    for request_id, entries in block_state.boundary_state_offloads.items():
        request = requests.get(request_id)
        if request is None:
            continue
        handoffs.append(
            StateCheckpointHandoff(
                request_id,
                tuple(request.block_hashes),
                tuple(StateCheckpointSource(*entry) for entry in entries),
            )
        )
    return tuple(handoffs)


def create_kv_pool_runtime(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolRuntime:
    """Compile vLLM startup state and bind its process-owned Worker resources."""

    program = compile_kv_pool_program(resolve_kv_pool_compilation_spec(vllm_config, kv_cache_config))
    backend_spec = resolve_backend_spec(program.backend_name)
    backend = backend_spec.backend_type(
        vllm_config.parallel_config,
        extra_config=vllm_config.kv_transfer_config.kv_connector_extra_config,
    )
    resources = KVPoolResources(backend, backend_spec, kv_cache_config.num_blocks, program.topology.groups)
    try:
        return KVPoolRuntime(program, resources)
    except BaseException:
        resources.close()
        raise


def compile_kv_pool_rule_binder(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> RuleBinder:
    """Compile θ from vLLM state; the returned callable accepts registered μ."""

    spec = resolve_kv_pool_compilation_spec(vllm_config, kv_cache_config)
    if not spec.schedule.requires_layerwise_backend:
        return compile_kv_pool_rules(spec)

    protocol = get_layerwise_protocol(spec.backend_name)
    if protocol is None:
        raise ValueError(f"Backend {spec.backend_name!r} has no Layerwise key protocol")
    topology = spec.topology
    kv_cache_groups = kv_cache_config.kv_cache_groups
    data_plane = get_layerwise_data_plane(protocol)
    use_hybrid = uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_groups) or (
        data_plane == "block_key" and len(kv_cache_groups) > 1
    )
    model_name = topology.groups[0].key_metadata.model_name
    key_builder = protocol.bind_layerwise_keys(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        model_name=model_name,
        use_hybrid=use_hybrid,
        grouped_block_size=[group.block_size for group in topology.groups],
    )
    make_partial_key = getattr(protocol, "make_partial_key", None)
    bound_partial_key = None
    if callable(make_partial_key):

        def bound_partial_key(
            request_id: str,
            block_index: int,
            end_token: int,
            *,
            group_id: int,
            head_rank: int,
            pp_rank: int,
        ) -> str:
            return make_partial_key(
                model_name,
                request_id,
                group_id,
                block_index,
                end_token,
                head_rank,
                pp_rank,
                topology.pp_size,
            )

    return compile_kv_pool_rules(
        spec,
        layerwise_full_key=key_builder.make_full_key,
        layerwise_partial_key=bound_partial_key,
    )


def resolve_kv_pool_compilation_spec(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> KVPoolCompilationSpec:
    """Capture vLLM-owned process state before compiling the KV Pool program."""

    topology = _resolve_kv_pool_topology(vllm_config, kv_cache_config)
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    return KVPoolCompilationSpec(
        topology=topology,
        backend_name=extra_config.get("backend", "mooncake").strip().lower(),
        schedule=_resolve_schedule(vllm_config, extra_config),
        max_model_len=vllm_config.model_config.max_model_len,
        use_eagle=_uses_eagle_block_drop(vllm_config),
        retention_interval=kv_cache_config.prefix_cache_retention_interval,
    )


def resolve_consumer_pipeline_partitions(vllm_config: VllmConfig) -> tuple[int, ...] | None:
    """Resolve the producer PP layer ranges required by a consumer that stores KV."""

    transfer_config = vllm_config.kv_transfer_config
    extra_config = transfer_config.kv_connector_extra_config
    if transfer_config.kv_role != "kv_consumer" or not extra_config.get("consumer_is_to_put", False):
        return None

    num_hidden_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    prefill_pp_size = int(extra_config.get("prefill_pp_size", 1))
    if prefill_pp_size <= 0:
        raise ValueError(f"prefill_pp_size must be positive, received {prefill_pp_size}")

    partition_config = extra_config.get("prefill_pp_layer_partition")
    if partition_config is not None:
        return _parse_pipeline_partitions(partition_config, prefill_pp_size, num_hidden_layers)

    layers_per_partition, remaining_layers = divmod(num_hidden_layers, prefill_pp_size)
    partitions = [layers_per_partition] * prefill_pp_size
    for index in range(2, remaining_layers + 2):
        partitions[-index] += 1
    return tuple(partitions)


def _resolve_kv_pool_topology(vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> KVPoolTopology:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_rank = get_tp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
    pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
    dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
    dcp_rank = get_dcp_group().rank_in_group if dcp_size > 1 else 0
    num_kv_heads = 1 if getattr(model_config, "use_mla", False) else model_config.get_total_num_kv_heads()
    put_step = tp_size // num_kv_heads if num_kv_heads < tp_size else 1
    head_or_tp_rank = tp_rank // put_step
    cache_transfer_granularity, hash_block_size = kv_cache_utils.resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )
    model_name = model_config.model.rstrip("/").split("/")[-1]
    base_layer_count = model_config.get_total_num_hidden_layers()
    hf_text_config = getattr(model_config, "hf_text_config", None)
    hf_config = getattr(model_config, "hf_config", hf_text_config)
    cache_family_hf_config = hf_text_config or hf_config
    compress_ratios = getattr(hf_text_config, "compress_ratios", None)
    if compress_ratios is None:
        compress_ratios = getattr(hf_config, "compress_ratios", None)
    kv_cache_groups = kv_cache_config.kv_cache_groups
    group_cache_families = infer_group_cache_families(
        kv_cache_groups,
        compress_ratios,
        cache_family_hf_config,
    )
    groups = []
    for group_id, group in enumerate(kv_cache_groups):
        kv_cache_spec = resolve_dcp_kv_cache_spec(group.kv_cache_spec, dcp_size)
        uses_align_state = kv_cache_spec_uses_align_state(kv_cache_spec)
        key_tp_rank = tp_rank if uses_align_state else head_or_tp_rank
        groups.append(
            KVPoolGroupTopology(
                group_id=group_id,
                kv_cache_spec=kv_cache_spec,
                layers=resolve_group_layers(group.layer_names, base_layer_count),
                key_metadata=KeyMetadata(
                    model_name,
                    key_tp_rank,
                    dcp_rank,
                    pp_rank,
                    kv_cache_group_id=group_id,
                    cache_family=group_cache_families[group_id],
                ),
                is_eagle_group=group.is_eagle_group,
            )
        )
    return KVPoolTopology(
        tp_rank,
        tp_size,
        pp_size,
        pcp_rank,
        pcp_size,
        dcp_size,
        put_step,
        cache_transfer_granularity,
        hash_block_size,
        _resolve_tp_partition(vllm_config),
        tuple(groups),
        tuple(kv_cache_config.transfer_group_ids),
        resolve_consumer_pipeline_partitions(vllm_config),
    )


def _resolve_tp_partition(vllm_config: VllmConfig) -> TPPartitionSpec:
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    tp_size = parallel_config.tensor_parallel_size
    use_mla = getattr(model_config, "use_mla", False)
    num_kv_heads = 1 if use_mla else model_config.get_total_num_kv_heads()
    mismatch_info = infer_tp_mismatch_info(
        vllm_config.kv_transfer_config.kv_role,
        vllm_config.kv_transfer_config.kv_connector_extra_config,
        tp_size,
        num_kv_heads,
        use_mla,
    )
    key_rank_count = mismatch_info.effective_tp_size if mismatch_info.enabled else min(tp_size, num_kv_heads)
    return TPPartitionSpec(mismatch_info.enabled, key_rank_count, mismatch_info.num_sub_keys)


def _resolve_schedule(vllm_config: VllmConfig, extra_config: dict[str, Any]) -> KVPoolSchedule:
    use_layerwise = bool(extra_config.get("use_layerwise", False))
    if use_layerwise:
        load_kind = LoadScheduleKind.LAYERWISE
    elif extra_config.get("load_async", False):
        load_kind = LoadScheduleKind.ASYNC
    else:
        load_kind = LoadScheduleKind.SYNC

    transfer_config = vllm_config.kv_transfer_config
    store_enabled = transfer_config.kv_role in ("kv_producer", "kv_both") or extra_config.get(
        "consumer_is_to_put", False
    )
    store_kind = None
    if store_enabled:
        store_kind = StoreScheduleKind.LAYERWISE if use_layerwise else StoreScheduleKind.ASYNC
    return KVPoolSchedule(load_kind, store_kind, _resolve_layerwise_prefetch_layers(extra_config, use_layerwise))


def _resolve_layerwise_prefetch_layers(extra_config: dict[str, Any], use_layerwise: bool) -> int:
    default_prefetch_layers = 2
    if not use_layerwise:
        return default_prefetch_layers
    configured_prefetch_layers = extra_config.get("layerwise_prefetch_layers", default_prefetch_layers)
    if isinstance(configured_prefetch_layers, bool):
        raise ValueError("layerwise_prefetch_layers must be a positive integer")
    try:
        prefetch_layers = int(configured_prefetch_layers)
    except (TypeError, ValueError) as error:
        raise ValueError("layerwise_prefetch_layers must be a positive integer") from error
    if prefetch_layers <= 0:
        raise ValueError("layerwise_prefetch_layers must be a positive integer")
    return prefetch_layers


def _parse_pipeline_partitions(
    partition_config: str,
    prefill_pp_size: int,
    num_hidden_layers: int,
) -> tuple[int, ...]:
    try:
        partitions = tuple(int(layer_count) for layer_count in partition_config.split(","))
    except ValueError as error:
        raise ValueError(f"Invalid prefill PP layer partition: {partition_config}") from error
    if len(partitions) != prefill_pp_size:
        raise ValueError(f"Partition count {len(partitions)} does not match prefill_pp_size {prefill_pp_size}")
    if sum(partitions) != num_hidden_layers:
        raise ValueError(f"Partition layer count {sum(partitions)} does not match model layers {num_hidden_layers}")
    return partitions


def _uses_eagle_block_drop(vllm_config: VllmConfig) -> bool:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if speculative_config is None:
        return False
    use_eagle_block_drop = getattr(speculative_config, "use_eagle_block_drop", None)
    if callable(use_eagle_block_drop):
        return bool(use_eagle_block_drop())
    use_eagle = getattr(speculative_config, "use_eagle", None)
    return bool(use_eagle()) if callable(use_eagle) else False
