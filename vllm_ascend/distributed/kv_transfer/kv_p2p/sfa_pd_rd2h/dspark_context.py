# SPDX-License-Identifier: Apache-2.0
"""DSpark P-to-D draft-KV readiness and cache-group helpers."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer import get_kv_transfer_group

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.spec_decode.dspark_utils import get_dspark_aux_layer_ids

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MAX_DSPARK_CONTEXT_CHUNK_TOKENS = 64


def uses_sfa_dspark_kv_transfer(vllm_config: VllmConfig) -> bool:
    """Opt into this backend's draft-KV protocol, including MultiConnector.

    P deliberately disables sparse offload, so connector selection, not the
    local offload flag, identifies participants in the P-to-D protocol.
    """
    speculative = getattr(vllm_config, "speculative_config", None)
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    if speculative is None or speculative.method != "dspark" or transfer is None:
        return False
    if not (transfer.is_kv_producer or transfer.is_kv_consumer):
        return False
    pending = [(transfer.kv_connector, transfer.kv_connector_extra_config or {})]
    while pending:
        name, extra = pending.pop()
        if name == "SfaRemoteD2HConnector":
            return True
        if name == "MultiConnector":
            pending.extend(
                (child.get("kv_connector"), child.get("kv_connector_extra_config") or {})
                for child in extra.get("connectors", ())
            )
    return False


def get_pd_dspark_aux_layer_ids(vllm_config: VllmConfig) -> tuple[int, ...]:
    """Resolve P-side DSpark context boundaries before model loading.

    The P worker must load the DSpark drafter so it can write draft KV locally
    before transferring those pages to D. This is intentionally different from
    a target-only feature-capture path.
    """
    if not uses_sfa_dspark_kv_transfer(vllm_config):
        return ()
    transfer = vllm_config.kv_transfer_config
    if transfer.is_kv_consumer or not transfer.is_kv_producer:
        return ()
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size * parallel.decode_context_parallel_size != 1:
        raise ValueError("P-side DSpark draft KV generation does not support context parallelism.")
    model_config = vllm_config.model_config
    if not model_config.enforce_eager:
        raise ValueError("P-side DSpark draft KV generation currently requires eager prefill.")
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "P-side DSpark requires local prefix caching disabled; paired AscendStore prefix reuse is separate."
        )
    layer_ids = get_dspark_aux_layer_ids(vllm_config)
    if not layer_ids:
        raise ValueError("P-side DSpark draft KV generation requires auxiliary boundaries from the draft checkpoint.")
    return layer_ids


def resident_mla_context_group_ids(groups: Sequence[Any]) -> tuple[int, ...]:
    """Resolve residency before scheduler conversion drops per-layer specs.

    A uniform scheduler block group may contain host target and resident draft
    tensors. Sharing block IDs does not mean sharing physical KV storage.
    """
    result = []
    for group_id, group in enumerate(groups):
        spec = group.kv_cache_spec
        layer_specs = getattr(spec, "kv_cache_specs", None)
        specs = layer_specs.values() if layer_specs is not None else (spec,)
        if any(isinstance(item, AscendMLAAttentionSpec) and not item.store_on_host for item in specs):
            result.append(group_id)
    return tuple(result)


def find_dspark_context_connector(connector: Any, metadata: Any) -> tuple[Any, Any]:
    """Select the unique PD child and its metadata without patching MultiConnector.

    Upstream exposes child connectors but does not forward model-specific
    methods. Keep metadata paired with its child, including nested wrappers.
    """
    pending = [(connector, metadata)]
    matches = []
    while pending:
        current, current_meta = pending.pop()
        children = getattr(current, "_connectors", ())
        if children:
            child_metadata = getattr(current_meta, "metadata", ())
            if len(children) != len(child_metadata):
                raise RuntimeError("DSpark MultiConnector children and metadata are not aligned")
            pending.extend(zip(children, child_metadata))
        elif callable(getattr(current, "get_dspark_context_descriptor", None)) and callable(
            getattr(current, "send_dspark_draft_kv", None)
        ):
            matches.append((current, current_meta))
    if len(matches) != 1:
        raise RuntimeError("DSpark prefill requires exactly one PD context connector")
    return matches[0]


def find_dspark_kv_connector(connector: Any) -> Any:
    """Find the unique SFA PD child that owns DSpark KV pages."""
    pending = [connector]
    matches = []
    while pending:
        current = pending.pop()
        children = getattr(current, "_connectors", ())
        if children:
            pending.extend(children)
        elif callable(getattr(current, "configure_dspark_draft_layers", None)) and callable(
            getattr(current, "send_dspark_draft_kv", None)
        ):
            matches.append(current)
    if len(matches) != 1:
        raise RuntimeError("DSpark KV transfer requires exactly one SFA PD connector")
    return matches[0]


def find_dspark_prefix_connector(connector: Any, metadata: Any) -> tuple[Any, Any] | None:
    """Pair the optional external draft-prefix store with this step's metadata."""
    pending = [(connector, metadata)]
    matches = []
    while pending:
        current, current_meta = pending.pop()
        children = getattr(current, "_connectors", ())
        if children:
            child_metadata = getattr(current_meta, "metadata", ())
            if len(children) != len(child_metadata):
                raise RuntimeError("DSpark MultiConnector children and metadata are not aligned")
            pending.extend(zip(children, child_metadata))
        elif getattr(current, "dspark_prefix_cache_enabled", False):
            matches.append((current, current_meta))
    if len(matches) > 1:
        raise RuntimeError("DSpark prefix reuse requires exactly one paired target/draft store")
    return matches[0] if matches else None


def configure_dspark_kv_transfer(
    vllm_config: VllmConfig, speculator: Any, kv_cache_config: Any, *, is_last_pp_rank: bool
) -> None:
    """Register loaded draft ownership only for this backend's PD protocol."""
    if not is_last_pp_rank or not uses_sfa_dspark_kv_transfer(vllm_config):
        return
    if speculator is None:
        raise RuntimeError("DSpark KV transfer requires the loaded draft model on the final PP rank")
    names = tuple(sorted(speculator.draft_attn_layer_names))
    if not names:
        raise RuntimeError("The loaded DSpark drafter did not expose its KV cache layer names")
    connector = find_dspark_kv_connector(get_kv_transfer_group())
    kv_cache_config.dspark_draft_layer_names = names
    connector.configure_dspark_draft_layers(names)


def bind_dspark_context_receiver(
    vllm_config: VllmConfig, *, sparse_offload_enabled: bool, is_last_pp_rank: bool, max_requests: int
) -> None:
    """Bind D readiness after cache allocation; the connector owns its receiver."""
    if not sparse_offload_enabled or not is_last_pp_rank or not uses_sfa_dspark_kv_transfer(vllm_config):
        return
    if not vllm_config.kv_transfer_config.is_kv_consumer:
        return
    receiver = DSparkContextReceiver(max_requests=max_requests)
    connector = find_dspark_kv_connector(get_kv_transfer_group())
    connector.bind_dspark_context_receiver(receiver)


@dataclass(frozen=True)
class DSparkContextDescriptor:
    request_id: str
    generation: str
    prompt_tokens: int
    aux_layer_ids: tuple[int, ...]
    hidden_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("DSpark context requires a nonempty external request ID")
        if not isinstance(self.generation, str) or not self.generation:
            raise ValueError("DSpark context requires request identity and allocation generation")
        if type(self.prompt_tokens) is not int or self.prompt_tokens <= 0:
            raise ValueError("DSpark prompt_tokens must be a positive integer")
        if type(self.hidden_size) is not int or self.hidden_size <= 0:
            raise ValueError("DSpark hidden_size must be a positive integer")
        if (
            not isinstance(self.aux_layer_ids, tuple)
            or not self.aux_layer_ids
            or any(type(layer) is not int or layer < 0 for layer in self.aux_layer_ids)
            or tuple(sorted(set(self.aux_layer_ids))) != self.aux_layer_ids
        ):
            raise ValueError("DSpark auxiliary boundary IDs must be a nonempty ordered unique tuple")

    @property
    def feature_width(self) -> int:
        return self.hidden_size * len(self.aux_layer_ids)


@dataclass(frozen=True)
class DSparkContextChunk:
    descriptor: DSparkContextDescriptor
    token_offset: int
    num_tokens: int


def send_dspark_prefill_kv(
    speculator: Any,
    batch: Any,
    aux_hidden_states: Sequence[torch.Tensor],
    requests: Mapping[str, Any],
    connector: Any,
    progress: dict[str, tuple[str, int]],
    finished_req_ids: Sequence[str],
    *,
    prefix_connector: Any = None,
    prefix_metadata: Any = None,
) -> None:
    """Project each P prefill chunk locally, then transfer completed draft pages.

    Target layerwise scratch buffers may be reused after this call, while
    draft pages remain persistent until the remote consumer acknowledges them.
    Only prompt rows are projected; target verification rows are not context.
    """
    for request_id in finished_req_ids:
        progress.pop(request_id, None)
    if speculator is None or not hasattr(speculator, "initialize_local_context"):
        raise RuntimeError("P-side DSpark prompt KV generation requires the loaded DSpark speculator")
    features = torch.cat(aux_hidden_states, dim=-1)
    draft_group_ids, _, block_sizes = speculator.get_draft_context_group_layout()
    for req_idx in range(batch.num_reqs):
        request_id = batch.req_ids[req_idx]
        prompt_tokens = int(batch.prefill_len_np[req_idx])
        offset = int(batch.num_computed_tokens_np[req_idx])
        scheduled = int(batch.num_scheduled_tokens[req_idx])
        context_tokens = min(scheduled, max(prompt_tokens - offset, 0))
        if context_tokens == 0:
            continue
        descriptor = connector.get_dspark_context_descriptor(request_id, prompt_tokens)
        if descriptor is None:
            continue
        if descriptor.prompt_tokens != prompt_tokens:
            raise RuntimeError("P/D DSpark prompt lengths disagree for a remote request")
        if len(aux_hidden_states) != len(descriptor.aux_layer_ids):
            raise RuntimeError("P produced a different number of DSpark auxiliary states than the configured schema")
        if features.ndim != 2 or features.shape[1] != descriptor.feature_width or features.dtype != torch.bfloat16:
            raise RuntimeError("P DSpark context must be a [tokens, aux_layers * hidden_size] BF16 tensor")
        row_begin = int(batch.query_start_loc_np[req_idx])
        if row_begin + scheduled > features.shape[0]:
            raise RuntimeError("P DSpark auxiliary rows do not cover the scheduled prompt chunk")
        local_block_ids = requests[request_id].local_block_ids
        source_blocks_by_group = {}
        for group_id in draft_group_ids:
            if group_id >= len(local_block_ids):
                raise RuntimeError(f"P request is missing DSpark draft KV group {group_id}")
            block_size = block_sizes[group_id]
            # Chunked prefill allocates pages incrementally; require coverage
            # through this chunk's end, not the entire prompt on its first step.
            needed = (offset + context_tokens + block_size - 1) // block_size
            source_blocks_by_group[group_id] = tuple(local_block_ids[group_id][:needed])
            if len(source_blocks_by_group[group_id]) != needed:
                raise RuntimeError(f"P DSpark draft KV group {group_id} does not cover the prefill chunk")
        previous = progress.get(request_id)
        expected_offset = 0 if previous is None else previous[1]
        if previous is None and offset and prefix_connector is not None:
            expected_offset = prefix_connector.restore_dspark_prefix(
                prefix_metadata, request_id, offset, source_blocks_by_group
            )
        if previous is not None and previous[0] != descriptor.generation:
            raise RuntimeError("P DSpark allocation generation changed during prompt prefill")
        if offset != expected_offset:
            raise RuntimeError(
                f"P DSpark prefill chunks are not contiguous for {request_id}: "
                f"expected offset {expected_offset}, received {offset}"
            )
        for sent in range(0, context_tokens, MAX_DSPARK_CONTEXT_CHUNK_TOKENS):
            count = min(MAX_DSPARK_CONTEXT_CHUNK_TOKENS, context_tokens - sent)
            speculator.initialize_local_context(
                DSparkContextChunk(descriptor, offset + sent, count),
                features[row_begin + sent : row_begin + sent + count].contiguous(),
                source_blocks_by_group,
            )
        next_offset = offset + context_tokens
        if prefix_connector is not None:
            prefix_connector.save_dspark_prefix(prefix_metadata, request_id, next_offset, source_blocks_by_group)
        progress[request_id] = (descriptor.generation, next_offset)
        if next_offset == prompt_tokens:
            connector.send_dspark_draft_kv(request_id, descriptor, source_blocks_by_group)
            progress.pop(request_id, None)


class DSparkContextSubmission(Enum):
    ACCEPTED = auto()
    BACKPRESSURE = auto()
    STALE = auto()


@dataclass
class _ContextProgress:
    descriptor: DSparkContextDescriptor
    initialized_tokens: int = 0
    target_kv_done: bool = False
    in_flight: bool = False
    failed: bool = False


class DSparkContextReceiver:
    """Track direct P-to-D draft-KV transfer and its join with target KV.

    The read thread copies already-computed draft KV directly into D's resident
    pages. This receiver only gates admission/completion; it never stages target
    hidden states or projects them on D.
    """

    def __init__(self, *, max_requests: int) -> None:
        if type(max_requests) is not int or max_requests <= 0:
            raise ValueError("DSpark request capacity must be a positive integer")
        self.max_requests = max_requests
        self._owner_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._contexts: dict[str, _ContextProgress] = {}
        # Keep completed generations until the request is freed so a lost ACK
        # can be retried without copying the same pages twice.
        self._completed: dict[str, DSparkContextDescriptor] = {}
        self._retired: OrderedDict[tuple[str, str], DSparkContextDescriptor] = OrderedDict()
        self._retired_capacity = max(16, max_requests * 4)

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("DSpark context lifecycle and draft KV writes require the model worker thread")

    def register_request(self, descriptor: DSparkContextDescriptor) -> None:
        self._require_owner()
        with self._lock:
            if descriptor.request_id in self._contexts or descriptor.request_id in self._completed:
                raise ValueError("Discard the previous DSpark allocation before registering a reused request ID")
            if len(self._contexts) >= self.max_requests:
                raise RuntimeError("DSpark context request capacity exhausted")
            self._contexts[descriptor.request_id] = _ContextProgress(descriptor)

    def get_descriptor(self, request_id: str) -> DSparkContextDescriptor | None:
        # The network reader may inspect request admission, but it never owns
        # lifecycle transitions or draft-model execution.
        with self._lock:
            state = self._contexts.get(request_id)
            return state.descriptor if state is not None else None

    def begin_direct_transfer(self, descriptor: DSparkContextDescriptor) -> tuple[DSparkContextSubmission, bool]:
        """Reserve one whole-prompt direct KV copy; duplicate retries are idempotent."""
        with self._lock:
            state = self._contexts.get(descriptor.request_id)
            if state is None:
                completed = self._completed.get(descriptor.request_id)
                if completed is not None:
                    if completed.generation != descriptor.generation:
                        return DSparkContextSubmission.STALE, False
                    if completed != descriptor:
                        raise ValueError("DSpark KV transfer schema changed within one allocation")
                    return DSparkContextSubmission.ACCEPTED, False
                retired = self._retired.get((descriptor.request_id, descriptor.generation))
                if retired is not None:
                    if retired != descriptor:
                        raise ValueError("Retired DSpark KV transfer schema changed")
                    return DSparkContextSubmission.ACCEPTED, False
                return DSparkContextSubmission.BACKPRESSURE, False
            if state.descriptor.generation != descriptor.generation:
                return DSparkContextSubmission.STALE, False
            if state.failed:
                raise RuntimeError("DSpark context allocation is quarantined after a transfer failure")
            if state.descriptor != descriptor:
                raise ValueError("DSpark KV transfer schema/prompt length changed within one allocation")
            if state.initialized_tokens == descriptor.prompt_tokens:
                return DSparkContextSubmission.ACCEPTED, False
            if state.in_flight:
                return DSparkContextSubmission.BACKPRESSURE, False
            if state.initialized_tokens:
                raise ValueError("Direct DSpark KV transfer cannot overlap another initialization path")
            state.in_flight = True
            return DSparkContextSubmission.ACCEPTED, True

    def finish_direct_transfer(self, descriptor: DSparkContextDescriptor, *, success: bool) -> None:
        """Publish transfer completion only after MemFabric reports success."""
        with self._lock:
            state = self._contexts.get(descriptor.request_id)
            if state is None or state.descriptor.generation != descriptor.generation:
                raise RuntimeError("DSpark allocation changed during direct KV transfer")
            if not state.in_flight:
                raise RuntimeError("DSpark direct KV transfer was not reserved")
            state.in_flight = False
            if not success:
                state.failed = True
                return
            state.initialized_tokens = descriptor.prompt_tokens

    def mark_target_kv_done(self, request_id: str, generation: str) -> None:
        self._require_owner()
        with self._lock:
            state = self._contexts.get(request_id)
            if state is not None and state.descriptor.generation == generation:
                state.target_kv_done = True

    def ready_requests(self) -> set[str]:
        """Local readiness only; the connector must still join all D TP ranks."""
        self._require_owner()
        with self._lock:
            return {
                req_id
                for req_id, state in self._contexts.items()
                if state.target_kv_done
                and not state.failed
                and not state.in_flight
                and state.initialized_tokens == state.descriptor.prompt_tokens
            }

    def retire_ready_request(self, request_id: str, generation: str) -> None:
        """Release ingress bookkeeping after every TP rank joined readiness.

        Resident draft KV and its block tables are owned by the runner, not
        this receiver. Keeping completed ingress allocations until decoding
        finishes can overlap the next scheduler admission: start_load_kv runs
        before finished-request cleanup in the same connector-only step.
        """
        self._require_owner()
        with self._lock:
            state = self._contexts.get(request_id)
            if state is None or state.descriptor.generation != generation:
                return
            if (
                not state.target_kv_done
                or state.failed
                or state.in_flight
                or state.initialized_tokens != state.descriptor.prompt_tokens
            ):
                raise RuntimeError("Cannot retire DSpark ingress before synchronized context and target KV readiness")
            del self._contexts[request_id]
            self._completed[request_id] = state.descriptor

    def _remember_retired(self, descriptor: DSparkContextDescriptor) -> None:
        key = (descriptor.request_id, descriptor.generation)
        self._retired[key] = descriptor
        self._retired.move_to_end(key)
        while len(self._retired) > self._retired_capacity:
            self._retired.popitem(last=False)

    def discard_request_id(self, request_id: str) -> None:
        """Free request bookkeeping while retaining a bounded lost-ACK tombstone."""
        self._require_owner()
        with self._lock:
            completed = self._completed.pop(request_id, None)
            if completed is not None:
                self._remember_retired(completed)
            state = self._contexts.get(request_id)
            if state is None:
                return
            if state.in_flight:
                raise RuntimeError("Cannot recycle DSpark blocks while a direct transfer is in flight")
            del self._contexts[request_id]
            if state.initialized_tokens == state.descriptor.prompt_tokens:
                self._remember_retired(state.descriptor)

    def discard_request(self, request_id: str, generation: str) -> None:
        """Cancel/retire an allocation after any direct read has completed."""
        self._require_owner()
        with self._lock:
            completed = self._completed.get(request_id)
            if completed is not None and completed.generation == generation:
                del self._completed[request_id]
                self._remember_retired(completed)
            state = self._contexts.get(request_id)
            if state is None or state.descriptor.generation != generation:
                return
            if state.in_flight:
                raise RuntimeError("Cannot recycle DSpark blocks while a direct transfer is in flight")
            del self._contexts[request_id]
            if state.initialized_tokens == state.descriptor.prompt_tokens:
                self._remember_retired(state.descriptor)


def build_draft_context_slot_mappings(
    chunk: DSparkContextChunk,
    *,
    draft_group_ids: tuple[int, ...],
    draft_block_ids_by_group: dict[int, tuple[int, ...]],
    block_sizes_by_group: dict[int, int],
    layer_group_ids: tuple[int, ...],
    device: torch.device,
) -> list[torch.Tensor]:
    """Map absolute prompt rows into the loader-discovered DSpark KV groups.

    ``layer_group_ids`` is in the loaded DSpark model's attention-layer order;
    callers must derive it from the actual speculative model and KV cache
    groups. No target, indexer, or inferred layer-index block IDs are accepted.
    """
    if not draft_group_ids or len(set(draft_group_ids)) != len(draft_group_ids):
        raise ValueError("DSpark context needs unique loaded draft KV cache-group IDs")
    if not layer_group_ids or any(group_id not in draft_group_ids for group_id in layer_group_ids):
        raise ValueError("Every DSpark attention layer must map to an actual resident draft cache group")
    if set(draft_group_ids) != set(draft_block_ids_by_group) or set(draft_group_ids) != set(block_sizes_by_group):
        raise ValueError("DSpark draft block tables must cover every and only resident draft cache group")

    positions = range(chunk.token_offset, chunk.token_offset + chunk.num_tokens)
    per_group: dict[int, torch.Tensor] = {}
    for group_id in draft_group_ids:
        block_ids = draft_block_ids_by_group[group_id]
        block_size = block_sizes_by_group[group_id]
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("DSpark draft KV block sizes must be positive integers")
        if len(block_ids) * block_size < chunk.token_offset + chunk.num_tokens:
            raise ValueError("DSpark draft block table does not cover the prefill chunk")
        if any(type(block_id) is not int or block_id < 0 for block_id in block_ids):
            raise ValueError("DSpark draft block table contains an invalid block ID")
        slots = [block_ids[position // block_size] * block_size + position % block_size for position in positions]
        per_group[group_id] = torch.tensor(slots, dtype=torch.int32, device=device)
    return [per_group[group_id] for group_id in layer_group_ids]


def initialize_draft_context_chunk(
    model: torch.nn.Module,
    chunk: DSparkContextChunk,
    aux_features: torch.Tensor,
    *,
    draft_group_ids: tuple[int, ...],
    draft_block_ids_by_group: dict[int, tuple[int, ...]],
    block_sizes_by_group: dict[int, int],
    layer_group_ids: tuple[int, ...],
    device: torch.device,
) -> None:
    """Project one received chunk and wait for its own MLA KV writes to finish."""
    if aux_features.device != device:
        raise ValueError("DSpark auxiliary staging tensor must be placed on the draft runner device")
    if aux_features.dtype != torch.bfloat16:
        raise ValueError("GLM MLA DSpark auxiliary context must retain BF16 checkpoint inputs")
    context_states = model.combine_hidden_states(aux_features)
    context_positions = torch.arange(
        chunk.token_offset,
        chunk.token_offset + chunk.num_tokens,
        dtype=torch.long,
        device=device,
    )
    slots = build_draft_context_slot_mappings(
        chunk,
        draft_group_ids=draft_group_ids,
        draft_block_ids_by_group=draft_block_ids_by_group,
        block_sizes_by_group=block_sizes_by_group,
        layer_group_ids=layer_group_ids,
        device=device,
    )
    model.precompute_and_store_context_kv(context_states, context_positions, slots)
    torch.npu.current_stream(device).synchronize()
