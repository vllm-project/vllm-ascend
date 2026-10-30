"""Bind remote query identities independently of local KV buffers and resources."""

from vllm.logger import logger

from ...backend.interfaces import KVStoreBackend
from ...backend.naming import WireKeyBuilder
from ...infrastructure.coordinator import ContiguousObservation, Coordinator, LookupObservation
from ...infrastructure.mapping import BlockHashMapping, build_lookup_keys
from ...infrastructure.topology import KVPoolTopology
from ...protocol.messages import LookupRequest, LookupResult
from ..data.keys import KeyIndex


class LookupService:
    def __init__(
        self,
        topology: KVPoolTopology,
        backend: KVStoreBackend,
        coordinator: Coordinator,
        *,
        key_builder: WireKeyBuilder | None = None,
    ) -> None:
        self._backend, self._coordinator = backend, coordinator
        self._group_ids = topology.config.transfer_group_ids
        self._block_hash_mappings = {
            group.group_id: BlockHashMapping(group.block_size, topology.config.hash_block_size)
            for group in topology.transfer_groups
        }
        self._keys = {}
        for group in topology.transfer_groups:
            head_count = (
                topology.config.tp_size if group.uses_align_state else topology.config.tp_partition.key_rank_count
            )
            self._keys[group.group_id] = (
                build_lookup_keys(topology, group.group_id)
                if key_builder is None
                else key_builder.lookup_parts(group.group_id, head_count)
            )
        self._fine_lookup = any(
            group.uses_align_state and group.block_size > topology.config.hash_block_size
            for group in topology.transfer_groups
        )

    def __call__(self, request: LookupRequest) -> LookupResult:
        if request.transfer_group_ids != self._group_ids:
            raise ValueError("Lookup groups do not match configured transfer groups")
        if not request.block_hashes or request.end_token <= 0:
            return LookupResult(0)
        if not 0 <= request.local_available_tokens <= request.end_token:
            raise ValueError("Local availability must lie within the Lookup range")
        fully_local = request.local_available_tokens == request.end_token and all(
            request.end_token % mapping.block_size == 0 for mapping in self._block_hash_mappings.values()
        )
        observations: list[LookupObservation | ContiguousObservation]
        if fully_local:
            empty: LookupObservation = (((), (), ()), ())
            observations = [empty] * len(self._group_ids)
        else:
            masks = self._coordinator.select_for_lookup(request.end_token)
            observations = []
            key_index: KeyIndex | None = None
            observed: list[bool] = []
            for (group_id, mapping), mask in zip(self._block_hash_mappings.items(), masks, strict=True):
                compact = self._coordinator.contiguous_lookup and mask is None
                if compact:
                    first = request.local_available_tokens // mapping.block_size
                    stop = min(len(request.block_hashes), request.end_token // mapping.block_size)
                    hashes = request.block_hashes[first:stop]
                    rows = None
                else:
                    starts, counts, selected_hashes = [], [], []
                    if mask != ():
                        for chunk in mapping.iter_hash_ranges(
                            request.block_hashes,
                            start_token=request.local_available_tokens,
                            end_token=request.end_token,
                            fine_grained=self._fine_lookup,
                        ):
                            position = chunk.start_token // mapping.block_size
                            if mask is not None and (position >= len(mask) or not mask[position]):
                                continue
                            starts.append(chunk.start_token)
                            counts.append(chunk.token_count)
                            selected_hashes.append(chunk.block_hash)
                    hashes = tuple(selected_hashes)
                    rows = (starts, counts, hashes)
                available: tuple[bool, ...] = ()
                if hashes:
                    if key_index is None:
                        key_index = KeyIndex()
                    # Consecutive identities need only an offset. Allocate a route on the first actual alias.
                    routes: list[int | list[int]] = []
                    for parts in self._keys[group_id]:
                        offset = len(key_index.keys)
                        route: list[int] | None = None
                        for position, block_hash in enumerate(hashes):
                            index = key_index.identify(parts, block_hash)
                            if route is None and index != offset + position:
                                route = list(range(offset, offset + position))
                            if route is not None:
                                route.append(index)
                        routes.append(offset if route is None else route)
                    keys = key_index.keys[len(observed) :]
                    try:
                        present = self._backend.exists(keys) if keys else ()
                    except Exception as error:
                        logger.error("Remote Lookup failed. type=%s, error=%s", type(error).__name__, error)
                        return LookupResult(0)
                    if len(present) != len(keys):
                        raise ValueError("Backend Lookup result count does not match its keys")
                    observed.extend(present)
                    if len(routes) == 1 and isinstance(routes[0], int):
                        offset = routes[0]
                        available = tuple(observed[offset : offset + len(hashes)])
                    else:
                        available = tuple(
                            all(
                                observed[route + position if isinstance(route, int) else route[position]]
                                for route in routes
                            )
                            for position in range(len(hashes))
                        )
                if compact:
                    observations.append(
                        ContiguousObservation(first * mapping.block_size, mapping.block_size, available)
                    )
                else:
                    assert rows is not None
                    observations.append((rows, available))
        reachable = self._coordinator.resolve_available_end(
            request.local_available_tokens, request.end_token, request.block_hashes, observations
        )
        return LookupResult(reachable.end_token, reachable.tail_key_boundaries)
