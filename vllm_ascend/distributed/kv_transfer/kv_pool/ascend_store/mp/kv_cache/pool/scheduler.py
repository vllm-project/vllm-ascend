"""Adapt KVPoolScheduler to the KVCacheServer process boundary."""

from collections.abc import Sequence

from vllm.v1.core.kv_cache_utils import BlockHash

from ....pool_scheduler import KVPoolScheduler
from ..registration import SchedulerIdentity, SchedulerRegistration, WorkerLookupHandler
from ..scheduler_view import SchedulerOutputView


class _BlockIdIndex:
    """Map block ids to themselves for inherited BlockPool bookkeeping.

    The inherited code only uses blocks[id] to pass the result directly to
    touch or free_blocks; it never reads a Block attribute.
    """

    def __getitem__(self, block_id: int) -> int:
        return block_id


class _BlockPoolProxy:
    """Record server-side BlockPool operations for its Scheduler owner.

    The inherited mamba bookkeeping reads blocks[id] and calls touch or
    free_blocks on the object occupying its _block_pool slot. Recording those
    ids lets the connector apply the operations to the real Scheduler-process
    BlockPool after the RPC returns.
    """

    def __init__(self):
        self.blocks = _BlockIdIndex()
        self._touch_ids: list[int] = []
        self._free_ids: list[int] = []

    def touch(self, block_ids) -> None:
        self._touch_ids.extend(block_ids)

    def free_blocks(self, block_ids) -> None:
        self._free_ids.extend(block_ids)

    def take_touch_ids(self) -> list[int]:
        ids = self._touch_ids
        self._touch_ids = []
        return ids

    def take_free_ids(self) -> list[int]:
        ids = self._free_ids
        self._free_ids = []
        return ids


class _WorkerLookupBridge:
    """Route KVPoolScheduler's non-layerwise lookup to its Worker service.

    KVPoolScheduler expects a LookupKeyClient. Inside KVCacheServer, the manager
    callback calls the Worker service on the executor thread assigned to that
    Worker, without another RPC. Layerwise lookup uses store_scheduler directly
    and never enters this bridge.
    """

    def __init__(self, identity: SchedulerIdentity, lookup_handler: WorkerLookupHandler):
        self._identity = identity
        self._lookup_handler = lookup_handler

    def lookup(
        self,
        token_len: int,
        block_hashes: Sequence[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        return self._lookup_handler(self._identity, token_len, block_hashes, kv_cache_group_ids, False, hbm_hit_tokens)


class MPKVPoolScheduler(KVPoolScheduler):
    """Run KVPoolScheduler semantics from server-side projections.

    KVPoolScheduler normally reads live vLLM request objects and a BlockPool
    owned by the Scheduler process. This adapter refreshes retained request
    views, routes non-layerwise Worker lookup through the Manager, and records
    BlockPool operations for the owning process to apply.
    """

    def __init__(self, registration: SchedulerRegistration, lookup_handler: WorkerLookupHandler):
        config = registration.config
        use_layerwise = config.kv_transfer_config.kv_connector_extra_config.get("use_layerwise", False)
        super().__init__(config, use_layerwise, kv_cache_config=config.build_kv_cache_config())
        self.client = _WorkerLookupBridge(registration.identity, lookup_handler)  # type: ignore[assignment]
        self._block_pool = _BlockPoolProxy()  # type: ignore[assignment]

    # ==============================
    # Scheduler request state across RPC
    # ==============================

    # A RequestView is registered before scheduling continues, while later RPCs
    # carry only the fields changed in that step. Refresh that same object before
    # entering inherited metadata orchestration so its request registry observes
    # the same evolving state as the in-process Scheduler.

    def build_connector_meta(self, scheduler_output: SchedulerOutputView):
        self._sync_request_views(scheduler_output)
        return super().build_connector_meta(scheduler_output)

    def _sync_request_views(self, output: SchedulerOutputView) -> None:
        """Apply this step's mutable Scheduler fields to registered views."""
        refreshed_new_reqs = []
        for new_req in output.scheduled_new_reqs:
            entry = self._unfinished_requests.get(new_req.req_id)
            view = entry[0] if entry is not None else None
            if view is None:
                # Keep the payload object; the inherited new-request path
                # raises the same "not in _unfinished_requests" error.
                refreshed_new_reqs.append(new_req)
                continue
            view.num_computed_tokens = new_req.num_computed_tokens
            view.block_ids = tuple(list(group) for group in new_req.block_ids_by_group)
            refreshed_new_reqs.append(view)
        output.scheduled_new_reqs = refreshed_new_reqs

        cached = output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            entry = self._unfinished_requests.get(req_id)
            if entry is None:
                continue
            view = entry[0]
            view.num_computed_tokens = cached.num_computed_tokens[index]
            view.all_token_ids.extend(cached.new_token_ids.get(req_id, []))

    # ==============================
    # BlockPool ownership across processes
    # ==============================

    # The inherited mamba bookkeeping runs in the server, but the real BlockPool
    # is owned by the vLLM Scheduler process. Its proxy records block-id operations;
    # each batch is returned once for the connector to apply after the RPC.

    def take_block_pool_commands(self) -> list[int]:
        """Return and clear block ids retained for in-flight mamba Store work."""
        return self._block_pool.take_touch_ids()

    def take_free_block_commands(self) -> list[int]:
        """Return and clear block ids whose Store work finished on every Worker."""
        return self._block_pool.take_free_ids()

    def close(self) -> None:
        """Satisfy the manager's uniform service close contract.

        Unlike the Worker service, the Scheduler owns no backend, transfer
        threads, or imported memory; its request views and recorded block-id
        commands die with this object.
        """
