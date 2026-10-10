"""MooncakeHeterogeneousXferAdapter: P-side protocol adapter for upstream vLLM MooncakeConnector wire protocol.

Design: learn/mooncake-layerwise-ascend-to-upstream-design.md section 6.
Verified (2026-09-14):
  - Upstream MooncakeXferMetadata matches design section 6.2 (mooncake_connector.py:379).
  - The P-side layerwise connector already uses the same zmq+msgspec stack, but with the
    opposite interaction direction: P pulls (sends GET_META_MSG to request MooncakeAgentMetadata),
    while upstream D pushes (sends MooncakeXferMetadata to P).
  - This module adapts the interaction direction: P starts by passively recving D's
    MooncakeXferMetadata, decodes it into internal send tasks, then drives the transfer.
  - Wire-format interop is verified (wire_protocol_test.py: 326-byte roundtrip).

Integration boundary (decoupled):
  - This module only handles protocol send/recv + metadata decoding + task state tracking + responses.
  - It does not call batch_transfer_sync_write (that is the connector's job).
  - on_metadata() returns the list of send tasks; the connector executes the transfer when KV
    is ready, then calls mark_task_done() to update state and trigger the response.
  - First-version limits (design section 7/8): TP=1, PP=1, PCP=DCP=1, non-MLA, BF16/FP16.

Note: this module has no NPU/GPU dependency; the protocol layer is unit-testable.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import msgspec
import zmq

logger = logging.getLogger(__name__)

# Wire protocol constants (aligned with upstream mooncake_connector.py).
# Not imported from vllm to avoid a circular dependency; self-contained for unit testing.


class MooncakeXferResponseStatus:
    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class MooncakeXferMetadata(msgspec.Struct, omit_defaults=True):
    """Transfer metadata pushed by upstream D (mirror of mooncake_connector.py:379)."""

    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict  # ReqId -> (TransferId, list[list[int]])
    kv_caches_base_addr: list
    block_lens: list
    kv_block_lens: list
    registered_layer_names: list = msgspec.field(default_factory=list)
    registered_layer_indices: list = msgspec.field(default_factory=list)
    registered_group_indices: list = msgspec.field(default_factory=list)


class MooncakeXferResponse(msgspec.Struct, omit_defaults=True):
    """P-side response back to D (mirror of MooncakeXferResponse)."""

    status: int
    ok_reqs: list | None = None
    err_reqs: list | None = None
    err_msg: str | None = None


# ReqId / TransferId are Newtypes of str upstream; kept as str here for simplicity.
ReqId = str
TransferId = str


@dataclass
class SendTask:
    """A layer/group send task executed by the connector after KV is ready.

    The connector takes src/dst/length, calls batch_transfer_sync_write,
    then adapter.mark_task_done(transfer_id, task_key, ok).
    """

    transfer_id: TransferId
    req_id: ReqId
    layer_index: int
    group_index: int
    d_rank: int
    # P source block ids / D target block ids — mapped in on_metadata.
    # First version TP=1: d_rank is always 0, block mapping is 1:1.
    src_block_ids: list[int]
    dst_block_ids: list[int]
    # Transfer params (the connector fills in the actual addresses before executing).
    # This module only does logical mapping; it holds no NPU pointers.
    done: bool = False
    failed: bool = False


@dataclass
class TransferState:
    # D flat region index i (1 region per layer, k/v interleaved); use i as the index.

    transfer_id: TransferId
    p_request_id: str
    d_request_ids: list[ReqId]
    # D is 1 region (4D k/v interleaved); do not expand. The connector merges P's k/v into
    # D's layout and transfers this 1 region.
    identity: bytes | None = None
    expected_tasks: dict[tuple, SendTask] = field(default_factory=dict)
    completed_tasks: set = field(default_factory=set)
    failed_tasks: set = field(default_factory=set)
    deadline: float = 0.0

    def is_all_done(self) -> bool:
        return len(self.completed_tasks) + len(self.failed_tasks) >= len(self.expected_tasks)

    def is_failed(self) -> bool:
        return bool(self.failed_tasks)


class MooncakeHeterogeneousXferAdapter:
    """P-side protocol adapter: passively receives upstream D's MooncakeXferMetadata,
    tracks transfer state, and sends responses.

    Thread model:
      - start_listener() starts a background ROUTER thread with a recv_multipart loop.
      - on_metadata() synchronously decodes + builds tasks and returns them (does not block listener).
      - The connector calls mark_task_done(); the response is sent once all tasks are done.

    First-version simplifications:
      - One FINISH per transfer (FINISH only after all layer tasks are done).
      - No pending-map timeout (design section 6.3 step 2); first version assumes the P request
        is already registered when metadata arrives (register_request before metadata).
    """

    def __init__(self, hostname: str, port: int, num_layers: int):
        self.hostname = hostname
        self.port = port
        self.num_layers = num_layers
        self._lock = threading.Lock()
        self._transfers: dict[TransferId, TransferState] = {}
        self._registered: dict[str, TransferId] = {}  # p_request_id -> transfer_id
        # D metadata cache: on_metadata stores the to_agent_metadata result + remote_hostname
        # for the worker to look up D's address (te_rpc_port + layer_metadata + remote_host) in start_load_kv.
        self._remote_metadata_cache: dict[TransferId, dict] = {}
        self._decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._encoder = msgspec.msgpack.Encoder()
        self._sock: zmq.Socket | None = None  # type: ignore[name-defined]
        self._listener_thread: threading.Thread | None = None
        self._stopped = False
        # Pending response queue: mark_task_done enqueues (identity, response); the listener thread drains and sends.
        # zmq sockets are not thread-safe; all send_multipart must happen in the listener thread.
        self._pending_responses: queue.Queue[tuple[bytes, MooncakeXferResponse]] = queue.Queue()

    def register_request(self, transfer_id: TransferId, p_request_id: str) -> None:
        """Register a P request (arrives before metadata)."""
        with self._lock:
            self._registered[p_request_id] = transfer_id

    def get_transfer_id_for_request(self, p_request_id: str) -> TransferId | None:
        """Look up the transfer_id for a P request (stored in register_request)."""
        with self._lock:
            return self._registered.get(p_request_id)

    def get_remote_metadata(self, transfer_id: TransferId) -> dict | None:
        """Look up D metadata (to_agent_metadata result + remote_hostname stored in on_metadata).

        Returns a dict with:
          remote_hostname: str — D host
          te_rpc_port: int — D TransferEngine rpc_port
          layer_metadata: dict[str, dict] — per-layer LayerMetadata-equivalent structure
        If D has not pushed metadata for this transfer yet, returns None (the worker should wait or skip).
        """
        with self._lock:
            return self._remote_metadata_cache.get(transfer_id)

    def start_listener(self) -> None:
        """Start a background ROUTER bind and loop recv of D metadata."""
        ctx = zmq.Context.instance()  # type: ignore[attr-defined]
        self._sock = ctx.socket(zmq.ROUTER)  # type: ignore[attr-defined]
        self._sock.bind(f"tcp://{self.hostname}:{self.port}")
        logger.info("MooncakeHeterogeneousXferAdapter listening on tcp://%s:%d", self.hostname, self.port)
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()

    def _listen_loop(self) -> None:
        """Listener loop: poll recv D metadata + drain pending responses with a timeout.

        Uses a 100ms Poller (not a blocking recv) so mark_task_done enqueued responses
        (FINISH/ERROR) can be handled between recvs. zmq sockets are not thread-safe; all
        send_multipart must run in the listener thread, so mark_task_done enqueues instead.
        """
        assert self._sock is not None
        poller = zmq.Poller()  # type: ignore[attr-defined]
        poller.register(self._sock, zmq.POLLIN)  # type: ignore[attr-defined]
        while not self._stopped:
            # 1. drain pending responses enqueued by mark_task_done (FINISH/ERROR)
            self._drain_pending_responses()
            # 2. poll for D metadata (100ms timeout, to drain promptly + check _stopped)
            events = dict(poller.poll(100))
            if self._sock not in events:
                continue
            try:
                identity, metadata_bytes = self._sock.recv_multipart(zmq.NOBLOCK)  # type: ignore[attr-defined]
            except zmq.ContextTerminated:  # type: ignore[attr-defined]
                break
            except zmq.Again:  # type: ignore[attr-defined]
                continue
            try:
                metadata = self._decoder.decode(metadata_bytes)
                tasks = self.on_metadata(metadata, identity)
                # Reply CONTINUE without ok_reqs: ok_reqs means "this req's KV is fully sent",
                # but the transfer has not started yet (tasks run async in the connector). D's
                # process_pulling_result decrements pull_tasks_count on ok_reqs and decodes when
                # it hits zero. If CONTINUE wrongly carried ok_reqs, D would decode before the KV
                # is written -> read stale blocks -> cross-prompt crosstalk. Only the FINISH
                # response from mark_all_tasks_done carries ok_reqs (after the transfer),
                resp = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.CONTINUE,
                )
                self._sock.send_multipart((identity, self._encoder.encode(resp)))
                logger.debug("Accepted metadata for %s, %d tasks queued", list(metadata.req_blocks.keys()), len(tasks))
            except Exception as e:
                logger.error("Error processing xfer metadata: %s", e)
                err_resp = MooncakeXferResponse(status=MooncakeXferResponseStatus.ERROR, err_msg=str(e))
                self._sock.send_multipart((identity, self._encoder.encode(err_resp)))

    def _drain_pending_responses(self) -> None:
        """Send responses enqueued by mark_task_done (FINISH/ERROR)."""
        assert self._sock is not None
        count = 0
        while True:
            try:
                identity, resp = self._pending_responses.get_nowait()
            except queue.Empty:
                break
            count += 1
            try:
                self._sock.send_multipart((identity, self._encoder.encode(resp)))
                logger.debug("Drained response status=%s to identity=%s", resp.status, identity[:8])
            except zmq.ZMQError as e:  # type: ignore[attr-defined]
                logger.error("Failed to send pending response: %s", e)
        if count:
            logger.debug("Drained %d pending responses", count)

    def on_metadata(self, metadata: MooncakeXferMetadata, identity: bytes | None = None) -> list[SendTask]:
        """Decode metadata and build send tasks; return the task list for the connector to execute.

        Args:
            metadata: MooncakeXferMetadata pushed by upstream D.
            identity: D-side ROUTER identity (passed by the listener, stored in state for
                mark_task_done to reply FINISH when all done). Optional in unit tests.

        Design section 6.2 steps:
          1. Limit metadata size, reject unknown/duplicate transfer_id (first version: log only).
          2. Verify the P request is registered (first version: do not block; log missing p_request_id).
          3. Build the (layer_index, group_index, region_index) mapping.
          4. Validate layer/group counts/dtype/block size (first version: assert layer count matches).
          5. D block ids -> Layerwise sender target block ids (first version TP=1, 1:1 mapping).
        """
        tasks: list[SendTask] = []
        with self._lock:
            for req_id, (transfer_id, d_block_ids) in metadata.req_blocks.items():
                state = self._transfers.get(transfer_id)
                if state is None:
                    state = TransferState(
                        transfer_id=transfer_id,
                        p_request_id=self._registered.get(req_id, ""),
                        d_request_ids=[req_id],
                        deadline=time.monotonic() + 300.0,
                    )
                    self._transfers[transfer_id] = state
                else:
                    if req_id not in state.d_request_ids:
                        state.d_request_ids.append(req_id)
                # Store D identity (the last metadata's identity overwrites, since D blocks on
                # recv for the same transfer, so identity is stable).
                if identity is not None:
                    state.identity = identity
                # Cache D metadata (to_agent_metadata result + remote_hostname) for the worker
                # to look up D's address in start_load_kv. Stored once per transfer_id.
                if transfer_id not in self._remote_metadata_cache:
                    agent_meta = self.to_agent_metadata(metadata)
                    agent_meta["remote_hostname"] = metadata.remote_hostname
                    # Keep D's req_blocks (per req_id D target block ids, per group) so the worker
                    # can compute dst offsets using D's block numbers (P/D blocks are independent, not 1:1).
                    agent_meta["req_blocks"] = dict(metadata.req_blocks)
                    self._remote_metadata_cache[transfer_id] = agent_meta

                # First version TP=1: one task per registered layer.
                # Block mapping 1:1 (P source block i -> D target block i).
                for layer_name, layer_idx, group_idx in zip(
                    metadata.registered_layer_names,
                    metadata.registered_layer_indices,
                    metadata.registered_group_indices,
                ):
                    # d_block_ids is per-layer list[list[int]]; first version takes the first group.
                    flat_blocks = d_block_ids[0] if d_block_ids else []
                    task_key = (layer_idx, group_idx, 0)
                    task = SendTask(
                        transfer_id=transfer_id,
                        req_id=req_id,
                        layer_index=layer_idx,
                        group_index=group_idx,
                        d_rank=0,
                        src_block_ids=flat_blocks,  # first version P source == D target (1:1)
                        dst_block_ids=flat_blocks,
                    )
                    state.expected_tasks[task_key] = task
                    tasks.append(task)
        logger.info("Built %d send tasks for transfer %s", len(tasks), transfer_id)
        return tasks

    @staticmethod
    def to_agent_metadata(metadata: MooncakeXferMetadata) -> dict:
        """Convert upstream MooncakeXferMetadata into the P-side connector's layer_metadata dict.

        (including te_rpc_port + layer_metadata: dict[str, LayerMetadata]). The adapter passively
        recvs upstream MooncakeXferMetadata and builds an equivalent dict with this method, so the
        connector's send path (_transfer_kv_cache / get_transfer_meta) is reused without changes.


        Heterogeneous KV layout adaptation (first version TP=1, single group, non-MLA):
          P (vllm-ascend/NPU) KV cache: k/v split, k_cache/v_cache each
            (num_blocks, block_size, num_kv_heads, head_size), 2 regions/layer.
          D (upstream vLLM 0.28 FA HND) KV cache: shape
            (num_blocks, num_kv_heads, block_size, 2*head_size) — 4D, k/v interleaved in the last dim
            ([..., :head_size]=k, [..., head_size:]=v; FA uses transpose(1,2).split(head_size, dim=-1)).
            block_len = 2*block_size*num_kv_heads*head_size*elem = 524288, 1 region/layer.
            intra-block = [num_kv_heads, block_size, 2*head_size]; each (head, slot)=[k_head, v_head] contiguous.
          # base + block_len//2 (the k or v half = 262144, for the connector to merge).
          This method keeps D's 1 region (not expanded); the connector merges P's split
          k/v cache into D's 4D HND layout.

        Returns a dict; layer_metadata per layer contains:
          kv_caches_base_addr: list[int] — D region base (1, the whole k/v-interleaved block)
          block_len: list[int]         — D region block stride (524288, k+v)
          kv_block_len: int            — block_len//2 (= k or v half = 262144, for the connector to merge)
          tensor_group_idx: list[int]
        """
        layer_metadata: dict[str, dict] = {}
        for i, layer_name in enumerate(metadata.registered_layer_names):
            # D flat region index i (1 region per layer, k/v interleaved); use i as the index.
            group = metadata.registered_group_indices[i] if i < len(metadata.registered_group_indices) else 0
            base_addr = metadata.kv_caches_base_addr[i] if i < len(metadata.kv_caches_base_addr) else 0
            block_len = metadata.block_lens[i] if i < len(metadata.block_lens) else 0
            kv_block_len = metadata.kv_block_lens[i] if i < len(metadata.kv_block_lens) else block_len // 2
            # D is 1 region (4D k/v interleaved); do not expand. The connector merges P's k/v into
            layer_metadata[layer_name] = {
                "tensor_group_idx": [group],
                "kv_caches_base_addr": [base_addr],
                "block_len": [block_len],
                "kv_block_len": kv_block_len,
                "block_size_scale": [1],
            }
        return {"te_rpc_port": metadata.remote_port, "layer_metadata": layer_metadata}

    def mark_task_done(
        self, transfer_id: TransferId, layer_index: int, group_index: int, d_rank: int = 0, ok: bool = True
    ) -> MooncakeXferResponse | None:
        """Called by the connector after batch_transfer_sync_write (layerwise model, per-layer mark).

        When all tasks are done, reply FINISH/ERROR to D with the stored identity and return the
        Response (for the caller); if not all done, return None. If the listener is not running
        (unit-test direct call) or identity is missing, return the Response without sending.
        """
        return self._mark_task(transfer_id, (layer_index, group_index, d_rank), ok)

    def mark_all_tasks_done(self, transfer_id: TransferId, ok: bool = True) -> MooncakeXferResponse | None:
        """Request-triggered model: mark all expected_tasks done at once (after transferring all layers).

        For non-layerwise connectors (e.g. MooncakeHeterogeneousConnector) that transfer all layers
        in one shot, then call this to mark all tasks done and trigger FINISH/ERROR back to D.
        """
        with self._lock:
            state = self._transfers.get(transfer_id)
            if state is None:
                logger.warning("mark_all_tasks_done: unknown transfer %s", transfer_id)
                return None
            task_keys = list(state.expected_tasks.keys())
        resp = None
        for task_key in task_keys:
            resp = self._mark_task(transfer_id, task_key, ok)
        return resp

    def _mark_task(self, transfer_id: TransferId, task_key: tuple, ok: bool) -> MooncakeXferResponse | None:
        """Internal: mark one task done; when all done, enqueue FINISH/ERROR."""
        with self._lock:
            state = self._transfers.get(transfer_id)
            if state is None:
                logger.warning("mark_task: unknown transfer %s", transfer_id)
                return None
            task = state.expected_tasks.get(task_key)
            if task is None:
                logger.warning("mark_task: unknown task %s", task_key)
                return None
            if ok:
                task.done = True
                state.completed_tasks.add(task_key)
            else:
                task.failed = True
                state.failed_tasks.add(task_key)

            if not state.is_all_done():
                return None

            if state.is_failed():
                resp = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_reqs=list({t.req_id for t in state.expected_tasks.values() if t.failed}),
                    err_msg="Some transfer tasks failed",
                )
            else:
                resp = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    ok_reqs=list(state.d_request_ids),
                )
            logger.info("Transfer %s complete: status=%s", transfer_id, resp.status)
            # All tasks done: release adapter-side state to avoid unbounded
            # growth on long-running P services (mirrors worker get_finished
            # cleanup of _completed/_failed_transfers).
            self._transfers.pop(transfer_id, None)
            self._remote_metadata_cache.pop(transfer_id, None)

        # Enqueue the response (drained and sent by the listener thread, to avoid zmq socket
        # cross-thread use). identity is stored by the listener in on_metadata. In unit tests
        if state.identity is not None and not self._stopped:
            self._pending_responses.put((state.identity, resp))
            logger.debug("Queued %s response for transfer %s identity=%s", resp.status, transfer_id, state.identity[:8])
        elif state.identity is None:
            logger.warning("mark_task: no identity for transfer %s, response not sent", transfer_id)
        return resp

    def stop(self) -> None:
        """Safe stop: set the flag so the listener loop exits, join the thread, then close the socket."""
        self._stopped = True
        # The listener polls with NOBLOCK (100ms), so it exits within ~200ms.
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2.0)
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None
