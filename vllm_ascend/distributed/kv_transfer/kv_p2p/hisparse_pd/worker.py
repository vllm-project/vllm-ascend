# SPDX-License-Identifier: Apache-2.0
"""Worker transfer orchestration; all KV allocation stays upstream."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

from .acl_stage import ACLStageBuffer
from .configuration import validate_transport_options
from .execution import (
    CacheBindingGuard,
    assert_transfer_boundary,
    draft_transfer_enabled,
    execution_contract,
    export_mirror_ready,
    validate_completion,
    validate_draft_state,
)
from .layout import build_layouts
from .protocol import (
    SOURCE_TRANSPORT,
    STAGE_BYTES,
    VERSION,
    LayerLayout,
    LeaseRegistry,
    batch_source_transfers,
    coalesce_transfers,
    encode_layouts,
    plan_reads,
    validate_wire,
)
from .stage import StageSlot
from .transport import (
    LeaseServer,
    MemfabricReader,
    RpcClient,
    TransferCancelled,
    acquire,
    advance_stage,
    prepare_stage,
    rpc,
)

logger = logging.getLogger(__name__)

# Keep failed workers, their real cache owners and the live ACK server until a
# successful close. Worker lifecycle operations run on the engine thread; the
# single receive thread may also retain its own worker when a read fails.
_retained_workers: dict[int, HiSparsePDWorker] = {}


class HiSparsePDWorker:
    def __init__(self, config, kv_cache_config, *, identity: str, producer: bool):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        extra = config.kv_transfer_config.kv_connector_extra_config
        self.config = config
        self.kv_cache_config = kv_cache_config
        self.identity = identity
        self.producer = producer
        self.rank = get_tensor_model_parallel_rank()
        self.world_size = config.parallel_config.tensor_parallel_size
        self.device = torch.npu.current_device()
        self.host = extra["host"]
        self.port = int(extra["control_port"]) + self.rank
        self.peer_host = extra.get("producer_host", self.host)
        self.peer_port = int(extra.get("producer_control_port", extra["control_port"])) + self.rank
        self.protocol, self.urma_swap_space_mib = validate_transport_options(extra)
        self.timeout = float(extra.get("transfer_timeout_seconds", 120))
        self.max_descriptors = int(extra.get("max_descriptors_per_read", 1024))
        if not 0 < self.max_descriptors <= 65536 or not 0 < self.timeout <= 3600:
            raise ValueError("Invalid HiSparse P/D transfer limits")
        self.registry = LeaseRegistry()
        self.server = None
        self.reader = None
        self.stage = None
        self.stage_buffer = None
        self.stage_sources = {}
        self.layouts = []
        self.owners = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hisparse-pd-read")
        self.pending = []
        self.futures = {}
        self.closed = False
        self.closing = False
        self._registered_caches = None
        self._binding_guard = None
        self._export_generation = 0
        self.include_draft = draft_transfer_enabled(config)
        self.transfer_contract = execution_contract(config)
        self.common_worker = None

    def register(self, kv_caches):
        if self.reader is not None or self.closing or self.closed:
            raise RuntimeError("HiSparse P/D caches can only be registered once")
        self._registered_caches = kv_caches
        context = self.config.compilation_config.static_forward_context
        self.layouts, regions, self.owners = build_layouts(
            kv_caches, self.kv_cache_config, context, producer=self.producer, include_draft=self.include_draft
        )
        self._binding_guard = CacheBindingGuard(self.layouts, regions, self.owners)
        try:
            if self.producer:
                self._create_stage(kv_caches, regions)
            else:
                # The one receive executor serializes every bounded SOURCE read.
                # Retain this owner before allocation or registration can fail.
                self.stage_buffer = ACLStageBuffer(self.device, host_owners=tuple(self.owners))
                self.owners.append(self.stage_buffer)
                self.stage_buffer.allocate()
                regions[self.stage_buffer.address, STAGE_BYTES] = "hisparse_receive_stage"
            self.reader = MemfabricReader(
                producer=self.producer,
                device=self.device,
                host=self.host,
                protocol=self.protocol,
                regions=regions,
                owners=self.owners,
                urma_swap_space_mib=self.urma_swap_space_mib,
            )
            self.reader.initialize()
            if self.producer:
                self.server = LeaseServer(f"tcp://{self.host}:{self.port}", self.registry, self.stage)
        except BaseException as initialize_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise initialize_error from cleanup_error
            raise

    def _create_stage(self, kv_caches, regions):
        # PTA owns the bounded tensor and stream. Retain it until device drain
        # and both transport teardown stages have completed.
        self.stage_buffer = ACLStageBuffer(self.device, host_owners=tuple(self.owners))
        self.owners.append(self.stage_buffer)  # Retain before allocate/query can fail.
        self.stage_buffer.allocate()
        for layout in self.layouts:
            if layout.role == "hisparse_source":
                cache = kv_caches[layout.name]
                if isinstance(cache, tuple):
                    (cache,) = cache
                self.stage_sources[layout.name] = cache
        regions[self.stage_buffer.address, STAGE_BYTES] = "hisparse_stage"
        self.stage = StageSlot(
            self.registry,
            self.stage_buffer.address,
            STAGE_BYTES,
            self._copy_source_to_stage,
            self._copy_sources_to_stage,
        )

    def _copy_source_to_stage(self, layer, address, length):
        """PTA selects the worker device and waits for the copy completion event."""
        source = self.stage_sources[layer]
        offset = address - source.data_ptr()
        if not offset >= 0 or offset + length > source.numel() * source.element_size():
            raise ValueError("Producer stage source is outside its retained Host layer")
        self.stage_buffer.copy_from_host(address, length)

    def _copy_sources_to_stage(self, layer, segments):
        source = self.stage_sources[layer]
        base, extent = source.data_ptr(), source.numel() * source.element_size()
        for address, length in segments:
            if address < base or address + length > base + extent:
                raise ValueError("Packed producer segment escapes its retained Host layer")
        self.stage_buffer.copy_from_host_segments(segments)

    def _validate_transfer_bindings(self):
        assert_transfer_boundary(
            capturing=torch.npu.is_current_stream_capturing(),
            closing=self.closing,
            closed=self.closed,
        )
        if self._binding_guard is None or self._registered_caches is None:
            raise RuntimeError("HiSparse P/D cache binding is unavailable")
        current = build_layouts(
            self._registered_caches,
            self.kv_cache_config,
            self.config.compilation_config.static_forward_context,
            producer=self.producer,
            include_draft=self.include_draft,
        )
        self._binding_guard.validate(*current)

    def _prepare_export(self):
        self._validate_transfer_bindings()
        # Common HiSparse finish_forward enqueues graph-era Host mirrors; its
        # restore/import events and TP completion remain upstream-owned. Keep
        # the conservative all-stream drain at this rare handoff boundary until
        # scoped export fences have separate model-level evidence.
        torch.npu.synchronize(self.device)
        self._export_generation += 1
        return {"contract": self.transfer_contract, "generation": self._export_generation}

    def queue(self, metadata):
        if self.closed or self.closing:
            raise RuntimeError("HiSparse P/D worker is closing")
        self.pending.append(metadata)

    def _publish_metadata_exports(self, metadata):
        completion = self._prepare_export()
        for export in metadata.exports:
            manifest = {
                "version": VERSION,
                "completion": completion,
                "source_transport": SOURCE_TRANSPORT,
                "stage_capacity": STAGE_BYTES,
                "stage_control_version": 2,
                "stage_address": self.stage_buffer.address,
                "identity": self.identity,
                "rank": self.rank,
                "world_size": self.world_size,
                "session": self.reader.session,
                "ticket": export["ticket"],
                "blocks": export["blocks"],
                "layers": encode_layouts(self.layouts),
            }
            self.registry.publish(export["ticket"]["lease_id"], export["request_id"], manifest)

    def publish_ready_exports(self):
        """Publish completed prefixes before an unrelated producer forward.

        The common sibling's start_step has submitted the previous draft
        mirrors. Keep the original completion fence and all source leases;
        regular poll still owns sending/receiving completion accounting.
        """
        if self.closed or self.closing:
            raise RuntimeError("HiSparse P/D worker is closing")
        if not self.producer or not any(metadata.exports for metadata in self.pending):
            return
        # A reversed/unknown sibling order retains the original poll path.
        # This is a CPU phase flag, never a device scalar extraction.
        if not getattr(self.common_worker, "_forward_mirror_active", False):
            return
        if not export_mirror_ready(self.common_worker, self.include_draft):
            return
        retained = []
        for metadata in self.pending:
            if not metadata.exports:
                retained.append(metadata)
                continue
            if metadata.imports:
                raise RuntimeError("A producer export must not also contain an import")
            self._publish_metadata_exports(metadata)
        self.pending = retained

    def poll(self, cancelled_ids: set[str]):
        if self.closed or self.closing:
            raise RuntimeError("HiSparse P/D worker is closing or has shut down")
        # Called after all MultiConnector start_load/finish_forward hooks, so
        # the existing HiSparse worker has already submitted host mirrors and
        # host block copies. Publication must follow their actual completion.
        deferred = []
        for metadata in self.pending:
            # Keep upstream's request/block lease alive until the next common
            # start_step has flushed the previous proposal's draft row mirrors.
            # Never flush another step's mapping or publish speculative lookahead.
            if metadata.exports and not export_mirror_ready(self.common_worker, self.include_draft):
                if metadata.imports:
                    raise RuntimeError("A producer export must not also contain an import")
                deferred.append(metadata)
                continue
            if metadata.exports:
                self._publish_metadata_exports(metadata)
            for job in metadata.imports:
                request_id = job["request_id"]
                if request_id in self.futures:
                    raise RuntimeError("Duplicate HiSparse P/D load")
                cancelled = threading.Event()
                future = self.executor.submit(self._load, job, cancelled)
                self.futures[request_id] = future, cancelled
        self.pending = deferred
        received, failed = set(), set()
        for request_id, (future, cancelled) in list(self.futures.items()):
            if request_id in cancelled_ids:
                cancelled.set()
            if not future.done():
                continue
            success = future.result()  # DMA errors remain fatal; never fake a drained completion.
            received.add(request_id)
            if not success:
                failed.add(request_id)
            del self.futures[request_id]
        return self.registry.take_finished(), received, failed

    def _load(self, job, cancelled):
        try:
            torch.npu.set_device(self.device)
            return self._load_request(job, cancelled)
        except BaseException:
            _retained_workers[id(self)] = self
            raise

    def _load_request(self, job, cancelled):
        endpoint = f"tcp://{self.peer_host}:{self.peer_port}"
        with RpcClient(endpoint) as control:
            return self._load_with_control(job, cancelled, control)

    def _load_with_control(self, job, cancelled, control):
        self._validate_transfer_bindings()
        endpoint = f"tcp://{self.peer_host}:{self.peer_port}"
        ticket = job["ticket"]
        consumer = f"{job['request_id']}:{self.rank}:{uuid.uuid4().hex}"
        message = {
            "version": VERSION,
            "source_transport": SOURCE_TRANSPORT,
            "lease_id": ticket["lease_id"],
            "consumer": consumer,
            "identity": self.identity,
            "rank": self.rank,
            "world_size": self.world_size,
        }
        try:
            manifest = acquire(endpoint, message, self.timeout, cancelled, client=control)
            validate_wire(manifest)
            validate_completion(manifest.get("completion"), self.transfer_contract)
            validate_draft_state(ticket.get("draft_state"), ticket["num_tokens"], self.include_draft)
            if manifest["version"] != VERSION or manifest["ticket"] != ticket:
                raise ValueError("Producer ticket differs from the requested prefix")
            if type(manifest.get("stage_capacity")) is not int or manifest["stage_capacity"] != STAGE_BYTES:
                raise ValueError("Producer HBM stage capacity differs")
            if (
                type(manifest.get("stage_address")) is not int
                or not 0 < manifest["stage_address"] < 2**64 - STAGE_BYTES
            ):
                raise ValueError("Producer HBM stage address is invalid")
            remote = [LayerLayout(**item) for item in manifest["layers"]]
            source_blocks = {int(key): value for key, value in manifest["blocks"].items()}
            local_blocks = {int(key): value for key, value in job["blocks"].items()}
            descriptors = plan_reads(
                remote,
                self.layouts,
                source_blocks,
                local_blocks,
                job["start_token"],
                ticket["num_tokens"],
                include_host=not self.kv_cache_config.hisparse_shared_host_pool or self.rank == 0,
                with_roles=True,
            )
            descriptors = coalesce_transfers(descriptors)
        except TransferCancelled:
            return False  # No acquire was sent and no DMA could have started.
        except (TimeoutError, ValueError, KeyError, TypeError) as error:
            # Even a lost acquire reply may have claimed the producer lease.
            # No read has been submitted yet, so releasing that claim is safe.
            try:
                rpc(
                    endpoint,
                    {"op": "release", **message, "drained": True},
                    min(self.timeout, 5),
                    client=control,
                )
            except (TimeoutError, ValueError):
                logger.warning(
                    "HiSparse P/D pre-read failure left an unconfirmed producer lease: %s",
                    error,
                )
            return False
        # Each call is synchronous. Only a successful return proves that its
        # remote reads no longer own the source. Do not ACK an uncertain error.
        self._read_batches(descriptors, manifest, endpoint, message, cancelled, control)
        # Every URMA read above is synchronous. SOURCE unpack also waits for
        # its own copy-stream event before reusing D or releasing the request.
        # Draining the whole device here would include unrelated live Decode
        # graphs, mirrors and collectives on this rank.
        try:
            response = rpc(endpoint, {"op": "release", **message, "drained": True}, self.timeout, client=control)
            if response["status"] != "released":
                raise ValueError("Producer did not acknowledge drained transfer")
        except (TimeoutError, ValueError):
            # All local reads are known drained. Report a failed request through
            # the common async-load path instead of crashing the engine. The
            # producer conservatively retains an unconfirmed acquired lease.
            logger.warning("HiSparse P/D completed its reads but could not confirm producer lease release")
            return False
        return not cancelled.is_set()

    @staticmethod
    def _source_chunk(message, index, layer, parts):
        segments = [[source, length] for destination, source, length in parts]
        count = sum(length for source, length in segments)
        return {
            **message,
            "chunk_id": f"{index}:0",
            "layer": layer,
            "source": segments[0][0],
            "length": count,
            "segments": segments,
        }

    def _release_stage_chunk(self, endpoint, chunk, prepared, control):
        released = {"op": "stage_release", **chunk, "generation": prepared["generation"], "drained": True}
        response = rpc(endpoint, released, self.timeout, client=control)
        if (
            response.get("status") != "stage_released"
            or response.get("chunk_id") != chunk["chunk_id"]
            or response.get("generation") != prepared["generation"]
        ):
            raise ValueError("Packed stage release is unconfirmed; source lease remains retained")

    def _read_batches(self, descriptors, manifest, endpoint, message, cancelled, control):
        batches = iter(enumerate(batch_source_transfers(descriptors)))
        current, prepared, indexer_reads = next(batches, None), None, []
        while current is not None:
            index, (role, layer, parts) = current
            if cancelled.is_set():
                if prepared is not None:
                    # The combined control already prepared this block, but no
                    # read has used it. Release its exact generation before the
                    # outer request lease can be returned.
                    chunk = self._source_chunk(message, index, layer[0], parts)
                    self._release_stage_chunk(endpoint, chunk, prepared, control)
                break
            following = next(batches, None)
            if role == "hisparse_source":
                if indexer_reads:
                    self.reader.read(manifest["session"], indexer_reads)
                    indexer_reads = []
                next_chunk = None
                if (
                    manifest.get("stage_control_version") == 2
                    and following is not None
                    and following[1][0] == "hisparse_source"
                ):
                    j, (_, next_layer, next_parts) = following
                    next_chunk = self._source_chunk(message, j, next_layer[0], next_parts)
                prepared = self._read_staged_batch(
                    endpoint,
                    message,
                    manifest["session"],
                    manifest["stage_address"],
                    index,
                    layer[0],
                    parts,
                    control,
                    prepared=prepared,
                    next_chunk=next_chunk,
                )
            else:
                indexer_reads.extend(parts)
                if len(indexer_reads) == self.max_descriptors:
                    self.reader.read(manifest["session"], indexer_reads)
                    indexer_reads = []
            current = following
        if indexer_reads and not cancelled.is_set():
            self.reader.read(manifest["session"], indexer_reads)

    def _read_staged_batch(
        self, endpoint, message, session, stage_address, index, layer, parts, control, *, prepared=None, next_chunk=None
    ):
        chunk = self._source_chunk(message, index, layer, parts)
        count = chunk["length"]
        if prepared is None:
            prepared = prepare_stage(endpoint, chunk, self.timeout, client=control)
        if prepared["stage_address"] != stage_address or self.stage_buffer is None:
            raise ValueError("Packed read stage differs from its registered owner")
        self.reader.read(session, [(self.stage_buffer.address, stage_address, count)])
        destinations = [(destination, length) for destination, source, length in parts]
        if next_chunk is not None and getattr(self, "_overlap_stage_copy", True):
            # The synchronous URMA read has drained the P source slot. D2H
            # uses the separate D receive slot, so P can prepare its successor
            # while the existing D copy stream completes. Never reuse the D
            # slot, its descriptors or Host owners before this exact fence.
            token = self.stage_buffer.begin_copy_to_host_segments(destinations)
            try:
                released = {**chunk, "generation": prepared["generation"], "drained": True}
                return advance_stage(endpoint, released, next_chunk, self.timeout, client=control)
            finally:
                # Also drain a submitted local copy when control fails. The
                # outer source lease stays retained on either uncertain error.
                self.stage_buffer.finish_copy_to_host(token)
        self.stage_buffer.copy_to_host_segments(destinations)
        if next_chunk is not None:
            released = {**chunk, "generation": prepared["generation"], "drained": True}
            return advance_stage(endpoint, released, next_chunk, self.timeout, client=control)
        self._release_stage_chunk(endpoint, chunk, prepared, control)
        return None

    def _read_staged_source(
        self,
        endpoint,
        message,
        session,
        stage_address,
        index,
        layer,
        destination,
        source,
        length,
        cancelled=None,
        control=None,
    ):
        for offset in range(0, length, STAGE_BYTES):
            if cancelled is not None and cancelled.is_set():
                return  # Every preceding chunk has an explicit drained release.
            count = min(length - offset, STAGE_BYTES)
            chunk = {
                **message,
                "chunk_id": f"{index}:{offset}",
                "layer": layer,
                "source": source + offset,
                "length": count,
            }
            # From this first prepare onwards an uncertain reply may own the P
            # stage. Errors must never enter the pre-read source-release path.
            prepared = prepare_stage(endpoint, chunk, self.timeout, client=control)
            if prepared["stage_address"] != stage_address:
                raise ValueError("Prepared stage differs from the acquired producer HBM allocation")
            if self.stage_buffer is None:
                raise RuntimeError("Consumer has no registered receive stage")
            self.reader.read(session, [(self.stage_buffer.address, prepared["stage_address"], count)])
            # Both the registered D2D read and this local D2H must succeed before
            # the source slot can be released. Errors retain both real owners.
            self.stage_buffer.copy_to_host(destination + offset, count)
            response = rpc(
                endpoint,
                {
                    "op": "stage_release",
                    **chunk,
                    "generation": prepared["generation"],
                    "drained": True,
                },
                self.timeout,
                client=control,
            )
            if (
                response.get("status") != "stage_released"
                or response.get("chunk_id") != chunk["chunk_id"]
                or response.get("generation") != prepared["generation"]
            ):
                raise ValueError("Producer stage release is unconfirmed; source lease remains retained")

    def close(self):
        if self.closed:
            return
        self.closing = True
        transfer_error = None
        try:
            for _, cancelled in self.futures.values():
                cancelled.set()
            self.executor.shutdown(wait=True, cancel_futures=False)
            for future, _ in self.futures.values():
                try:
                    future.result()
                except BaseException as error:
                    if transfer_error is None:
                        transfer_error = error
            # A failed read still needs both device drain and complete transport
            # teardown. Do not let future.result() prevent those safe retries.
            torch.npu.set_device(self.device)
            torch.npu.synchronize(self.device)
            # Failed admission leaves the ACK server operational for active
            # remote readers. Only a drained ACK can release their source lease.
            self.registry.begin_shutdown()
            if self.stage is not None:
                self.stage.begin_shutdown()
            if self.server is not None:
                self.server.close()
            if self.reader is not None:
                self.reader.close(device_drained=True)
            if self.stage_buffer is not None:
                # Includes failed registration: only complete transport teardown
                # proves that the raw allocation is no longer remotely owned.
                self.stage_buffer.close(device_drained=True, transport_closed=True)
        except BaseException as cleanup_error:
            _retained_workers[id(self)] = self
            if transfer_error is not None:
                raise transfer_error from cleanup_error
            raise
        self.closed = True
        self.owners.clear()
        self._registered_caches = None
        self._binding_guard = None
        self.stage_sources.clear()
        self.stage = None
        self.stage_buffer = None
        self.pending.clear()
        self.futures.clear()
        _retained_workers.pop(id(self), None)
        if transfer_error is not None:
            # Memory is now safe to release, but the request still failed.
            raise transfer_error
