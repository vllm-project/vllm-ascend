# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Common scheduler-side logic for Mooncake KV transfer connectors."""

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import msgspec
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from zmq.constants import SocketOption, SocketType
from zmq.error import Again

from vllm_ascend.ascend_config import (
    get_ascend_config,
    init_ascend_config,
)
from vllm_ascend.core.kv_cache_interface import is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePCPTransferMetadata,
    MooncakePPTransferMetadata,
    MooncakeTPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import zmq_ctx

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


GET_META_MSG = b"get_meta_msg"
_METADATA_TIMEOUT_MS = 1000


class MooncakeMetadataThread(threading.Thread):
    """Common scheduler metadata aggregation and ROUTER query service."""

    def __init__(
        self,
        host: str,
        port: int,
        engine_id: str,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
        tp_size: int,
        pp_size: int,
        pcp_size: int,
        dcp_size: int,
        use_kv_pp: bool,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name=type(self).__name__)
        encoder = msgspec.msgpack.Encoder()
        self.host = host
        self.port = port
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.pcp_size = pcp_size
        self.dcp_size = dcp_size
        self.engine_id = engine_id
        self.use_kv_pp = use_kv_pp
        if self.use_kv_pp and self.dcp_size != 1:
            raise ValueError(f"Mooncake KVPP cannot be combined with DCP, got dcp_size={self.dcp_size}")
        metadata_by_pp_rank = self._merge_metadata_by_pp_rank(metadata)
        self.encoded_metadata = encoder.encode(
            MooncakeTransferMetadataGroups(
                engine_id=engine_id,
                scheduler_host=host,
                scheduler_port=port,
                pp_size=pp_size,
                pcp_size=pcp_size,
                dcp_size=dcp_size,
                tp_size=tp_size,
                use_kv_pp=self.use_kv_pp,
                metadata_by_pp_rank=metadata_by_pp_rank,
            )
        )
        logger.info(
            "Initialized Mooncake scheduler handshake metadata: engine_id=%s, "
            "worker_count=%d, pp_ranks=%s, tp_size=%d, pp_size=%d, "
            "pcp_size=%d, dcp_size=%d, use_kv_pp=%s",
            engine_id,
            len(metadata),
            sorted(metadata_by_pp_rank),
            tp_size,
            pp_size,
            pcp_size,
            dcp_size,
            self.use_kv_pp,
        )
        self.ready_event = ready_event
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.address = make_zmq_path("tcp", host, port)

    def _merge_metadata_by_pp_rank(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> dict[int, MooncakePPTransferMetadata]:
        """Merge worker metadata into one PP-wide layer table.

        LayerSplit may assign different layers to any PCP/TP worker. Per-layer
        fields are therefore aligned by layer name instead of requiring every
        worker to expose the same layer list. Worker-private address tables are
        padded to the PP-union layer order, while ``layer_indices`` records the
        entries physically owned by each PCP/TP worker.
        """
        # pp_rank -> pcp_rank -> tp_rank -> worker handshake metadata.
        workers_by_pp_rank: dict[int, dict[int, dict[int, MooncakeTransferMetadata]]] = {}
        for metadata_key, rank_metadata in metadata.items():
            if isinstance(metadata_key, int):
                pp_rank, pcp_rank, tp_rank = 0, 0, metadata_key
            elif len(metadata_key) == 2:
                pp_rank, tp_rank = metadata_key
                pcp_rank = 0
            elif len(metadata_key) == 3:
                pp_rank, pcp_rank, tp_rank = metadata_key
            else:
                raise ValueError(
                    "Mooncake handshake metadata key must be tp_rank, "
                    f"(pp_rank, tp_rank), or (pp_rank, pcp_rank, tp_rank), got {metadata_key!r}"
                )

            if not isinstance(rank_metadata, MooncakeTransferMetadata):
                raise ValueError(
                    "Mooncake scheduler expects MooncakeTransferMetadata, "
                    f"got {type(rank_metadata).__name__} for key {metadata_key!r}"
                )
            if rank_metadata.engine_id != self.engine_id:
                raise ValueError(
                    "Mooncake worker metadata engine ID mismatch: expected "
                    f"{self.engine_id!r}, got {rank_metadata.engine_id!r}"
                )

            workers_by_tp_rank = workers_by_pp_rank.setdefault(pp_rank, {}).setdefault(pcp_rank, {})
            if tp_rank in workers_by_tp_rank:
                raise ValueError(
                    f"Duplicate Mooncake metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, TP rank {tp_rank}"
                )
            workers_by_tp_rank[tp_rank] = rank_metadata

        expected_pp_ranks = set(range(self.pp_size))
        if set(workers_by_pp_rank) != expected_pp_ranks:
            raise ValueError(
                "Mooncake worker metadata has incomplete PP ranks: expected "
                f"{sorted(expected_pp_ranks)}, got {sorted(workers_by_pp_rank)}"
            )

        expected_pcp_ranks = set(range(self.pcp_size))
        expected_tp_ranks = set(range(self.tp_size))
        merged: dict[int, MooncakePPTransferMetadata] = {}
        for pp_rank, workers_by_pcp_rank in sorted(workers_by_pp_rank.items()):
            if set(workers_by_pcp_rank) != expected_pcp_ranks:
                raise ValueError(
                    "Mooncake worker metadata has incomplete PCP ranks for "
                    f"PP rank {pp_rank}: expected {sorted(expected_pcp_ranks)}, "
                    f"got {sorted(workers_by_pcp_rank)}"
                )
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                if set(workers_by_tp_rank) != expected_tp_ranks:
                    raise ValueError(
                        "Mooncake worker metadata has incomplete TP ranks for "
                        f"PP rank {pp_rank}, PCP rank {pcp_rank}: "
                        f"expected {sorted(expected_tp_ranks)}, got {sorted(workers_by_tp_rank)}"
                    )

            reference_pcp_rank = min(workers_by_pcp_rank)
            reference_tp_rank = min(workers_by_pcp_rank[reference_pcp_rank])
            reference = workers_by_pcp_rank[reference_pcp_rank][reference_tp_rank]
            for pcp_rank, workers_by_tp_rank in workers_by_pcp_rank.items():
                for tp_rank, worker_metadata in workers_by_tp_rank.items():
                    mismatched_fields = [
                        field_name
                        for field_name in ("block_size", "num_blocks", "cp_kv_cache_interleave_size")
                        if getattr(worker_metadata, field_name) != getattr(reference, field_name)
                    ]
                    if mismatched_fields:
                        raise ValueError(
                            "Mooncake worker metadata differs across PCP/TP ranks for "
                            f"PP rank {pp_rank}: PCP/TP {reference_pcp_rank}/{reference_tp_rank} and "
                            f"{pcp_rank}/{tp_rank} mismatch in {mismatched_fields}"
                        )

            # layer_name -> (first owning worker metadata, its local layer index).
            # The source supplies PP-shared fields after all duplicate owners
            # have been checked against the corresponding structural signature.
            layer_source_by_name: dict[str, tuple[MooncakeTransferMetadata, int]] = {}
            # layer_name -> (group index, block size, strides, lengths, shapes,
            #                block-size scales, indexer replication size).
            layer_signature_by_name: dict[str, tuple[object, ...]] = {}
            reference_layer_names: set[str] | None = None
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    worker_layer_names = set(worker_metadata.layer_names)
                    if len(worker_layer_names) != len(worker_metadata.layer_names):
                        raise ValueError(
                            f"Mooncake worker metadata for PP rank {pp_rank}, PCP rank {pcp_rank}, "
                            f"TP rank {tp_rank} contains duplicate layers"
                        )
                    if reference_layer_names is None:
                        reference_layer_names = worker_layer_names
                    elif not self.use_kv_pp and worker_layer_names != reference_layer_names:
                        raise ValueError(
                            "Mooncake workers expose different KV-cache layers while KVPP is disabled: "
                            f"PP rank {pp_rank}, PCP/TP rank {pcp_rank}/{tp_rank}"
                        )
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_signature = (
                            worker_metadata.group_indices[local_layer_index],
                            worker_metadata.dcp_sharded[local_layer_index],
                            worker_metadata.layer_block_sizes[local_layer_index],
                            worker_metadata.block_strides[local_layer_index],
                            worker_metadata.block_lens[local_layer_index],
                            worker_metadata.block_shapes[local_layer_index],
                            worker_metadata.block_size_scales[local_layer_index],
                            worker_metadata.indexer_replication_sizes[local_layer_index],
                        )
                        previous_signature = layer_signature_by_name.get(layer_name)
                        if previous_signature is not None and previous_signature != layer_signature:
                            layer_field_names = (
                                "group_indices",
                                "dcp_sharded",
                                "layer_block_sizes",
                                "block_strides",
                                "block_lens",
                                "block_shapes",
                                "block_size_scales",
                                "indexer_replication_sizes",
                            )
                            mismatched_layer_fields = [
                                field_name
                                for field_name, previous_value, value in zip(
                                    layer_field_names, previous_signature, layer_signature
                                )
                                if previous_value != value
                            ]
                            raise ValueError(
                                "Mooncake worker metadata differs across PCP/TP ranks for "
                                f"PP rank {pp_rank}, layer {layer_name!r}: mismatch in {mismatched_layer_fields}"
                            )
                        if previous_signature is None:
                            layer_source_by_name[layer_name] = (worker_metadata, local_layer_index)
                            layer_signature_by_name[layer_name] = layer_signature
            layer_names = sorted(layer_source_by_name)
            layer_block_sizes: list[int] = []
            group_indices: list[int] = []
            dcp_sharded: list[bool] = []
            block_strides: list[list[int]] = []
            block_lens: list[list[int]] = []
            block_shapes: list[list[tuple[int, ...]]] = []
            block_size_scales: list[list[int]] = []
            indexer_replication_sizes: list[int] = []
            for layer_name in layer_names:
                worker_metadata, local_layer_index = layer_source_by_name[layer_name]
                layer_block_sizes.append(worker_metadata.layer_block_sizes[local_layer_index])
                group_indices.append(worker_metadata.group_indices[local_layer_index])
                dcp_sharded.append(worker_metadata.dcp_sharded[local_layer_index])
                block_strides.append(worker_metadata.block_strides[local_layer_index])
                block_lens.append(worker_metadata.block_lens[local_layer_index])
                block_shapes.append(worker_metadata.block_shapes[local_layer_index])
                block_size_scales.append(worker_metadata.block_size_scales[local_layer_index])
                indexer_replication_sizes.append(worker_metadata.indexer_replication_sizes[local_layer_index])

            layer_index_by_name = {layer_name: layer_index for layer_index, layer_name in enumerate(layer_names)}
            metadata_by_pcp_rank: dict[int, MooncakePCPTransferMetadata] = {}
            for pcp_rank, workers_by_tp_rank in sorted(workers_by_pcp_rank.items()):
                metadata_by_tp_rank: dict[int, MooncakeTPTransferMetadata] = {}
                for tp_rank, worker_metadata in sorted(workers_by_tp_rank.items()):
                    layer_indices = sorted(
                        layer_index_by_name[layer_name] for layer_name in worker_metadata.layer_names
                    )
                    # PP-union layer index -> this worker's per-cache-tensor
                    # addresses; unowned entries remain empty.
                    aligned_base_addrs: list[list[int]] = [[] for _ in layer_names]
                    for local_layer_index, layer_name in enumerate(worker_metadata.layer_names):
                        layer_index = layer_index_by_name[layer_name]
                        aligned_base_addrs[layer_index] = worker_metadata.kv_caches_base_addr[local_layer_index]
                    metadata_by_tp_rank[tp_rank] = MooncakeTPTransferMetadata(
                        te_rpc_port=worker_metadata.te_rpc_port,
                        layer_indices=layer_indices,
                        kv_caches_base_addr=aligned_base_addrs,
                        local_ip=worker_metadata.local_ip,
                        handshake_port=worker_metadata.handshake_port,
                    )
                metadata_by_pcp_rank[pcp_rank] = MooncakePCPTransferMetadata(metadata_by_tp_rank=metadata_by_tp_rank)

            merged[pp_rank] = MooncakePPTransferMetadata(
                block_size=reference.block_size,
                cp_kv_cache_interleave_size=reference.cp_kv_cache_interleave_size,
                num_blocks=reference.num_blocks,
                layer_names=layer_names,
                layer_block_sizes=layer_block_sizes,
                group_indices=group_indices,
                dcp_sharded=dcp_sharded,
                block_strides=block_strides,
                block_lens=block_lens,
                block_shapes=block_shapes,
                block_size_scales=block_size_scales,
                metadata_by_pcp_rank=metadata_by_pcp_rank,
                indexer_replication_sizes=indexer_replication_sizes,
            )
        return merged

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.ident is not None:
            self.join()

    def run(self) -> None:
        try:
            with zmq_ctx(SocketType.ROUTER, self.address) as sock:
                sock.setsockopt(SocketOption.SNDTIMEO, _METADATA_TIMEOUT_MS)
                sock.setsockopt(SocketOption.RCVTIMEO, _METADATA_TIMEOUT_MS)
                self.address = sock.getsockopt(SocketOption.LAST_ENDPOINT).decode()
                self.ready_event.set()
                self._run_busy_loop(sock)
        except Exception as exc:
            self.error = exc
            logger.exception("Mooncake scheduler metadata thread failed on %s", self.address)
        finally:
            self.ready_event.set()

    def _run_busy_loop(self, sock: Any) -> None:
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while not self.stop_event.is_set():
            try:
                frames = sock.recv_multipart()
            except Again:
                continue
            identity = frames[0]
            payload = [frame for frame in frames[1:] if frame]
            if len(payload) != 1:
                logger.warning("Invalid Mooncake scheduler control frames: %s", frames)
                continue
            try:
                response = self._handle_message(decoder.decode(payload[0]))
                if response is not None:
                    sock.send_multipart((identity, b"", response))
            except Exception:
                logger.exception("Failed to handle Mooncake scheduler control message")

    def _handle_message(self, message: tuple) -> bytes | None:
        if message and message[0] == GET_META_MSG:
            return self.encoded_metadata
        return None


class MooncakeBaseConnectorScheduler:
    """Scheduler logic shared by Mooncake transfer modes."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        assert vllm_config.kv_transfer_config is not None

        self.vllm_config = vllm_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        if self.kv_transfer_config.is_kv_consumer == self.kv_transfer_config.is_kv_producer:
            raise ValueError(
                f"Mooncake scheduler requires exactly one KV transfer role, got {self.kv_transfer_config.kv_role!r}"
            )
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config is not None else 0
        )

        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()

        self.local_ip = get_ip()
        self.side_channel_host = self.local_ip
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.max_device_id = (
            self.tp_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Worker handshake ports occupy
        # [kv_port, kv_port + dp_size * pp_size * pcp_size * tp_size).
        # Keep one scheduler control port per DP rank immediately after that
        # range so it cannot collide with a worker handshake socket.
        # Dense engines reset their compute DP rank to zero; the serving
        # index still identifies each independent replica.
        self.side_channel_port = (
            self.kv_transfer_config.kv_port + self.max_device_id + vllm_config.parallel_config.data_parallel_index
        )

        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.group_block_size = [group.kv_cache_spec.block_size for group in self.kv_cache_groups]
        self.group_unique_specs = [self._get_group_unique_specs(group) for group in self.kv_cache_groups]
        self.need_truncate = self._needs_prefill_token_truncation()
        self._logged_mamba_allocation = False

        logger.info("Initializing Mooncake Scheduler %s", engine_id)

    def _log_mamba_request_allocation(self, request_id: str, blocks: "KVCacheBlocks") -> None:
        if self._logged_mamba_allocation:
            return
        for group_id, group_specs in enumerate(self.group_unique_specs):
            for spec in group_specs:
                if not isinstance(spec, MambaSpec):
                    continue
                block_ids = [block.block_id for block in blocks.blocks[group_id] if not block.is_null]
                logger.info(
                    "Mamba request allocation: engine=%s request=%s group=%s "
                    "num_speculative_blocks=%s allocated_blocks=%s block_ids=%s "
                    "page_bytes_per_layer=%s content_bytes_per_layer=%s allocated_bytes_per_layer=%s",
                    self.engine_id,
                    request_id,
                    group_id,
                    spec.num_speculative_blocks,
                    len(block_ids),
                    block_ids,
                    spec.page_size_bytes,
                    spec.state_content_size_bytes,
                    len(block_ids) * spec.page_size_bytes,
                )
        # Observe one real request before transfer-block trimming, outside steady-state loads.
        self._logged_mamba_allocation = True

    def _needs_prefill_token_truncation(self) -> bool:
        """Return whether Prefill must leave the last prompt token to Decode."""
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        uses_compressed_cache = isinstance(compress_ratios, (list, tuple, dict))
        has_state_group = any(
            isinstance(spec, MambaSpec) for group_specs in self.group_unique_specs for spec in group_specs
        )
        return uses_compressed_cache or has_state_group

    @staticmethod
    def _get_group_unique_specs(group: Any) -> list[Any]:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            return [group.kv_cache_spec]

        specs = []
        for layer_name in group.layer_names:
            layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return prompt blocks while retaining evicted-block padding.

        Attention groups are clipped by their block size. Circular groups keep
        their single state page. Mamba groups are not prompt-block aligned, so
        only their speculative tail blocks are removed.
        """
        if not block_ids:
            return block_ids

        assert len(block_ids) == len(self.group_unique_specs), "Number of KV cache groups must match"

        transfer_block_ids: list[list[int]] = []
        for blocks, block_size, group_specs in zip(
            block_ids,
            self.group_block_size,
            self.group_unique_specs,
        ):
            if all(is_circular_kv_cache_spec(spec) for spec in group_specs):
                # Circular caches own one request-lifetime ring block. It is
                # state, not a prompt-aligned attention-block sequence.
                transfer_block_ids.append(blocks)
            elif any(isinstance(spec, MambaSpec) for spec in group_specs):
                # RecoverSSM stores its whole verify window in one state page,
                # so draft tokens do not imply extra checkpoint blocks.
                num_speculative_blocks = max(
                    spec.num_speculative_blocks for spec in group_specs if isinstance(spec, MambaSpec)
                )
                transfer_block_ids.append(blocks[:-num_speculative_blocks] if num_speculative_blocks else blocks)
            else:
                cp_size = max(1, self.dcp_size) if any(spec.dcp_sharded for spec in group_specs) else 1
                num_prompt_blocks = cdiv(prompt_len, block_size * cp_size)
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """Return the prompt token count transferred for stateful models."""
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """Drop the last P-side prompt token for stateful model transfer."""
        params = request.kv_transfer_params
        if (
            params is None
            or not self.need_truncate
            or params.get("_p_side_truncated")
            or request.num_prompt_tokens <= 1
        ):
            return

        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return

        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        # Request already hashed the full prompt. Rebuild for the shortened one.
        request.block_hashes.clear()
        request.update_block_hashes()
        params["_p_side_truncated"] = True

    def on_new_request(self, request: "Request") -> None:
        # Truncate before local prefix-cache lookup so its hit count stays
        # consistent with the prompt length used by the scheduler.
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

    def shutdown(self) -> None:
        """Release scheduler-side resources when the engine shuts down."""
        raise NotImplementedError

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        raise NotImplementedError

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """Return prompt tokens that will be loaded from a remote producer."""
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            actual = self._state_prefill_token_count(len(token_ids))
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count > 0:
                return count, True

        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        raise NotImplementedError

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError

    def set_xfer_handshake_metadata_from_workers(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Set worker transfer metadata on a concrete scheduler."""
        raise NotImplementedError

    def set_xfer_handshake_metadata(
        self,
        metadata: Mapping[int | tuple[int, ...], KVConnectorHandshakeMetadata],
    ) -> None:
        """Handle the legacy port-offset-keyed handshake entry point."""
        self.set_xfer_handshake_metadata_from_workers(metadata)


__all__ = ["MooncakeBaseConnectorScheduler"]
