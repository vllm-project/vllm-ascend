# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Worker-side implementation entry point for Mooncake pull transfers."""

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

import msgspec
import torch
import zmq
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakeConnectorMetadata,
    MooncakePPTransferMetadata,
    MooncakeTransferMetadata,
    MooncakeTransferMetadataGroups,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import (
    SizedDict,
    ensure_zmq_recv,
    ensure_zmq_send,
    zmq_ctx,
)


class MooncakePullRecvingThread(threading.Thread):
    """D-side execution thread for Mooncake pull requests.

    The queue/result lifecycle is intentionally implemented separately from
    the transfer algorithm. Metadata lookup, block mapping, and Mooncake READ
    submission belong in _handle_requests.
    """

    def __init__(
        self,
        engine: Any,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        layer_name_to_group_index: dict[str, int],
        layer_name_to_spec_index: dict[str, int],
        local_metadata: MooncakeTransferMetadata,
        tp_rank: int,
        tp_size: int,
        pp_rank: int,
        pp_size: int,
        dp_rank: int,
        dp_size: int,
        pcp_rank: int,
        pcp_size: int,
        dcp_rank: int,
        dcp_size: int,
        device: Any,
        ready_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True, name="MooncakePullRecvingThread")
        self.engine = engine
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.kv_cache_config = kv_cache_config
        self.kv_cache_specs = kv_cache_specs
        self.layer_name_to_group_index = layer_name_to_group_index
        self.layer_name_to_spec_index = layer_name_to_spec_index
        self.local_metadata = local_metadata

        self.block_size = local_metadata.block_size
        self.layer_names = local_metadata.layer_names
        self.layer_block_sizes = local_metadata.layer_block_sizes
        self.group_indices = local_metadata.group_indices
        self.spec_indices = [layer_name_to_spec_index[layer_name] for layer_name in self.layer_names]
        self.kv_caches_base_addr = local_metadata.kv_caches_base_addr
        self.block_strides = local_metadata.block_strides
        self.block_lens = local_metadata.block_lens
        self.block_shapes = local_metadata.block_shapes
        self.block_size_scales = local_metadata.block_size_scales

        self.use_mla = self.model_config.is_deepseek_mla
        hf_text_config = self.model_config.hf_text_config
        self.num_key_value_heads = getattr(hf_text_config, "num_key_value_heads", 0)
        speculative_config = vllm_config.speculative_config
        self.num_speculative_tokens = speculative_config.num_speculative_tokens if speculative_config is not None else 0

        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.pcp_rank = pcp_rank
        self.pcp_size = pcp_size
        self.dcp_rank = dcp_rank
        self.dcp_size = dcp_size
        assert self.pcp_size == 1, f"Mooncake pull worker temporarily requires pcp_size=1, got {self.pcp_size}"
        self.device = device
        self.ready_event = ready_event
        self.executor = ThreadPoolExecutor(
            max_workers=32,
            initializer=torch.npu.set_device,
            initargs=(self.device,),
        )
        self.can_report_invalid_block_ids = len(kv_cache_config.kv_cache_groups) == 1
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeTransferMetadataGroups)
        self.remote_metadata: SizedDict[str, MooncakeTransferMetadataGroups] = SizedDict()
        # Candidate producer TP ranks cached independently for each engine:
        # engine_id -> remote_pp_rank -> (local_layer_index, remote_layer_index)
        # -> candidate_groups. Each outer candidate group represents one
        # remote head/TP piece required by this local worker. With remote DCP=1,
        # ranks inside a group are interchangeable replicas and one is selected;
        # with remote DCP>1, they are the DCP shards that jointly provide it.
        self.remote_tp_rank_groups: SizedDict[str, dict[int, dict[tuple[int, int], list[list[int]]]]] = SizedDict()
        # engine_id -> remote_pp_rank
        # -> [(local_layer_index, remote_layer_index), ...]
        self.remote_layer_index_pairs: SizedDict[str, dict[int, list[tuple[int, int]]]] = SizedDict()
        self.request_queue: queue.Queue[tuple[str, dict[str, ReqMeta]]] = queue.Queue()
        self.finished_requests: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.invalid_block_ids: set[int] = set()
        self.invalid_block_ids_lock = threading.Lock()
        assert self.local_metadata is not None

    def add_requests(
        self,
        remote_engine_id: str,
        requests: dict[str, ReqMeta],
    ) -> None:
        """Queue requests for one remote engine."""
        if requests:
            self.request_queue.put((remote_engine_id, requests))

    def get_and_clear_finished_requests(self) -> set[str]:
        """Drain requests whose transfer attempt has completed."""
        finished: set[str] = set()
        while True:
            try:
                finished.add(self.finished_requests.get_nowait())
            except queue.Empty:
                return finished

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Drain local block IDs affected by failed pull attempts."""
        with self.invalid_block_ids_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def run(self) -> None:
        """Consume queued requests and invoke the transfer implementation."""
        torch.npu.set_device(self.device)
        self.ready_event.set()
        while True:
            remote_engine_id, requests = self.request_queue.get()
            request_endpoints = {(request.remote_host, request.remote_port) for request in requests.values()}
            try:
                remote_host, remote_port = self._get_remote_endpoint(request_endpoints)
                failed_request_ids = self._handle_requests(
                    remote_engine_id,
                    remote_host,
                    remote_port,
                    requests,
                )
                for request_id in failed_request_ids:
                    self._mark_request_failed(requests[request_id])
            except Exception as exc:
                if self.can_report_invalid_block_ids:
                    for request_metadata in requests.values():
                        self._mark_request_failed(request_metadata)
                logger.exception(
                    "Mooncake pull failed for remote engine %s at %s: %s",
                    remote_engine_id,
                    sorted(request_endpoints),
                    exc,
                )
            finally:
                for request_id in requests:
                    self.finished_requests.put(request_id)
                self.request_queue.task_done()

    def _mark_request_failed(self, request_metadata: ReqMeta) -> None:
        if not self.can_report_invalid_block_ids:
            return
        with self.invalid_block_ids_lock:
            for group_block_ids in request_metadata.local_block_ids:
                self.invalid_block_ids.update(group_block_ids)

    @staticmethod
    def _get_remote_endpoint(
        endpoints: set[tuple[str, int]],
    ) -> tuple[str, int]:
        if len(endpoints) != 1:
            raise ValueError(
                f"Requests for one remote engine must share one scheduler endpoint, got {sorted(endpoints)}"
            )
        return next(iter(endpoints))

    def _build_remote_transfer_layout(
        self,
        remote_metadata: MooncakeTransferMetadataGroups,
    ) -> tuple[
        dict[int, dict[tuple[int, int], list[list[int]]]],
        dict[int, list[tuple[int, int]]],
    ]:
        """Build per-PP layer matches and producer TP candidates.

        Candidate groups use the same layout as ``remote_tp_rank_groups``:
        one outer entry per required remote head/TP piece and one inner list of
        replica candidates (DCP=1) or participating DCP shards (DCP>1).
        """
        if remote_metadata.use_kv_pp and remote_metadata.dcp_size != 1:
            raise ValueError("Mooncake producer cannot enable KV parallel and DCP together")
        groups_by_pp_rank: dict[int, dict[tuple[int, int], list[list[int]]]] = {}
        layer_pairs_by_pp_rank: dict[int, list[tuple[int, int]]] = {}
        matched_local_layer_indices: set[int] = set()
        # local_spec_index -> topology-derived candidate groups before the
        # per-layer KVPP owner filter. Layers sharing one spec reuse this value.
        raw_tp_rank_groups_by_spec: dict[int, list[list[int]]] = {}

        for remote_pp_rank, pp_metadata in sorted(remote_metadata.metadata_by_pp_rank.items()):
            layer_index_pairs: list[tuple[int, int]] = []
            groups_by_layer_pair: dict[tuple[int, int], list[list[int]]] = {}
            remote_layer_index_by_name = {
                layer_name: layer_index for layer_index, layer_name in enumerate(pp_metadata.layer_names)
            }
            # remote_layer_index -> TP ranks that physically own this layer.
            owner_tp_ranks_by_layer_index: list[set[int]] = [set() for _ in pp_metadata.layer_names]
            if remote_metadata.use_kv_pp:
                for remote_tp_rank, tp_metadata in pp_metadata.metadata_by_tp_rank.items():
                    for owned_remote_layer_index in tp_metadata.layer_indices:
                        owner_tp_ranks_by_layer_index[owned_remote_layer_index].add(remote_tp_rank)

            for local_layer_index, layer_name in enumerate(self.layer_names):
                remote_layer_index = remote_layer_index_by_name.get(layer_name)
                if remote_layer_index is None:
                    continue

                layer_pair = (local_layer_index, remote_layer_index)
                layer_index_pairs.append(layer_pair)
                matched_local_layer_indices.add(local_layer_index)
                local_spec_index = self.spec_indices[local_layer_index]
                raw_tp_rank_groups = raw_tp_rank_groups_by_spec.get(local_spec_index)
                if raw_tp_rank_groups is None:
                    raw_tp_rank_groups = self._get_layer_remote_tp_rank_groups(
                        local_layer_index,
                        remote_layer_index,
                        self.kv_cache_specs[local_spec_index],
                        pp_metadata,
                        remote_metadata.tp_size,
                        remote_metadata.dcp_size,
                    )
                    raw_tp_rank_groups_by_spec[local_spec_index] = raw_tp_rank_groups

                remote_tp_rank_groups = raw_tp_rank_groups
                if remote_metadata.use_kv_pp:
                    owner_tp_ranks = owner_tp_ranks_by_layer_index[remote_layer_index]
                    remote_tp_rank_groups = [
                        [tp_rank for tp_rank in group if tp_rank in owner_tp_ranks] for group in remote_tp_rank_groups
                    ]
                    if any(not group for group in remote_tp_rank_groups):
                        raise ValueError(
                            "Mooncake KV parallel metadata has no producer TP owning "
                            f"layer {layer_name!r} for every required TP group on PP rank {remote_pp_rank}"
                        )
                groups_by_layer_pair[layer_pair] = remote_tp_rank_groups

            layer_pairs_by_pp_rank[remote_pp_rank] = layer_index_pairs
            groups_by_pp_rank[remote_pp_rank] = groups_by_layer_pair

        missing_local_layer_indices = set(range(len(self.layer_names))) - matched_local_layer_indices
        if missing_local_layer_indices:
            missing_local_layers = [
                self.layer_names[layer_index] for layer_index in sorted(missing_local_layer_indices)
            ]
            raise ValueError(
                f"Mooncake producer metadata is missing layers required by this worker: {missing_local_layers}"
            )
        return groups_by_pp_rank, layer_pairs_by_pp_rank

    def _get_layer_remote_tp_rank_groups(
        self,
        local_layer_index: int,
        remote_layer_index: int,
        spec: KVCacheSpec,
        remote_metadata: MooncakePPTransferMetadata,
        remote_tp_size: int,
        remote_dcp_size: int,
    ) -> list[list[int]]:
        """Infer one matched layer's TP strategy and remote rank groups."""
        if isinstance(spec, MambaSpec):
            return self._get_mamba_remote_tp_rank_groups(remote_tp_size)

        fixed_total_num_kv_heads = None
        if isinstance(spec, (AscendSFAIndexerCacheSpec, SlidingWindowMLASpec)):
            local_dcp_size = remote_dcp_size = 1
            local_num_kv_heads = remote_num_kv_heads = 1
            fixed_total_num_kv_heads = 1
        elif isinstance(spec, MLAAttentionSpec):
            local_dcp_size = self.dcp_size
            local_num_kv_heads = remote_num_kv_heads = 1
            fixed_total_num_kv_heads = 1
        # For FA or SWA with kv_heads > 1, must use HND kv_cache layout.
        elif isinstance(spec, SlidingWindowSpec):
            local_dcp_size = remote_dcp_size = 1
            local_num_kv_heads = self.block_shapes[local_layer_index][0][0]
            remote_num_kv_heads = remote_metadata.block_shapes[remote_layer_index][0][0]
        elif isinstance(spec, FullAttentionSpec):
            local_dcp_size = self.dcp_size
            local_num_kv_heads = self.block_shapes[local_layer_index][0][0]
            remote_num_kv_heads = remote_metadata.block_shapes[remote_layer_index][0][0]
        else:
            raise NotImplementedError(f"Mooncake pull has no TP grouping rule for KV cache spec {type(spec).__name__}")

        total_num_kv_heads = self._infer_total_num_kv_heads(
            local_num_kv_heads=local_num_kv_heads,
            remote_num_kv_heads=remote_num_kv_heads,
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_dcp_size,
            remote_dcp_size=remote_dcp_size,
            fixed_total_num_kv_heads=fixed_total_num_kv_heads,
        )
        return self._get_attention_remote_tp_rank_groups(
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_dcp_size,
            remote_dcp_size=remote_dcp_size,
            total_num_kv_heads=total_num_kv_heads,
        )

    def _get_mamba_remote_tp_rank_groups(
        self,
        remote_tp_size: int,
    ) -> list[list[int]]:
        """Map Mamba ranks like non-replicated FullAttention TP shards."""
        larger_tp_size = max(self.tp_size, remote_tp_size)
        smaller_tp_size = min(self.tp_size, remote_tp_size)
        if smaller_tp_size <= 0 or larger_tp_size % smaller_tp_size != 0:
            raise ValueError(
                f"Mooncake Mamba TP sizes must have an integer ratio, got local={self.tp_size}, remote={remote_tp_size}"
            )
        tp_ratio = larger_tp_size // smaller_tp_size
        if remote_tp_size >= self.tp_size:
            start_rank = self.tp_rank * tp_ratio
            return [[remote_tp_rank] for remote_tp_rank in range(start_rank, start_rank + tp_ratio)]
        return [[self.tp_rank // tp_ratio]]

    def _infer_total_num_kv_heads(
        self,
        local_num_kv_heads: int,
        remote_num_kv_heads: int | None,
        remote_tp_size: int,
        local_dcp_size: int,
        remote_dcp_size: int,
        fixed_total_num_kv_heads: int | None,
    ) -> int:
        """Infer and validate the total KV heads for one transfer spec."""
        if fixed_total_num_kv_heads is not None:
            return fixed_total_num_kv_heads
        if remote_num_kv_heads is None:
            raise ValueError("Remote KV head count is required when it cannot be inferred from the cache spec")
        assert max(self.tp_size, remote_tp_size) % min(self.tp_size, remote_tp_size) == 0
        assert max(local_num_kv_heads, remote_num_kv_heads) % min(local_num_kv_heads, remote_num_kv_heads) == 0
        local_head_tp_size = self.tp_size // local_dcp_size
        remote_head_tp_size = remote_tp_size // remote_dcp_size
        inferred_total_heads: set[int] = set()
        if local_num_kv_heads > 1:
            inferred_total_heads.add(local_num_kv_heads * self.tp_size)
        if remote_num_kv_heads > 1:
            inferred_total_heads.add(remote_num_kv_heads * remote_tp_size)
        if local_dcp_size > 1:
            inferred_total_heads.add(local_head_tp_size)
        if remote_dcp_size > 1:
            inferred_total_heads.add(remote_head_tp_size)

        if not inferred_total_heads:
            inferred_total_heads.add(min(local_head_tp_size, remote_head_tp_size))
        if len(inferred_total_heads) != 1:
            raise ValueError(f"Mooncake inferred inconsistent total KV head counts: {sorted(inferred_total_heads)}")

        return inferred_total_heads.pop()

    def _get_attention_remote_tp_rank_groups(
        self,
        remote_tp_size: int,
        local_dcp_size: int,
        remote_dcp_size: int,
        total_num_kv_heads: int,
    ) -> list[list[int]]:
        """Build remote TP groups from an already inferred head topology.

        Each result entry covers one intersection between this local rank's
        head interval and a remote head shard. Its rank list contains either
        interchangeable replicas or all DCP ranks for that remote head shard.
        """
        local_head_tp_size = self.tp_size // local_dcp_size
        remote_head_tp_size = remote_tp_size // remote_dcp_size
        local_head_tp_rank = self.tp_rank // local_dcp_size
        local_head_interval = self._get_head_interval(local_head_tp_rank, local_head_tp_size, total_num_kv_heads)

        ranks_by_head_piece: dict[tuple[int, int], list[int]] = {}
        for remote_head_tp_rank in range(remote_head_tp_size):
            remote_head_interval = self._get_head_interval(remote_head_tp_rank, remote_head_tp_size, total_num_kv_heads)
            head_start = max(local_head_interval[0], remote_head_interval[0])
            head_end = min(local_head_interval[1], remote_head_interval[1])
            if head_start >= head_end:
                continue
            remote_rank_start = remote_head_tp_rank * remote_dcp_size
            ranks_by_head_piece.setdefault((head_start, head_end), []).extend(
                range(remote_rank_start, remote_rank_start + remote_dcp_size)
            )

        if not ranks_by_head_piece:
            raise ValueError(
                "MooncakeConnector found no remote TP group for local rank "
                f"{self.tp_rank}, local_tp={self.tp_size}, remote_tp={remote_tp_size}, "
                f"total_heads={total_num_kv_heads}, local_dcp={local_dcp_size}, remote_dcp={remote_dcp_size}"
            )
        return [ranks for _, ranks in sorted(ranks_by_head_piece.items())]

    @staticmethod
    def _get_head_interval(
        tp_rank: int,
        tp_size: int,
        total_num_kv_heads: int,
    ) -> tuple[int, int]:
        if total_num_kv_heads >= tp_size:
            heads_per_rank = total_num_kv_heads // tp_size
            start = tp_rank * heads_per_rank
            return start, start + heads_per_rank

        replication_size = tp_size // total_num_kv_heads
        head_index = tp_rank // replication_size
        return head_index, head_index + 1

    def _handle_requests(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_port: int,
        requests: dict[str, ReqMeta],
    ) -> set[str]:
        """Build and execute transfer buckets for one producer engine."""
        remote_metadata = self._get_remote_metadata(remote_engine_id, remote_host, remote_port)
        tp_rank_groups_by_pp_rank = self.remote_tp_rank_groups[remote_engine_id]
        layer_pairs_by_pp_rank = self.remote_layer_index_pairs[remote_engine_id]
        # Cache block mappings for this producer-engine batch:
        # request_id -> (local_spec_index, normalized_remote_tp_rank_groups) ->
        # [(selected_remote_tp_rank, local_kernel_block_ids,
        #   remote_kernel_block_ids), ...].
        # Layers with the same spec and candidate topology share their block
        # mapping across PP ranks. Candidate groups are part of the key because
        # KVPP layers using the same spec can have different owner TP ranks.
        transfer_block_ids_by_spec: dict[
            str,
            dict[
                tuple[int, tuple[tuple[int, ...], ...]],
                list[tuple[int, list[int], list[int]]],
            ],
        ] = {}
        # future -> (remote_pp_rank, remote_tp_rank, affected_request_ids).
        # A failed TP task marks only the requests represented in that bucket.
        future_to_task: dict[Future[None], tuple[int, int, set[str]]] = {}
        submission_error: Exception | None = None
        try:
            for remote_pp_rank, layer_pairs in layer_pairs_by_pp_rank.items():
                pp_metadata = remote_metadata.metadata_by_pp_rank[remote_pp_rank]
                transfer_block_buckets, request_ids_by_remote_tp_rank = self._build_transfer_block_buckets(
                    pp_metadata,
                    layer_pairs,
                    tp_rank_groups_by_pp_rank[remote_pp_rank],
                    remote_metadata.dcp_size,
                    requests,
                    transfer_block_ids_by_spec,
                )
                for remote_tp_rank, transfer_entries_by_spec in transfer_block_buckets.items():
                    future = self.executor.submit(
                        self._execute_tp_transfer_bucket,
                        remote_pp_rank,
                        remote_tp_rank,
                        remote_metadata.tp_size,
                        remote_metadata.dcp_size,
                        pp_metadata,
                        transfer_entries_by_spec,
                    )
                    future_to_task[future] = (
                        remote_pp_rank,
                        remote_tp_rank,
                        request_ids_by_remote_tp_rank[remote_tp_rank],
                    )
        except Exception as exc:
            submission_error = exc

        failed_request_ids: set[str] = set()
        for future in as_completed(future_to_task):
            remote_pp_rank, remote_tp_rank, request_ids = future_to_task[future]
            try:
                future.result()
            except Exception:
                logger.exception(
                    "Mooncake transfer task failed for remote PP rank %s, TP rank %s, requests=%s",
                    remote_pp_rank,
                    remote_tp_rank,
                    sorted(request_ids),
                )
                if self.can_report_invalid_block_ids:
                    failed_request_ids.update(request_ids)
                else:
                    logger.warning(
                        "Ignoring Mooncake transfer failure for hybrid KV cache requests %s because "
                        "vLLM invalid block reporting currently supports only one KV cache group",
                        sorted(request_ids),
                    )

        if submission_error is not None:
            raise submission_error
        return failed_request_ids

    def _build_transfer_block_buckets(
        self,
        remote_metadata: MooncakePPTransferMetadata,
        layer_pairs: list[tuple[int, int]],
        tp_rank_groups_by_layer: dict[tuple[int, int], list[list[int]]],
        remote_dcp_size: int,
        requests: dict[str, ReqMeta],
        transfer_block_ids_by_spec: dict[
            str,
            dict[
                tuple[int, tuple[tuple[int, ...], ...]],
                list[tuple[int, list[int], list[int]]],
            ],
        ],
    ) -> tuple[
        dict[int, dict[int, dict[tuple[int, int], list[tuple[str, list[int], list[int]]]]]],
        dict[int, set[str]],
    ]:
        """Bucket one PP rank's requests by remote TP, spec index, and layer pair."""
        # remote_tp_rank -> local_spec_index -> (local_layer_index, remote_layer_index) ->
        # [(request_id, local_kernel_block_ids, remote_kernel_block_ids), ...]
        transfer_block_buckets: dict[
            int,
            dict[
                int,
                dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
            ],
        ] = {}
        # remote_tp_rank -> requests with at least one transfer entry in its
        # bucket; used to attribute a failed TP task to affected requests only.
        request_ids_by_remote_tp_rank: dict[int, set[str]] = {}
        for selection_index, (request_id, request_metadata) in enumerate(requests.items()):
            request_block_ids_by_spec = transfer_block_ids_by_spec.setdefault(request_id, {})
            for local_layer_index, remote_layer_index in layer_pairs:
                spec_index = self.spec_indices[local_layer_index]
                spec = self.kv_cache_specs[spec_index]
                layer_pair = (local_layer_index, remote_layer_index)
                remote_tp_rank_groups = tp_rank_groups_by_layer[layer_pair]
                cache_key = (
                    spec_index,
                    tuple(tuple(group) for group in remote_tp_rank_groups),
                )
                transfer_block_ids = request_block_ids_by_spec.get(cache_key)
                if transfer_block_ids is None:
                    group_index = self.group_indices[local_layer_index]
                    if not isinstance(spec, MambaSpec):
                        local_kernel_block_size = (
                            self.layer_block_sizes[local_layer_index] // self.block_size_scales[local_layer_index][0]
                        )
                        remote_kernel_block_size = (
                            remote_metadata.layer_block_sizes[remote_layer_index]
                            // remote_metadata.block_size_scales[remote_layer_index][0]
                        )
                        assert local_kernel_block_size == remote_kernel_block_size, (
                            "MooncakeConnector does not support different local and remote kernel block size %s | %s.",
                            local_kernel_block_size,
                            remote_kernel_block_size,
                        )
                    transfer_block_ids = self._compute_group_block_ids(
                        request_id,
                        remote_tp_rank_groups,
                        remote_dcp_size,
                        spec_index,
                        self.layer_block_sizes[local_layer_index],
                        remote_metadata.layer_block_sizes[remote_layer_index],
                        request_metadata.local_block_ids[group_index],
                        request_metadata.local_full_block_ids[group_index],
                        request_metadata.remote_block_ids[group_index],
                        request_metadata.local_num_prompt_tokens,
                        request_metadata.remote_num_prompt_tokens,
                        request_metadata.num_computed_tokens,
                        self.block_size_scales[local_layer_index][0],
                        remote_metadata.block_size_scales[remote_layer_index][0],
                        spec,
                        selection_index,
                    )
                    request_block_ids_by_spec[cache_key] = transfer_block_ids

                for remote_tp_rank, local_block_ids, remote_block_ids in transfer_block_ids:
                    request_ids_by_remote_tp_rank.setdefault(remote_tp_rank, set()).add(request_id)
                    transfer_entries_by_spec = transfer_block_buckets.setdefault(remote_tp_rank, {})
                    transfer_entries_by_layer = transfer_entries_by_spec.setdefault(spec_index, {})
                    transfer_entries_by_layer.setdefault(layer_pair, []).append(
                        (request_id, local_block_ids, remote_block_ids)
                    )
        return transfer_block_buckets, request_ids_by_remote_tp_rank

    @staticmethod
    def _expand_block_ids(block_ids: list[int], scale: int) -> list[int]:
        """Expand logical block IDs into contiguous kernel block IDs."""
        if scale == 1:
            return block_ids
        return [block_id * scale + offset for block_id in block_ids for offset in range(scale)]

    @staticmethod
    def _select_remote_tp_rank(candidate_tp_ranks: list[int], selection_index: int) -> int:
        """Select a replicated producer TP rank using round-robin."""
        return candidate_tp_ranks[selection_index % len(candidate_tp_ranks)]

    def _compute_group_block_ids(
        self,
        request_id: str,
        remote_tp_rank_groups: list[list[int]],
        remote_dcp_size: int,
        spec_index: int,
        local_block_size: int,
        remote_block_size: int,
        local_group_block_ids: list[int],
        local_full_group_block_ids: list[int],
        remote_group_block_ids: list[int],
        local_num_prompt_tokens: int,
        remote_num_prompt_tokens: int,
        num_computed_tokens: int,
        local_block_size_scale: int,
        remote_block_size_scale: int,
        spec: KVCacheSpec,
        selection_index: int,
    ) -> list[tuple[int, list[int], list[int]]]:
        """Pair remote TP ranks with local and remote kernel block IDs."""
        is_dcp_transfer = (
            (self.dcp_size > 1 or remote_dcp_size > 1)
            and isinstance(spec, FullAttentionSpec)
            and not isinstance(spec, AscendSFAIndexerCacheSpec)
        )
        if isinstance(spec, SlidingWindowSpec):
            assert local_block_size == remote_block_size, "Mooncake SWA requires the same P/D logical block size."
            local_unhashed_start_idx = len(local_full_group_block_ids) - len(local_group_block_ids)
            local_kernel_block_ids = self._expand_block_ids(local_group_block_ids, local_block_size_scale)
            remote_kernel_block_ids = self._expand_block_ids(
                remote_group_block_ids[local_unhashed_start_idx:], remote_block_size_scale
            )
        elif isinstance(spec, FullAttentionSpec):
            if is_dcp_transfer:
                local_kernel_block_size = local_block_size // local_block_size_scale
                remote_kernel_block_size = remote_block_size // remote_block_size_scale
                if local_kernel_block_size != remote_kernel_block_size:
                    raise ValueError(
                        "Mooncake DCP requires the same P/D kernel block size: "
                        f"local={local_kernel_block_size}, remote={remote_kernel_block_size}"
                    )

                local_virtual_block_size = local_block_size * self.dcp_size
                remote_virtual_block_size = remote_block_size * remote_dcp_size
                local_start_block_idx = num_computed_tokens // local_virtual_block_size
                transfer_block_ids: list[tuple[int, list[int], list[int]]] = []

                if local_block_size != remote_block_size:
                    kernel_block_size = local_kernel_block_size
                    transfer_start_token = num_computed_tokens
                    transfer_end_token = min(local_num_prompt_tokens - 1, remote_num_prompt_tokens)
                    if transfer_end_token <= transfer_start_token:
                        return []

                    local_virtual_kernel_blocks: list[tuple[int, int]] = []
                    for local_block_offset, local_block_id in enumerate(local_group_block_ids):
                        local_block_idx = local_start_block_idx + local_block_offset
                        local_dcp_token_start = (
                            local_block_idx * local_virtual_block_size + self.dcp_rank * local_block_size
                        )
                        for kernel_offset in range(local_block_size_scale):
                            token_start = local_dcp_token_start + kernel_offset * kernel_block_size
                            if token_start + kernel_block_size <= transfer_start_token:
                                continue
                            if token_start >= transfer_end_token:
                                break
                            local_kernel_block_id = local_block_id * local_block_size_scale + kernel_offset
                            virtual_kernel_block_idx = token_start // kernel_block_size
                            local_virtual_kernel_blocks.append((local_kernel_block_id, virtual_kernel_block_idx))

                for candidate_tp_ranks in remote_tp_rank_groups:
                    if remote_dcp_size == 1:
                        remote_tp_rank = (
                            candidate_tp_ranks[0]
                            if len(candidate_tp_ranks) == 1
                            else self._select_remote_tp_rank(candidate_tp_ranks, selection_index)
                        )
                        remote_tp_ranks = [remote_tp_rank]
                    else:
                        if len(candidate_tp_ranks) != remote_dcp_size:
                            raise ValueError(
                                "Mooncake DCP candidate rank count must equal remote DCP size: "
                                f"ranks={candidate_tp_ranks}, remote_dcp={remote_dcp_size}"
                            )
                        remote_tp_ranks = candidate_tp_ranks

                    # remote_tp_rank -> (local_kernel_block_ids,
                    #                    remote_kernel_block_ids).
                    # DCP remapping can split one logical request across ranks.
                    block_ids_by_remote_tp_rank: dict[int, tuple[list[int], list[int]]] = {}
                    if local_block_size != remote_block_size:
                        for local_kernel_block_id, virtual_kernel_block_idx in local_virtual_kernel_blocks:
                            token_start = virtual_kernel_block_idx * kernel_block_size
                            remote_block_idx = token_start // remote_virtual_block_size
                            if remote_block_idx >= len(remote_group_block_ids):
                                break
                            remote_virtual_offset = token_start % remote_virtual_block_size
                            remote_dcp_rank = remote_virtual_offset // remote_block_size
                            remote_kernel_offset = (
                                remote_virtual_offset % remote_block_size
                            ) // remote_kernel_block_size
                            remote_tp_rank = remote_tp_ranks[remote_dcp_rank]
                            local_kernel_block_ids, remote_kernel_block_ids = block_ids_by_remote_tp_rank.setdefault(
                                remote_tp_rank, ([], [])
                            )
                            local_kernel_block_ids.append(local_kernel_block_id)
                            remote_kernel_block_ids.append(
                                remote_group_block_ids[remote_block_idx] * remote_block_size_scale
                                + remote_kernel_offset
                            )
                    else:
                        for local_block_offset, local_block_id in enumerate(local_group_block_ids):
                            local_block_idx = local_start_block_idx + local_block_offset
                            global_block_idx = local_block_idx * self.dcp_size + self.dcp_rank
                            remote_block_idx = global_block_idx // remote_dcp_size
                            if remote_block_idx >= len(remote_group_block_ids):
                                break
                            remote_dcp_rank = global_block_idx % remote_dcp_size
                            remote_tp_rank = remote_tp_ranks[remote_dcp_rank]
                            local_kernel_block_ids, remote_kernel_block_ids = block_ids_by_remote_tp_rank.setdefault(
                                remote_tp_rank, ([], [])
                            )
                            local_kernel_block_ids.extend(
                                local_block_id * local_block_size_scale + offset
                                for offset in range(local_block_size_scale)
                            )
                            remote_block_id = remote_group_block_ids[remote_block_idx]
                            remote_kernel_block_ids.extend(
                                remote_block_id * remote_block_size_scale + offset
                                for offset in range(remote_block_size_scale)
                            )

                    transfer_block_ids.extend(
                        (remote_tp_rank, local_kernel_block_ids, remote_kernel_block_ids)
                        for remote_tp_rank, (
                            local_kernel_block_ids,
                            remote_kernel_block_ids,
                        ) in block_ids_by_remote_tp_rank.items()
                    )
                return transfer_block_ids
            local_kernel_block_ids = self._expand_block_ids(local_group_block_ids, local_block_size_scale)
            remote_kernel_block_ids = self._expand_block_ids(remote_group_block_ids, remote_block_size_scale)
            kernel_block_size = local_block_size // local_block_size_scale
            remote_start_block_idx = num_computed_tokens // kernel_block_size
            remote_kernel_block_ids = remote_kernel_block_ids[remote_start_block_idx:]
        elif isinstance(spec, MambaSpec):
            local_kernel_block_ids = [local_group_block_ids[-self.num_speculative_tokens - 1]]
            remote_kernel_block_ids = [remote_group_block_ids[-1]]
        else:
            raise NotImplementedError(f"Mooncake block ID expansion does not support {type(spec).__name__}")

        num_kernel_blocks = min(len(local_kernel_block_ids), len(remote_kernel_block_ids))
        local_kernel_block_ids = local_kernel_block_ids[:num_kernel_blocks]
        remote_kernel_block_ids = remote_kernel_block_ids[:num_kernel_blocks]
        return [
            (
                candidate_tp_ranks[0]
                if len(candidate_tp_ranks) == 1
                else self._select_remote_tp_rank(candidate_tp_ranks, selection_index),
                local_kernel_block_ids,
                remote_kernel_block_ids,
            )
            for candidate_tp_ranks in remote_tp_rank_groups
        ]

    def _append_mamba_transfer_addresses(
        self,
        spec: MambaSpec,
        remote_tp_rank: int,
        remote_tp_size: int,
        transfer_entries_by_layer: dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
        remote_metadata: MooncakePPTransferMetadata,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
    ) -> None:
        """Append Mamba conv/SSM slices for one producer TP rank."""
        larger_tp_size = max(self.tp_size, remote_tp_size)
        smaller_tp_size = min(self.tp_size, remote_tp_size)
        if larger_tp_size % smaller_tp_size != 0:
            raise ValueError(
                f"Mooncake Mamba TP sizes must have an integer ratio, got local={self.tp_size}, remote={remote_tp_size}"
            )
        tp_ratio = larger_tp_size // smaller_tp_size
        if remote_tp_size >= self.tp_size:
            rank_offset = remote_tp_rank % tp_ratio
            if remote_tp_rank // tp_ratio != self.tp_rank:
                raise ValueError(
                    f"Mooncake Mamba remote TP {remote_tp_rank} does not belong to local TP {self.tp_rank}"
                )
        else:
            rank_offset = self.tp_rank % tp_ratio
            if self.tp_rank // tp_ratio != remote_tp_rank:
                raise ValueError(
                    f"Mooncake Mamba local TP {self.tp_rank} does not belong to remote TP {remote_tp_rank}"
                )

        remote_tp_metadata = remote_metadata.metadata_by_tp_rank[remote_tp_rank]
        for (local_layer_index, remote_layer_index), transfer_entries in transfer_entries_by_layer.items():
            local_base_addrs = self.kv_caches_base_addr[local_layer_index]
            remote_base_addrs = remote_tp_metadata.kv_caches_base_addr[remote_layer_index]
            local_shapes = self.block_shapes[local_layer_index]
            remote_shapes = remote_metadata.block_shapes[remote_layer_index]
            if not (
                len(local_base_addrs)
                == len(remote_base_addrs)
                == len(local_shapes)
                == len(remote_shapes)
                == len(spec.dtypes)
            ):
                raise ValueError(
                    f"Mooncake Mamba cache metadata count mismatch for layer {self.layer_names[local_layer_index]!r}"
                )

            for cache_index, (local_base_addr, remote_base_addr) in enumerate(zip(local_base_addrs, remote_base_addrs)):
                local_block_stride = self.block_strides[local_layer_index][cache_index]
                remote_block_stride = remote_metadata.block_strides[remote_layer_index][cache_index]
                local_block_len = self.block_lens[local_layer_index][cache_index]
                remote_block_len = remote_metadata.block_lens[remote_layer_index][cache_index]
                local_shape = local_shapes[cache_index]
                remote_shape = remote_shapes[cache_index]

                address_slices: list[tuple[int, int, int]]
                if self.tp_size == remote_tp_size:
                    if remote_tp_rank != self.tp_rank or local_block_len != remote_block_len:
                        raise ValueError(
                            f"Mooncake Mamba equal-TP metadata mismatch for layer "
                            f"{self.layer_names[local_layer_index]!r}, cache {cache_index}"
                        )
                    address_slices = [(0, 0, local_block_len)]
                elif cache_index == 0 and len(local_shape) == len(remote_shape) == 2:
                    if local_shape[0] != remote_shape[0]:
                        raise ValueError(
                            f"Mooncake Mamba conv row mismatch for layer {self.layer_names[local_layer_index]!r}: "
                            f"local={local_shape}, remote={remote_shape}"
                        )
                    dtype_size = torch.tensor([], dtype=spec.dtypes[cache_index]).element_size()
                    mamba_type = spec.mamba_type
                    local_projection_widths: tuple[int, ...]
                    remote_projection_widths: tuple[int, ...]
                    if mamba_type == MambaAttentionBackendEnum.MAMBA1:
                        local_projection_widths = (local_shape[1],)
                        remote_projection_widths = (remote_shape[1],)
                    elif mamba_type in (MambaAttentionBackendEnum.MAMBA2, MambaAttentionBackendEnum.GDN_ATTN):
                        if len(local_shapes) < 2 or len(remote_shapes) < 2:
                            raise ValueError(
                                f"Mooncake {mamba_type.name} conv transfer requires a temporal state shape"
                            )
                        local_state_shape = local_shapes[1]
                        remote_state_shape = remote_shapes[1]
                        local_state_width = local_state_shape[0] * local_state_shape[1]
                        remote_state_width = remote_state_shape[0] * remote_state_shape[1]
                        local_remainder = local_shape[1] - local_state_width
                        remote_remainder = remote_shape[1] - remote_state_width
                        if local_remainder <= 0 or remote_remainder <= 0 or local_remainder % 2 or remote_remainder % 2:
                            raise ValueError(
                                f"Mooncake {mamba_type.name} conv shape cannot be decomposed: "
                                f"local={local_shape}, remote={remote_shape}, "
                                f"local_state={local_state_shape}, remote_state={remote_state_shape}"
                            )
                        if mamba_type == MambaAttentionBackendEnum.MAMBA2:
                            local_projection_widths = (
                                local_state_width,
                                local_remainder // 2,
                                local_remainder // 2,
                            )
                            remote_projection_widths = (
                                remote_state_width,
                                remote_remainder // 2,
                                remote_remainder // 2,
                            )
                        else:
                            local_projection_widths = (
                                local_remainder // 2,
                                local_remainder // 2,
                                local_state_width,
                            )
                            remote_projection_widths = (
                                remote_remainder // 2,
                                remote_remainder // 2,
                                remote_state_width,
                            )
                    else:
                        raise NotImplementedError(
                            f"Mooncake unequal-TP conv transfer does not support Mamba type {mamba_type!r}"
                        )
                    if (
                        sum(local_projection_widths) != local_shape[1]
                        or sum(remote_projection_widths) != remote_shape[1]
                    ):
                        raise ValueError(
                            f"Mooncake Mamba conv projections do not match cache shapes for layer "
                            f"{self.layer_names[local_layer_index]!r}: local={local_shape}, remote={remote_shape}"
                        )

                    address_slices = []
                    local_projection_offset = 0
                    remote_projection_offset = 0
                    for local_projection_width, remote_projection_width in zip(
                        local_projection_widths, remote_projection_widths
                    ):
                        if remote_tp_size > self.tp_size:
                            if local_projection_width != remote_projection_width * tp_ratio:
                                raise ValueError("Mooncake Mamba conv projection TP ratio mismatch")
                            local_slice_offset = local_projection_offset + rank_offset * remote_projection_width
                            remote_slice_offset = remote_projection_offset
                            transfer_width = remote_projection_width
                        else:
                            if remote_projection_width != local_projection_width * tp_ratio:
                                raise ValueError("Mooncake Mamba conv projection TP ratio mismatch")
                            local_slice_offset = local_projection_offset
                            remote_slice_offset = remote_projection_offset + rank_offset * local_projection_width
                            transfer_width = local_projection_width
                        for row_index in range(local_shape[0]):
                            address_slices.append(
                                (
                                    (row_index * local_shape[1] + local_slice_offset) * dtype_size,
                                    (row_index * remote_shape[1] + remote_slice_offset) * dtype_size,
                                    transfer_width * dtype_size,
                                )
                            )
                        local_projection_offset += local_projection_width
                        remote_projection_offset += remote_projection_width
                elif remote_tp_size > self.tp_size:
                    if local_block_len != remote_block_len * tp_ratio:
                        raise ValueError(
                            f"Mooncake Mamba state TP ratio mismatch for layer "
                            f"{self.layer_names[local_layer_index]!r}, cache {cache_index}"
                        )
                    address_slices = [(rank_offset * remote_block_len, 0, remote_block_len)]
                else:
                    if remote_block_len != local_block_len * tp_ratio:
                        raise ValueError(
                            f"Mooncake Mamba state TP ratio mismatch for layer "
                            f"{self.layer_names[local_layer_index]!r}, cache {cache_index}"
                        )
                    address_slices = [(0, rank_offset * local_block_len, local_block_len)]

                for _, local_block_ids, remote_block_ids in transfer_entries:
                    for local_block_id, remote_block_id in zip(local_block_ids, remote_block_ids):
                        local_block_addr = local_base_addr + local_block_id * local_block_stride
                        remote_block_addr = remote_base_addr + remote_block_id * remote_block_stride
                        for local_inner_offset, remote_inner_offset, transfer_len in address_slices:
                            src_list.append(local_block_addr + local_inner_offset)
                            dst_list.append(remote_block_addr + remote_inner_offset)
                            length_list.append(transfer_len)

    def _append_spec_transfer_addresses(
        self,
        spec_index: int,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        transfer_entries_by_layer: dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
        remote_metadata: MooncakePPTransferMetadata,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
    ) -> None:
        """Append all layers and requests belonging to one local spec."""
        for (local_layer_index, _), transfer_entries in transfer_entries_by_layer.items():
            for request_id, local_block_ids, remote_block_ids in transfer_entries:
                if len(local_block_ids) != len(remote_block_ids):
                    raise ValueError(
                        f"Mooncake block ID count mismatch for request {request_id!r}, "
                        f"layer {self.layer_names[local_layer_index]!r}: "
                        f"local={len(local_block_ids)}, remote={len(remote_block_ids)}"
                    )
        if not any(
            local_block_ids
            for transfer_entries in transfer_entries_by_layer.values()
            for _, local_block_ids, _ in transfer_entries
        ):
            return

        spec = self.kv_cache_specs[spec_index]
        if isinstance(spec, MambaSpec):
            self._append_mamba_transfer_addresses(
                spec,
                remote_tp_rank,
                remote_tp_size,
                transfer_entries_by_layer,
                remote_metadata,
                src_list,
                dst_list,
                length_list,
            )
            return
        remote_tp_metadata = remote_metadata.metadata_by_tp_rank[remote_tp_rank]
        transfer_whole_block = isinstance(
            spec,
            (MLAAttentionSpec, SlidingWindowMLASpec, AscendSFAIndexerCacheSpec),
        )
        if transfer_whole_block:
            for (local_layer_index, remote_layer_index), transfer_entries in transfer_entries_by_layer.items():
                local_base_addrs = self.kv_caches_base_addr[local_layer_index]
                remote_base_addrs = remote_tp_metadata.kv_caches_base_addr[remote_layer_index]
                if len(local_base_addrs) != len(remote_base_addrs):
                    raise ValueError(
                        f"Mooncake KV tensor count mismatch for layer {self.layer_names[local_layer_index]!r}: "
                        f"local={len(local_base_addrs)}, remote={len(remote_base_addrs)}"
                    )
                for cache_index, (local_base_addr, remote_base_addr) in enumerate(
                    zip(local_base_addrs, remote_base_addrs)
                ):
                    local_block_stride = self.block_strides[local_layer_index][cache_index]
                    remote_block_stride = remote_metadata.block_strides[remote_layer_index][cache_index]
                    local_block_len = self.block_lens[local_layer_index][cache_index]
                    remote_block_len = remote_metadata.block_lens[remote_layer_index][cache_index]
                    if local_block_len != remote_block_len:
                        raise ValueError(
                            f"Mooncake whole-block length mismatch for layer "
                            f"{self.layer_names[local_layer_index]!r}, cache {cache_index}: "
                            f"local={local_block_len}, remote={remote_block_len}"
                        )
                    for _, local_block_ids, remote_block_ids in transfer_entries:
                        for local_block_id, remote_block_id in zip(local_block_ids, remote_block_ids):
                            src_list.append(local_base_addr + local_block_id * local_block_stride)
                            dst_list.append(remote_base_addr + remote_block_id * remote_block_stride)
                            length_list.append(local_block_len)
            return

        if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            raise NotImplementedError(f"Mooncake transfer address calculation does not support {type(spec).__name__}")

        first_local_layer_index, first_remote_layer_index = next(iter(transfer_entries_by_layer))
        if isinstance(spec, SlidingWindowSpec):
            local_attention_dcp_size = remote_attention_dcp_size = 1
        else:
            local_attention_dcp_size = self.dcp_size
            remote_attention_dcp_size = remote_dcp_size
        local_num_heads = self.block_shapes[first_local_layer_index][0][0]
        remote_num_heads = remote_metadata.block_shapes[first_remote_layer_index][0][0]
        total_num_kv_heads = self._infer_total_num_kv_heads(
            local_num_kv_heads=local_num_heads,
            remote_num_kv_heads=remote_num_heads,
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_attention_dcp_size,
            remote_dcp_size=remote_attention_dcp_size,
            fixed_total_num_kv_heads=None,
        )
        local_head_tp_size = self.tp_size // local_attention_dcp_size
        remote_head_tp_size = remote_tp_size // remote_attention_dcp_size
        local_head_tp_rank = self.tp_rank // local_attention_dcp_size
        remote_head_tp_rank = remote_tp_rank // remote_attention_dcp_size
        local_head_start, local_head_end = self._get_head_interval(
            local_head_tp_rank, local_head_tp_size, total_num_kv_heads
        )
        remote_head_start, remote_head_end = self._get_head_interval(
            remote_head_tp_rank, remote_head_tp_size, total_num_kv_heads
        )
        transfer_head_start = max(local_head_start, remote_head_start)
        transfer_head_end = min(local_head_end, remote_head_end)
        if transfer_head_start >= transfer_head_end:
            raise ValueError(
                f"Mooncake found no head overlap for local TP {self.tp_rank} and remote TP {remote_tp_rank}"
            )
        if local_head_end - local_head_start != local_num_heads:
            raise ValueError(
                f"Mooncake local HND head shape does not match inferred topology for spec {spec_index}: "
                f"shape={local_num_heads}, interval={local_head_end - local_head_start}"
            )
        if remote_head_end - remote_head_start != remote_num_heads:
            raise ValueError(
                f"Mooncake remote HND head shape does not match inferred topology for spec {spec_index}: "
                f"shape={remote_num_heads}, interval={remote_head_end - remote_head_start}"
            )

        local_head_offset = transfer_head_start - local_head_start
        remote_head_offset = transfer_head_start - remote_head_start
        num_transfer_heads = transfer_head_end - transfer_head_start
        for (local_layer_index, remote_layer_index), transfer_entries in transfer_entries_by_layer.items():
            local_base_addrs = self.kv_caches_base_addr[local_layer_index]
            remote_base_addrs = remote_tp_metadata.kv_caches_base_addr[remote_layer_index]
            if len(local_base_addrs) != len(remote_base_addrs):
                raise ValueError(
                    f"Mooncake KV tensor count mismatch for layer {self.layer_names[local_layer_index]!r}: "
                    f"local={len(local_base_addrs)}, remote={len(remote_base_addrs)}"
                )
            if self.block_shapes[local_layer_index][0][0] != local_num_heads:
                raise ValueError(f"Mooncake local head count differs within spec {spec_index}")
            if remote_metadata.block_shapes[remote_layer_index][0][0] != remote_num_heads:
                raise ValueError(f"Mooncake remote head count differs within local spec {spec_index}")

            for cache_index, (local_base_addr, remote_base_addr) in enumerate(zip(local_base_addrs, remote_base_addrs)):
                local_block_shape = self.block_shapes[local_layer_index][cache_index]
                remote_block_shape = remote_metadata.block_shapes[remote_layer_index][cache_index]
                if not local_block_shape or local_block_shape[0] != local_num_heads:
                    raise ValueError(
                        f"Mooncake local K/V head counts differ for layer {self.layer_names[local_layer_index]!r}"
                    )
                if not remote_block_shape or remote_block_shape[0] != remote_num_heads:
                    raise ValueError(
                        f"Mooncake remote K/V head counts differ for layer {self.layer_names[local_layer_index]!r}"
                    )

                local_block_stride = self.block_strides[local_layer_index][cache_index]
                remote_block_stride = remote_metadata.block_strides[remote_layer_index][cache_index]
                local_block_len = self.block_lens[local_layer_index][cache_index]
                remote_block_len = remote_metadata.block_lens[remote_layer_index][cache_index]
                if local_block_len % local_num_heads or remote_block_len % remote_num_heads:
                    raise ValueError(
                        f"Mooncake HND block length is not divisible by head count for layer "
                        f"{self.layer_names[local_layer_index]!r}, cache {cache_index}"
                    )
                local_head_len = local_block_len // local_num_heads
                remote_head_len = remote_block_len // remote_num_heads
                if local_head_len != remote_head_len:
                    raise ValueError(
                        f"Mooncake P/D head length mismatch for layer {self.layer_names[local_layer_index]!r}, "
                        f"cache {cache_index}: local={local_head_len}, remote={remote_head_len}"
                    )
                transfer_len = num_transfer_heads * local_head_len
                local_inner_offset = local_head_offset * local_head_len
                remote_inner_offset = remote_head_offset * remote_head_len
                for _, local_block_ids, remote_block_ids in transfer_entries:
                    for local_block_id, remote_block_id in zip(local_block_ids, remote_block_ids):
                        src_list.append(local_base_addr + local_block_id * local_block_stride + local_inner_offset)
                        dst_list.append(remote_base_addr + remote_block_id * remote_block_stride + remote_inner_offset)
                        length_list.append(transfer_len)

    def _execute_tp_transfer_bucket(
        self,
        remote_pp_rank: int,
        remote_tp_rank: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_metadata: MooncakePPTransferMetadata,
        transfer_entries_by_spec: dict[
            int,
            dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
        ],
    ) -> None:
        """Calculate addresses and execute one remote PP/TP transfer bucket.

        ``transfer_entries_by_spec`` maps local spec index to layer-index pairs,
        then to ``(request_id, local_block_ids, remote_block_ids)`` records.
        """
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        for spec_index, transfer_entries_by_layer in transfer_entries_by_spec.items():
            self._append_spec_transfer_addresses(
                spec_index=spec_index,
                remote_tp_rank=remote_tp_rank,
                remote_tp_size=remote_tp_size,
                remote_dcp_size=remote_dcp_size,
                transfer_entries_by_layer=transfer_entries_by_layer,
                remote_metadata=remote_metadata,
                src_list=src_list,
                dst_list=dst_list,
                length_list=length_list,
            )

        if not src_list:
            return
        tp_metadata = remote_metadata.metadata_by_tp_rank[remote_tp_rank]
        session_id = f"{tp_metadata.local_ip}:{tp_metadata.te_rpc_port}"
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            raise RuntimeError(
                f"Mooncake KV transfer failed for remote PP rank {remote_pp_rank}, TP rank {remote_tp_rank}, ret={ret}"
            )

    @staticmethod
    def _validate_remote_metadata(
        metadata: MooncakeTransferMetadataGroups,
        expected_engine_id: str,
    ) -> None:
        if metadata.engine_id != expected_engine_id:
            raise ValueError(
                "Mooncake producer metadata engine ID mismatch: expected "
                f"{expected_engine_id!r}, got {metadata.engine_id!r}"
            )
        if not metadata.metadata_by_pp_rank:
            raise ValueError("Mooncake producer scheduler returned no PP metadata")
        if metadata.pcp_size != 1:
            raise ValueError(f"Mooncake pull temporarily requires remote pcp_size=1, got {metadata.pcp_size}")

        empty_pp_ranks = [
            pp_rank
            for pp_rank, pp_metadata in metadata.metadata_by_pp_rank.items()
            if not pp_metadata.metadata_by_tp_rank
        ]
        if empty_pp_ranks:
            raise ValueError(
                f"Mooncake producer scheduler returned no TP metadata for PP ranks {sorted(empty_pp_ranks)}"
            )
        invalid_tp_ranks = {
            tp_rank
            for pp_metadata in metadata.metadata_by_pp_rank.values()
            for tp_rank in pp_metadata.metadata_by_tp_rank
            if tp_rank < 0 or tp_rank >= metadata.tp_size
        }
        if invalid_tp_ranks:
            raise ValueError(f"Mooncake producer scheduler returned invalid TP ranks {sorted(invalid_tp_ranks)}")

    def _get_remote_metadata(
        self,
        remote_engine_id: str,
        remote_host: str,
        remote_port: int,
    ) -> MooncakeTransferMetadataGroups:
        cached_metadata = self.remote_metadata.get(remote_engine_id)
        if cached_metadata is not None:
            return cached_metadata

        path = make_zmq_path("tcp", remote_host, remote_port)
        with zmq_ctx(zmq.REQ, path) as sock:  # type: ignore[attr-defined]
            sock.setsockopt(zmq.SNDTIMEO, 1000)  # type: ignore[attr-defined]
            sock.setsockopt(zmq.RCVTIMEO, 1000)  # type: ignore[attr-defined]
            ensure_zmq_send(
                sock,
                self.encoder.encode((b"get_meta_msg",)),
                path,
            )
            metadata_bytes = ensure_zmq_recv(sock, path)

        if not metadata_bytes:
            raise RuntimeError(
                f"Mooncake producer scheduler returned no transfer metadata for engine {remote_engine_id!r} from {path}"
            )

        transfer_metadata = self.decoder.decode(metadata_bytes)
        self._validate_remote_metadata(transfer_metadata, remote_engine_id)
        remote_tp_rank_groups, remote_layer_index_pairs = self._build_remote_transfer_layout(transfer_metadata)
        self.remote_metadata[remote_engine_id] = transfer_metadata
        self.remote_tp_rank_groups[remote_engine_id] = remote_tp_rank_groups
        self.remote_layer_index_pairs[remote_engine_id] = remote_layer_index_pairs
        logger.debug(
            "Mooncake remote transfer layout for engine %s: TP candidates=%s, layer index pairs=%s",
            remote_engine_id,
            remote_tp_rank_groups,
            remote_layer_index_pairs,
        )
        return transfer_metadata


class MooncakePullConnectorWorker(MooncakeBaseConnectorWorker):
    """Worker-side framework for Mooncake pull transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._recving_thread: MooncakePullRecvingThread | None = None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register caches and start the D-side pull execution thread."""
        super().register_kv_caches(kv_caches)
        if self.kv_transfer_config.is_kv_consumer:
            if not isinstance(self.xfer_handshake_metadata, MooncakeTransferMetadata):
                raise RuntimeError("Mooncake KV caches were registered without transfer metadata")
            ready_event = threading.Event()
            self._recving_thread = MooncakePullRecvingThread(
                engine=self.engine,
                vllm_config=self.vllm_config,
                kv_cache_config=self.kv_cache_config,
                kv_cache_specs=self.kv_cache_specs,
                layer_name_to_group_index=self.layer_name_to_group_index,
                layer_name_to_spec_index=self.layer_name_to_spec_index,
                local_metadata=self.xfer_handshake_metadata,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                dp_rank=self.dp_rank,
                dp_size=self.dp_size,
                pcp_rank=self.pcp_rank,
                pcp_size=self.pcp_size,
                dcp_rank=self.dcp_rank,
                dcp_size=self.dcp_size,
                device=torch.npu.current_device(),
                ready_event=ready_event,
            )
            self._recving_thread.start()
            if not ready_event.wait(timeout=10):
                raise RuntimeError("Timed out starting Mooncake pull receiving thread")
            if not self._recving_thread.is_alive():
                raise RuntimeError("Mooncake pull receiving thread failed to start")

    def start_load_kv(self, metadata: MooncakeConnectorMetadata) -> None:
        """Start the pull operations described by scheduler metadata."""
        if not metadata.requests:
            return
        if self.kv_transfer_config.is_kv_consumer:
            assert self._recving_thread is not None
            request_groups: dict[str, dict[str, ReqMeta]] = {}
            for request_id, request_metadata in metadata.requests.items():
                requests = request_groups.setdefault(request_metadata.remote_engine_id, {})
                requests[request_id] = request_metadata

            for remote_engine_id, requests in request_groups.items():
                self._recving_thread.add_requests(remote_engine_id, requests)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return requests with completed receive and send operations."""
        finished_recving = (
            self._recving_thread.get_and_clear_finished_requests() if self._recving_thread is not None else set()
        )
        return set(), finished_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return local block IDs whose pull operations failed."""
        if self._recving_thread is None:
            return set()
        return self._recving_thread.get_and_clear_invalid_block_ids()


__all__ = [
    "MooncakePullConnectorWorker",
    "MooncakePullRecvingThread",
]
