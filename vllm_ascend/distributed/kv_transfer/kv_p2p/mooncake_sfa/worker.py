# SPDX-License-Identifier: Apache-2.0
"""SFA cache registration and receive policy, reusing the Mooncake Pull lifecycle."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)

from vllm_ascend.core.kv_cache_interface import AscendSFAIndexerCacheSpec, is_circular_kv_cache_spec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import (
    MooncakePPTransferMetadata,
    MooncakeTransferMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_worker import (
    MooncakePullConnectorWorker,
    MooncakePullRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.utils import group_concurrent_contiguous
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAPDState
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.host_partition import partition_shared_block_groups
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import (
    MooncakeSFAConnectorMetadata,
    MooncakeSFAReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.staging import MooncakeSFAStagingReader

_STAGING_DRAIN_RETRY_INTERVAL_SECONDS = 10


class MooncakeSFAPullRecvingThread(MooncakePullRecvingThread):
    def __init__(self, *args, staging_reader: MooncakeSFAStagingReader, **kwargs) -> None:
        if staging_reader is None:
            raise ValueError("Mooncake SFA receiving requires a staging reader")
        super().__init__(*args, **kwargs)
        self.staging_reader = staging_reader
        self._sfa_batch_requests: dict[str, ReqMeta] | None = None
        self._sfa_host_kv = staging_reader.host_kv
        self._transfer_targets = staging_reader.get_transfer_targets(self.layer_names, self.kv_caches_base_addr)

    def _get_remote_metadata(self, *args):
        metadata = super()._get_remote_metadata(*args)
        if hasattr(self._sfa_host_kv, "append_prefill_reads") and (metadata.pcp_size != 1 or metadata.dcp_size != 1):
            raise ValueError("Mooncake V2 SFA PD hot/tail reads require a producer without context parallelism")
        return metadata

    def _handle_requests(
        self, remote_engine_id: str, remote_host: str, remote_port: int, requests: dict[str, ReqMeta]
    ) -> set[str]:
        """Retain request resources until every submitted Host landing has stopped."""
        # The inherited loop handles one batch at a time and joins every
        # submitted bucket before returning. Keep this read-only context
        # available to executor workers until their landing has drained.
        self._sfa_batch_requests = requests
        try:
            return super()._handle_requests(remote_engine_id, remote_host, remote_port, requests)
        finally:
            # The shared loop marks failures and reports completion only after
            # this returns. A failed wait must retain owners and keep retrying.
            self._wait_for_pending_landing()
            self._sfa_batch_requests = None

    def _wait_for_pending_landing(self) -> None:
        while True:
            try:
                self.staging_reader.drain_pending_landing()
                return
            except Exception:
                logger.exception(
                    "Mooncake Host landing drain failed; retaining request resources and retrying in %s seconds",
                    _STAGING_DRAIN_RETRY_INTERVAL_SECONDS,
                )
                time.sleep(_STAGING_DRAIN_RETRY_INTERVAL_SECONDS)

    def _make_block_group_builder(self, block_ids_by_request):
        block_groups_by_mode: dict[Any, dict[str, tuple[list[list[int]], list[list[int]]]]] = {}
        host_kv = getattr(self, "_sfa_host_kv", None)

        def get_block_groups(coalesce, local_base_addr, local_target):
            offloaded = getattr(local_target, "offloaded", None)
            shared_host = host_kv is not None and (
                host_kv.contains(local_base_addr) if offloaded is None else offloaded
            )
            key = (coalesce, shared_host)
            prepared = block_groups_by_mode.get(key)
            if prepared is None:
                prepared = {}
                for request_id, (local_block_ids, remote_block_ids) in block_ids_by_request.items():
                    if shared_host:
                        assert host_kv is not None
                        # Ownership already sorts and coalesces the original pairs;
                        # grouping them before that only creates work to undo.
                        groups = partition_shared_block_groups(
                            [local_block_ids], [remote_block_ids], host_kv.rank, host_kv.size, coalesce=coalesce
                        )
                    elif coalesce:
                        groups = group_concurrent_contiguous(local_block_ids, remote_block_ids)
                    else:
                        groups = (
                            [[block_id] for block_id in local_block_ids],
                            [[block_id] for block_id in remote_block_ids],
                        )
                    prepared[request_id] = groups
                block_groups_by_mode[key] = prepared
            return prepared

        return get_block_groups

    @staticmethod
    def _append_block_transfer_addresses(
        local_block_groups: list[list[int]],
        remote_block_groups: list[list[int]],
        local_base_addr: int,
        remote_base_addr: int,
        local_block_stride: int,
        remote_block_stride: int,
        transfer_len: int,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        local_inner_offset: int = 0,
        remote_inner_offset: int = 0,
        transfer_targets: list[Any] | None = None,
        local_target: Any | None = None,
    ) -> None:
        """Append addresses from block groups prepared for one request."""
        if transfer_len <= 0:
            return
        for local_block_group, remote_block_group in zip(local_block_groups, remote_block_groups):
            src_list.append(local_base_addr + local_block_group[0] * local_block_stride + local_inner_offset)
            dst_list.append(remote_base_addr + remote_block_group[0] * remote_block_stride + remote_inner_offset)
            length_list.append(transfer_len * len(local_block_group))
            if transfer_targets is not None:
                assert local_target is not None
                transfer_targets.append(local_target)

    def _append_spec_transfer_addresses(
        self,
        spec_index: int,
        remote_pcp_rank: int,
        remote_tp_rank: int,
        remote_pcp_size: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        transfer_entries_by_layer: dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
        remote_metadata: MooncakePPTransferMetadata,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        transfer_targets: list[Any] | None = None,
    ) -> None:
        """Append all layers and requests belonging to one local spec."""
        spec = self.kv_cache_specs[spec_index]
        if isinstance(spec, MambaSpec):
            self._append_mamba_transfer_addresses(
                spec,
                remote_pcp_rank,
                remote_tp_rank,
                remote_tp_size,
                transfer_entries_by_layer,
                remote_metadata,
                src_list,
                dst_list,
                length_list,
            )
            return

        block_ids_by_request: dict[str, tuple[list[int], list[int]]] = {}
        for (local_layer_index, _), transfer_entries in transfer_entries_by_layer.items():
            for request_id, local_block_ids, remote_block_ids in transfer_entries:
                if len(local_block_ids) != len(remote_block_ids):
                    raise ValueError(
                        f"Mooncake block ID count mismatch for request {request_id!r}, "
                        f"layer {self.layer_names[local_layer_index]!r}: "
                        f"local={len(local_block_ids)}, remote={len(remote_block_ids)}"
                    )
                block_ids = (local_block_ids, remote_block_ids)
                previous_block_ids = block_ids_by_request.setdefault(request_id, block_ids)
                if previous_block_ids != block_ids:
                    raise ValueError(f"Mooncake block IDs differ within spec {spec_index} for request {request_id!r}")

        if not any(local_block_ids for local_block_ids, _ in block_ids_by_request.values()):
            return

        # Resolve policy once per spec; per-range address construction stays direct.
        get_block_groups = self._make_block_group_builder(block_ids_by_request)
        target_views = self._get_transfer_target_views() if transfer_targets is not None else None
        append_addresses = self._append_block_transfer_addresses

        remote_tp_metadata = remote_metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
        transfer_whole_block = is_circular_kv_cache_spec(spec) or isinstance(
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
                    if is_circular_kv_cache_spec(spec):
                        local_shape = self.block_shapes[local_layer_index][cache_index]
                        remote_shape = remote_metadata.block_shapes[remote_layer_index][cache_index]
                        if local_shape != remote_shape:
                            raise ValueError(
                                f"Mooncake circular cache layout mismatch for layer "
                                f"{self.layer_names[local_layer_index]!r}, cache {cache_index}: "
                                f"local={local_shape}, remote={remote_shape}"
                            )
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
                    if local_block_len <= 0:
                        continue
                    local_target = None
                    if transfer_targets is not None:
                        assert target_views is not None
                        local_target = target_views[local_layer_index][cache_index]
                    block_ids_to_transfer = get_block_groups(
                        local_block_stride == remote_block_stride == local_block_len, local_base_addr, local_target
                    )
                    for request_id, _, _ in transfer_entries:
                        local_block_groups, remote_block_groups = block_ids_to_transfer[request_id]
                        append_addresses(
                            local_block_groups=local_block_groups,
                            remote_block_groups=remote_block_groups,
                            local_base_addr=local_base_addr,
                            remote_base_addr=remote_base_addr,
                            local_block_stride=local_block_stride,
                            remote_block_stride=remote_block_stride,
                            transfer_len=local_block_len,
                            src_list=src_list,
                            dst_list=dst_list,
                            length_list=length_list,
                            transfer_targets=transfer_targets,
                            local_target=local_target,
                        )
            return

        if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)):
            raise NotImplementedError(f"Mooncake transfer address calculation does not support {type(spec).__name__}")

        first_local_layer_index, first_remote_layer_index = next(iter(transfer_entries_by_layer))
        local_num_heads = self.block_shapes[first_local_layer_index][0][0]
        remote_num_heads = remote_metadata.block_shapes[first_remote_layer_index][0][0]
        local_attention_dcp_size, remote_attention_dcp_size = self._get_attention_dcp_sizes(
            spec, first_remote_layer_index, remote_metadata, remote_dcp_size
        )
        total_num_kv_heads = self._infer_total_num_kv_heads(
            local_num_kv_heads=local_num_heads,
            remote_num_kv_heads=remote_num_heads,
            local_pcp_size=self.pcp_size,
            remote_pcp_size=remote_pcp_size,
            remote_tp_size=remote_tp_size,
            local_dcp_size=local_attention_dcp_size,
            remote_dcp_size=remote_attention_dcp_size,
            fixed_total_num_kv_heads=None,
        )
        local_dcp_tp_size = 1 if local_attention_dcp_size == 1 else local_attention_dcp_size // self.pcp_size
        remote_dcp_tp_size = 1 if remote_attention_dcp_size == 1 else remote_attention_dcp_size // remote_pcp_size
        local_head_tp_size = self.tp_size // local_dcp_tp_size
        remote_head_tp_size = remote_tp_size // remote_dcp_tp_size
        local_head_tp_rank = self.tp_rank // local_dcp_tp_size
        remote_head_tp_rank = remote_tp_rank // remote_dcp_tp_size
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
                if transfer_len <= 0:
                    continue
                local_inner_offset = local_head_offset * local_head_len
                remote_inner_offset = remote_head_offset * remote_head_len
                local_target = None
                if transfer_targets is not None:
                    assert target_views is not None
                    local_target = target_views[local_layer_index][cache_index]
                block_ids_to_transfer = get_block_groups(
                    local_block_stride == local_block_len == transfer_len
                    and remote_block_stride == remote_block_len == transfer_len,
                    local_base_addr,
                    local_target,
                )
                for request_id, _, _ in transfer_entries:
                    local_block_groups, remote_block_groups = block_ids_to_transfer[request_id]
                    append_addresses(
                        local_block_groups=local_block_groups,
                        remote_block_groups=remote_block_groups,
                        local_base_addr=local_base_addr,
                        remote_base_addr=remote_base_addr,
                        local_block_stride=local_block_stride,
                        remote_block_stride=remote_block_stride,
                        transfer_len=transfer_len,
                        local_inner_offset=local_inner_offset,
                        remote_inner_offset=remote_inner_offset,
                        src_list=src_list,
                        dst_list=dst_list,
                        length_list=length_list,
                        transfer_targets=transfer_targets,
                        local_target=local_target,
                    )

    def _execute_tp_transfer_bucket(
        self,
        remote_pp_rank: int,
        remote_pcp_rank: int,
        remote_tp_rank: int,
        remote_pcp_size: int,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_metadata: MooncakePPTransferMetadata,
        transfer_entries_by_spec: dict[
            int,
            dict[tuple[int, int], list[tuple[str, list[int], list[int]]]],
        ],
        *,
        requests: dict[str, ReqMeta] | None = None,
    ) -> None:
        """Calculate addresses and execute one remote PP/PCP/TP transfer bucket.

        ``transfer_entries_by_spec`` maps local spec index to layer-index pairs,
        then to ``(request_id, local_block_ids, remote_block_ids)`` records.
        """
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        # Preserve local target identities alongside the computed addresses.
        # Mamba's distinct state-copy layout keeps its existing generic path.
        transfer_targets = self._create_transfer_targets(transfer_entries_by_spec)
        for spec_index, transfer_entries_by_layer in transfer_entries_by_spec.items():
            self._append_spec_transfer_addresses(
                spec_index=spec_index,
                remote_pcp_rank=remote_pcp_rank,
                remote_tp_rank=remote_tp_rank,
                remote_pcp_size=remote_pcp_size,
                remote_tp_size=remote_tp_size,
                remote_dcp_size=remote_dcp_size,
                transfer_entries_by_layer=transfer_entries_by_layer,
                remote_metadata=remote_metadata,
                src_list=src_list,
                dst_list=dst_list,
                length_list=length_list,
                transfer_targets=transfer_targets,
            )

        if requests is None:
            requests = getattr(self, "_sfa_batch_requests", None)
        self._append_additional_reads(
            remote_metadata,
            remote_pcp_rank,
            remote_tp_rank,
            transfer_entries_by_spec,
            requests,
            src_list,
            dst_list,
            length_list,
            transfer_targets,
        )
        if not src_list:
            return
        tp_metadata = remote_metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
        session_id = f"{tp_metadata.local_ip}:{tp_metadata.te_rpc_port}"
        request_ids = sorted(
            {
                request_id
                for transfer_entries_by_layer in transfer_entries_by_spec.values()
                for transfer_entries in transfer_entries_by_layer.values()
                for request_id, _, _ in transfer_entries
            }
        )
        transfer_start_time = time.perf_counter()
        ret = self._read_transfer_bucket(session_id, src_list, dst_list, length_list, request_ids, transfer_targets)
        if ret < 0:
            raise RuntimeError(
                "Mooncake KV transfer failed for remote "
                f"PP/PCP/TP rank {remote_pp_rank}/{remote_pcp_rank}/{remote_tp_rank}, ret={ret}"
            )
        transfer_elapsed_ms = (time.perf_counter() - transfer_start_time) * 1000
        logger.info(
            "KV cache transfer for requests %s took %.2f ms (%d ranges, %d bytes). "
            "local_ip %s local_device_id %s remote_session_id %s",
            request_ids,
            transfer_elapsed_ms,
            len(length_list),
            sum(length_list),
            self.local_metadata.local_ip,
            self.tp_rank,
            session_id,
        )

    def _create_transfer_targets(self, transfer_entries_by_spec) -> list[Any] | None:
        return (
            []
            if getattr(self, "_transfer_targets", None) is not None
            and not any(isinstance(self.kv_cache_specs[i], MambaSpec) for i in transfer_entries_by_spec)
            else None
        )

    def _get_transfer_target_views(self):
        return self._transfer_targets

    def _append_additional_reads(
        self,
        remote_metadata,
        remote_pcp_rank,
        remote_tp_rank,
        transfer_entries_by_spec,
        requests,
        src_list,
        dst_list,
        length_list,
        transfer_targets,
    ) -> None:
        if requests is None or not hasattr(getattr(self, "_sfa_host_kv", None), "append_prefill_reads"):
            return
        tp_metadata = remote_metadata.metadata_by_pcp_rank[remote_pcp_rank].metadata_by_tp_rank[remote_tp_rank]
        host_kv = self._sfa_host_kv
        for entries_by_layer in transfer_entries_by_spec.values():
            for (local_index, remote_index), entries in entries_by_layer.items():
                name = self.layer_names[local_index]
                if name not in host_kv.hot_caches:
                    continue
                for request_id in dict.fromkeys(entry[0] for entry in entries):
                    host_kv.append_prefill_reads(
                        name,
                        requests[request_id],
                        tp_metadata.kv_caches_base_addr[remote_index],
                        remote_metadata.block_strides[remote_index],
                        remote_metadata.block_lens[remote_index],
                        src_list,
                        dst_list,
                        length_list,
                        transfer_targets,
                    )

    def _read_transfer_bucket(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        request_ids: list[str],
        transfer_targets: list[Any] | None,
    ) -> int:
        # Success includes landing and event drain, before completion can
        # release the producer's blocks or make Host KV visible to decode.
        return self.staging_reader.batch_transfer_sync_read(
            session_id,
            src_list,
            dst_list,
            length_list,
            ",".join(request_ids),
            **({"transfer_targets": transfer_targets} if transfer_targets is not None else {}),
        )


class MooncakeSFAConnectorWorker(MooncakePullConnectorWorker):
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]) -> None:
        """Project SFA cache views before entering the shared registration lifecycle."""
        super().register_kv_caches(self._prepare_transfer_caches(kv_caches))

    def _prepare_transfer_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Select the persistent cache views published in the handshake."""
        self._sfa_host_kv = None
        sparse_cfg = getattr(self.ascend_config, "sparse_kv_offload_config", None)
        if sparse_cfg is not None and sparse_cfg.enabled and self.kv_transfer_config.is_kv_consumer:
            # MRV2 has already registered the manager before connector setup.
            # Only persistent Host main KV and device indexer views go on wire.
            from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.adapter import MooncakeSFAHostKV
            from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
                get_sparse_kv_offload_manager,
            )

            self._sfa_host_kv = MooncakeSFAHostKV(get_sparse_kv_offload_manager(), kv_caches)
            kv_caches = self._sfa_host_kv.transfer_caches
        return kv_caches

    def _register_transfer_buffers(self, ptrs: list[int], lengths: list[int]) -> None:
        """Select physical regions to register without changing wire metadata."""
        # Manager-owned Host KV is not an HBM READ destination. Keep its
        # addresses in metadata, but register only device storage with the transfer engine.
        device_regions = [
            (ptr, length)
            for ptr, length in zip(ptrs, lengths)
            if self._sfa_host_kv is None or self._sfa_host_kv.registry.get_register_location(ptr, length) != "cpu"
        ]
        if self._sfa_host_kv is not None:
            device_regions = list(dict.fromkeys([*device_regions, *self._sfa_host_kv.device_register_regions()]))
        super()._register_transfer_buffers([ptr for ptr, _ in device_regions], [length for _, length in device_regions])

    def _create_recving_thread(self, ready_event: threading.Event) -> MooncakeSFAPullRecvingThread:
        host_kv = self._sfa_host_kv
        if host_kv is None:
            raise RuntimeError("Mooncake SFA consumer requires enabled sparse KV offload")
        metadata = self.xfer_handshake_metadata
        if not isinstance(metadata, MooncakeTransferMetadata):
            raise RuntimeError("Mooncake KV caches were registered without transfer metadata")
        reader = MooncakeSFAStagingReader(
            self.engine,
            host_kv,
            metadata.block_lens,
            staging_config=self.ascend_config.mooncake_staging_config,
        )
        self._sfa_pd_state = MooncakeSFAPDState(host_kv.main_group_id)
        return self._build_recving_thread(MooncakeSFAPullRecvingThread, ready_event, staging_reader=reader)

    def prepare_sparse_kv_offload(self, metadata: MooncakeSFAConnectorMetadata) -> None:
        """Publish row reservations before MRV2 prepares attention metadata."""
        state = getattr(self, "_sfa_pd_state", None)
        if state is not None:
            state.bind_metadata(metadata)

    def prepare_copy_sfa_decode(self, input_batch, slots, block_tables, dense_fills, *, restore_tails: bool):
        state = getattr(self, "_sfa_pd_state", None)
        if state is None:
            return restore_tails, dense_fills, None
        host_kv = self._sfa_host_kv
        if host_kv is None:
            raise RuntimeError("Mooncake SFA PD state requires registered Host KV")
        return state.prepare_decode(
            input_batch,
            slots,
            block_tables,
            dense_fills,
            restore_tails=restore_tails,
            manager=host_kv.manager,
            block_size=self.vllm_config.cache_config.block_size,
            hot_tokens=self.ascend_config.sparse_kv_offload_config.topk_buffer_size,
        )

    def start_load_kv(self, metadata) -> None:
        self.prepare_sparse_kv_offload(metadata)
        state = getattr(self, "_sfa_pd_state", None)
        if state is not None:
            for request_id, request in metadata.requests.items():
                if not isinstance(request, MooncakeSFAReqMeta) or request.copy_sfa_slot != state.bindings[request_id]:
                    raise ValueError("Mooncake V2 SFA PD transfer differs from its reserved hot row")
        super().start_load_kv(metadata)

    def get_transfer_results(self):
        results = super().get_transfer_results()
        state = getattr(self, "_sfa_pd_state", None)
        if state is not None:
            state.record_received(results.finished_recving, results.failed_recving)
        return results
