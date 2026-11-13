# SPDX-License-Identifier: Apache-2.0
"""SFA cache registration and receive policy, reusing the Mooncake Pull lifecycle."""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
from vllm.logger import logger
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.metadata import MooncakeTransferMetadata
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
        self._sfa_host_kv = staging_reader.host_kv
        self._transfer_targets = staging_reader.get_transfer_targets(self.layer_names, self.kv_caches_base_addr)

    def _get_remote_metadata(self, *args):
        metadata = super()._get_remote_metadata(*args)
        if hasattr(self._sfa_host_kv, "append_prefill_reads") and (metadata.pcp_size != 1 or metadata.dcp_size != 1):
            raise ValueError("Mooncake V2 SFA PD hot/tail reads require a producer without context parallelism")
        return metadata

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
        local_block_groups,
        remote_block_groups,
        local_base_addr,
        remote_base_addr,
        local_block_stride,
        remote_block_stride,
        transfer_len,
        src_list,
        dst_list,
        length_list,
        local_inner_offset=0,
        remote_inner_offset=0,
        transfer_targets=None,
        local_target=None,
        shared_block_groups=None,
        host_kv=None,
    ) -> None:
        # Direct callers may supply unpartitioned pairs. Per-spec plans already
        # partition once in _make_block_group_builder and do not pass host_kv here.
        if transfer_len <= 0:
            return
        if host_kv is not None and host_kv.contains(local_base_addr):
            coalesce = local_block_stride == remote_block_stride == transfer_len
            key = (host_kv.rank, host_kv.size, coalesce)
            prepared = shared_block_groups.get(key) if shared_block_groups is not None else None
            if prepared is None:
                prepared = partition_shared_block_groups(
                    local_block_groups, remote_block_groups, host_kv.rank, host_kv.size, coalesce=coalesce
                )
                if shared_block_groups is not None:
                    shared_block_groups[key] = prepared
            local_block_groups, remote_block_groups = prepared
            if not local_block_groups:
                return
        MooncakePullRecvingThread._append_block_transfer_addresses(
            local_block_groups,
            remote_block_groups,
            local_base_addr,
            remote_base_addr,
            local_block_stride,
            remote_block_stride,
            transfer_len,
            src_list,
            dst_list,
            length_list,
            local_inner_offset,
            remote_inner_offset,
            transfer_targets,
            local_target,
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

    def _get_transfer_register_regions(self, register_regions) -> list[tuple[int, int]]:
        """Select physical regions to register without changing wire metadata."""
        # Manager-owned Host KV is not an HBM READ destination. Keep its
        # addresses in metadata, but register only device storage with the transfer engine.
        device_regions = [
            (ptr, length)
            for ptr, length in zip(register_regions.ptrs, register_regions.lengths)
            if self._sfa_host_kv is None or self._sfa_host_kv.registry.get_register_location(ptr, length) != "cpu"
        ]
        if self._sfa_host_kv is not None:
            device_regions = list(dict.fromkeys([*device_regions, *self._sfa_host_kv.device_register_regions()]))
        return device_regions

    def _create_recving_thread(self, ready_event: threading.Event) -> MooncakeSFAPullRecvingThread:
        host_kv = self._sfa_host_kv
        if host_kv is None:
            raise RuntimeError("Mooncake SFA consumer requires enabled sparse KV offload")
        metadata = self.xfer_handshake_metadata
        if not isinstance(metadata, MooncakeTransferMetadata):
            raise RuntimeError("Mooncake KV caches were registered without transfer metadata")
        reader = MooncakeSFAStagingReader(self.engine, host_kv, metadata.block_lens)
        self._sfa_pd_state = MooncakeSFAPDState(host_kv.main_group_id)
        return self._build_recving_thread(MooncakeSFAPullRecvingThread, ready_event, staging_reader=reader)

    def prepare_sparse_kv_offload(self, metadata: MooncakeSFAConnectorMetadata) -> None:
        """Publish row reservations before MRV2 prepares attention metadata."""
        state = getattr(self, "_sfa_pd_state", None)
        if state is not None:
            state.bind_metadata(metadata)

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
