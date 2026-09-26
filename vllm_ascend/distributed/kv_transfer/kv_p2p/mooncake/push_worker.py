# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""P-side worker for heterogeneous Mooncake KV transfer (910B NPU -> GPU decode).

Inherits the V2 ``MooncakeBaseConnectorWorker`` to reuse topology collection
(TP/PP/DP/PCP/DCP), spec dedup (``_build_kv_cache_spec_mappings``), and stats,
while overriding the heterogeneous-specific pieces:

  * ``__init__`` pre-creates the TransferEngine with ``heterogeneous=True``
    (HeterogeneousRdmaTransport build check) and wires the ZMQ ROUTER adapter.
  * ``register_kv_caches`` registers the merged KV storage (NHD) and builds
    per-layer ``LayerMeta`` instead of the V2 configured-region path.
  * ``start_load_kv`` performs the request-triggered one-shot transfer:
    NHD->HND layout conversion + ``npu.synchronize`` +
    ``batch_transfer_sync_write`` + ``mark_all_tasks_done`` (FINISH).

Pairs with upstream vLLM's V1 ``MooncakeConnector`` on the D side (P-push wire
protocol); the D side is unchanged.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import (
    MooncakeBaseConnectorWorker,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.push_adapter import (
    MooncakeHeterogeneousXferAdapter,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    global_te,
)
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    validate_register_region_count,
)

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

ReqId = str
TransferId = str


@dataclass
class LayerMeta:
    """Per-layer KV cache address metadata (simplified LayerMetadata)."""

    kv_caches_base_addr: list[int]
    block_len: list[int]


@dataclass
class MooncakeHeterogeneousConnectorMetadata(KVConnectorMetadata):
    """Send requests passed from scheduler to worker.

    reqs_to_send: p_req_id -> (transfer_id, local_block_ids)
      transfer_id: D-assigned transfer id (from kv_transfer_params)
      local_block_ids: P-side computed block ids (filled in request_finished)
    """

    reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = field(default_factory=dict)
    reqs_not_processed: list[TransferId] = field(default_factory=list)


class MooncakeHeterogeneousPushConnectorWorker(MooncakeBaseConnectorWorker):
    """P-side worker: adapter integration + one-shot heterogeneous transfer.

    Transfer is triggered in ``start_load_kv`` (before the forward, same timing
    as upstream ``record_send_reqs``).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Pre-create the heterogeneous engine BEFORE super().__init__() so the
        # base's get_transfer_engine() call returns this same singleton. The
        # heterogeneous build check (HeterogeneousRdmaTransport symbols) runs once
        # here; the base call is a no-op singleton return.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        te_device = extra.get("device_name") or envs.VLLM_ASCEND_KV_TRANSFER_DEVICE
        te_host = extra.get("rdma_host") or envs.VLLM_ASCEND_KV_TRANSFER_RDMA_HOST or get_ip()
        global_te.get_transfer_engine(te_host, device_name=te_device, heterogeneous=True)

        super().__init__(vllm_config, engine_id, kv_cache_config)

        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.layer_metadata: dict[str, LayerMeta] = {}
        self.kv_caches: dict[str, torch.Tensor] = {}
        # Per-layer KV cache spec (standard attention / MLA / Mamba) for
        # _do_transfer dispatch, indexed by layer_name for the request-triggered
        # traversal. Filled in register_kv_caches from the base spec dedup.
        self.kv_cache_specs_by_name: dict[str, KVCacheSpec] = {}

        # bind_host: adapter ROUTER bind address (0.0.0.0 for all interfaces).
        # Config precedence (mirrors upstream MooncakeConnector
        # kv_connector_extra_config): extra_config > ENV > default. Users set
        # device_name/rdma_host/advertise_host in the kv-transfer-config JSON
        # extra_config.
        self.bind_host = "0.0.0.0"
        kv_port = vllm_config.kv_transfer_config.kv_port
        # advertise_host: address registered to bootstrap for D to connect — only
        # needs to be TCP-reachable from D; can be a jump/routed IP. Default =
        # te_host.
        self.advertise_host = extra.get("advertise_host") or envs.VLLM_ASCEND_KV_TRANSFER_HOST or te_host
        # TP>1: each rank binds its own port; D connects to the right P rank via
        # bootstrap by tp_rank. PORT_OFFSET shifts per-instance ports on a host.
        self.adapter_port = kv_port + self.tp_rank + envs.VLLM_ASCEND_HETEROGENEOUS_CONNECTOR_PORT_OFFSET

        # Adapter (wire protocol; passively recvs D's MooncakeXferMetadata).
        self.adapter = MooncakeHeterogeneousXferAdapter(self.bind_host, self.adapter_port, self.total_layers)
        self._completed_transfers: set[TransferId] = set()
        self._failed_transfers: set[TransferId] = set()
        self._transfer_to_req: dict[TransferId, ReqId] = {}
        self._reported_finished: set[ReqId] = set()
        self._pending_hnd_bufs: dict[TransferId, list[torch.Tensor]] = {}
        self.bootstrap_server: MooncakeBootstrapServer | None = None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> None:
        """Register KV memory to the TransferEngine, build layer_metadata, start the adapter.

        Reuses the base ``_build_kv_cache_spec_mappings`` for spec dedup, but
        registers the merged KV storage (NHD) and builds per-layer LayerMeta
        instead of the V2 configured-region + MooncakeTransferMetadata path.
        """
        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        self._build_kv_cache_spec_mappings()
        self.kv_cache_specs_by_name = {
            layer_name: self.kv_cache_specs[idx] for layer_name, idx in self.layer_name_to_spec_index.items()
        }

        for layer_name, kv_cache_tuple in kv_caches.items():
            if not isinstance(kv_cache_tuple, (list, tuple)):
                kv_cache_tuple = [kv_cache_tuple]
            layer_meta = LayerMeta(kv_caches_base_addr=[], block_len=[])
            for single_kv_cache in kv_cache_tuple:
                block_start_rank = 1
                block_shape = single_kv_cache.shape[block_start_rank:]
                layer_meta.kv_caches_base_addr.append(single_kv_cache.data_ptr())
                layer_meta.block_len.append(single_kv_cache.element_size() * math.prod(block_shape))
            self.layer_metadata[layer_name] = layer_meta
            logger.info(
                "Registered layer %s: %d regions, block_len=%s",
                layer_name,
                len(layer_meta.kv_caches_base_addr),
                layer_meta.block_len,
            )

        # Register the merged contiguous memory region (same approach as the
        # layerwise connector).
        register_regions = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        # Start the adapter listener (passively recvs D's MooncakeXferMetadata).
        self.adapter.start_listener()
        logger.info(
            "MooncakeHeterogeneousPushConnector worker ready: te_rpc_port=%d, adapter_port=%d",
            self.te_rpc_port,
            self.adapter_port,
        )

        # Bootstrap: P registers its adapter address so D's /query can discover it.
        self._start_bootstrap()

    def _start_bootstrap(self) -> None:
        """P starts the bootstrap server (rank 0) and registers its adapter address (all ranks).

        The bootstrap server binds 0.0.0.0 (cross-end D reaches it via P's
        external IP), overriding the 127.0.0.1 (local-only) that
        get_mooncake_bootstrap_addr returns.
        """
        import httpx

        bs_host, bs_port = get_mooncake_bootstrap_addr(self.vllm_config)
        if should_launch_bootstrap_server(self.vllm_config):
            self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", bs_port)
            self.bootstrap_server.start()
            logger.info(
                "Bootstrap server started at 0.0.0.0:%d (cross-end reachable)",
                bs_port,
            )

        worker_addr = make_zmq_path("tcp", self.advertise_host, self.adapter_port)
        bs_url = make_zmq_path("http", bs_host, bs_port) + "/register"
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            tp_rank=self.tp_rank,
            pp_rank=0,  # PP=1 for now
            addr=worker_addr,
        )
        for _ in range(30):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(bs_url, json=payload.model_dump())
                    resp.raise_for_status()
                logger.info(
                    "Registered adapter addr %s with bootstrap at %s",
                    worker_addr,
                    bs_url,
                )
                return
            except httpx.ConnectError:
                time.sleep(1)
            except Exception as e:
                logger.error("Bootstrap register failed: %s", e)
                return
        logger.error("Bootstrap register timeout: server at %s not ready", bs_url)

    def start_load_kv(self, metadata: MooncakeHeterogeneousConnectorMetadata) -> None:
        """Transfer trigger point (before the forward, same timing as upstream record_send_reqs).

        For each reqs_to_send: query D's address from the adapter, compute
        (src,dst,length), batch_transfer_sync_write all layers in one shot,
        mark_task_done triggers FINISH.
        """
        for req_id, (transfer_id, local_block_ids) in metadata.reqs_to_send.items():
            # D pull metadata reaches the adapter via zmq with a delay; the first
            # request may arrive before start_load_kv. Do not skip it (D would
            # decode on empty KV -> garbage); short-poll instead.
            remote_meta = None
            for _ in range(50):  # poll up to ~5s (50x100ms)
                remote_meta = self.adapter.get_remote_metadata(transfer_id)
                if remote_meta is not None:
                    break
                time.sleep(0.1)
            if remote_meta is None:
                logger.warning(
                    "No remote metadata for transfer %s (req %s) after waiting, "
                    "marking transfer failed to free P blocks",
                    transfer_id,
                    req_id,
                )
                self._transfer_to_req[transfer_id] = req_id
                self._failed_transfers.add(transfer_id)
                continue
            self._do_transfer(req_id, transfer_id, local_block_ids, remote_meta)

    def _do_transfer(
        self,
        req_id: ReqId,
        transfer_id: TransferId,
        local_block_ids: list[int],
        remote_meta: dict,
    ) -> None:
        """Transfer all layers' KV to D in one shot (heterogeneous layout adaptation).

        Mirrors upstream _expand_transfer_regions + _align_transfer_regions +
        _build_transfer_params address computation (TP=1, single group, non-MLA):
          - P (NPU) has 2 regions per layer (k, v split), block_len=kv_block_len.
          - D (GPU FA blocks-first) has 1 merged region per layer;
            to_agent_metadata expands it into 2 virtual regions, block_len covers
            k+v, kv_block_len is the k or v half.
          - Align the two sides by occurrence: 0=k, 1=v.
          - Address formula (TP=1, src/dst_region_offset=0):
              src = P_base[occ] + p_bid * P_block_len[occ]
              dst = D_base[occ] + d_bid * D_block_len[occ]
              length = P_kv_block_len (whole k or v block)
          - Block mapping: P source uses local_block_ids, D target uses D's
            req_blocks (P/D independently allocated, not 1:1). Single group, first.
        """
        remote_host = remote_meta["remote_hostname"]
        remote_te_rpc_port = remote_meta["te_rpc_port"]
        remote_layer_metadata = remote_meta["layer_metadata"]
        session_id = f"{remote_host}:{remote_te_rpc_port}"

        # Record transfer_id -> req_id for get_finished block release. Must be
        # assigned before all early-return paths, else get_finished iterating
        # _failed_transfers finds no req_id and P blocks leak forever.
        self._transfer_to_req[transfer_id] = req_id

        # D target block ids (per group). req_blocks: d_req_id ->
        # (transfer_id, list[list[int]]). Pick D blocks for the current
        # transfer_id exactly (mirrors upstream _build_transfer_params picking by
        # d_req_id), so a single D-side ZMQ message batching multiple reqs does
        # not let next(iter()) grab another req's D blocks -> crosstalk.
        req_blocks: dict = remote_meta.get("req_blocks", {})
        d_block_ids: list[int] = []
        for _d_req_id, (_tid, _d_groups) in req_blocks.items():
            if _tid == transfer_id:
                d_block_ids = list(_d_groups[0]) if _d_groups else []
                break
        if not d_block_ids:
            logger.warning(
                "No D block ids in req_blocks for transfer %s (req %s), skipping",
                transfer_id,
                req_id,
            )
            self.adapter.mark_all_tasks_done(transfer_id, ok=False)
            self._failed_transfers.add(transfer_id)
            return

        # P/D block count alignment: upstream takes P's trailing n_remote blocks
        # on partial prefix cache hit. Single group for now; P count should == D
        # count; if P > D take the tail (consistent with upstream).
        if len(local_block_ids) > len(d_block_ids):
            local_block_ids = local_block_ids[-len(d_block_ids) :]
        if len(local_block_ids) < len(d_block_ids):
            logger.error(
                "req %s: P blocks(%d) < D blocks(%d), cannot transfer",
                req_id,
                len(local_block_ids),
                len(d_block_ids),
            )
            self.adapter.mark_all_tasks_done(transfer_id, ok=False)
            self._failed_transfers.add(transfer_id)
            return

        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        # Heterogeneous layout: D (vllm 0.28 FA) KV cache is 4D (num_blocks,
        # num_kv_heads, block_size, 2*head_size) with k/v interleaved in the last
        # dim ([..., :head_size]=k, [..., head_size:]=v). P keeps k_cache/v_cache
        # split as (num_blocks, block_size, num_kv_heads, head_size). For each
        # layer, merge P's k/v into D's 4D HND layout, register, and transfer as
        # 1 region. hnd_buf is kept in _pending_hnd_bufs[transfer_id] and released
        # after get_finished confirms FINISH, so the async RDMA src memory is not
        # GC'd mid-transfer.
        hnd_buffers = self._pending_hnd_bufs.setdefault(transfer_id, [])
        for layer_name, local_lm in self.layer_metadata.items():
            if layer_name not in remote_layer_metadata:
                logger.warning("Layer %s not in remote metadata, skipping", layer_name)
                continue
            remote_lm = remote_layer_metadata[layer_name]
            d_base = remote_lm["kv_caches_base_addr"][0] if remote_lm["kv_caches_base_addr"] else 0
            d_block_len = remote_lm["block_len"][0] if remote_lm["block_len"] else 0
            p_kv = self.kv_caches.get(layer_name)
            # Dispatch layout conversion by model spec. Standard attention only
            # for now; MLA/Mamba reserved.
            spec = self.kv_cache_specs_by_name.get(layer_name)
            if isinstance(spec, MambaSpec):
                # Mamba SSM state transfer (stage 4, not implemented). Mamba has no
                # standard k/v; cache = (conv, ssm). D only registers conv. The
                # same-vendor layerwise path transfers conv+ssm, involving
                # mamba_cache_mode/num_speculative_tokens/local_transfer_idx —
                # complex, and the region count differs, which needs real
                # hardware to clarify. Deferred to a Mamba-specific implementation.
                raise NotImplementedError("Mamba SSM state transfer not supported yet (stage 4)")
            if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                # MLA (incl. SlidingWindowMLA variant): P NPU cache is a 2-tuple
                # (kv_c, k_pe); D GPU is a single tensor kv_c_and_k_pe_cache
                # (kv_c first + k_pe after, concatenated along head_dim). Same
                # structure as standard attention's (k, v) 2-tuple: both are
                # 2-tuple 4D tensors -> index_select -> permute NHD->HND -> cat
                # along last dim. So reuse the standard attention hnd_buf logic
                # below.
                logger.debug(
                    "Layer %s: MLA/SlidingWindow spec, reusing standard hnd_buf branch",
                    layer_name,
                )
            # Merge k/v into D's 4D HND: k_cache[occ0], v_cache[occ1] each
            # (num_blocks, bs, h, d) -> permute(0,2,1,3) -> (num_blocks, h, bs, d)
            # -> cat dim=-1 -> (num_blocks, h, bs, 2d). MLA (kv_c, k_pe) has the
            # same structure and falls into this branch.
            if (
                isinstance(p_kv, (list, tuple))
                and len(p_kv) >= 2
                and isinstance(p_kv[0], torch.Tensor)
                and isinstance(p_kv[1], torch.Tensor)
                and p_kv[0].dim() == 4
            ):
                k_cache, v_cache = p_kv[0], p_kv[1]
                idx_tensor = torch.tensor(local_block_ids, device=k_cache.device)
                k_gathered = k_cache.index_select(0, idx_tensor).permute(0, 2, 1, 3)
                v_gathered = v_cache.index_select(0, idx_tensor).permute(0, 2, 1, 3)
                hnd_buf = torch.cat([k_gathered, v_gathered], dim=-1).contiguous()
                hnd_buffers.append(hnd_buf)
                # hnd_buf does not need register_buffer: PR #759
                # HeterogeneousRdmaTransport's registerLocalMemory is a no-op for
                # NPU HBM (aclrtPtrAttributes.type != 0); submitTransfer relies on
                # aclrtMemcpyAsync(D2H) to an already-registered hostAddr_ then
                # RDMA. Also, global_te.register_buffer's is_register_buffer flag
                # means "register KV cache storage once"; misusing it for a
                # temporary hnd_buf would set the flag before KV cache registration
                # and short-circuit the later KV cache registration.
                # src stride uses the P hnd_buf's actual block bytes
                # (stride(0)*element_size), not an assumption that it equals D
                # block_len. They are equal under the same model config, but
                # diverge when P/D dtype or head config differ — use the real
                # stride to avoid silent misalignment.
                transfer_len = hnd_buf.stride(0) * hnd_buf.element_size()
                if transfer_len != d_block_len:
                    logger.warning(
                        "Layer %s: P hnd_buf block stride (%d) != D block_len (%d), "
                        "using P stride for src; verify P/D model config consistency",
                        layer_name,
                        transfer_len,
                        d_block_len,
                    )
                hnd_base = hnd_buf.data_ptr()
                for i, d_bid in enumerate(d_block_ids):
                    src = hnd_base + i * transfer_len
                    dst = d_base + d_bid * d_block_len
                    src_list.append(src)
                    dst_list.append(dst)
                    length_list.append(transfer_len)
            else:
                # Fallback: raw NHD passthrough (defensive; should not reach here).
                num_regions = min(
                    len(local_lm.kv_caches_base_addr),
                    len(remote_lm["kv_caches_base_addr"]),
                )
                for occ in range(num_regions):
                    p_base = local_lm.kv_caches_base_addr[occ]
                    p_block_len = local_lm.block_len[occ] if occ < len(local_lm.block_len) else 0
                    for p_bid, d_bid in zip(local_block_ids, d_block_ids):
                        src = p_base + p_bid * p_block_len
                        dst = d_base + d_bid * d_block_len
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(p_block_len)

        if not src_list:
            logger.warning("No transfer entries for req %s, marking done as failed", req_id)
            self.adapter.mark_all_tasks_done(transfer_id, ok=False)
            self._failed_transfers.add(transfer_id)
            return

        logger.info(
            "Transfer req %s (transfer %s): %d entries to %s",
            req_id,
            transfer_id,
            len(src_list),
            session_id,
        )
        # The heterogeneous transport (PR #759 HeterogeneousRdmaTransport) runs D2H
        # on a separate aclrt stream from the NPU compute stream that builds
        # hnd_buf (index_select/permute/cat). Without a sync, D2H may read a
        # half-written hnd_buf in HBM -> RDMA sends partial/stale data -> D
        # receives wrong values. torch.npu.synchronize() ensures the compute
        # stream is done so D2H reads a complete hnd_buf.
        torch.npu.synchronize()
        ret = self.engine.batch_transfer_sync_write(session_id, src_list, dst_list, length_list)
        ok = ret >= 0
        if not ok:
            logger.error("batch_transfer_sync_write failed for req %s, ret=%d", req_id, ret)
            self._failed_transfers.add(transfer_id)
        # mark_all_tasks_done: in the request-triggered model, mark all tasks
        # done after transferring all layers, which triggers FINISH/ERROR back to
        # D (adapter's expected_tasks are per-layer; all must be marked for FINISH).
        self.adapter.mark_all_tasks_done(transfer_id, ok=ok)
        if ok:
            self._completed_transfers.add(transfer_id)

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return finished req_ids (those that reached FINISH state, for the scheduler to free blocks).

        Iterate _completed_transfers / _failed_transfers, find the req_ids that
        have not been reported yet, and return them as save_finished_req_ids
        (the first item, load_finished, is always empty since P does not load).
        """
        save_finished: set[str] = set()
        for transfer_id in list(self._completed_transfers) + list(self._failed_transfers):
            req_id = self._transfer_to_req.get(transfer_id)
            if req_id is not None and req_id not in self._reported_finished:
                save_finished.add(req_id)
                self._reported_finished.add(req_id)
            # Release this transfer's hnd_buf (RDMA done; src memory can be reclaimed).
            self._pending_hnd_bufs.pop(transfer_id, None)
            # Discard processed transfer state to avoid linear growth on a
            # long-running P service. transfer_id is discarded so get_finished no
            # longer iterates it; its req_id dedup mark can also go.
            self._completed_transfers.discard(transfer_id)
            self._failed_transfers.discard(transfer_id)
            self._transfer_to_req.pop(transfer_id, None)
        self._reported_finished -= save_finished
        return set(), save_finished

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def shutdown(self) -> None:
        self.adapter.stop()
        if self.bootstrap_server is not None:
            self.bootstrap_server.shutdown()


__all__ = [
    "LayerMeta",
    "MooncakeHeterogeneousConnectorMetadata",
    "MooncakeHeterogeneousPushConnectorWorker",
]
