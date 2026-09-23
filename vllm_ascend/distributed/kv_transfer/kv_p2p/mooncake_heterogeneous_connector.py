"""MooncakeHeterogeneousConnector: P-side connector for heterogeneous PD KV transfer.

Pairs with upstream vLLM's MooncakeConnector on the D side (910B NPU prefill ->
GPU decode over Mooncake RDMA P2P).

Design note: upstream's D-side ``wait_for_layer_load`` is a no-op
(mooncake_connector.py:576), so D cannot receive layer-by-layer — it blocks on
a single FINISH before decoding. This connector therefore drops layerwise and
uses a request-triggered model (mirroring upstream P-side ``send_kv_to_decode``
:1193): on D's pull, transfer all layers at once once P has computed the KV in
``start_load_kv`` (before the forward), then send FINISH.

Timing (mirrors upstream ``record_send_reqs`` :2028):
  request_finished (scheduler marks computed + fills block_ids)
  -> build_connector_meta (puts reqs_to_send into metadata)
  -> start_load_kv (worker: compute addrs + batch_transfer_sync_write
     + mark_task_done -> FINISH)

Coverage:
- Standard attention (k/v 2-tuple NHD->HND): verified.
- MLA (kv_c/k_pe 2-tuple, DeepSeek): reuses the standard branch; layout on
  DeepSeek not yet verified on hardware.
- TP>1: wired (per-rank port kv_port+tp_rank, bootstrap advertises by tp_rank,
  ROUTER-identity multi-rank routing, per-rank hnd_buf from own heads); TP=2
  not yet verified on hardware.
- Mamba (SSM state): not implemented (raises NotImplementedError).
Limits: PP=1, PCP=DCP=1, BF16/FP16, no resharding.

Integration:
  - Reuses MooncakeHeterogeneousXferAdapter for the wire protocol.
  - Reuses global_te + NPU HBM transfer.
  - Does not touch MooncakeLayerwiseConnector (both connectors coexist).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, make_zmq_path
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend import envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_heterogeneous_adapter import (
    MooncakeHeterogeneousXferAdapter,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    collect_storage_merged_register_regions,
    validate_register_region_count,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.worker.forward_context import ForwardContext

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


class MooncakeHeterogeneousConnectorScheduler:
    """P-side scheduler (pure producer; does not receive KV)."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.block_size = vllm_config.cache_config.block_size
        # p_req_id -> (transfer_id, request, local_block_ids)
        self._reqs_need_send: dict[ReqId, tuple[TransferId, Any, list[int]]] = {}

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int, bool]:
        """P is the source, not a consumer, so it never matches external KV."""
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        """Record a request to send (D address comes from kv_transfer_params)."""
        kv_transfer_params = request.kv_transfer_params or {}
        transfer_id = kv_transfer_params.get("transfer_id", "")
        if transfer_id:
            self._reqs_need_send[request.request_id] = (transfer_id, request, [])

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Put requests whose block_ids are filled (computed) into reqs_to_send."""
        meta = MooncakeHeterogeneousConnectorMetadata()
        finished: list[ReqId] = []
        for req_id, (transfer_id, request, local_block_ids) in self._reqs_need_send.items():
            # request_filled already filled local_block_ids when computed
            if local_block_ids:
                meta.reqs_to_send[req_id] = (transfer_id, local_block_ids)
                finished.append(req_id)
        for req_id in finished:
            del self._reqs_need_send[req_id]
        return meta

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict | None]:
        """Mark a request computed and fill block_ids; delay block release until
        the transfer is confirmed (only when tracked and has blocks).

        Mirrors upstream MooncakeConnector.request_finished (:838-892):
          - no transfer_id / not in _reqs_need_send -> release immediately
            (return False), else the scheduler keeps the block but the worker
            never reports completion -> permanent block leak.
          - empty block_ids -> release immediately (return False).
          - tracked and non-empty -> fill block_ids + return True (delayed
            release until get_finished).
        """
        req_id = request.request_id
        if req_id not in self._reqs_need_send:
            return False, None
        if not block_ids:
            return False, None
        transfer_id, request_obj, _ = self._reqs_need_send[req_id]
        self._reqs_need_send[req_id] = (transfer_id, request_obj, list(block_ids))
        return True, None  # delayed block release until get_finished confirms transfer

    def request_finished_all_groups(self, request: Any, block_ids: tuple[list[int], ...]) -> tuple[bool, dict | None]:
        """SupportsHMA contract. Single group for now; use the first group."""
        first_group = block_ids[0] if block_ids else []
        return self.request_finished(request, first_group)


class MooncakeHeterogeneousConnectorWorker:
    """P-side worker: adapter integration + one-shot transfer.

    Transfer is triggered in start_load_kv (before the forward, same timing as
    upstream record_send_reqs :2028).
    """

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.layer_metadata: dict[str, LayerMeta] = {}
        self.num_blocks = kv_cache_config.num_blocks
        self.kv_caches: dict[str, torch.Tensor] = {}
        # Per-layer KV cache spec (standard attention / MLA / Mamba) for _do_transfer dispatch.
        # Build layer_name -> KVCacheSpec from kv_cache_config.kv_cache_groups (same source as
        # layerwise connector :291 self.kv_cache_specs, but indexed by layer_name for the
        # request-triggered layer_name traversal). Standard attention only for now; MLA/Mamba reserved.
        self.kv_cache_specs_by_name: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.kv_cache_groups:
            for ln in group.layer_names:
                self.kv_cache_specs_by_name[ln] = group.kv_cache_spec

        # bind_host: adapter ROUTER bind address (must be local, 0.0.0.0 for all interfaces).
        # Config precedence (mirrors upstream MooncakeConnector :928-932 kv_connector_extra_config):
        #   kv-transfer-config extra_config > ENV > default. Users set device_name/rdma_host/
        #   advertise_host in the kv-transfer-config JSON extra_config (single place, aligned
        #   with upstream); ENV is override/back-compat only.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        # advertise_host: address registered to bootstrap for D to connect — only needs to be
        #   TCP-reachable from D; can be a jump/routed IP (P need not bind it). Default = rdma_host.
        self.bind_host = "0.0.0.0"
        kv_port = vllm_config.kv_transfer_config.kv_port
        # TransferEngine (NPU HBM transfer):
        #   te_host = RPC bind identity (local_server_name); must be a P NIC that is both locally
        #   bindable and RDMA-reachable from D — typically the P IP on the cross-end RDMA subnet
        #   (e.g. the P IP on 910B, reachable from D on the same subnet). Do NOT reuse the
        #   advertise_host jump IP (the jump/routed addr is not a local P IP; bind would fail).
        #   Default get_ip() (same-vendor scenario).
        #   te_device = RDMA NIC whitelist (Mooncake buildDeviceFilter splits on comma). For
        #   cross-end, point it at a NIC on a subnet D can reach (e.g. 910B mlx5_2);
        #   default None enumerates all NICs and D may pick an unreachable one.
        te_device = extra.get("device_name") or envs.VLLM_ASCEND_KV_TRANSFER_DEVICE
        te_host = extra.get("rdma_host") or envs.VLLM_ASCEND_KV_TRANSFER_RDMA_HOST or get_ip()
        self.advertise_host = extra.get("advertise_host") or envs.VLLM_ASCEND_KV_TRANSFER_HOST or te_host
        # TP>1: each rank binds its own port; D connects to the right P rank via bootstrap by tp_rank
        # (TP=2 not yet verified on hardware). PORT_OFFSET shifts per-instance ports on a shared host.
        self.adapter_port = kv_port + self.tp_rank + envs.VLLM_ASCEND_HETEROGENEOUS_CONNECTOR_PORT_OFFSET

        # heterogeneous=True triggers the HeterogeneousRdmaTransport build check (910B->H20 specific;
        # same-vendor NPU connectors do not pass it, avoiding a false check that would fail startup).
        self.engine = global_te.get_transfer_engine(te_host, device_name=te_device, heterogeneous=True)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Adapter (wire protocol; passively recvs D's MooncakeXferMetadata) — binds 0.0.0.0
        self.adapter = MooncakeHeterogeneousXferAdapter(self.bind_host, self.adapter_port, self.total_layers)
        self._completed_transfers: set[TransferId] = set()
        self._failed_transfers: set[TransferId] = set()
        self._transfer_to_req: dict[TransferId, ReqId] = {}  # reverse map for get_finished block release
        self._reported_finished: set[ReqId] = set()  # reqs already reported via get_finished (dedup)
        self._pending_hnd_bufs: dict[TransferId, list[torch.Tensor]] = {}  # hnd_bufs kept until RDMA completes
        self.bootstrap_server: MooncakeBootstrapServer | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register KV memory to the TransferEngine, build layer_metadata, start the adapter."""
        self.kv_caches = kv_caches
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

        # Register the merged contiguous memory region (same approach as layerwise connector :1331)
        register_regions = collect_storage_merged_register_regions(kv_caches)
        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        # Start the adapter listener (passively recvs D's MooncakeXferMetadata)
        self.adapter.start_listener()
        logger.info(
            "MooncakeHeterogeneousConnector worker ready: te_rpc_port=%d, adapter_port=%d",
            self.te_rpc_port,
            self.adapter_port,
        )

        # Bootstrap: P registers its adapter address so D's /query can discover it.
        # Reuses upstream MooncakeBootstrapServer + RegisterWorkerPayload (mooncake_utils.py:44).
        self._start_bootstrap()

    def _start_bootstrap(self) -> None:
        """P starts the bootstrap server (rank 0) and registers its adapter address (all ranks).

        The bootstrap server binds 0.0.0.0 (cross-end D reaches it via P's external IP),
        overriding the 127.0.0.1 (local-only) that get_mooncake_bootstrap_addr returns.
        """
        import time

        import httpx

        bs_host, bs_port = get_mooncake_bootstrap_addr(self.vllm_config)
        # rank 0 starts the bootstrap server, binding 0.0.0.0 so cross-end D can reach it.
        if should_launch_bootstrap_server(self.vllm_config):
            self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", bs_port)
            self.bootstrap_server.start()
            logger.info("Bootstrap server started at 0.0.0.0:%d (cross-end reachable)", bs_port)

        # All P workers register their adapter address (D's /query returns it).
        worker_addr = make_zmq_path("tcp", self.advertise_host, self.adapter_port)
        bs_url = make_zmq_path("http", bs_host, bs_port) + "/register"
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.vllm_config.parallel_config.data_parallel_rank,
            tp_rank=self.tp_rank,
            pp_rank=0,  # PP=1 for now
            addr=worker_addr,
        )
        # Retry registration (the bootstrap server may not be up yet).
        for _ in range(30):
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.post(bs_url, json=payload.model_dump())
                    resp.raise_for_status()
                logger.info("Registered adapter addr %s with bootstrap at %s", worker_addr, bs_url)
                return
            except httpx.ConnectError:
                time.sleep(1)
            except Exception as e:
                logger.error("Bootstrap register failed: %s", e)
                return
        logger.error("Bootstrap register timeout: server at %s not ready", bs_url)

    def start_load_kv(self, metadata: MooncakeHeterogeneousConnectorMetadata) -> None:
        """Transfer trigger point (before the forward, same timing as upstream record_send_reqs).

        For each reqs_to_send: query D's address from the adapter, compute (src,dst,length),
        batch_transfer_sync_write all layers in one shot, mark_task_done triggers FINISH.
        """
        for req_id, (transfer_id, local_block_ids) in metadata.reqs_to_send.items():
            # D pull metadata reaches the adapter via zmq with a delay; the first request may
            # arrive before start_load_kv. Do not skip it (D would decode on empty KV -> garbage);
            # short-poll instead.
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
                # Record the transfer->req map and mark failed so get_finished reports this req
                # as done and the scheduler frees P's KV blocks (request_finished delayed release).
                # Without this the req never enters _completed/_failed_transfers, get_finished
                # never reports it, and P blocks leak forever. D times out on its own via
                # PullReqMeta.expire_time.
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
        _build_transfer_params address computation (TP=1, single group, non-MLA for now):
          - P (NPU) has 2 regions per layer (k, v split), block_len=kv_block_len=262144.
          - D (GPU FA blocks-first) has 1 merged region per layer; to_agent_metadata expands
            it into 2 virtual regions (k@base, v@base+kv_block_len), block_len=524288,
            kv_block_len=262144.
          - Align the two sides by occurrence: 0=k, 1=v.
          - Address formula (TP=1, src/dst_region_offset=0, transfer_len=P kv_block_len):
              src = P_base[occ] + p_bid * P_block_len[occ]
              dst = D_base[occ] + d_bid * D_block_len[occ]
              length = P_kv_block_len   (whole k or v block)

          - Block mapping: P source uses local_block_ids (P's own blocks), D target uses D's
            req_blocks (D's blocks; P/D independently allocated, not 1:1). Single group, first group.
        """
        remote_host = remote_meta["remote_hostname"]
        remote_te_rpc_port = remote_meta["te_rpc_port"]
        remote_layer_metadata = remote_meta["layer_metadata"]
        session_id = f"{remote_host}:{remote_te_rpc_port}"

        # Record transfer_id -> req_id for get_finished block release. Must be assigned before
        # all early-return paths, else get_finished iterating _failed_transfers finds no req_id
        # and P blocks leak forever.
        self._transfer_to_req[transfer_id] = req_id

        # D target block ids (per group). req_blocks: d_req_id -> (transfer_id, list[list[int]]).
        # Pick D blocks for the current transfer_id exactly (mirrors upstream
        # _build_transfer_params picking by d_req_id), so a single D-side ZMQ message batching
        # multiple reqs does not let next(iter()) grab another req's D blocks -> crosstalk.
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

        # P/D block count alignment: upstream takes P's trailing n_remote blocks on partial
        # prefix cache hit. Single group for now; P count should == D count; if P > D take the
        # tail (consistent with upstream).
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
        # Heterogeneous layout: D (vllm 0.28 FA) KV cache is 4D (num_blocks, num_kv_heads,
        # block_size, 2*head_size) with k/v interleaved in the last dim
        # ([..., :head_size]=k, [..., head_size:]=v). P keeps k_cache/v_cache split as
        # (num_blocks, block_size, num_kv_heads, head_size). For each layer, merge P's k/v into
        # D's 4D HND layout, register, and transfer as 1 region. hnd_buf is kept in
        # self._pending_hnd_bufs[transfer_id] and released after get_finished confirms FINISH, so
        # the async RDMA src memory is not GC'd mid-transfer.
        hnd_buffers = self._pending_hnd_bufs.setdefault(transfer_id, [])
        for layer_name, local_lm in self.layer_metadata.items():
            if layer_name not in remote_layer_metadata:
                logger.warning("Layer %s not in remote metadata, skipping", layer_name)
                continue
            remote_lm = remote_layer_metadata[layer_name]
            d_base = remote_lm["kv_caches_base_addr"][0] if remote_lm["kv_caches_base_addr"] else 0
            d_block_len = remote_lm["block_len"][0] if remote_lm["block_len"] else 0
            p_kv = self.kv_caches.get(layer_name)
            # Dispatch layout conversion by model spec (reuses the layerwise approach, see
            # docs/hetero_pd_layerwise_reuse_plan.md). Standard attention only for now; MLA/Mamba reserved.
            spec = self.kv_cache_specs_by_name.get(layer_name)
            if isinstance(spec, MambaSpec):
                # Mamba SSM state transfer (stage 4, not implemented). Mamba has no standard k/v;
                # cache = (conv, ssm). D only registers conv (upstream get_transfer_cache_regions
                # for Mamba returns [conv]). The same-vendor layerwise path transfers conv+ssm
                # (:312-321), involving mamba_cache_mode/num_speculative_tokens/local_transfer_idx
                # — complex, and the region count differs (layerwise 2 regions vs D-side 1 conv),
                # which needs real hardware to clarify. Deferred to a Mamba-specific implementation.
                # Refs: layerwise get_transfer_meta :297-321, upstream utils.py :603-607.
                raise NotImplementedError(
                    "Mamba SSM state transfer not supported yet (stage 4, see docs/hetero_pd_layerwise_reuse_plan.md)"
                )
            if isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec)):
                # MLA (incl. SlidingWindowMLA variant, aligned with upstream :1697): P NPU cache
                # is a 2-tuple (kv_c, k_pe); D GPU is a single tensor kv_c_and_k_pe_cache (kv_c
                # first + k_pe after, concatenated along head_dim, see upstream flashattn_mla.py:338-339).
                # Same structure as standard attention's (k, v) 2-tuple: both are 2-tuple 4D tensors
                # -> index_select target blocks -> permute NHD->HND -> cat along last dim (first
                # tensor before, second after). So reuse the standard attention hnd_buf logic below.
                # TODO(hardware-verify): verify D-side MLA region block_len/address structure is
                # compatible with standard attention (transfer_len from remote_lm.block_len should
                # adapt to MLA's kv_lora_rank+qk_rope_head_dim head_dim). See stage 2 in
                # docs/hetero_pd_layerwise_reuse_plan.md.
                logger.debug(
                    "Layer %s: MLA/SlidingWindow spec, reusing standard hnd_buf branch",
                    layer_name,
                )
            # Merge k/v into D's 4D HND: k_cache[occ0], v_cache[occ1] each (num_blocks, bs, h, d)
            # -> permute(0,2,1,3) -> (num_blocks, h, bs, d) -> cat dim=-1 -> (num_blocks, h, bs, 2d).
            # MLA (kv_c, k_pe) has the same structure and falls into this branch (see MLA note above).
            if (
                isinstance(p_kv, (list, tuple))
                and len(p_kv) >= 2
                and isinstance(p_kv[0], torch.Tensor)
                and isinstance(p_kv[1], torch.Tensor)
                and p_kv[0].dim() == 4
            ):
                k_cache, v_cache = p_kv[0], p_kv[1]
                idx_tensor = torch.tensor(local_block_ids, device=k_cache.device)
                k_gathered = k_cache.index_select(0, idx_tensor).permute(0, 2, 1, 3)  # (n, h, bs, d)
                v_gathered = v_cache.index_select(0, idx_tensor).permute(0, 2, 1, 3)  # (n, h, bs, d)
                hnd_buf = torch.cat([k_gathered, v_gathered], dim=-1).contiguous()  # (n, h, bs, 2d)
                hnd_buffers.append(hnd_buf)
                # hnd_buf does not need register_buffer: PR #759 HeterogeneousRdmaTransport's
                # registerLocalMemory is a no-op for NPU HBM (aclrtPtrAttributes.type != 0);
                # submitTransfer relies on aclrtMemcpyAsync(D2H) to an already-registered hostAddr_
                # then RDMA, not on hnd_buf's own registration. Also, global_te.register_buffer's
                # is_register_buffer flag means "register KV cache storage once"; misusing it for a
                # temporary hnd_buf would set the flag before KV cache registration and short-circuit
                # the later KV cache registration (hidden risk), so it is removed.
                # src stride uses the P hnd_buf's actual block bytes (stride(0)*element_size),
                # not an assumption that it equals D block_len. They are equal under the same model
                # config, but diverge when P/D dtype or head config differ — use the real stride to
                # avoid silent misalignment.
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
        # The heterogeneous transport (PR #759 HeterogeneousRdmaTransport) runs D2H on a separate
        # aclrt stream from the NPU compute stream that builds hnd_buf (index_select/permute/cat).
        # Without a sync, D2H may read a half-written hnd_buf in HBM -> RDMA sends partial/stale
        # data -> D receives wrong values. torch.npu.synchronize() ensures the compute stream is
        # done so D2H reads a complete hnd_buf.
        torch.npu.synchronize()
        ret = self.engine.batch_transfer_sync_write(session_id, src_list, dst_list, length_list)
        ok = ret >= 0
        if not ok:
            logger.error("batch_transfer_sync_write failed for req %s, ret=%d", req_id, ret)
            self._failed_transfers.add(transfer_id)
        # mark_all_tasks_done: in the request-triggered model, mark all tasks done after
        # transferring all layers, which triggers FINISH/ERROR back to D (adapter's expected_tasks
        # are per-layer; all must be marked for FINISH).
        self.adapter.mark_all_tasks_done(transfer_id, ok=ok)
        if ok:
            self._completed_transfers.add(transfer_id)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        """Return finished req_ids (for the scheduler to free blocks).

        Iterate _completed_transfers / _failed_transfers, find the req_ids that have not been
        reported yet, and return them as save_finished_req_ids (the first item, load_finished,
        is always None since P does not load).
        """
        save_finished: set[str] = set()
        for transfer_id in list(self._completed_transfers) + list(self._failed_transfers):
            req_id = self._transfer_to_req.get(transfer_id)
            if req_id is not None and req_id not in self._reported_finished:
                save_finished.add(req_id)
                self._reported_finished.add(req_id)
            # Release this transfer's hnd_buf (RDMA done; src memory can be reclaimed).
            self._pending_hnd_bufs.pop(transfer_id, None)
            # Discard processed transfer state to avoid linear growth on a long-running P service.
            # transfer_id is discarded so get_finished no longer iterates it; its req_id dedup
            # mark can also go (no transfer to trigger means no duplicate report).
            self._completed_transfers.discard(transfer_id)
            self._failed_transfers.discard(transfer_id)
            self._transfer_to_req.pop(transfer_id, None)
        self._reported_finished -= save_finished
        return None, (save_finished or None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def save_kv_layer(self, *args, **kwargs) -> None:
        """Non-layerwise; no-op (mirrors upstream P-side :580)."""
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """P is the sender; no loading needed."""
        pass

    def wait_for_save(self) -> None:
        """Transfer completes synchronously in start_load_kv; no-op here."""
        pass

    def shutdown(self) -> None:
        self.adapter.stop()
        if self.bootstrap_server is not None:
            self.bootstrap_server.shutdown()


class MooncakeHeterogeneousConnector(KVConnectorBase_V1, SupportsHMA):
    """P-side connector facade: delegates to scheduler (scheduler side) + worker (worker side)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._is_kv_producer = vllm_config.kv_transfer_config.is_kv_producer
        self.connector_scheduler: MooncakeHeterogeneousConnectorScheduler | None = None
        self.connector_worker: MooncakeHeterogeneousConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER and kv_cache_config is not None:
            self.connector_scheduler = MooncakeHeterogeneousConnectorScheduler(
                vllm_config, kv_cache_config, str(vllm_config.kv_transfer_config.engine_id)
            )
        elif role == KVConnectorRole.WORKER and kv_cache_config is not None:
            self.connector_worker = MooncakeHeterogeneousConnectorWorker(
                vllm_config, kv_cache_config, str(vllm_config.kv_transfer_config.engine_id)
            )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict) -> bool:
        """Non-layerwise (one-shot); no piecewise support needed."""
        return False

    # ---- Scheduler-side (delegated) ----
    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int) -> tuple[int, bool]:
        if self.connector_scheduler:
            return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)
        return 0, False

    def update_state_after_alloc(self, request: Any, blocks: Any, num_external_tokens: int) -> None:
        if self.connector_scheduler:
            self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        if self.connector_scheduler:
            return self.connector_scheduler.build_connector_meta(scheduler_output)
        return MooncakeHeterogeneousConnectorMetadata()

    def request_finished(self, request: Any, block_ids: list[int]) -> tuple[bool, dict | None]:
        if self.connector_scheduler:
            return self.connector_scheduler.request_finished(request, block_ids)
        return False, None

    def request_finished_all_groups(self, request: Any, block_ids: tuple[list[int], ...]) -> tuple[bool, dict | None]:
        if self.connector_scheduler:
            return self.connector_scheduler.request_finished_all_groups(request, block_ids)
        return False, None

    # ---- Worker-side (delegated) ----
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if self.connector_worker:
            self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        if self.connector_worker:
            return self.connector_worker.get_finished(finished_req_ids)
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.connector_worker:
            return self.connector_worker.get_block_ids_with_load_errors()
        return set()

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        if self.connector_worker and isinstance(self._connector_metadata, MooncakeHeterogeneousConnectorMetadata):
            self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        pass

    def wait_for_save(self) -> None:
        pass
