# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UCX link bootstrap and registry for the LWD host data plane.

``data_plane_comm: host`` has no global comm domain: the edge and the cloud run
as separate torch worlds, so link endpoints derive from identity alone --
``lwd_config.role``, ``instance_id`` and the per-connection ``LwdDPConnection``
from ``lwd_config.connections`` (whose ``connection_id`` carries edge/cloud/dp
and whose DPs carry addr/data_port; topology ranks are absent in host mode, and
in edge_share the view spans every logical edge of the shared engine). No
global ranks, no LwdParallelGroupsSpec, no world collectives.

Link model:
  * per DP connection, TWO UHC1 links (UP/DOWN), each an independent
    LwdUcxChannel with its own UCP endpoint, IO thread and block pool;
  * the cloud DP binds ONE shared ``LwdUcxListener`` on its ``data_port``,
    reused by every edge connected to that DP; both links of each connection
    attach via ``serve_via``; the handshake token (blake2b over
    "edge_id:cloud_id:dp_idx:channel") routes each incoming TCP connection to
    its channel;
  * the edge never listens: its UP/send and DOWN/recv IOs connect out (UHC1 is
    full-duplex, so which side binds the handshake socket is independent of
    the data direction);
  * endpoint process within an instance: dp_rank == dp_idx and TP rank 0;
  * warmup: only the edge waits for its links (``wait_ready``, budget
    HS_TIMEOUT_S + 5). The cloud starts before the edges, so it just reports
    the listener bound and adopts edges asynchronously; no cross-world barrier
    exists to join.

Block sizing comes from the topology (``deployment.data_plane_host``):
``up_buffer_size`` (None = derived per model by ``ucx_block_policy``) and
``down_buffer_size`` (used as configured).
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.config.lwd.topology import LwdHostDataPlane, LwdQosDataPlane
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import logger
from vllm.v1.lwd_comm.data_comm.ucx_channel import LwdUcxListener

from vllm_ascend.distributed.lwd_data_plane.host import ucx_block_policy
from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import (
    BLOCKS_PER_CHANNEL,
    LwdNpuUcxChannelIo,
    LwdNpuUcxChannelIoConfig,
)
from vllm_ascend.distributed.lwd_data_plane.lwd_types import LwdChannelType

# (edge_id, cloud_id, dp_idx, channel, op) -> LwdNpuUcxChannelIo
_UCX_IOS: dict[tuple[int, int, int, LwdChannelType, str], LwdNpuUcxChannelIo] = {}
# (cloud_id, dp_idx) -> the cloud DP's shared listener (handshake entry only;
# data flows over RoCE). There is no stop(): the accept thread is a daemon and
# dies with the process.
_UCX_LISTENERS: dict[tuple[int, int], LwdUcxListener] = {}
_UCX_CHANNELS_INITIALIZED = False


# Cached at channel-init time: the config is only set inside the init context,
# and get_current_vllm_config() raises at request time after vLLM clears it, so
# request-path readers must use the cache.
_CACHED_COMM: str | None = None


def lwd_data_plane_comm() -> str:
    """Transport comm selected by the topology file ('device'|'host').
    The YAML is the single source of truth -- there is no env override.

    The first call MUST happen during channel init (worker init_device),
    where the current config is guaranteed; it caches the value for the
    request path, where the global config is no longer set."""
    global _CACHED_COMM
    if _CACHED_COMM is not None:
        return _CACHED_COMM
    lwd_config = get_current_vllm_config().lwd_config
    if lwd_config is None:
        comm = "device"
    else:
        comm = lwd_config.topology.deployment.data_plane_comm
    _CACHED_COMM = comm
    return comm


def lwd_data_plane_is_host() -> bool:
    return lwd_data_plane_comm() == "host"


def _link_token(edge_id: int, cloud_id: int, dp_idx: int, channel: LwdChannelType) -> int:
    """Channel token, identical on both peers of a link and mismatch-proof.

    Derived from "edge_id:cloud_id:dp_idx:channel" (blake2b-8), with dp_idx
    standing in for the engine index.
    """
    raw = f"{edge_id}:{cloud_id}:{dp_idx}:{channel.value}:1"
    token = int.from_bytes(hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest(), "little")
    return token or 1


def _my_dp_identity() -> tuple[int, int]:
    """(dp_rank, tp_rank) of this process within its OWN instance world --
    in host comm these are instance-local (the edge and cloud worlds are
    independent), never global LWD ranks."""
    parallel = get_current_vllm_config().parallel_config
    # `data_parallel_index` keeps the launched DP rank, while
    # `data_parallel_rank` is reset to 0 by
    # `reconfigure_for_independent_dp_rank` in every non-MoE DP engine process.
    # Reading the reset field would make all cloud DPs claim dp_rank 0 and bind
    # the same listener port.
    dp_rank = getattr(parallel, "data_parallel_index", None)
    if dp_rank is None:
        dp_rank = getattr(parallel, "data_parallel_rank", 0)
    return int(dp_rank or 0), int(get_tensor_model_parallel_rank())


def _hf_attr(cfg: Any, name: str) -> Any:
    """Attribute of an hf_config (or its nested text_config), or None.

    Dict configs are supported.
    """
    if isinstance(cfg, dict):
        return cfg.get(name)
    return getattr(cfg, name, None)


def _qos_data_plane() -> LwdQosDataPlane:
    """The top-level ``qos.data_plane`` block of the topology, or an
    empty default when omitted (UCX defaults then apply)."""
    qos = get_current_vllm_config().lwd_config.topology.qos
    if qos is None or qos.data_plane is None:
        return LwdQosDataPlane()
    return qos.data_plane


def _host_plane() -> LwdHostDataPlane:
    """The topology's ``deployment.data_plane_host`` block; every field
    (and the block itself) is optional -- defaults apply when omitted."""
    host_plane = get_current_vllm_config().lwd_config.topology.deployment.data_plane_host
    if host_plane is None:
        host_plane = LwdHostDataPlane()
    return host_plane


def _resolve_up_block_mb() -> int:
    """Both peers derive the same UP block from the same model config.

    Multimodal/MoE models (e.g. qwen3_5) keep the real values inside
    ``hf_config.text_config``, so every sizing attribute falls back to it when
    the top level is absent.
    """
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        raise RuntimeError("[LWD] host UP block auto-sizing failed: model_config.hf_config is unavailable")
    text_config = _hf_attr(hf_config, "text_config")

    def attr(name: str) -> Any:
        value = _hf_attr(hf_config, name)
        if value is None and text_config is not None:
            value = _hf_attr(text_config, name)
        return value

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    if max_num_batched_tokens is None:
        raise RuntimeError(
            "[LWD] host UP block auto-sizing failed: scheduler_config.max_num_batched_tokens is unavailable"
        )
    try:
        plan = ucx_block_policy.resolve(
            model_type=attr("model_type") or "",
            hidden_size=attr("hidden_size"),
            num_hash_layers=attr("num_hash_layers") or 0,
            num_experts_per_tok=attr("num_experts_per_tok") or 0,
            max_num_batched_tokens=max_num_batched_tokens,
            explicit_up_block_mb=_host_plane().up_buffer_size,
        )
    except ucx_block_policy.LwdUcxBlockPolicyError as exc:
        raise RuntimeError(f"[LWD] host UP block auto-sizing failed: {exc}") from exc
    # Keep this record format stable: the log-analysis tooling parses it.
    for warning in plan.warnings:
        logger.warning("[LWD][ucx-block] %s", warning)
    logger.info(
        "[LWD][ucx-block] model_type=%s max_num_batched_tokens=%d "
        "per_token=%d raw=%.1fMiB auto=%dMiB source=%s up_buffer_size=%d",
        plan.model_type or "<unknown>",
        plan.max_num_batched_tokens,
        plan.per_token_elements,
        plan.raw_mib,
        plan.auto_up_block_mb,
        plan.source,
        plan.up_block_mb,
    )
    return plan.up_block_mb


# Per-rank host data-plane memory budget in MiB: an estimate above it fails
# startup. 12 GiB covers the largest UP/DOWN block pair in use today.
_HOST_MEM_BUDGET_MB = 12 * 1024


def _host_memory_need_mb(channels: int, *, up_block_mb: int, down_block_mb: int) -> int:
    """Host data-plane memory estimate: one pool per channel.

    A send direction reuses its channel's pool as D2H staging, so pools =
    channels; ``channels=2`` (UP/DOWN per connection) gives
    BLOCKS_PER_CHANNEL * (up + down) per connection pair.
    """
    up_mb = int(up_block_mb)
    down_mb = int(down_block_mb)
    pairs = channels // 2
    return pairs * BLOCKS_PER_CHANNEL * (up_mb + down_mb)


def lwd_init_host_channels() -> None:
    """Create this process's UCX links from its role identity. Idempotent.

    Called by ``lwd_channels.lwd_init_channels`` when the topology
    selects ``data_plane_comm: host``. Identity-driven: an edge connects out
    to every linked cloud DP of every logical edge it serves (in edge_share
    the shared engine owns them all); a cloud instance binds one
    listener per its own DP and adopts edges asynchronously. The cloud
    endpoint process owns the connections of its own DP (dp_rank == dp_idx);
    the edge owns every connection of its instance. In both cases only TP rank
    0 creates IO objects.
    """
    global _UCX_CHANNELS_INITIALIZED
    if _UCX_CHANNELS_INITIALIZED:
        return
    lwd_config = get_current_vllm_config().lwd_config
    if lwd_config is None:
        raise RuntimeError("[LWD] the host data plane requires lwd_config")
    role = lwd_config.role
    instance_id = lwd_config.instance_id
    # The LwdConfig view, not the per-instance topology view: in edge_share
    # the ONE shared edge engine must own every logical edge's connections,
    # as the scheduler, control plane and queue consumers do. A per-instance
    # view here would leave the other frontends' connections without IO and
    # fail their submits with "no host send IO".
    connections = lwd_config.connections
    if not connections:
        raise RuntimeError(f"[LWD] no DP connections for role={role} instance={instance_id}")
    dp_rank, tp_rank = _my_dp_identity()
    up_block_mb = _resolve_up_block_mb()
    down_block_mb = _host_plane().down_buffer_size
    # Sidecar deployment: a dedicated transport daemon owns the listeners and
    # engines ATTACH through the ctl socket ("daemon" mode of open_transport).
    _sidecar = _host_plane().deployment == "sidecar"

    # Host memory budget gate: every link owns one pool of BLOCKS_PER_CHANNEL
    # blocks (a send direction reuses its channel's pool as D2H staging, no
    # extra arena), so need = links/2 * BLOCKS_PER_CHANNEL * (up + down).
    #
    # A cloud engine owns exactly one DP (its own); ONE edge engine serves every
    # DP connection of its instance, so it holds them all no matter what its own
    # (always 0) DP index says. TP rank 0 is the endpoint process of each
    # connection; the other TP ranks own no IO.
    local_conns = [conn for conn in connections if role == "edge" or conn.connection_id.dp_idx == dp_rank]
    endpoint_conns = local_conns if tp_rank == 0 else []
    links = 2 * len(endpoint_conns)
    need_mb = _host_memory_need_mb(links, up_block_mb=up_block_mb, down_block_mb=down_block_mb)
    limit_mb = _HOST_MEM_BUDGET_MB
    if need_mb > limit_mb:
        raise RuntimeError(
            "host data plane memory budget exceeded: "
            f"need={need_mb}MiB limit={limit_mb}MiB per rank "
            f"(links={links}, pools={links}, blocks_per_channel={BLOCKS_PER_CHANNEL}, "
            f"up_buffer_size={up_block_mb}, down_buffer_size={down_block_mb}); lower "
            "the block sizes in the topology file"
        )
    logger.info(
        "[LWD] host memory plan links=%d pools=%d need=%dMiB limit=%dMiB",
        links,
        links,
        need_mb,
        limit_mb,
    )

    qos = _qos_data_plane()
    for conn in endpoint_conns:
        connection = conn.connection_id
        if conn.cloud.data_port is None:
            raise RuntimeError(
                f"[LWD] host data plane requires 'data_port' on the cloud DP "
                f"(cloud={connection.cloud_id}, dp_idx={connection.dp_idx}); set it in "
                "the topology file (edges never listen; ctrl_port is "
                "control-plane only)"
            )
        addr = conn.cloud.addr
        port = conn.cloud.data_port  # used exactly as configured
        # Cloud DP: one listener shared by every connection of that DP;
        # both links (UP recv / DOWN send) attach to it via serve_via,
        # token-routed. A second edge of the same DP must REUSE it: the
        # listener binds in its constructor, so building another one on
        # the same data_port would fail the bind.
        # Daemon mode is the exception: the listener lives INSIDE the
        # transport daemon (spawned on demand by ucx_port over the ctl
        # socket); the engine-side channels ATTACH there instead, so no
        # engine-side listener may be built (old-branch semantics).
        listener = None
        if role == "cloud" and not _sidecar:
            listener_key = (connection.cloud_id, connection.dp_idx)
            listener = _UCX_LISTENERS.get(listener_key)
            if listener is None:
                listener = LwdUcxListener(addr, port)
                _UCX_LISTENERS[listener_key] = listener
                logger.info(
                    "[LWD] host listener bound for cloud=%d dp_idx=%d on %s:%d",
                    connection.cloud_id,
                    connection.dp_idx,
                    addr,
                    port,
                )
        for channel in (LwdChannelType.UP, LwdChannelType.DOWN):
            # Edge: UP sender / DOWN receiver, both connect out (the edge
            # never listens -- UHC1 is full-duplex). Cloud: UP receiver /
            # DOWN sender, both on the shared listener.
            if role == "edge":
                op = "send" if channel is LwdChannelType.UP else "recv"
                serve = False
            else:
                op = "recv" if channel is LwdChannelType.UP else "send"
                serve = True
            _UCX_IOS[(connection.edge_id, connection.cloud_id, connection.dp_idx, channel, op)] = LwdNpuUcxChannelIo(
                channel,
                torch.npu.Stream(priority=-1),
                LwdNpuUcxChannelIoConfig(
                    op=op,
                    up_block_mb=up_block_mb,
                    down_block_mb=down_block_mb,
                    host=addr,
                    port=port,
                    serve=serve,
                    token=_link_token(connection.edge_id, connection.cloud_id, connection.dp_idx, channel),
                    listener=listener,
                    transport_mode="daemon" if _sidecar else "embedded",
                    traffic_class=qos.traffic_class,
                    sl=qos.sl,
                ),
            )
    # Listeners bind at construction; accept loops start only after every
    # channel has registered its token (LwdUcxListener.register rejects
    # duplicates, so registration order does not matter). Daemon mode has
    # no engine-side listeners (they are owned by the transport daemon).
    for listener in _UCX_LISTENERS.values():
        listener.start()
    if role == "cloud" and _sidecar:
        logger.info(
            "[LWD] host sidecar deployment: listeners owned by the transport daemon (ports %s)",
            sorted({conn.cloud.data_port for conn in endpoint_conns}),
        )
    logger.info(
        "[LWD] host data-plane links created for role=%s instance=%d dp_rank=%d tp_rank=%d (ios=%s)",
        role,
        instance_id,
        dp_rank,
        tp_rank,
        sorted(f"e{e}c{c}/{d}/{ch.value}/{op}" for e, c, d, ch, op in _UCX_IOS),
    )
    if role == "edge":
        # Only the EDGE waits for its links (old-branch semantics: the
        # cloud starts before edges -- its listener is bound and each edge
        # is adopted asynchronously as it connects; blocking here would
        # deadlock a cloud-first deployment within the warmup timeout).
        lwd_warm_up_host_channels()
    else:
        logger.info(
            "[LWD] host cloud listener ready; waiting for %d edge link(s) asynchronously",
            len(_UCX_IOS),
        )
    _UCX_CHANNELS_INITIALIZED = True


def _io(edge_id: int, cloud_id: int, dp_idx: int, channel: LwdChannelType, op: str) -> LwdNpuUcxChannelIo:
    key = (edge_id, cloud_id, dp_idx, channel, op)
    try:
        return _UCX_IOS[key]
    except KeyError:
        raise RuntimeError(
            f"[LWD] this process holds no UCX {op} IO of {channel.value} on "
            f"(edge={edge_id}, cloud={cloud_id}, dp_idx={dp_idx}); only the "
            f"endpoint process (dp_rank==dp_idx, tp_rank==0) of the "
            "connection does"
        ) from None


def lwd_warm_up_host_channels(timeout: float | None = None) -> None:
    """Wait for every LOCAL link's readiness -- no cross-world barrier
    exists to join in host comm (old-branch semantics: the cloud reports
    its listener bound and adopts edges asynchronously; the edge waits
    for every cloud link, bounded by HS_TIMEOUT_S + 5)."""
    if timeout is None:
        from vllm_ascend.distributed.lwd_data_plane.host.ucx_io import (
            _HS_TIMEOUT_S,
        )

        timeout = max(5.0, _HS_TIMEOUT_S + 5.0)
    if not _UCX_IOS:
        logger.info("[LWD] host data-plane: no local links (non-endpoint process)")
        return
    import time as _time

    t0 = _time.perf_counter()
    for (edge_id, cloud_id, dp_idx, channel, op), io in _UCX_IOS.items():
        if not io.wait_ready(timeout=timeout):
            raise RuntimeError(
                f"[LWD][handshake] host link not ready: "
                f"edge={edge_id} cloud={cloud_id} dp_idx={dp_idx} "
                f"channel={channel.value} op={op} within {timeout:.0f}s"
            )
    logger.info(
        "[LWD][handshake] OK data_plane=host links=%d %.1fms (edge-side wait_ready only, no cross-world barrier)",
        len(_UCX_IOS),
        (_time.perf_counter() - t0) * 1000.0,
    )


def lwd_host_channels_initialized() -> bool:
    return _UCX_CHANNELS_INITIALIZED


def lwd_get_host_io(
    channel: LwdChannelType,
    dp_idx: int = 0,
    op: str = "send",
    edge_id: int | None = None,
    cloud_id: int | None = None,
) -> LwdNpuUcxChannelIo:
    """The queue factory's lookup. Single-connection callers omit the
    instance ids; the (edge, cloud) pair disambiguates multi-instance
    topologies (token identity is per (edge, cloud, dp, channel))."""
    if edge_id is None or cloud_id is None:
        if len({(e, c) for e, c, _d, _ch, _op in _UCX_IOS}) != 1:
            raise RuntimeError(
                "[LWD] ambiguous UCX link lookup: multiple (edge, cloud) connections exist; pass edge_id/cloud_id"
            )
        ((edge_id, cloud_id),) = {(e, c) for e, c, _d, _ch, _op in _UCX_IOS}
    return _io(edge_id, cloud_id, dp_idx, channel, op)


def lwd_destroy_host_channels(timeout: float = 2.0) -> None:
    global _UCX_CHANNELS_INITIALIZED
    for key, io in list(_UCX_IOS.items()):
        try:
            io.shutdown(timeout=timeout)
        except Exception:
            logger.warning("[LWD] failed to shut down the UCX IO for %s", key)
    _UCX_IOS.clear()
    for key, listener in list(_UCX_LISTENERS.items()):
        try:
            listener.close()
        except Exception:
            logger.warning("[LWD] failed to close host listener %s", key)
    _UCX_LISTENERS.clear()
    global _CACHED_COMM
    _CACHED_COMM = None
    _UCX_CHANNELS_INITIALIZED = False
