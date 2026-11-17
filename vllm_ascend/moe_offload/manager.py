"""MoE expert weight offload — v0.18 H2D implementation
ported onto vLLM-Ascend v0.23.0.

The H2D path (mirror, streams, events, slot rotation, validation) is the
v0.18 `expert_offload_manager.py` implementation kept as-is; only thin
shims adapt it to the v0.23.0 callers (MoEOffloadManager.get_instance /
register_layer / is_enabled, worker-supplied PP-aware LayerTopology).

EPLB coexistence (D2D + staging + async D2H writeback scheme):
  - All 6 mirrored params (w13/w2/scales/fused scales) live in flat
    pinned CPU mirrors; the mirror is the authoritative layout.
  - Per-layer EPLB updates are handled by EplbD2DManager
    (vllm_ascend/moe_offload/eplb_d2d.py): NPU D2D exchange on the
    original HCCL P2P framework, a per-layer NPU staging buffer holding
    the complete NEW layout, and an async D2H writeback of the staging
    buffer into the mirror.
  - This manager integrates three hooks for that scheme:
      * `_enqueue_layer_copy`: if a staging buffer exists for the layer,
        preload goes NPU-to-NPU D2D (staging -> slot) instead of the
        mirror H2D; if a D2H writeback is still in flight, the H2D path
        waits (stream-level) for its completion event first.
      * `finish_layer_and_preload`: skips preloading a layer whose
        mirror is mid-writeback and has no staging buffer available.
      * `_wait_layer_copy`: notifies the D2D manager when a staging
        buffer has been consumed by a slot load.
"""

import logging
import os
import re
import time
import weakref
from dataclasses import dataclass
from typing import Optional

import torch

# NOTE: `acl.rt` / `torch_npu` are imported lazily at their use sites so
# that CPU-only environments (e.g. x86 UT runners without torch_npu) can
# import this module without crashing. They are only touched on the real
# offload paths, which require an NPU runtime anyway.

logger = logging.getLogger(__name__)

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
NZ_FORMAT = 29  # FRACTAL_NZ

# ---------------------------------------------------------------------------
# Diag counters (host-only, no NPU allocation)
# ---------------------------------------------------------------------------
_diag_time = time.monotonic
_DIAG = {
    "n": 0,
    "prep": 0.0,
    "sub": 0.0,
    "p0": 0.0,
    "p1": 0.0,
    "fin": 0.0,
    "dma_stall": 0.0,
    "boundary": None,
    "_step_count": 0,
}

# Saved by _numa_bind_for_mirror() before restricting the main thread's
# affinity; restored by finalize_and_prime() after all mirrors are allocated.
_ORIGINAL_CPU_AFFINITY = None


def _numa_bind_for_mirror():
    """Bind the MAIN thread to the NPU-local CPU pool BEFORE any pinned
    mirror allocation, so the mirrors' first-touch lands on the NPU's
    NUMA node (migratepages later cannot move mlocked pages).

    All pool math and hardware dispatch are DELEGATED to vllm-ascend's
    cpu_binding.CpuAlloc — the exact allocator bind_cpus() runs at the
    end of load_model — so the early binding is idempotent with the
    later one and inherits the per-device role split (A2/A3/A5/310P each
    get their own allocation mode inside CpuAlloc.allocate). The later
    bind_cpus() additionally binds acl/release threads, IRQs, and memory
    migration; re-binding the same main pool is a no-op.
    """
    if getattr(_numa_bind_for_mirror, "_done", False):
        return
    _numa_bind_for_mirror._done = True
    logger.info("NUMA mirror-bind: enter pid=%s", os.getpid())
    try:
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            if not get_ascend_config().enable_cpu_binding:
                logger.info("NUMA mirror-bind skipped: vllm-ascend cpu binding disabled")
                return
        except Exception:
            pass  # config unavailable: default is enabled

        # --- Get local_rank (device-local rank for CPU binding) ---
        # Priority: LOCAL_RANK env > TP group rank > global dist rank.
        # LOCAL_RANK is set by the multiprocess executor and is correct
        # in ALL parallelism configs (TP/PP/DP). TP group rank is WRONG
        # under DPxTP (returns 0-7 per DP group, but DP1 workers use
        # chips 8-15). Global rank is wrong for multi-process configs.
        local_rank = -1
        _src = "none"
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        _src = "env"
        if local_rank < 0:
            try:
                from vllm.distributed.parallel_state import get_tp_group

                _tg = get_tp_group()
                if hasattr(_tg, "device_group"):
                    local_rank = _tg.device_group.rank()
                    _src = "tp_group"
            except Exception:
                pass
        if local_rank < 0:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    local_rank = dist.get_rank()
                    _src = "dist_global"
            except Exception:
                pass
        logger.info(
            "NUMA mirror-bind resolve: local_rank=%s src=%s visible=%s",
            local_rank,
            _src,
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"),
        )
        if local_rank < 0:
            logger.info("NUMA mirror-bind skipped: cannot determine local_rank")
            return

        # --- Delegate pool math / hardware dispatch to the upstream ---
        from vllm_ascend.cpu_binding import CpuAlloc, is_arm_cpu

        if not is_arm_cpu():
            logger.info("NUMA mirror-bind skipped: non-ARM CPU detected")
            return

        binder = None
        try:
            # v0.26+ signature: CpuAlloc(rank_id, npu_id) — derive the
            # npu_id the same way the 0.23 allocator does internally
            # (index into the sorted visible set).
            visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
            if visible:
                running_npus = sorted(int(x) for x in visible.split(","))
            else:
                running_npus = list(range(16))
            if local_rank >= len(running_npus):
                logger.info("NUMA mirror-bind skipped: local_rank=%s >= npu count=%s", local_rank, len(running_npus))
                return
            binder = CpuAlloc(local_rank, npu_id=running_npus[local_rank])
        except TypeError:
            binder = CpuAlloc(local_rank)  # v0.23 signature

        # build_cpu_pools() dispatches per device (A2/A3/A5/310P each
        # get their own pool mode) and fills npu_cpu_pool; allocate()
        # then applies the role split into assign_main/acl/rel.
        binder.build_cpu_pools()
        binder.allocate()

        current_npu = getattr(binder, "current_npu", None)
        if current_npu is None:
            current_npu = binder.device_info.running_npu_list[local_rank]
        main_cpus = binder.assign_main.get(current_npu) or []
        if not main_cpus:
            logger.info("NUMA mirror-bind skipped: empty main pool for NPU=%s", current_npu)
            return

        # Save the original affinity so finalize_and_prime() can restore
        # it after all mirrors are allocated. Without this, vllm-ascend's
        # bind_cpus() reads a narrowed /proc/self/status Cpus_allowed_list
        # and fails with "Insufficient CPUs".
        global _ORIGINAL_CPU_AFFINITY
        _ORIGINAL_CPU_AFFINITY = os.sched_getaffinity(0)

        os.sched_setaffinity(0, set(main_cpus))
        _cpu_node = getattr(binder, "cpu_node", {}) or {}
        nodes = {_cpu_node[c] for c in main_cpus if c in _cpu_node}
        node_str = str(nodes) if nodes else "?"
        logger.info(
            "NUMA mirror-bind: local_rank=%s NPU=%s main(%dcpus) NUMA=%s pool=%d-%d",
            local_rank,
            current_npu,
            len(main_cpus),
            node_str,
            min(main_cpus),
            max(main_cpus),
        )
    except Exception as e:
        logger.warning("NUMA mirror-bind failed (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# 0.23 shim topology (constructed by worker.py, PP-aware)
# ---------------------------------------------------------------------------
@dataclass
class LayerTopology:
    moe_layer_offset: int
    moe_layer_count: int
    moe_slot_count: int
    num_local_experts: int


# ---------------------------------------------------------------------------
# v0.18 original data structures
# ---------------------------------------------------------------------------
@dataclass
class FoldedTopology:
    rotating_moe_layer_count: int
    resident_tail_layer: int | None
    slots: int


def compute_folded_topology(offset: int, count: int, slots: int):
    """v0.18 original: odd layer counts leave the last layer resident."""
    if count % slots == 0:
        return FoldedTopology(count, None, slots)
    resident_tail = offset + count - 1
    return FoldedTopology(count - 1, resident_tail, slots)


@dataclass
class FullLayerExpertWeights:
    params: list  # flat pinned CPU mirrors: w13, w2, scales...
    expert_count: int


@dataclass
class FullLayerTransferState:
    copy_done_event: object
    slot_idx: int
    expert_count: int
    reason: str
    # Keepalive for the source mirror params: an EPLB COW rebuild swaps
    # full_layer_expert_params to NEW flats; without this reference the
    # OLD flats are GC'd while the copy stream still has in-flight (or
    # not-yet-started, gated on reader_done) DMAs reading them -> device
    # hang. Released by _reap_transfers once the copy completes.
    src_params: list | None = None


# ---------------------------------------------------------------------------
# v0.18 MoEOffloadManager (H2D path kept original) + 0.23 shims
# ---------------------------------------------------------------------------
class MoEOffloadManager:
    _instance: Optional["MoEOffloadManager"] = None

    # ------------------------------------------------------------------ #
    # construction / singleton (0.23 shim signature)                     #
    # ------------------------------------------------------------------ #
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def _initialize(self, topology, pin_mirror: bool) -> None:
        self.topology = topology
        self.pin_mirror = bool(pin_mirror)
        # v0.18 fields, sourced from the worker-supplied (PP-aware) topology
        self.moe_layer_offset = topology.moe_layer_offset
        self.moe_layer_count = topology.moe_layer_count
        self.moe_slot_count = topology.moe_slot_count
        self.initial_folded_moe = True
        self.is_expert_offload_enabled = True
        self._offset = self.moe_layer_offset

        _folded = compute_folded_topology(self.moe_layer_offset, self.moe_layer_count, self.moe_slot_count)
        self._folded_topology = _folded
        self.rotating_moe_layer_count = _folded.rotating_moe_layer_count
        self.resident_tail_layer = _folded.resident_tail_layer

        # v0.18 original: single dedicated copy stream
        self.expert_copy_stream = torch.npu.Stream()

        self.moe_modules: dict = {}
        self.layer_prefixes: list = []
        self.layer_to_prefix: dict = {}
        self.full_layer_expert_params: dict = {}
        self.local_to_global: dict = {}
        self.param_names: dict = {}
        self._active_transfers: list = []
        # EPLB D2D + staging + D2H writeback scheme hooks (see module
        # docstring). `_eplb_d2d_mgr` is created lazily by the EPLB
        # patch after warmup; until then everything runs the stock H2D
        # path with zero overhead.
        self._eplb_d2d_mgr = None
        self._eplb_pending_d2h: set = set()
        self.pending_transfers: dict = {}
        self.reader_done_events: dict = {}
        self.slot_loaded_layers: dict = {}
        self.initial_folded_weights_finalized = False
        # NUMA-first-touch guard: bind the MAIN thread (the one that
        # allocates the pinned mirrors in register_layer) to the
        # NPU-local CPU pool BEFORE any mirror allocation. bind_cpus()
        # re-binds the same set at the end of load_model (idempotent);
        # doing it early is required because migratepages cannot move
        # mlocked (pinned) pages once they are first-touched elsewhere.
        # Universal placement: covers every quant method registering
        # mirrors through this manager (w4a8 / w8a8 / future).
        _numa_bind_for_mirror()
        logger.info(
            "MoE offload: init: offset=%d layers=%d rotating=%d slots=%d resident_tail=%s",
            self.moe_layer_offset,
            self.moe_layer_count,
            self.rotating_moe_layer_count,
            self.moe_slot_count,
            self.resident_tail_layer,
        )

    @classmethod
    def get_instance(cls, topology=None, pin_mirror=True):
        if cls._instance is None:
            cls._instance = cls()
        if not getattr(cls._instance, "_initialized", False):
            cls._instance._initialize(topology, pin_mirror)
            cls._instance._initialized = True
        return cls._instance

    @classmethod
    def is_enabled(cls) -> bool:
        try:
            from vllm_ascend.ascend_config import get_ascend_config

            return bool(get_ascend_config().moe_offload_config.enabled)
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # v0.18 original helpers (kept as-is)                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_layer_idx(layer_prefix: str) -> int:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_prefix)
        if match is None:
            raise ValueError(f"Cannot parse model layer index from {layer_prefix!r}")
        return int(match.group(1))

    def _is_resident_layer(self, layer_prefix: str) -> bool:
        return self.resident_tail_layer is not None and self._parse_layer_idx(layer_prefix) == self.resident_tail_layer

    def is_offload_managed(self, layer_prefix: str) -> bool:
        """True if this MoE layer is inside the offload-managed range
        (rotating layers + resident tail). False for layers outside,
        e.g. the MTP head layer, whose weights stay resident clones."""
        idx = self._parse_layer_idx(layer_prefix)
        return self.moe_layer_offset <= idx < self.moe_layer_offset + self.moe_layer_count

    def _slot_idx(self, layer_prefix: str) -> int:
        layer_idx = self._parse_layer_idx(layer_prefix)
        if self._is_resident_layer(layer_prefix):
            return 0
        rel = layer_idx - self.moe_layer_offset
        if rel < 0 or rel >= self.rotating_moe_layer_count:
            raise RuntimeError(f"{layer_prefix} is not a rotating MoE layer: rel={rel}")
        return rel % self.moe_slot_count

    def _next_layer_in_slot(self, layer_prefix: str) -> str:
        layer_idx = self._parse_layer_idx(layer_prefix)
        self._slot_idx(layer_prefix)
        # Generalized same-slot rotation (v0.18 hardcoded 2 slots where
        # "+slots then -rotating" happened to stay in-slot; it breaks for
        # slot_count >= 3). Correct rule: next rel with the SAME slot,
        # wrapping to the first occurrence in the rotation.
        rel = layer_idx - self.moe_layer_offset
        slot = rel % self.moe_slot_count
        nxt_rel = rel + self.moe_slot_count
        if nxt_rel >= self.rotating_moe_layer_count:
            nxt_rel = slot  # first occurrence of this slot in rotation
        if nxt_rel == rel:
            raise RuntimeError(f"no next same-slot layer for {layer_prefix}")
        candidate = self.moe_layer_offset + nxt_rel
        for p in self.layer_prefixes:
            if self._parse_layer_idx(p) == candidate:
                return p
        raise RuntimeError(f"Next same-slot layer {candidate} is not registered")

    def _live_module(self, layer_prefix: str):
        module_ref = self.moe_modules.get(layer_prefix)
        if module_ref is None:
            raise RuntimeError(f"MoE module {layer_prefix} is not registered")
        module = module_ref()
        if module is None:
            raise RuntimeError(f"MoE module {layer_prefix} was garbage collected")
        return module

    def register_moe_module(self, layer_prefix: str, module) -> None:
        existing = self.moe_modules.get(layer_prefix)
        if existing is not None and existing() is not module:
            raise RuntimeError(f"Different modules registered for {layer_prefix}")
        self.moe_modules[layer_prefix] = weakref.ref(module)
        self.layer_to_prefix[self._parse_layer_idx(layer_prefix)] = layer_prefix
        if layer_prefix not in self.layer_prefixes:
            self.layer_prefixes.append(layer_prefix)
            self.layer_prefixes.sort(key=self._parse_layer_idx)

    @staticmethod
    def _raw_storage_size(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().nbytes())

    def _validate_packed_nz_tensor(self, layer_prefix: str, weight_name: str, tensor: torch.Tensor) -> None:
        if tensor.device.type != "npu" and tensor.device.type != "cpu":
            raise RuntimeError(f"{layer_prefix} {weight_name} is not on npu/cpu")
        import torch_npu

        if int(torch_npu.get_npu_format(tensor)) != NZ_FORMAT:
            # pinned CPU mirrors are uint8 1-D views over NZ bytes; only
            # the NPU destination must be NZ.
            if tensor.device.type == "npu":
                raise RuntimeError(f"{layer_prefix} {weight_name} is not FRACTAL_NZ")

    def _copy_packed_nz_storage_to_cpu(
        self,
        layer_prefix: str,
        weight_name: str,
        source: torch.Tensor,
        check_nz: bool = True,
    ) -> torch.Tensor:
        if check_nz:
            self._validate_packed_nz_tensor(layer_prefix, weight_name, source)
        import acl.rt

        size_bytes = self._raw_storage_size(source)
        destination = torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=self.pin_mirror,
        )
        ret = acl.rt.memcpy(
            destination.data_ptr(),
            size_bytes,
            source.data_ptr(),
            size_bytes,
            ACL_MEMCPY_DEVICE_TO_HOST,
        )
        if ret != 0:
            raise RuntimeError(
                "Packed-NZ raw D2H memcpy failed: "
                f"layer={layer_prefix} weight={weight_name} "
                f"bytes={size_bytes} acl_error={ret}"
            )
        return destination

    def _validate_cpu_mirror(
        self,
        layer_prefix: str,
        weight_name: str,
        source: torch.Tensor,
        destination: torch.Tensor,
        check_nz: bool = True,
    ) -> None:
        if check_nz:
            self._validate_packed_nz_tensor(layer_prefix, weight_name, destination)
        if (
            source.device.type != "cpu"
            or source.dtype != torch.uint8
            or source.dim() != 1
            or source.storage_offset() != 0
        ):
            raise RuntimeError(f"{layer_prefix} {weight_name} CPU mirror must be 1-D uint8")
        if not source.is_pinned():
            # [V18PORT] pageable allowed for experiments; warn once
            if not getattr(self, "_pageable_warned", False):
                logger.warning("MoE offload: CPU mirror is PAGEABLE (pin off)")
                self._pageable_warned = True
        source_bytes = self._raw_storage_size(source)
        destination_bytes = self._raw_storage_size(destination)
        if source.numel() != source_bytes or source_bytes != destination_bytes:
            raise RuntimeError(
                f"{layer_prefix} {weight_name} raw storage mismatch: "
                f"source_numel={source.numel()} "
                f"source_bytes={source_bytes} "
                f"destination_bytes={destination_bytes}"
            )

    # ------------------------------------------------------------------ #
    # 0.23 shim registration: only w13/w2 are mirrored                   #
    # ------------------------------------------------------------------ #
    def register_layer(self, layer_prefix, expert_params, param_names=None, global_ids=None):
        w13_t = expert_params[0]
        w2_t = expert_params[1]
        if w13_t is None or w2_t is None:
            raise RuntimeError(f"{layer_prefix}: w13/w2 tensors required for v18 port")
        torch.npu.synchronize()
        mirrors = []
        for t in expert_params:
            if t is not None and torch.is_tensor(t):
                flat = t.reshape(-1) if t.dim() != 1 else t
                # NZ check only for the packed weights (p0/p1); scales and
                # fused scales are ordinary ND tensors.
                mirrors.append(
                    self._copy_packed_nz_storage_to_cpu(
                        layer_prefix,
                        f"p{len(mirrors)}",
                        flat,
                        check_nz=(len(mirrors) < 2),
                    )
                )
            else:
                mirrors.append(None)
        self.full_layer_expert_params[layer_prefix] = FullLayerExpertWeights(
            params=mirrors,
            expert_count=int(w13_t.shape[0]),
        )
        self.layer_to_prefix[self._parse_layer_idx(layer_prefix)] = layer_prefix
        self.param_names[layer_prefix] = list(param_names) if param_names is not None else ["w13_weight", "w2_weight"]
        if global_ids is not None:
            self.local_to_global[layer_prefix] = list(global_ids)
        else:
            self.local_to_global[layer_prefix] = list(range(int(w13_t.shape[0])))
        if layer_prefix not in self.layer_prefixes:
            self.layer_prefixes.append(layer_prefix)
            self.layer_prefixes.sort(key=self._parse_layer_idx)

    # ------------------------------------------------------------------ #
    # v0.18 original H2D submit path (kept as-is, dst via _offload_ attr) #
    # ------------------------------------------------------------------ #
    def _copy_stream(self):
        copy_stream = self.expert_copy_stream
        if copy_stream is None:
            raise RuntimeError("Expert copy stream requested outside initial-folded mode")
        compute_stream = torch.npu.current_stream()
        if copy_stream.npu_stream == compute_stream.npu_stream:
            raise RuntimeError("Expert H2D copy stream must differ from compute stream")
        return copy_stream, compute_stream

    def _enqueue_raw_h2d(
        self,
        layer_prefix,
        weight_name,
        slot_idx,
        source,
        destination,
        copy_stream,
        check_nz=True,
    ):
        self._validate_cpu_mirror(layer_prefix, weight_name, source, destination, check_nz)
        import acl.rt

        size_bytes = self._raw_storage_size(destination)
        ret = acl.rt.memcpy_async(
            destination.data_ptr(),
            size_bytes,
            source.data_ptr(),
            size_bytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
            copy_stream.npu_stream,
        )
        if ret != 0:
            raise RuntimeError(
                "Packed-NZ raw H2D memcpy_async failed: "
                f"layer={layer_prefix} weight={weight_name} "
                f"slot={slot_idx} bytes={size_bytes} acl_error={ret}"
            )

    def _enqueue_layer_copy(
        self,
        layer_prefix,
        slot_idx,
        reader_done_event,
        reason,
    ):
        source = self.full_layer_expert_params[layer_prefix]
        module = self._live_module(layer_prefix)
        copy_stream, _ = self._copy_stream()
        names = self.param_names.get(layer_prefix, ["w13_weight", "w2_weight"])
        d2d = self._eplb_d2d_mgr
        staging_flats = None
        if d2d is not None:
            staging_flats = d2d.get_staging_for_preload(layer_prefix)
        import acl.rt

        _s0 = _diag_time()
        with torch.npu.stream(copy_stream):
            if reader_done_event is not None:
                copy_stream.wait_event(reader_done_event)
            if staging_flats is not None:
                # EPLB fast path: preload from the NPU staging buffer
                # (holds the complete NEW layout) via raw D2D instead of
                # mirror H2D. Same-stream ordering covers the copy-stream
                # part of the staging fill; the compute-stream part
                # (recv -> staging) needs an explicit cross-stream wait.
                recv_ev = d2d.get_staging_recv_event(layer_prefix)
                if recv_ev is not None:
                    copy_stream.wait_event(recv_ev)
                for pid, src_flat in enumerate(staging_flats):
                    if src_flat is None or pid >= len(names):
                        continue
                    dst = getattr(module, f"_offload_{names[pid]}", None)
                    if dst is None:
                        continue
                    size_bytes = self._raw_storage_size(dst)
                    ret = acl.rt.memcpy_async(
                        dst.data_ptr(),
                        size_bytes,
                        src_flat.data_ptr(),
                        size_bytes,
                        ACL_MEMCPY_DEVICE_TO_DEVICE,
                        copy_stream.npu_stream,
                    )
                    if ret != 0:
                        raise RuntimeError(
                            "Staging D2D memcpy_async failed: "
                            f"layer={layer_prefix} pid={pid} "
                            f"slot={slot_idx} bytes={size_bytes} "
                            f"acl_error={ret}"
                        )
                copy_done_event = torch.npu.current_stream().record_event()
                # Record the preload event on the staging slot so its
                # next reuse orders against this read.
                if d2d is not None:
                    d2d.note_staging_preload(layer_prefix, copy_done_event)
                _DIAG["sub"] += _diag_time() - _s0
                transfer = FullLayerTransferState(
                    copy_done_event=copy_done_event,
                    slot_idx=slot_idx,
                    expert_count=source.expert_count,
                    reason=reason + "/staging",
                    src_params=list(staging_flats),
                )
                self._active_transfers.append(transfer)
                return transfer
            # Mirror H2D path: if an EPLB D2H writeback for this layer
            # is still in flight, the mirror read must wait for it
            # (stream-level wait; the host never blocks).
            if layer_prefix in self._eplb_pending_d2h:
                ev = d2d.get_d2h_event(layer_prefix) if d2d else None
                if ev is not None:
                    copy_stream.wait_event(ev)
            # 0.23 shim: destinations via the _offload_ attributes that
            # w4a8/w8a8 set (non-slot layers share the slot layer storage).
            first = True
            for pid, src_t in enumerate(source.params):
                if src_t is None or pid >= len(names):
                    continue
                dst = getattr(module, f"_offload_{names[pid]}", None)
                if dst is None:
                    continue
                _m0 = _diag_time()
                self._enqueue_raw_h2d(
                    layer_prefix,
                    f"p{pid}",
                    slot_idx,
                    src_t,
                    dst,
                    copy_stream,
                    check_nz=(pid < 2),
                )
                _dt = _diag_time() - _m0
                if first:
                    _DIAG["p0"] += _dt
                    first = False
                else:
                    _DIAG["p1"] += _dt
            copy_done_event = torch.npu.current_stream().record_event()
        _DIAG["sub"] += _diag_time() - _s0
        transfer = FullLayerTransferState(
            copy_done_event=copy_done_event,
            slot_idx=slot_idx,
            expert_count=source.expert_count,
            reason=reason,
            src_params=list(source.params),
        )
        self._active_transfers.append(transfer)
        return transfer

    def _wait_layer_copy(self, layer_prefix, transfer):
        if not transfer.copy_done_event.query():
            _DIAG["dma_stall"] += 1
        torch.npu.current_stream().wait_event(transfer.copy_done_event)
        self.slot_loaded_layers[transfer.slot_idx] = layer_prefix
        if transfer.reason.endswith("/staging") and self._eplb_d2d_mgr is not None:
            # The staging buffer's content is now in the slot; mark it
            # consumed (clears the pending-D2H guard for this layer too,
            # since the slot serves the new layout from here on).
            self._eplb_d2d_mgr.on_layer_loaded(layer_prefix)

    def _drop_pending_for_slot(self, slot_idx):
        stale = [p for p, t in self.pending_transfers.items() if t.slot_idx == slot_idx]
        for p in stale:
            self.pending_transfers.pop(p, None)

    # ------------------------------------------------------------------ #
    # finalize (v0.18 original logic, shim entry point)                  #
    # ------------------------------------------------------------------ #
    def _reap_transfers(self):
        """Drop completed transfers; their src_params keepalive is then
        freeable (the copy has physically completed)."""
        if not self._active_transfers:
            return
        alive = []
        for t in self._active_transfers:
            try:
                done = bool(t.copy_done_event.query())
            except Exception:
                done = True
            if not done:
                alive.append(t)
        self._active_transfers = alive

    def finalize_and_prime(self):
        if self.initial_folded_weights_finalized:
            return
        # Expected registrations cover the ROTATING layers only — the
        # resident tail (odd MoE count) stays exclusively resident on
        # the NPU per v0.18 semantics and never registers a mirror.
        expected = set(
            range(
                self.moe_layer_offset,
                self.moe_layer_offset + self.rotating_moe_layer_count,
            )
        )
        registered = {self._parse_layer_idx(p) for p in self.full_layer_expert_params}
        if registered != expected:
            missing = expected - registered
            if missing:
                logger.warning("MoE offload: missing layers: %s", missing)
            if registered - expected:
                raise RuntimeError(
                    f"folded MoE registration mismatch: registered={sorted(registered)} expected={sorted(expected)}"
                )
        rotating = [p for p in self.layer_prefixes if not self._is_resident_layer(p)]
        if len(rotating) != self.rotating_moe_layer_count:
            raise RuntimeError(
                f"folded MoE topology mismatch: rotating={len(rotating)} expected={self.rotating_moe_layer_count}"
            )
        torch.npu.config.allow_internal_format = True
        for layer_prefix in rotating[: self.moe_slot_count]:
            slot_idx = self._slot_idx(layer_prefix)
            self._enqueue_layer_copy(layer_prefix, slot_idx, None, "initial-prime")
            self.slot_loaded_layers[slot_idx] = layer_prefix
        copy_stream, _ = self._copy_stream()
        copy_stream.synchronize()
        self.pending_transfers.clear()
        self.reader_done_events.clear()
        self.initial_folded_weights_finalized = True
        # Restore the main thread's original CPU affinity now that all
        # mirrors are allocated. bind_cpus() runs after load_model() and
        # must see the FULL CPU set.
        if _ORIGINAL_CPU_AFFINITY is not None:
            try:
                os.sched_setaffinity(0, _ORIGINAL_CPU_AFFINITY)
                logger.info("NUMA affinity restored (%d cpus)", len(_ORIGINAL_CPU_AFFINITY))
            except Exception as e:
                logger.warning("NUMA affinity restore failed: %s", e)
        logger.info(
            "MoE offload: ready: mirrors=%d rotating=%d slots=%d resident_tail=%s primed=%s",
            len(self.full_layer_expert_params),
            len(rotating),
            self.moe_slot_count,
            self.resident_tail_layer,
            rotating[: self.moe_slot_count],
        )

    # ------------------------------------------------------------------ #
    # v0.18 original hot path                                            #
    # ------------------------------------------------------------------ #
    def prepare_layer(self, layer_prefix):
        _t0 = _diag_time()
        if not self.initial_folded_weights_finalized:
            raise RuntimeError("Folded MoE forward started before mirrors were committed")
        if self._is_resident_layer(layer_prefix):
            return
        layer_idx = self._parse_layer_idx(layer_prefix)
        if layer_idx == self.moe_layer_offset:
            now = _diag_time()
            d = _DIAG
            d["_step_count"] += 1
            if d["boundary"] is not None:
                _w = (now - d["boundary"]) * 1000
                logger.debug(
                    "[offload-step] #%d wall=%.1fms prep=%.1f sub=%.1f p0=%.1f p1=%.1f fin=%.1f miss=%.0f",
                    d["_step_count"],
                    _w,
                    d["prep"] * 1000,
                    d["sub"] * 1000,
                    d["p0"] * 1000,
                    d["p1"] * 1000,
                    d["fin"] * 1000,
                    d["dma_stall"],
                )
            d.update(
                n=0,
                prep=0.0,
                sub=0.0,
                p0=0.0,
                p1=0.0,
                fin=0.0,
                dma_stall=0.0,
                boundary=now,
            )
        slot_idx = self._slot_idx(layer_prefix)
        self._reap_transfers()
        transfer = self.pending_transfers.pop(layer_prefix, None)
        if transfer is not None:
            if transfer.slot_idx != slot_idx:
                raise RuntimeError(
                    f"Preload slot mismatch for {layer_prefix}: scheduled={transfer.slot_idx} actual={slot_idx}"
                )
            self._wait_layer_copy(layer_prefix, transfer)
            _DIAG["prep"] += _diag_time() - _t0
            return
        if self.slot_loaded_layers.get(slot_idx) == layer_prefix:
            _DIAG["prep"] += _diag_time() - _t0
            return
        self._drop_pending_for_slot(slot_idx)
        transfer = self._enqueue_layer_copy(
            layer_prefix,
            slot_idx,
            self.reader_done_events.get(slot_idx),
            "on-demand",
        )
        self._wait_layer_copy(layer_prefix, transfer)
        _DIAG["prep"] += _diag_time() - _t0

    def finish_layer_and_preload(self, layer_prefix):
        _f0 = _diag_time()
        if not self.initial_folded_weights_finalized:
            return
        if self._is_resident_layer(layer_prefix):
            return
        slot_idx = self._slot_idx(layer_prefix)
        if self.slot_loaded_layers.get(slot_idx) != layer_prefix:
            raise RuntimeError(f"Cannot record reader for unloaded layer {layer_prefix} in slot {slot_idx}")
        _, compute_stream = self._copy_stream()
        reader_done_event = compute_stream.record_event()
        self.reader_done_events[slot_idx] = reader_done_event
        next_layer_prefix = self._next_layer_in_slot(layer_prefix)
        d2d = self._eplb_d2d_mgr
        if d2d is not None:
            staging_ok = d2d.get_staging_for_preload(next_layer_prefix) is not None
            if next_layer_prefix in self._eplb_pending_d2h and not staging_ok:
                # Mirror is mid-writeback and no staging buffer is
                # available: skip the preload — prepare_layer will load
                # on demand after waiting for the D2H event.
                _DIAG["fin"] += _diag_time() - _f0
                return
        self._drop_pending_for_slot(slot_idx)
        transfer = self._enqueue_layer_copy(
            next_layer_prefix,
            slot_idx,
            reader_done_event,
            "preload",
        )
        self.pending_transfers[next_layer_prefix] = transfer
        if len(self.pending_transfers) > self.moe_slot_count:
            raise RuntimeError("Packed-NZ pending transfer state exceeded slot count")
        _DIAG["fin"] += _diag_time() - _f0
