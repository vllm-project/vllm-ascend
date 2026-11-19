"""
Patch EPLB to support weight offload mode via the D2D + staging +
async D2H writeback scheme (vllm_ascend/moe_offload/eplb_d2d.py).

When MoE weight offload is enabled (moe_offload_config.enabled):

- forward_before (offload-managed layer): builds the HCCL P2P exchange
  manually — isend sources are per-expert views into a dedicated static
  send buffer kept prefetched one layer ahead on a dedicated stream
  (the stock weight_list views point into the shared slot storage,
  which is NOT authoritative at EPLB update time); irecv targets are
  the stock adaptor buffers. The exchange is launched asynchronously,
  exactly like the stock loader does.

- forward_end (offload-managed layer): waits for the exchange, then
  hands the update to EplbD2DManager.complete_layer_update: assembles
  the complete NEW layout into an NPU staging buffer (recv blocks via
  raw D2D + locally-kept blocks via raw H2D from the CPU mirror),
  submits an async D2H writeback of the staging buffer into the mirror,
  switches the routing maps, invalidates the layer's slot, and
  advances the send-buffer prefetch pipeline to the next layer.

- Layers that are NOT offload-managed (e.g. the MTP layer, whose MoE
  weights stay resident on the NPU as independent per-expert clones)
  keep the ORIGINAL EPLB NPU D2D path unchanged (stock loader
  generate/async/update calls).

The original EPLB logic is preserved for when weight offload is
disabled. No modification to EPLB's source code — this patch is
applied via monkey-patching at import time.
"""

from vllm_ascend.eplb.eplb_updator import EplbUpdator

# Save originals for fallback when weight offload is disabled.
_original_forward_before = EplbUpdator.forward_before
_original_forward_end = EplbUpdator.forward_end
_original_warm_up_eplb = EplbUpdator.warm_up_eplb


def _is_offload_managed_layer(layer_id):
    """True if this EPLB layer_id maps to a layer registered with the
    weight-offload manager (the rotating MoE layers). False for the MTP
    layer — its weights are resident per-expert clones and are never
    offloaded, so the original NPU D2D redistribution applies to it.
    """
    from vllm_ascend.moe_offload import MoEOffloadManager

    mgr = MoEOffloadManager.get_instance()
    if mgr is None or not hasattr(mgr, "layer_to_prefix"):
        return False
    topo = getattr(mgr, "topology", None)
    offset = getattr(topo, "moe_layer_offset", 0) if topo is not None else 0
    return mgr.layer_to_prefix.get(layer_id + offset) is not None


def _get_d2d_manager():
    from vllm_ascend.moe_offload import MoEOffloadManager

    mgr = MoEOffloadManager.get_instance()
    if getattr(mgr, "_eplb_d2d_mgr", None) is None:
        from vllm_ascend.moe_offload.eplb_d2d import EplbD2DManager

        mgr._eplb_d2d_mgr = EplbD2DManager(mgr)
    return mgr._eplb_d2d_mgr


def _patched_warm_up_eplb(self):
    """Stock EPLB warmup + send-buffer allocation & priming for the
    D2D+D2H scheme."""
    _original_warm_up_eplb(self)
    from vllm_ascend.moe_offload import MoEOffloadManager

    if not MoEOffloadManager.is_enabled():
        return
    d2d = _get_d2d_manager()
    d2d.init_send_buffer_and_prime()


def _to_tensor(x):
    import numpy
    import torch

    if torch.is_tensor(x):
        return x
    return torch.from_numpy(numpy.array(x))


def _patched_forward_before(self):
    """EPLB forward_before with the weight-offload D2D branch."""
    from vllm_ascend.moe_offload import MoEOffloadManager

    if not MoEOffloadManager.is_enabled():
        # Original EPLB logic: NPU D2D + copy_
        _original_forward_before(self)
        return

    if self.get_update_info_flag():
        self.update_info_all = self.eplb_process.block_update_q.get()
    if not self.update_expert_weight_flag():
        return

    (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = self.update_info_all.pop(0)

    if _is_offload_managed_layer(layer_id):
        # Offload layer: launch the exchange with send-buffer sources
        # (recv side = stock adaptor buffers). Weight assembly and map
        # switch happen in forward_end.
        d2d = _get_d2d_manager()
        updated_map_t = _to_tensor(updated_expert_map)
        self.reqs = d2d.launch_d2d_exchange(layer_id, expert_send_info, expert_recv_info, self)
        self._offload_layer_id = layer_id
        self._offload_update_info = (
            expert_send_info,
            expert_recv_info,
            updated_map_t,
            _to_tensor(log2phy_map),
            layer_id,
        )
    else:
        # Non-offload layer (MTP): stock path — loader generate +
        # async launch (resident per-expert clones, safe to read).
        self._offload_layer_id = None
        self._offload_update_info = None
        import numpy
        import torch
        from vllm.v1.utils import record_function_or_nullcontext

        with record_function_or_nullcontext("EPLB generate p2p task"):
            log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
            self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(numpy.array(updated_expert_map))
            self.eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info,
                expert_recv_info,
                updated_expert_map_this_rank,
                layer_id,
            )
            self.reqs = []
            self.eplb_loader.asyn_expert_weight_transfer(self.reqs)


# Resident pinned scratch for the log2phy map update (NPU<-host copy
# must never allocate at runtime — see the staging-pool lesson). Sized
# on first use and reused for every layer/cycle afterwards.
_PINNED_L2P_SCRATCH = {"t": None}


def _get_pinned_l2p_like(t):
    import torch

    scr = _PINNED_L2P_SCRATCH["t"]
    if scr is None or scr.numel() < t.numel() or scr.dtype != t.dtype:
        scr = torch.empty(t.numel(), dtype=t.dtype, pin_memory=True)
        _PINNED_L2P_SCRATCH["t"] = scr
    return scr[: t.numel()]


def _patched_forward_end(self, eplb_heat_collection_status: bool = True):
    """EPLB forward_end with the weight-offload D2D+D2H branch."""
    from vllm_ascend.moe_offload import MoEOffloadManager

    if not MoEOffloadManager.is_enabled():
        # Original EPLB logic: req.wait + copy_ + map updates
        _original_forward_end(self, eplb_heat_collection_status)
        return

    # Heat collection still runs normally.
    if self.wakeup_eplb_worker_flag():
        from vllm.v1.utils import record_function_or_nullcontext

        with record_function_or_nullcontext("EPLB gather moe load"):
            self.compute_and_set_moe_load()
            self.wakeup_eplb_worker()

    if self.update_expert_weight_flag() and self.expert_map_record_path is None:
        if getattr(self, "_offload_layer_id", None) is not None:
            # Offload-managed layer: D2D wait + staging + D2H writeback
            # + immediate map switch (EplbD2DManager).
            (expert_send_info, expert_recv_info, updated_map_t, log2phy_t, layer_id) = self._offload_update_info
            self._offload_layer_id = None
            self._offload_update_info = None

            def _apply_maps(_adaptor=self.adaptor, _lid=layer_id, _m=updated_map_t, _l=log2phy_t):
                _adaptor.do_update_expert_map(_lid, _m)
                if _l.device.type == "cpu" and not _l.is_pinned():
                    scr = _get_pinned_l2p_like(_l)
                    scr.copy_(_l)
                    _l = scr
                _adaptor.do_update_log2phy_map(_lid, _l)

            d2d = _get_d2d_manager()
            d2d.complete_layer_update(
                layer_id,
                expert_send_info,
                expert_recv_info,
                updated_map_t,
                self,
                self.reqs,
                _apply_maps,
                log2phy_map_t=log2phy_t,
            )
            self.reqs = []
        else:
            # Non-offload layer (MTP): original D2D completion — wait,
            # copy buffers into the weight lists and switch maps.
            self.eplb_loader.update_expert_map_and_weight(self.reqs)
            self.reqs = []

    # Iteration counter still updates normally.
    if self.cur_iterations >= self.expert_heat_collection_interval - 1 or eplb_heat_collection_status:
        self.update_iteration()


# Apply patch at import time. Monkey-patching is the point of this
# module; silence the method-assignment checker accordingly.
EplbUpdator.forward_before = _patched_forward_before  # type: ignore[method-assign]
EplbUpdator.forward_end = _patched_forward_end  # type: ignore[method-assign]
EplbUpdator.warm_up_eplb = _patched_warm_up_eplb  # type: ignore[method-assign]
