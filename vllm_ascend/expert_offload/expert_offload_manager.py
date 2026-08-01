"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch_npu
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


_SUBSCRIBED_COMPUTE_STREAMS = set()
def get_subscribed_compute_streams() -> set:
    return _SUBSCRIBED_COMPUTE_STREAMS


class ExpertOffloadManager:
    """Singleton manager for expert weight offloading.

    Stores all expert weights on CPU and pages the needed experts to NPU
    during forward based on routing topk_ids.
    """

    _instance: "ExpertOffloadManager | None" = None

    # Parallel weight-load pool. The strided transpose-copy in load_w13/
    # load_w2 is single-threaded (~0.2 GB/s into pinned memory); fanning the
    # ~99k shard copies out over this many workers hits ~2-4 GB/s.
    _LOAD_POOL_WORKERS = 32
    # Bound on in-flight futures before a partial drain (releases owned clones
    # early so transient memory stays small). >> workers, so no starvation.
    _LOAD_POOL_DRAIN_EVERY = 2048

    @classmethod
    def get_instance(cls) -> "ExpertOffloadManager":
        assert cls._instance is not None, "ExpertOffloadManager not initialized"
        return cls._instance

    def __init__(self, vllm_config: VllmConfig):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        self.num_device_experts = self.offload_config.num_device_experts
        self.topk = vllm_config.model_config.hf_config.num_experts_per_tok
        self.offload_threshold = self.num_device_experts // self.topk

        # Multi-card EP offload (stages 1-2). ep_rank/ep_size are resolved
        # lazily on first read because the EP process group is not initialized
        # yet at manager construction time (model_runner.__init__ runs before
        # init_distributed_environment completes).
        self.enable_multi_card = self.offload_config.enable_multi_card
        self._ep_size = 1
        self._ep_rank = 0
        self._ep_info_resolved = False
        # Multi-card decode resident cache: per layer_idx, {slot: expert_id} of
        # the experts currently loaded in THIS rank's device slots. Used to turn
        # the per-step full H2D into skip-on-hit (only load misses) and to log
        # hit/miss. Keyed by slot (the planner assigns expert->slot via log2phy,
        # so a hit = same expert already in the same slot).
        self._mc_resident = {}
        # Two-timescale LRU: per-step local freq tracking + every-N-step gloo
        # all_reduce -> global hotness. Stable-slot placement uses prev_log2phy
        # (keep experts on same rank in their slot) + hotness (order new experts).
        self._mc_prev_log2phy = {}      # layer_idx -> prev step's log2phy (CPU)
        # LRC hotness policy (same one single-card uses: recent freq + EMA +
        # age), fed the GLOBAL active set each step. Built lazily in
        # _gather_global_counts_and_hotness. Replaces the old crude local-freq
        # + 32-step all_reduce hotness, which was stale and had no EMA/age.
        self._mc_lrc = None

        # Per-layer cap on experts actually H2D-loaded by _do_prefetch: only
        # the top-N highest-confidence predicted experts are loaded, the rest
        # are left to update_weights()'s reactive fallback. Clamped to
        # [1, topk]; >topk has no extra effect since the router selects at
        # most topk experts per token.
        self.expert_prefetch_num = self.offload_config.expert_prefetch_num
        self.prefetch_topk = max(1, min(self.topk, self.expert_prefetch_num))

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.scale_bias_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary per-expert storage for w13 scale/offset shard assembly.
        # Key: (layer_moe_idx, expert_id, attr_name), value: first shard.
        # Scale/offset arrive as w1 + w3 shards; we stash one until the
        # other arrives, then assemble and copy into scale_cpu_buffers.
        self._scale_shard_temp: dict[tuple[int, int, str], torch.Tensor] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in init_layer_cpu_buffers
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        # Master debug switch for expert-offload diagnostics — UPDATE-W cache
        # trace, per-prefill-load logs, prefetch/update slot shortfalls.
        # Flipping it on surfaces them at info level (no need for global
        # VLLM_LOGGING_LEVEL=DEBUG).
        self._debug = self.offload_config.moe_offload_debug

        # Diagnostic: wall time of the parallel weight-load phase (safetensors
        # → pinned CPU buffers). Logged in _finalize_offload.
        self._weight_load_secs: float = 0.0
        self._weight_load_calls: int = 0

        # Deferred weight-load pool. load_w13/load_w2/_load_scale_shard clone
        # loaded_weight synchronously (while the safetensors mmap is still
        # mapped) and submit the strided transpose-copy to this pool. The
        # deferred copy reads the owned clone, so it stays correct after the
        # safetensors mmap is unmapped (which happens before _finalize_offload).
        # drain_load_pool() is called from _finalize_offload before the buffers
        # are read by process_weights_after_loading().
        self._load_pool: ThreadPoolExecutor | None = None
        self._load_futures: list = []
        self._load_phase_start: float = 0.0
        self._saved_num_threads: int | None = None

        ExpertOffloadManager._instance = self

        self.load_stream = torch_npu.npu.Stream()

        self._init_prefill_pool_state()
        self._is_prefetch: bool = False
        self._init_prefetch_state()

    def _init_prefill_pool_state(self) -> None:
        """Prefill-pool attribute init (ndl layers × all experts on NPU)."""
        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []       # W8A8 / W4A8_DYNAMIC
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8 / W4A8_DYNAMIC
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        # W4A8_DYNAMIC scale_bias (float32), per-channel new_quant_version only.
        # Allocated lazily in create_prefill_pool when the layer has
        # w13_scale_bias / w2_scale_bias parameters.
        self._prefill_w13_scale_bias: list[torch.Tensor] = []
        self._prefill_w2_scale_bias: list[torch.Tensor] = []
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

    def _init_prefetch_state(self) -> None:
        """Next-layer expert-prefetch infrastructure init."""
        # Next-layer expert prefetch infrastructure
        self._prefetch_stream = torch_npu.npu.Stream()
        # NPU copy of gate weights for graph-capturable on-device prediction
        # (predict_next_layer_experts_npu). Kept in fp32.
        self._gate_weights_npu: list[torch.Tensor | None] = []

        # Prefetch state: _prefetch_state_lock guards _prefetch_layer_npu_event,
        # which carries load_done_event from trigger_next_layer_prefetch into
        # update_weights' stream-join (capture-stream invariant — must stay).
        self._prefetch_state_lock = threading.Lock()
        self._prefetch_layer_npu_event: dict[int, torch_npu.npu.Event] = {}

        # Pinned CPU staging buffer for graph-mode prefetch: trigger_next_
        # layer_prefetch stages the next layer's log2phy here with
        # non_blocking D2H around the host callback, mirroring update_weights
        # (blocking .cpu() on a live graph tensor would deadlock on replay).
        # Allocated lazily in _finalize_offload (num_total_experts is only
        # known after MoE layers register).
        self._prefetch_log2phy_h: torch.Tensor | None = None
        self._prefetch_log2phy_np = None

    def _resolve_ep_info(self) -> None:
        """Lazily resolve ep_rank/ep_size from the EP group on first access.

        No-op (stays ep_size=1, ep_rank=0) when ``enable_multi_card`` is False,
        so the single-card path is unchanged.
        """
        if self._ep_info_resolved:
            return
        if self.enable_multi_card:
            from vllm.distributed.parallel_state import get_ep_group
            ep_group = get_ep_group()
            self._ep_size = ep_group.world_size
            self._ep_rank = ep_group.rank_in_group
        self._ep_info_resolved = True

    @property
    def ep_size(self) -> int:
        self._resolve_ep_info()
        return self._ep_size

    @property
    def ep_rank(self) -> int:
        self._resolve_ep_info()
        return self._ep_rank

    # ------------------------------------------------------------------ #
    #  Lifecycle: called during model init and after weight loading       #
    # ------------------------------------------------------------------ #

    def init_layer_cpu_buffers(self, layer, layer_moe_idx: int):
        """Allocate CPU weight + scale/offset buffers for one MoE layer.

        Called from AscendFusedMoE.__init__ after device tensors are set up,
        so CPU buffers exist before the safetensors weight loader runs.
        """
        ntotal = layer.global_num_experts
        if self.num_total_experts is None:
            self.num_total_experts = ntotal
        assert ntotal == self.num_total_experts, \
            f"MoE layers must have same expert count: {ntotal} vs {self.num_total_experts}"

        params_dtype = layer.w13_weight.dtype
        w13_shape = (layer.w13_weight.shape[2], layer.w13_weight.shape[1])
        w2_shape = (layer.w2_weight.shape[2], layer.w2_weight.shape[1])

        use_shard = self.offload_config.shard_per_rank
        if use_shard:
            # shard-per-rank: each rank holds ONLY its EP shard of weight
            # experts (ntotal // ep_size), as per-expert pinned tensors (like
            # the non-shared path, but shard-sized). No mmap, no cross-process
            # sharing, no staging — H2D reads each expert's own pinned storage.
            # Scales/offsets stay full-size (negligible memory). Placement must
            # be constrained to EP ownership (expert e → rank e // shard) so a
            # rank only loads experts it actually holds.
            shard = ntotal // max(1, self.ep_size)
            self._shard_size = shard
            self._shard_base = self.ep_rank * shard
            w13_list = [
                torch.empty(w13_shape, dtype=params_dtype, device="cpu",
                            pin_memory=True) for _ in range(shard)
            ]
            w2_list = [
                torch.empty(w2_shape, dtype=params_dtype, device="cpu",
                            pin_memory=True) for _ in range(shard)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        else:
            w13_list = [
                torch.empty(w13_shape, dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(ntotal)
            ]
            w2_list = [
                torch.empty(w2_shape, dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(ntotal)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)

        # Per-expert storage size (works for both list[0] and big_tensor[0]).
        first_w13 = self.w13_weights_cpu[-1][0]
        first_w2 = self.w2_weights_cpu[-1][0]
        self.w13_expert_size_bytes = first_w13.nelement() * first_w13.element_size()
        self.w2_expert_size_bytes = first_w2.nelement() * first_w2.element_size()

        # Scale / offset CPU buffers (W8A8)
        self._init_layer_scale_buffers(layer, layer_moe_idx, ntotal)

        self.moe_layers.append(layer)
        # If the cache policy was already built (this layer is registered
        # after _finalize_offload, e.g. an MTP draft MoE layer loaded after
        # the target model), extend the policy and per-layer stats so LRC
        # eviction applies uniformly to target and draft layers. Keeps the
        # invariant that every registered MoE layer has a matching cache
        # state and stats slot.
        self._extend_cache_for_layer()
        # Same post-finalize path for prefetch gate weights: register this
        # layer's gate so _gate_weights_npu stays index-aligned with
        # moe_layers. Without it, len(moe_layers) > len(_gate_weights_npu)
        # and predict_next_layer_experts_npu returns None for the boundary
        # layer. Pre-finalize layers are covered in bulk by
        # register_gate_weights(); the cache_policy sentinel skips them.
        self._register_layer_gate(layer)

    def _extend_cache_for_layer(self):
        """Grow cache_policy and stats lists to cover one more MoE layer.

        No-op before _finalize_offload has built the policy (the target
        layers are all covered in one shot there). Afterwards each newly
        registered layer (e.g. the MTP draft layer) gets its own fresh
        LRC state, so draft-layer hotness is tracked independently from
        the target layers.
        """
        if self.cache_policy is None:
            return
        new_idx = self.cache_policy.add_layer()
        self.cache_requests.append(0)
        self.cache_hits.append(0)
        self.cache_misses.append(0)
        self.cache_calls.append(0)
        self.last_hit_experts.append([])
        self.last_miss_experts.append([])
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] extended cache policy to layer=%d "
            "(total_layers=%d)",
            new_idx, len(self.cache_policy.layer_states))

    def _init_layer_scale_buffers(self, layer, layer_moe_idx: int,
                                   ntotal: int):
        """Allocate CPU scale/offset buffers for a single MoE layer."""
        attr_specs = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
            ("scale_bias_cpu_buffers", "w13_scale_bias"),
            ("scale_bias_cpu_buffers", "w2_scale_bias"),
        ]
        for buffer_dict_name, attr_name in attr_specs:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            dtype = dev_tensor.dtype
            if "scale_bias" in attr_name:
                per_expert_shape = tuple(dev_tensor.shape[1:])
            elif dtype.itemsize == 1:
                from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                    apply_mxfp4_weight_scale_layout)
                dtype = torch.uint8
                per_expert_shape = tuple(
                    apply_mxfp4_weight_scale_layout(dev_tensor[0].view(torch.uint8)).shape)
            else:
                per_expert_shape = (dev_tensor[0].numel(),)
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(ntotal):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype,
                                device="cpu", pin_memory=True))

    def _finalize_offload(self, model):
        """Post-weight-loading finalization.

        Must be called AFTER get_model() has finished loading all weights.
        Performs NZ format conversion, cache policy init, forward buffer
        init, fp32 scale refresh, prefill pool creation, and gate weight
        registration.
        """
        if not self.moe_layers:
            return
        # Barrier: ensure all deferred load_w13/load_w2/_load_scale_shard
        # copies have landed before process_weights_after_loading reads them.
        self.drain_load_pool()
        t0 = time.perf_counter()
        logger.info(
            "[OFFLOAD] weight load (safetensors→CPU buffer): %.1fs over %d calls",
            self._weight_load_secs, self._weight_load_calls)
        t1 = time.perf_counter()
        self.process_weights_after_loading()
        t2 = time.perf_counter()

        num_moe_layers = len(self.moe_layers)
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=self.num_total_experts,
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )
        t3 = time.perf_counter()

        ntotal = self.num_total_experts
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32, device="cpu", pin_memory=True)
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32, device="cpu", pin_memory=True)
        # Per-rank active-token mask (1=real, 0=pad), mirrored to pinned CPU so
        # the multi-card host callback can drop pad rows before counting. Under
        # single-batch TP, ranks past the real-token count route zero-hidden
        # PAD tokens whose topk is garbage; without this filter they pollute
        # global_counts (placement), the LRU freq, and hit/miss stats.
        self.mc2_mask_h = torch.zeros(
            self.offload_threshold, dtype=torch.int32,
            device="cpu", pin_memory=True)
        self.log2phy_h = torch.zeros(ntotal, dtype=torch.int32,
                                     device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()
        t4 = time.perf_counter()

        self.refresh_fp32_scales()
        t5 = time.perf_counter()
        self.create_prefill_pool()
        t6 = time.perf_counter()
        if self.offload_config.expert_prefetch_enabled:
            self.register_gate_weights(model)
            # Pinned staging buffer for graph-mode prefetch: trigger_next_
            # layer_prefetch stages the next layer's log2phy here with
            # non_blocking D2H before launching the host callback, mirroring
            # update_weights (blocking .cpu() on a live graph tensor would
            # deadlock during replay).
            self._prefetch_log2phy_h = torch.zeros(
                self.num_total_experts, dtype=torch.int32,
                device='cpu', pin_memory=True)
            self._prefetch_log2phy_np = self._prefetch_log2phy_h.numpy()
        t7 = time.perf_counter()
        logger.info(
            "[OFFLOAD] finalize breakdown: process_weights=%.1fs "
            "cache_policy=%.1fs buffers=%.1fs init_device=%.1fs "
            "prefill_pool=%.1fs gate=%.1fs | total=%.1fs",
            t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6, t7 - t0)

    def process_weights_after_loading(self):
        """Convert resident CPU expert buffers to the on-device weight format.

        W8A8 (int8): fractal NZ cast on the transpose-after buffer layout
        (the device path transposes first, then casts NZ).
        W4A8_DYNAMIC (int8 cpu, int32 device): mirror the device path which
        transposes, casts NZ, then packs 4 int8 → 1 int32.
        W4A8_MXFP (uint8): mirror the device process_weights_after_loading,
        which casts29 (mxfp4) on the *pre-transpose* shape and then
        transposes — so we restore the pre-transpose shape first, cast, and
        transpose back to match the device slot layout byte-for-byte.

        After this runs each w13/w2 CPU tensor still reports its original
        shape but its storage holds on-device-format bytes — a "liar tensor".
        Touch it only via untyped_storage() slicing, never through the tensor
        view. No-op for non-quantized (other dtype) models.
        """
        first_w13 = self.w13_weights_cpu[0][0]
        first_layer = self.moe_layers[0]
        # Detect W4A8_DYNAMIC by scale dtype: process_scale converts float32
        # to int64 on the device for both modelslim and compressed_tensors
        # paths. The weight dtype is NOT a reliable signal — modelslim leaves
        # it as int8 (no pack_to_int32) while compressed_tensors packs to
        # int32. Without this check, modelslim W4A8 would be misrouted to the
        # W8A8 branch, skipping scale encoding and producing garbled output.
        is_w4a8 = (hasattr(first_layer, 'w13_weight_scale') and
                   first_layer.w13_weight_scale.dtype == torch.int64)
        if first_w13.dtype == torch.int8:
            first_dev = first_layer.w13_weight
            if first_dev.dtype == torch.int32:
                # compressed_tensors W4A8: weight packed to int32
                self._cast_cpu_weights_to_device_format(w4a8_dynamic=True)
            else:
                # W8A8 or modelslim W4A8: weight is int8, just NZ cast
                self._cast_cpu_weights_to_device_format(mxfp4=False)
            # Scale encoding for W4A8 (both modelslim and compressed_tensors).
            # Must run AFTER _cast_cpu_weights_to_device_format so the CPU
            # buffers are already in device byte layout.
            if is_w4a8:
                self._process_scale_bias_cpu_buffers()
                self._encode_w4a8_dynamic_weight_scales()
        elif first_w13.dtype == torch.uint8:
            self._cast_cpu_weights_to_device_format(mxfp4=True)
        # else: non-quantized model, no-op

    def _cast_cpu_weights_to_device_format(self, mxfp4: bool = False,
                                            w4a8_dynamic: bool = False):
        """Relayout resident CPU w13/w2 expert buffers into the device format.

        NZ (W8A8) and format-29 mxfp4 (W4A8_MXFP) are equal-length relayouts,
        so per-expert on-device bytes == nelement * element_size; we still
        recompute from the cast storage to stay correct if that ever changes.

        W4A8_DYNAMIC mirrors the device path: transpose → NZ cast →
        pack 4 int8 into 1 int32 (view as int32).  The storage size is
        preserved (4× fewer elements at 4× element size), so the CPU buffer
        can hold the packed bytes without reallocation.
        """
        num_moe_layers = len(self.w13_weights_cpu)
        num_experts = len(self.w13_weights_cpu[0])
        use_shard = self.offload_config.shard_per_rank
        for layer_id in range(num_moe_layers):
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            if mxfp4:
                w13 = w13.transpose(1, 2).contiguous()
                w2 = w2.transpose(1, 2).contiguous()
                w13 = torch_npu.npu_format_cast(
                    w13.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w2 = torch_npu.npu_format_cast(
                    w2.view(torch.uint8), 29,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                w13 = w13.transpose(1, 2)
                w2 = w2.transpose(1, 2)
            elif w4a8_dynamic:
                # CPU buffer is already (E, H, dim2) — _copy_w13_shard stored
                # owned.t(), matching the post-transpose device layout. The
                # device path (process_weights_after_loading_modelslim) does
                # transpose(1,2) → NZ cast → pack_to_int32, where transpose
                # converts (E, dim2, H) → (E, H, dim2). Since the CPU buffer
                # is already (E, H, dim2), we NZ-cast directly — NO extra
                # transpose. A double transpose here would apply NZ blocking
                # to (dim2, H) instead of (H, dim2), producing wrong block
                # layout and garbled output.
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
                w13 = w13.view(torch.int32)
                w2 = w2.view(torch.int32)
            else:
                w13 = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
                w2 = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w13_storage = w13.untyped_storage()
            w2_storage = w2.untyped_storage()
            per_w13 = w13_storage.nbytes() // num_experts
            per_w2 = w2_storage.nbytes() // num_experts
            self.w13_expert_size_bytes = per_w13
            self.w2_expert_size_bytes = per_w2
            for local_i in range(num_experts):
                # _expert_dst_storage takes a GLOBAL eid (shard-per-rank
                # remaps it to the local slot); w13/w2_storage are indexed by
                # the local position in the stacked tensor (== global id when
                # not sharded).
                geid = (self._shard_base + local_i) if use_shard else local_i
                self._expert_dst_storage(layer_id, geid, 'w13').copy_(
                    w13_storage[local_i * per_w13 : (local_i + 1) * per_w13]
                )
                self._expert_dst_storage(layer_id, geid, 'w2').copy_(
                    w2_storage[local_i * per_w2 : (local_i + 1) * per_w2]
                )

    def _process_scale_bias_cpu_buffers(self):
        """Apply update_bias transformation to scale_bias CPU buffers.

        Mirrors the device-side update_bias for W4A8_DYNAMIC new_quant_version:
        w13_scale_bias: (D1, 1) -> transpose -> (1, D1) -> sum(axis=0) -> (D1,)
        w2_scale_bias: (D1, D2) -> transpose -> (D2, D1) -> sum(axis=0) -> (D1,)
        """
        for attr_name, layer_buffers in self.scale_bias_cpu_buffers.items():
            for layer_idx, expert_buffers in enumerate(layer_buffers):
                new_buffers = []
                for buf in expert_buffers:
                    transformed = buf.transpose(0, 1).contiguous().sum(dim=0)
                    new_buffers.append(transformed)
                layer_buffers[layer_idx] = new_buffers

    def _encode_w4a8_dynamic_weight_scales(self):
        """Encode W4A8_DYNAMIC weight_scale CPU buffers to device int64 format.

        The safetensors checkpoint stores ``w13_weight_scale`` /
        ``w2_weight_scale`` as float32 tensors, but the device-side
        ``AscendW4A8DynamicFusedMoEMethod.process_scale`` reinterprets the
        float32 bytes as uint32 and zero-extends to int64 before storing it
        on the NPU. The decode-path H2D ``copy_`` therefore must write
        int64-encoded bytes — copying raw float32 into an int64 device tensor
        would corrupt the kernel's scale decoding.

        This method mirrors the per-channel branch of ``process_scale``
        (the only branch supported by expert offload today): float32 →
        uint32 bit-reinterpret → int64 zero-extension. Each expert buffer is
        encoded independently (per-channel encoding is element-wise), so the
        transformation is applied per-expert without cross-expert ops.

        After this runs the CPU buffer dtype changes from float32 to int64,
        matching ``layer.w13_weight_scale.dtype`` on the NPU.
        """
        import numpy as np
        for attr_name in ("w13_weight_scale", "w2_weight_scale"):
            if attr_name not in self.scale_cpu_buffers:
                continue
            for layer_idx, expert_buffers in enumerate(
                    self.scale_cpu_buffers[attr_name]):
                encoded_buffers = []
                for buf in expert_buffers:
                    # buf: float32, shape per-expert (e.g. (2*IN,) for w13)
                    scale_np = np.ascontiguousarray(
                        buf.cpu().numpy()).astype(np.float32)
                    # Bit-reinterpret float32 bytes as uint32, then
                    # zero-extend to int64 — identical to device process_scale
                    # per-channel branch.
                    scale_np.dtype = np.uint32
                    encoded = scale_np.astype(np.int64)
                    encoded_buf = torch.from_numpy(np.ascontiguousarray(
                        encoded.copy()))
                    encoded_buffers.append(encoded_buf)
                self.scale_cpu_buffers[attr_name][layer_idx] = encoded_buffers

    # ------------------------------------------------------------------ #
    #  Deferred weight-load pool                                          #
    # ------------------------------------------------------------------ #
    #
    # Weight loading is callback-driven: the safetensors loader calls
    # load_w13/load_w2/_load_scale_shard once per shard (~99k calls), serially
    # in the main thread. The per-call strided transpose-copy into pinned
    # memory is ~0.2 GB/s single-threaded, which dominated startup (~9 min).
    #
    # Strategy: each loader callback (a) owns the shard via a synchronous
    # .clone() while the safetensors mmap is still mapped, then (b) submits
    # the strided transpose-copy to a worker pool and returns immediately.
    # The main thread keeps pulling shards while the pool churns through
    # copies concurrently. drain_load_pool() barriers before _finalize_offload
    # reads the buffers. Because the deferred copy reads the owned clone (not
    # the mmap view), it stays correct after the safetensors mmap is unmapped
    # (which happens before _finalize_offload runs).

    def _get_load_pool(self) -> ThreadPoolExecutor:
        if self._load_pool is None:
            # Pin torch intra-op threads to 1: otherwise each copy_ spawns
            # nproc libgomp threads and 32 workers x 640 cores exhausts the
            # thread limit (EAGAIN). Parallelism comes from the pool itself.
            self._saved_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            self._load_pool = ThreadPoolExecutor(
                max_workers=self._LOAD_POOL_WORKERS,
                thread_name_prefix="offload-load")
            self._load_phase_start = time.perf_counter()
            logger.info(
                "[OFFLOAD] starting parallel weight load (workers=%d)",
                self._LOAD_POOL_WORKERS)
        return self._load_pool

    def _track_load_future(self, fut) -> None:
        self._load_futures.append(fut)
        if len(self._load_futures) >= self._LOAD_POOL_DRAIN_EVERY:
            self._drain_futures()

    def _drain_futures(self) -> None:
        if not self._load_futures:
            return
        # f.result() re-raises any worker exception (e.g. shape mismatch).
        for f in self._load_futures:
            f.result()
        self._load_futures.clear()

    def drain_load_pool(self) -> None:
        """Wait for all deferred weight copies to finish.

        Safe to call after the safetensors mmap is unmapped: deferred copies
        read owned clones, not mmap views.
        """
        self._drain_futures()
        if self._load_pool is not None:
            self._load_pool.shutdown(wait=True)
            self._load_pool = None
            if self._saved_num_threads is not None:
                torch.set_num_threads(self._saved_num_threads)
                self._saved_num_threads = None
            self._weight_load_secs = time.perf_counter() - self._load_phase_start

    # -- int4 packing helper for W4A8_DYNAMIC checkpoints -- #

    @staticmethod
    def _pack_int4_dim0(weight: torch.Tensor) -> torch.Tensor:
        """Pack pairs of int4 values along dim 0.

        W4A8_DYNAMIC (msModelSlim new_quant_version) checkpoint weights store
        one int4 value per int8 element along the output dimension.  The
        device tensor expects two int4 values packed into one int8 byte,
        halving dim 0.  This helper performs that packing.

        For w1/w3 shard ``(IN, H)`` -> ``(IN // 2, H)``.
        For w2 ``(H, IN)`` -> ``(H // 2, IN)``.
        """
        if weight.dtype != torch.int8:
            weight = weight.to(torch.int8)
        assert weight.shape[0] % 2 == 0, (
            f"dim 0 must be even for int4 packing, got {weight.shape[0]}")
        pairs = weight.reshape(weight.shape[0] // 2, 2, *weight.shape[1:])
        lo = pairs[:, 0] & 0x0F
        hi = pairs[:, 1] & 0x0F
        return ((hi << 4) | lo).contiguous()

    # -- worker copy kernels (static: no self, no shared mutable state) -- #

    @staticmethod
    def _copy_w13_shard(cpu: torch.Tensor, owned: torch.Tensor,
                        shard_id: str, intermed: int) -> None:
        if shard_id == "w1":
            cpu[:, :intermed].copy_(owned.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + owned.shape[0]].copy_(owned.t())

    @staticmethod
    def _copy_w2(dst: torch.Tensor, owned: torch.Tensor) -> None:
        dst.copy_(owned.t())

    @staticmethod
    def _copy_scale_assembled(target: torch.Tensor,
                              w1: torch.Tensor, w3: torch.Tensor) -> None:
        assembled = torch.cat([w1, w3], dim=0)
        if target.dtype == torch.uint8:
            # W4A8_MXFP: store the post-layout bytes so the 1D buffer matches
            # the post-process device slot element order (the device path
            # applies reshape(...,k//2,2).transpose to the e8m0 scale).
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            assembled = apply_mxfp4_weight_scale_layout(assembled.view(torch.uint8))
        target.copy_(assembled.reshape(target.shape))

    @staticmethod
    def _copy_scale_direct(target: torch.Tensor, owned: torch.Tensor) -> None:
        if target.dtype == torch.uint8:
            from vllm_ascend.quantization.methods.w4a8_mxfp4 import (
                apply_mxfp4_weight_scale_layout)
            owned = apply_mxfp4_weight_scale_layout(owned.view(torch.uint8))
        target.copy_(owned.reshape(target.shape))

    # ------------------------------------------------------------------ #
    #  Weight-load entry points (called by the safetensors loader)        #
    # ------------------------------------------------------------------ #

    def register_gate_weights(self, model):
        """Store an fp32 NPU copy of gate.weight for each MoE layer.

        Called from _finalize_offload() after all MoE layers are registered.
        Used by predict_next_layer_experts_npu() so prediction runs on-device
        and can be captured in a CUDA/NPU graph.
        """
        from vllm_ascend.models.deepseek_v4 import DeepseekV4MoE
        moe_wrappers = [m for m in model.modules()
                        if isinstance(m, DeepseekV4MoE)]
        for wrapper in moe_wrappers:
            gate_param = wrapper.gate.weight.data
            # fp32 clone on the gate's own device for on-device,
            # graph-capturable prediction.
            self._gate_weights_npu.append(gate_param.float().clone())
        logger.info("[PREFETCH] registered gate weights for %d MoE layers",
                    len(self._gate_weights_npu))

    def _register_layer_gate(self, layer):
        """Stage one MoE layer's gate.weight for prefetch prediction.

        Single-layer counterpart to register_gate_weights(), for layers
        registered after _finalize_offload (e.g. the MTP draft MoE). Keeps
        _gate_weights_npu index-aligned with moe_layers so
        predict_next_layer_experts_npu can look up
        _gate_weights_npu[next_idx] for every registered layer.

        No-op before _finalize_offload has built the cache policy (the
        target layers are covered in bulk there); mirrors the
        _extend_cache_for_layer() sentinel.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if self.cache_policy is None:
            return
        gate = getattr(layer, 'gate', None)
        if gate is None:
            return
        gate_param = gate.weight.data
        self._gate_weights_npu.append(gate_param.float().clone())
        logger.info(
            "[PREFETCH] registered gate weight for post-finalize layer "
            "(total gates=%d, moe_layers=%d)",
            len(self._gate_weights_npu), len(self.moe_layers))

    def _shard_local(self, global_eid: int) -> int | None:
        """Map a GLOBAL expert id to the CPU weight buffer's local index.

        shard-per-rank: this rank owns shard [base, base+shard); return the
        local index, or None if the expert isn't owned here (caller skips the
        load). Other modes: identity — the buffer is full (global-indexed) or
        shared (global-indexed mmap slice).
        """
        if not self.offload_config.shard_per_rank:
            return global_eid
        local = global_eid - self._shard_base
        return local if 0 <= local < self._shard_size else None

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._shard_local(expert_id)
        if idx is None:
            return  # shard-per-rank: expert not owned by this rank
        cpu = self.w13_weights_cpu[layer_moe_idx][idx]
        intermed = cpu.shape[1] // 2
        if loaded_weight.ndim > 0 and loaded_weight.shape[0] > intermed:
            if loaded_weight.shape[0] == 2 * intermed:
                loaded_weight = self._pack_int4_dim0(loaded_weight)
            else:
                loaded_weight = loaded_weight.narrow(0, 0, intermed)
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(
            self._copy_w13_shard, cpu, owned, shard_id, intermed)
        self._track_load_future(fut)

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (transposed) via the load pool."""
        self._weight_load_calls += 1
        idx = self._shard_local(expert_id)
        if idx is None:
            return  # shard-per-rank: expert not owned by this rank
        dst = self.w2_weights_cpu[layer_moe_idx][idx]
        owned = loaded_weight.cpu().clone()
        fut = self._get_load_pool().submit(self._copy_w2, dst, owned)
        self._track_load_future(fut)

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _load_scale_shard(self, layer_moe_idx: int, expert_id: int,
                          attr_name: str, shard_id: str,
                          loaded_weight: torch.Tensor):
        """Load a scale/offset shard into its CPU buffer via the load pool.

        w13 scale/offset arrives as two shards (w1, w3) that must be
        concatenated along dim 0. We stash the first-arriving owned clone in
        _scale_shard_temp and assemble when the second shard arrives.
        w2 scale/offset is a single shard — clone and defer directly.
        """
        self._weight_load_calls += 1
        assert shard_id in ("w1", "w2", "w3"), f"unexpected shard_id: {shard_id}"
        if "scale_bias" in attr_name:
            target_dict = self.scale_bias_cpu_buffers
        elif "scale" in attr_name:
            target_dict = self.scale_cpu_buffers
        else:
            target_dict = self.offset_cpu_buffers
        target = target_dict[attr_name][layer_moe_idx][expert_id]
        if attr_name.startswith("w13_"):
            key = (layer_moe_idx, expert_id, attr_name)
            pending_shard = self._scale_shard_temp.pop(key, None)
            if pending_shard is not None:
                # Second shard — own it, then defer cat + copy.
                cur_shard = loaded_weight.cpu().clone()
                if shard_id == "w1":
                    w1, w3 = cur_shard, pending_shard
                else:
                    w1, w3 = pending_shard, cur_shard
                fut = self._get_load_pool().submit(
                    self._copy_scale_assembled, target, w1, w3)
                self._track_load_future(fut)
            else:
                # First shard — stash an owned clone.
                self._scale_shard_temp[key] = loaded_weight.cpu().clone()
        else:
            # w2 scale/offset — single shard.
            owned = loaded_weight.cpu().clone()
            fut = self._get_load_pool().submit(
                self._copy_scale_direct, target, owned)
            self._track_load_future(fut)

    def refresh_fp32_scales(self):
        """Recompute the derived fp32 per-expert scale after weight loading.

        Device experts are already in place (loaded by the weight loader and
        process_weights_after_loading); this only refreshes
        w13_weight_scale_fp32 from the freshly-loaded w13_weight_scale.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _finalize_offload() after decode buffers are set up.
        Creates ndl device tensors each holding all experts (e.g. 128).
        These are used when num_tokens > offload_threshold (large-batch
        prefill), loaded via full-overwrite in _prefill_load_layer.
        """
        if self._prefill_initialized:
            return
        if not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_layer = self.moe_layers[0]
        dev = pool_layer.w13_weight.device
        dt = pool_layer.w13_weight.dtype
        # Size the pool to the per-rank EP shard, NOT the global expert count.
        # The All2All prefill GMM (aclnnGroupedMatmulWeightNz) requires
        # groupList == weight.dim0; groupList = shard, so the pool must hold
        # exactly `shard` experts per rank. mc_shard_size == num_total//ep_size,
        # which is ntotal for single-card (ep_size=1) — so single-card is
        # unchanged (pool holds all experts), multi-card holds the rank's shard.
        ntotal = self.mc_shard_size

        for _ in range(ndl):
            self._alloc_prefill_pool_slot(pool_layer, dev, dt, ntotal)

        # Cast prefill pool weight tensors to the on-device format (kernel
        # requires it). Must happen BEFORE loading data — same order as decode
        # path: create → format-cast → copy_(cpu → npu).
        self._cast_prefill_pool_format(dev, dt)

        # Prefill log2phy: identity — all experts mapped to their slots
        self._prefill_log2phy = torch.arange(ntotal, dtype=torch.int32, device=dev)

        # Pre-initialize all pool slots with layer 0 weights so that
        # profile_run / _dummy_run (which may use prefill path) has
        # valid data.  Subsequent _prefill_load_layer calls will
        # overwrite with the correct per-layer weights.
        self._init_prefill_pool_data(dev, ntotal, ndl)
        self._prefill_initialized = True
        logger.info("[PREFILL_POOL] allocated %d layers × %d experts, "
                    "w13[0].shape=%s w2[0].shape=%s",
                    ndl, ntotal,
                    tuple(self._prefill_w13[0].shape),
                    tuple(self._prefill_w2[0].shape))

    def _alloc_prefill_pool_slot(self, pool_layer, dev, dt, ntotal: int):
        """Append one prefill-pool slot (weights always; scales/offsets/scale_bias
        only if the layer carries them). Weights use the layer dtype `dt`;
        per-channel fp32 scales use float32; the rest use their source dtype."""
        # (target_attr, source_attr, dtype_override)
        quant_specs = [
            ("_prefill_w13_scale", "w13_weight_scale", None),
            ("_prefill_w13_scale_fp32", "w13_weight_scale_fp32", torch.float32),
            ("_prefill_w13_offset", "w13_weight_offset", None),
            ("_prefill_w2_scale", "w2_weight_scale", None),
            ("_prefill_w2_offset", "w2_weight_offset", None),
            ("_prefill_w13_scale_bias", "w13_scale_bias", None),
            ("_prefill_w2_scale_bias", "w2_scale_bias", None),
        ]
        self._prefill_w13.append(torch.empty(
            (ntotal,) + tuple(pool_layer.w13_weight.shape[1:]), dtype=dt, device=dev))
        self._prefill_w2.append(torch.empty(
            (ntotal,) + tuple(pool_layer.w2_weight.shape[1:]), dtype=dt, device=dev))
        for tgt, src, dtype_override in quant_specs:
            if not hasattr(pool_layer, src):
                continue
            src_t = getattr(pool_layer, src)
            dtype = dtype_override if dtype_override is not None else src_t.dtype
            getattr(self, tgt).append(torch.empty(
                (ntotal,) + tuple(src_t.shape[1:]), dtype=dtype, device=dev))

    def _cast_prefill_pool_format(self, dev, dt):
        """Cast prefill-pool weight tensors to the on-device (kernel) format.

        Must run BEFORE data is loaded (same create → format-cast ordering as
        the decode path). dtype-dispatched:
          - int8 (W8A8): straight FRACTAL_NZ cast.
          - int32 (W4A8_DYNAMIC): rebuild via int8 backing → NZ → view int32
            (an empty int32 tensor can't be NZ-cast directly).
          - uint8 (W4A8_MXFP): cast29 on the pre-transpose shape, then transpose.
        """
        n = len(self._prefill_w13)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)
        elif dt == torch.int32:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(n):
                t13 = self._prefill_w13[i]
                t2 = self._prefill_w2[i]
                t13_nz = torch_npu.npu_format_cast(
                    torch.empty(t13.shape[:-1] + (t13.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                t2_nz = torch_npu.npu_format_cast(
                    torch.empty(t2.shape[:-1] + (t2.shape[-1] * 4,),
                                dtype=torch.int8, device=dev),
                    ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w13[i] = t13_nz.view(torch.int32)
                self._prefill_w2[i] = t2_nz.view(torch.int32)
        elif dt == torch.uint8:
            for i in range(n):
                for attr in ("_prefill_w13", "_prefill_w2"):
                    t = getattr(self, attr)[i]
                    t = torch_npu.npu_format_cast(
                        t.transpose(1, 2).contiguous().view(torch.uint8), 29,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=torch_npu.float4_e2m1fn_x2,
                    )
                    getattr(self, attr)[i] = t.transpose(1, 2)

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Use simple per-expert copy_() — same pattern
        as the decode path's _update_weights.
        """
        has_scales = bool(self._prefill_w13_scale)
        has_offsets = bool(self._prefill_w13_offset)
        has_scale_bias = bool(self._prefill_w13_scale_bias)

        for slot in range(ndl):
            for eid in range(min(ntotal, len(self.w13_weights_cpu[0]))):
                # _expert_src_storage takes a GLOBAL eid (shard-per-rank remaps
                # it to the local shard slot); the prefill-pool slot stays local.
                geid = (self._shard_base + eid) if self.offload_config.shard_per_rank else eid
                self._prefill_w13[slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self._expert_src_storage(0, geid, 'w13')
                )
                self._prefill_w2[slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self._expert_src_storage(0, geid, 'w2')
                )

            # Initialize scale/offset buffers with layer 0 data (W8A8)
            if has_scales:
                for scale_name, prefill_list, cpu_buffers in [
                    ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                    ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
                ]:
                    if (scale_name in cpu_buffers and
                            0 < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][0]))):
                            src = cpu_buffers[scale_name][0][eid]
                            prefill_list[slot][eid].copy_(
                                src.reshape(prefill_list[slot][eid].shape))
            if has_offsets:
                for offset_name, prefill_list, cpu_buffers in [
                    ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                    ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
                ]:
                    if (offset_name in cpu_buffers and
                            0 < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][0]))):
                            src = cpu_buffers[offset_name][0][eid]
                            prefill_list[slot][eid].copy_(
                                src.reshape(prefill_list[slot][eid].shape))
            # Initialize scale_bias buffers with layer 0 data (W4A8_DYNAMIC)
            if has_scale_bias:
                for sb_name, prefill_list in [
                    ("w13_scale_bias", self._prefill_w13_scale_bias),
                    ("w2_scale_bias", self._prefill_w2_scale_bias),
                ]:
                    cpu_buffers = self.scale_bias_cpu_buffers
                    if (sb_name in cpu_buffers and
                            len(cpu_buffers[sb_name]) > 0 and
                            slot < len(prefill_list)):
                        for eid in range(min(ntotal, len(cpu_buffers[sb_name][0]))):
                            src = cpu_buffers[sb_name][0][eid]
                            prefill_list[slot][eid].copy_(
                                src.reshape(prefill_list[slot][eid].shape))
            # Initialize fp32 scale (convert from scale)
            if has_scales and slot < len(self._prefill_w13_scale_fp32):
                for eid in range(min(ntotal, self._prefill_w13_scale[slot].shape[0])):
                    self._prefill_w13_scale_fp32[slot][eid].copy_(
                        self._prefill_w13_scale[slot][eid].to(torch.float32))

    def _prefill_load_layer(self, layer_idx: int, log2phy: torch.Tensor):
        """Load ALL experts for model layer layer_idx into the prefill pool.

        For W8A8: loads into normal-format scratch, then casts to NZ.
        For unquantized: loads directly into pool tensors via copy_().
        Full-overwrite into pool_slot = layer_idx % ndl.  No slot_owner
        tracking needed — log2phy is set to identity for prefill.
        """
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        dev = self._prefill_w13[pool_slot].device
        ntotal = self.num_total_experts
        is_w8a8 = self._prefill_w13[pool_slot].dtype == torch.int8

        if self._debug:
            logger.info("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                        layer_idx, pool_slot, ntotal, is_w8a8)

        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        with torch_npu.npu.stream(self.load_stream):
            for eid in range(ntotal):
                self._prefill_w13[pool_slot].untyped_storage()[eid * self.w13_expert_size_bytes : (eid + 1) * self.w13_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w13')
                )
                self._prefill_w2[pool_slot].untyped_storage()[eid * self.w2_expert_size_bytes : (eid + 1) * self.w2_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w2')
                )

            # W8A8 scale/offset — load into prefill buffers
            for scale_name, prefill_list, cpu_buffers in [
                ("w13_weight_scale", self._prefill_w13_scale, self.scale_cpu_buffers),
                ("w2_weight_scale", self._prefill_w2_scale, self.scale_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (scale_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[scale_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[scale_name][layer_idx]))):
                            src = cpu_buffers[scale_name][layer_idx][eid]
                            prefill_list[pool_slot][eid].copy_(
                                src.reshape(prefill_list[pool_slot][eid].shape))
            for offset_name, prefill_list, cpu_buffers in [
                ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (offset_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][layer_idx]))):
                            src = cpu_buffers[offset_name][layer_idx][eid]
                            prefill_list[pool_slot][eid].copy_(
                                src.reshape(prefill_list[pool_slot][eid].shape))

            # W4A8_DYNAMIC scale_bias — load into prefill buffers
            for sb_name, prefill_list in [
                ("w13_scale_bias", self._prefill_w13_scale_bias),
                ("w2_scale_bias", self._prefill_w2_scale_bias),
            ]:
                cpu_buffers = self.scale_bias_cpu_buffers
                if pool_slot < len(prefill_list):
                    if (sb_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[sb_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[sb_name][layer_idx]))):
                            src = cpu_buffers[sb_name][layer_idx][eid]
                            prefill_list[pool_slot][eid].copy_(
                                src.reshape(prefill_list[pool_slot][eid].shape))

            # Refresh fp32 scale for prefill pool
            if (pool_slot < len(self._prefill_w13_scale_fp32) and
                    pool_slot < len(self._prefill_w13_scale)):
                # Copy scale data from freshly loaded scale to fp32
                for eid in range(min(ntotal, self._prefill_w13_scale[pool_slot].shape[0])):
                    self._prefill_w13_scale_fp32[pool_slot][eid].copy_(
                        self._prefill_w13_scale[pool_slot][eid].to(torch.float32))

            self.load_stream.synchronize()

        # NOTE: Do NOT modify the layer's own log2phy here — decode path
        # relies on it staying with 32-expert mapping.  Prefill path in
        # apply() explicitly uses self._prefill_log2phy instead.

    # ------------------------------------------------------------------ #
    #  Multi-card prefill: per-rank EP shard into the prefill pool        #
    # ------------------------------------------------------------------ #
    @property
    def mc_shard_size(self) -> int:
        """Experts per rank in standard EP shard (num_total_experts // ep_size)."""
        return self.num_total_experts // max(1, self.ep_size)

    def _get_shard_expert_map(self) -> torch.Tensor:
        """Standard EP shard expert_map for THIS rank, len = num_total_experts.
        Maps global experts in this rank's shard [base, base+shard) to local
        index [0..shard), everything else to -1. Consumed by the AllGather
        dispatcher (it masks topk_ids to local via expert_map != -1 and uses
        active_expert_range = [rank*nel, rank*nel+nel]).
        """
        if getattr(self, '_mc_shard_expert_map', None) is not None:
            return self._mc_shard_expert_map
        shard = self.mc_shard_size
        base = self.ep_rank * shard
        emap = torch.full((self.num_total_experts,), -1, dtype=torch.int32)
        for i in range(shard):
            emap[base + i] = i
        self._mc_shard_expert_map = emap
        return emap

    def _prefill_load_layer_shard(self, layer_idx: int):
        """Multi-card prefill: load THIS rank's EP shard into the prefill pool.

        Standard EP AllGather has rank r own global experts [r*shard:(r+1)*shard]
        and compute them in LOCAL slots [0:shard]. So we load only the rank's
        shard (not all experts) into pool local slots [0:shard]. The pool buffer
        is sized for num_total_experts; slots [shard:] stay unused this forward.
        Mirrors _prefill_load_layer but sharded + per-rank.
        """
        if not self._prefill_initialized or not self.moe_layers:
            return
        ndl = self.num_device_layers
        pool_slot = layer_idx % ndl
        shard = self.mc_shard_size
        base = self.ep_rank * shard

        with torch_npu.npu.stream(self.load_stream):
            for local_i in range(shard):
                eid = base + local_i
                self._prefill_w13[pool_slot].untyped_storage()[local_i * self.w13_expert_size_bytes : (local_i + 1) * self.w13_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w13'))
                self._prefill_w2[pool_slot].untyped_storage()[local_i * self.w2_expert_size_bytes : (local_i + 1) * self.w2_expert_size_bytes].copy_(
                    self._expert_src_storage(layer_idx, eid, 'w2'))
            # quant scales / offsets / scale_bias (w4a8) — shard only
            for scale_name, prefill_list in [("w13_weight_scale", self._prefill_w13_scale),
                                             ("w2_weight_scale", self._prefill_w2_scale)]:
                if pool_slot < len(prefill_list) and scale_name in self.scale_cpu_buffers \
                        and layer_idx < len(self.scale_cpu_buffers[scale_name]):
                    for local_i in range(min(shard, len(self.scale_cpu_buffers[scale_name][layer_idx]))):
                        src = self.scale_cpu_buffers[scale_name][layer_idx][base + local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for off_name, prefill_list in [("w13_weight_offset", self._prefill_w13_offset),
                                           ("w2_weight_offset", self._prefill_w2_offset)]:
                if pool_slot < len(prefill_list) and off_name in self.offset_cpu_buffers \
                        and layer_idx < len(self.offset_cpu_buffers[off_name]):
                    for local_i in range(min(shard, len(self.offset_cpu_buffers[off_name][layer_idx]))):
                        src = self.offset_cpu_buffers[off_name][layer_idx][base + local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            for sb_name, prefill_list in [("w13_scale_bias", self._prefill_w13_scale_bias),
                                          ("w2_scale_bias", self._prefill_w2_scale_bias)]:
                if pool_slot < len(prefill_list) and sb_name in self.scale_bias_cpu_buffers \
                        and layer_idx < len(self.scale_bias_cpu_buffers[sb_name]):
                    for local_i in range(min(shard, len(self.scale_bias_cpu_buffers[sb_name][layer_idx]))):
                        src = self.scale_bias_cpu_buffers[sb_name][layer_idx][base + local_i]
                        prefill_list[pool_slot][local_i].copy_(src.reshape(prefill_list[pool_slot][local_i].shape))
            # Sync the load stream so the pool data is valid before the GMM reads
            # it (the apply path continues on the default stream after we return;
            # the host-side block here guarantees the copies are done before the
            # next op is queued).
            self.load_stream.synchronize()

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None,
                        hidden_states: torch.Tensor | None = None) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Routes to prefill pool (full-overwrite) when num_tokens exceeds
        offload_threshold, otherwise uses per-expert paging (decode path).

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.
            topk_weights: Optional routing weights for cache policy.
            hidden_states: Optional [num_tokens, hidden_dim] tensor used
                           for next-layer expert prefetch prediction.

        Returns: number of CPU→NPU copies performed (decode path),
                 0 for prefill path (full-overwrite via pool).
        """
        # Multi-card offload sets routed layers' log2phy=None (standard-EP
        # dispatch) and uses update_weights_multi_card instead. Shared experts
        # (per-card replicated, not dispatched) may still reach here with
        # log2phy=None — bail out to avoid copy_(None). TODO: give shared
        # experts a proper single-card log2phy so they still page in.
        if log2phy is None:
            return 0
        num_tokens = topk_ids.size(0)
        if num_tokens > self.offload_threshold:
            # Prefill: layerwise reuse + full-overwrite of all experts
            if (self._prefill_initialized
                    and not self._skip_prefill):
                try:
                    layer_idx = self.moe_layers.index(layer)
                except ValueError:
                    return 0
                self._prefill_load_layer(layer_idx, log2phy)
                return 0
            else:
                # Profile run or pool not ready — bail out gracefully
                return 0

        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        # Wait for prefetch NPU copies to complete before using the weights.
        # Use stream wait (graphable) instead of host synchronize.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)

        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None 
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)
        self._is_prefetch = False
        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
            self._is_prefetch,
        )
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream,
                self._update_weights,
                args,
            )
        else:
            self._update_weights(args)

        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _mc_handle_prefill_regime(self, layer_idx) -> bool:
        """Multi-card PREFILL (non-MC2 comm): load this rank's EP shard into the
        prefill pool. Returns True when handled so the caller returns early.

        We load even during profile_run: the decode buffer is too small for the
        EP shard, so multi-card prefill MUST use the pool, and real weights
        avoid GMM errors on garbage scales (single-card can skip during profile
        because its AllGather reuses decode weights; multi-card All2All cannot).
        """
        from vllm_ascend.ascend_forward_context import MoECommType
        if _EXTRA_CTX.moe_comm_type == MoECommType.MC2:
            return False
        if self._prefill_initialized:
            self._prefill_load_layer_shard(layer_idx)
            if self._debug:
                base = self.ep_rank * self.mc_shard_size
                logger.info(
                    "[MC_OBS] rank=%s L=%s PREFILL: loaded EP shard "
                    "experts[%d..%d] (%d experts) into pool on rank%d "
                    "(static shard, reloaded each prefill forward)",
                    self.ep_rank, layer_idx, base, base + self.mc_shard_size - 1,
                    self.mc_shard_size, self.ep_rank)
        return True

    def update_weights_multi_card(self, layer, topk_ids, log2phy,
                                  topk_weights=None, hidden_states=None,
                                  mc2_mask=None):
        """Multi-card EP offload: planner decides global placement; this rank
        H2D-loads only its assigned experts and writes the placement into
        ``log2phy`` (which the MC2 dispatcher then consumes).

        MVP: full H2D of this rank's assigned experts every layer. No
        skip-if-resident, no LRC victim selection, no hot pool yet (those are
        later stages). Determinism comes from the planner (decision 8): every
        rank feeds the same all-reduced counts and gets the same placement.
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return
        # Wait for this layer's prefetch (if any) to finish H2D before reading
        # the device slots — mirror the single-card update_weights stream-join
        # (graphable: stream wait_event, not host sync). Without it the reactive
        # GMM could read a slot before the prefetch's load_stream H2D lands.
        with self._prefetch_state_lock:
            npu_event = self._prefetch_layer_npu_event.pop(layer_idx, None)
        if npu_event is not None:
            torch_npu.npu.current_stream().wait_event(npu_event)
        # num_device_experts is the TOTAL slot count; physical_range (the global
        # physical-id space MC2 routes over) == total, and each rank holds
        # num_device_experts//ep_size slots (== planner capacity per rank).
        physical_range = self.num_device_experts
        num_tokens = topk_ids.size(0)

        # NOTE: the decode profile dummy must use the real dynamic placement
        # (NOT a spread shortcut) — spread maps dummy topk all to rank0 and
        # deadlocks MC2. Debug observability + the overflow-spread fallback now
        # live in _update_weights_multi_card (graph-safe: they read the pinned
        # CPU topk_ids_h, not the live NPU tensor).

        # PREFILL regime: the MC2 dispatch kernel caps at 512 tokens, so prefill
        # uses AllGather + a per-rank EP shard loaded into the prefill pool
        # (selected by select_moe_comm_method -> ALLGATHER for multi-card large
        # batches). Drive off the comm TYPE (MC2=decode, else prefill) — the
        # single source of truth — so this stays in lockstep with apply().
        if self._mc_handle_prefill_regime(layer_idx):
            return
        # ---- DECODE (MC2) branch: graph-aware (mirror single-card) ----
        # Dynamic placement varies per step (router-driven) and is incompatible
        # with cudagraph's fixed op sequence. Mirror single-card update_weights:
        # D2H router outputs into pinned CPU buffers, run planning+H2D as a host
        # callback (_launch_host_func, re-executed every replay with the current
        # topk) or inline (eager), H2D log2phy back. The host callback's
        # load_stream.synchronize() gates the compute stream until H2D is done.
        # The cross-rank expert-count all_reduce uses gloo cpu_group
        # (get_ep_group().cpu_group) — stream-independent, so it runs inside the
        # host callback. HCCL all_reduce CANNOT be a captured graph op, and the
        # planner needs the counts on host anyway.
        per_rank_slots = self.num_device_experts // self.ep_size
        topk_ids_h = self.topk_ids_h[:num_tokens]
        log2phy_h = self.log2phy_h
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)
        # Mirror the per-rank active-token mask to pinned CPU on the same
        # stream as topk_ids_h, so the host callback (graph replay) reads it
        # after the copy lands — same ordering contract as topk_ids_h. None
        # means all-active (e.g. non-uniform global_bs path): no filtering,
        # fully backward compatible.
        if mc2_mask is not None:
            mc2_mask_h = self.mc2_mask_h[:num_tokens]
            # Cast bool->int32 on the NPU first: a direct bool D2H on the
            # captured stream forces a sync ("stream is captured", rtMemcpy
            # 107027) because Ascend has no async bool memcpy path. int32 D2H
            # is the same async path topk_ids_h.copy_ already uses, so it
            # records cleanly into the graph.
            mc2_mask_h.copy_(mc2_mask.to(torch.int32),
                             non_blocking=_EXTRA_CTX.capturing)
        else:
            mc2_mask_h = None
        current_compute_stream = torch_npu.npu.current_stream()
        subscribed = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed.add(current_compute_stream)
        args = (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots, False,
                mc2_mask_h)
        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream, self._update_weights_multi_card, args)
        else:
            self._update_weights_multi_card(args)
        # Copy the (host-func-mutated) log2phy_h back to the NPU tensor so the
        # MC2 dispatcher reads the fresh placement.
        log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _expert_src_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's bytes as UntypedStorage for H2D **read**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own (already pinned) storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.offload_config.shard_per_rank:
            return cpu_buf[eid - self._shard_base].untyped_storage()
        return cpu_buf[eid].untyped_storage()

    def _expert_dst_storage(self, layer_idx, eid, which='w13'):
        """Return expert eid's storage for fill **write**.

        shard-per-rank: eid is GLOBAL; remap to this rank's local shard slot.
        Otherwise: the expert tensor's own storage, global eid.
        """
        cpu_buf = getattr(self, f'{which}_weights_cpu')[layer_idx]
        if self.offload_config.shard_per_rank:
            return cpu_buf[eid - self._shard_base].untyped_storage()
        return cpu_buf[eid].untyped_storage()

    def _copy_quant_attrs_into_slot(self, layer, layer_idx, eid, slot):
        """Copy one expert's scale/offset/scale_bias CPU buffers into a device slot.

        Shared by the single-card and multi-card decode H2D load loops (both
        needed the same 3-attr copy). Iterates the quant buffer dicts that
        exist for this model — W8A8 has scale+offset, W4A8_DYNAMIC adds
        scale_bias, W4A8_MXFP has none — and silently skips any that are
        absent for this layer/expert, so callers don't branch on quant type.
        """
        for buffer_dict in (self.scale_cpu_buffers,
                            self.offset_cpu_buffers,
                            self.scale_bias_cpu_buffers):
            for attr_name, buffers in buffer_dict.items():
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                dev_tensor = getattr(layer, attr_name, None)
                if dev_tensor is None:
                    continue
                src = buffers[layer_idx][eid]
                dev_tensor.data[slot].copy_(
                    src.reshape(dev_tensor.data[slot].shape), non_blocking=True)

    def _update_weights_multi_card(self, args):
        """Host callback (graph replay) / inline (eager) for multi-card DECODE
        placement + H2D. Reads the pinned CPU topk_ids_h, does CPU bincount +
        gloo all_reduce (cpu_group) for global expert counts, plans the
        load-balanced placement, H2D-loads misses on load_stream (synced to gate
        the compute stream), and writes placement.log2phy into the pinned
        log2phy_h (the wrapper H2D-copies it back to the NPU tensor).
        """
        (topk_ids_h, log2phy_h, layer, layer_idx, per_rank_slots, is_prefetch,
         mc2_mask_h) = args
        from vllm_ascend.expert_offload.multi_card_planner import plan_placement
        from vllm.distributed.parallel_state import get_ep_group
        cpu_group = get_ep_group().cpu_group if self.ep_size > 1 else None

        # Drop pad-token rows (mc2_mask==0) BEFORE any counting / placing / LRU.
        # Under single-batch TP the ranks past the real-token count hold PAD
        # tokens (zero hidden) whose topk is garbage; counting them inflates
        # global_counts (-> wrong placement + wasted H2D), corrupts the LRU
        # freq, and distorts hit/miss stats. An all-pad rank contributes an
        # empty [0, topk] view -> zero local counts; the all_reduce still
        # carries the real ranks' counts, and the global placement still
        # assigns that rank the real experts MC2 dispatches to it. mc2_mask_h
        # None -> all-active (backward compatible).
        if mc2_mask_h is not None:
            topk_for_count = topk_ids_h[mc2_mask_h.bool()]
        else:
            topk_for_count = topk_ids_h

        if self._debug:
            self._log_mc_router_observation(layer_idx, topk_for_count)

        global_counts, cache_on, hotness, prev_log2phy = \
            self._gather_global_counts_and_hotness(layer_idx, topk_for_count,
                                                    cpu_group)

        # Plan placement: stable slot (prev_log2phy) + hotness (new order).
        # shard-per-rank: force EP ownership (expert → e//shard) so a rank only
        # places experts it holds in its shard CPU buffer.
        force_shard = getattr(self, '_shard_size', None) if self.offload_config.shard_per_rank else None
        placement = plan_placement(global_counts, self.ep_size, per_rank_slots,
                                   prev_log2phy, hotness, force_shard=force_shard)
        if cache_on:
            self._mc_prev_log2phy[layer_idx] = placement.log2phy.clone()
        if self._debug:
            logger.info(
                "[MC_OBS] rank=%s L=%s DECODE placement per_rank_experts=%s "
                "per_rank_load=%s",
                self.ep_rank, layer_idx, placement.per_rank_experts,
                placement.per_rank_load)

        # Capacity overflow: spread log2phy so MC2 sees only valid physical ids
        # (avoids 561002). Correctness degraded for this layer but run survives.
        if placement.unassigned:
            spread = torch.arange(self.num_total_experts, dtype=torch.int32) \
                % self.num_device_experts
            log2phy_h.copy_(spread)
            logger.warning(
                "[multi_card_offload] rank=%s L=%s overflow(%d unassigned) "
                "spread log2phy (physical_range=%s) — routing degraded",
                self.ep_rank, layer_idx, len(placement.unassigned),
                self.num_device_experts)
            return

        my_experts = placement.per_rank_experts[self.ep_rank]
        # active_set = this step's token topk (the NEEDED experts). Only these
        # count in the hit/miss metric — retained-but-unneeded experts stay
        # cached but aren't counted as hits — matching single-card's
        # needed-based rate so multi vs single hit rates are comparable now
        # that placement retains a persistent hot set across steps.
        active_set = (set(global_counts.nonzero(as_tuple=True)[0].tolist())
                      if global_counts is not None else None)
        resident_map, hits, misses = self._compute_resident_hits(
            layer_idx, my_experts, cache_on, active_set)
        if misses:
            self._h2d_load_mc_misses(layer, layer_idx, misses, resident_map)
        if self._debug:
            self._log_mc_decode_cache(layer_idx, my_experts, hits, misses,
                                      resident_map, is_prefetch)
        # Write the FULL global placement into the pinned log2phy_h; the wrapper
        # H2D-copies it back to the NPU log2phy. Must be the FULL placement (not
        # just this rank) so MC2 routes tokens cross-rank correctly — writing only
        # my_experts would leave remote experts at -1 -> clamp 0 -> zero cross-
        # rank traffic -> MC2 uniform-mode dispatch deadlocks.
        log2phy_h.copy_(placement.log2phy)

    def _log_mc_router_observation(self, layer_idx, topk_ids_h):
        if not self._debug:
            return
        num_tokens = topk_ids_h.size(0)
        topk = topk_ids_h.size(1) if topk_ids_h.dim() > 1 else 1
        uniq = sorted({int(e) for e in topk_ids_h.reshape(-1).tolist()})
        if num_tokens <= 8:
            logger.info(
                "[MC_OBS] rank=%s L=%s router: tokens=%s topk=%s "
                "uniq_experts(%d)=%s topk_ids=%s",
                self.ep_rank, layer_idx, num_tokens, topk, len(uniq), uniq,
                topk_ids_h.tolist())
        else:
            from collections import Counter
            cnt = Counter(int(e) for e in topk_ids_h.reshape(-1).tolist())
            top = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[:8]
            logger.info(
                "[MC_OBS] rank=%s L=%s router: tokens=%s topk=%s "
                "uniq_experts=%d/%d top8(expert:count)=%s uniq_list=%s",
                self.ep_rank, layer_idx, num_tokens, topk, len(cnt),
                self.num_total_experts, top, uniq)

    def _gather_global_counts_and_hotness(self, layer_idx, topk_ids_h, cpu_group):
        """CPU bincount + gloo all_reduce -> global expert counts, then update
        the LRC hotness policy (the same one single-card uses: recent freq +
        EMA + age) from the GLOBAL active set and return its per-expert
        hotness. global_counts is all-reduced EVERY step (identical across
        ranks after the mc2_mask filter), so the LRC state + hotness are too
        -> placement/eviction stay deterministic. gloo runs on cpu_group
        (stream/graph-independent, safe inside the host callback)."""
        import numpy as np
        from vllm_ascend.expert_offload.multi_card_planner import (
            local_expert_counts_cpu, gather_global_counts_cpu)
        local_counts = local_expert_counts_cpu(topk_ids_h, self.num_total_experts)
        global_counts = gather_global_counts_cpu(local_counts, cpu_group)
        cache_on = self.offload_config.cache_policy_enabled
        if not cache_on:
            return global_counts, cache_on, None, None
        # Build the per-layer LRC policy lazily. Fed the GLOBAL active set
        # (== the token's topk for single-batch decode), it evolves
        # identically to single-card's policy -> same hotness -> the
        # multi-card placement/eviction decisions match single-card quality.
        # Replaces the old crude local-freq + 32-step all_reduce hotness
        # (stale, no EMA/age/router).
        if self._mc_lrc is None:
            from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy
            # num_layers=1 then grow via add_layer() (LRC requires >= 1).
            self._mc_lrc = LRCExpertCachePolicy(
                num_layers=1, num_experts=self.num_total_experts,
                cache_size=self.num_device_experts, topk=self.topk)
        while len(self._mc_lrc.layer_states) <= layer_idx:
            self._mc_lrc.add_layer()
        active = global_counts.nonzero(as_tuple=True)[0].tolist()
        if active:
            self._mc_lrc.observe(layer_idx, [tuple(int(e) for e in active)])
        hotness = self._mc_lrc.hotness_array(layer_idx)
        return (global_counts, cache_on, hotness,
                self._mc_prev_log2phy.get(layer_idx))

    def _compute_resident_hits(self, layer_idx, my_experts, cache_on,
                               active_set=None):
        """Split this rank's ACTIVE placed experts into cache hits (expert
        already resident in its assigned slot) vs misses (need H2D). Returns
        (resident_map, hits, misses).

        Only ACTIVE experts (in ``active_set`` = this step's token topk) are
        counted: retained-but-not-needed experts stay cached but don't count
        as hits, mirroring single-card's (needed ∩ on_device)/needed so the
        hit rate is comparable across configs. ``active_set=None`` counts all
        (backward-compatible fallback)."""
        if not cache_on:
            # No cache: every (active) expert is a miss (full H2D every step).
            misses = [(s, int(e)) for s, e in enumerate(my_experts)
                      if e >= 0 and (active_set is None or int(e) in active_set)]
            return {}, [], misses
        resident_map = self._mc_resident.setdefault(layer_idx, {})
        hits, misses = [], []
        for slot, eid in enumerate(my_experts):
            if eid < 0:
                continue
            eid = int(eid)
            if active_set is not None and eid not in active_set:
                continue  # retained but not needed this step: cached, not counted
            (hits if resident_map.get(slot) == eid else misses).append((slot, eid))
        return resident_map, hits, misses

    def _load_expert_weights_into_slot(self, layer, layer_idx, eid, slot):
        """H2D-copy one expert's w13/w2 + quant scale/offset/scale_bias from the
        pinned CPU buffer into the device slot, on the caller's load_stream."""
        layer.w13_weight.data.untyped_storage()[
            slot * self.w13_expert_size_bytes:(slot + 1) * self.w13_expert_size_bytes
        ].copy_(self._expert_src_storage(layer_idx, eid, 'w13'), non_blocking=True)
        layer.w2_weight.data.untyped_storage()[
            slot * self.w2_expert_size_bytes:(slot + 1) * self.w2_expert_size_bytes
        ].copy_(self._expert_src_storage(layer_idx, eid, 'w2'), non_blocking=True)
        self._copy_quant_attrs_into_slot(layer, layer_idx, eid, slot)

    def _h2d_load_mc_misses(self, layer, layer_idx, misses, resident_map):
        """H2D-load missed experts (w13/w2 + quant attrs) into their slots on
        load_stream, then synchronize to gate the compute stream."""
        with torch_npu.npu.stream(self.load_stream):
            for slot, eid in misses:
                self._load_expert_weights_into_slot(layer, layer_idx, eid, slot)
                resident_map[slot] = eid
            self.load_stream.synchronize()

    def _log_mc_decode_cache(self, layer_idx, my_experts, hits, misses,
                             resident_map, is_prefetch=False):
        if not self._debug:
            return
        logger.info(
            "[MC_OBS] rank=%s L=%s DECODE cache: placed=%d hit=%d miss=%d | "
            "resident_experts_on_rank%d=%s | miss_load_from_cpu{slot->expert}=%s "
            "| prefetch=%s",
            self.ep_rank, layer_idx, len(my_experts), len(hits), len(misses),
            self.ep_rank, sorted(resident_map.values()),
            {s: e for s, e in misses}, is_prefetch)

    def _update_weights(self, args):
        (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
            is_prefetch,
        ) = args
        with torch_npu.npu.stream(self.load_stream):
            # Hotness observation only on the reactive (non-prefetch) H2D path
            # with LRC policy enabled.
            if not is_prefetch and self.cache_policy is not None:
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
            else:
                needed = set(topk_ids_h.reshape(-1).tolist())

            l2p_list = log2phy_np.tolist()
            slot_owner = {s: e for e, s in enumerate(l2p_list) if s >= 0}
            on_device = set(slot_owner.values())

            if is_prefetch:
                # Prefetch: only load the truly-missing top-N predicted experts.
                ordered_misses = [e for e in topk_ids_h.reshape(-1).tolist() if e not in on_device]
                need_to_load = set(ordered_misses[:self.prefetch_topk])
            else:
                need_to_load = needed - on_device
            already_there = needed & on_device              # for cache_stats / debug

            if self.cache_policy is not None:
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self._debug:
                flag = '[PREFETCH-W]' if is_prefetch else '[UPDATE-W]'
                already_there_layer = set(topk_ids_h[0].tolist()) & on_device
                logger.info("%s l=%d expert_hit=%s expert_miss=%s hit_rate=%.2f layer_expert_hit=%s needed=%s topk_ids_h=%s" ,
                            flag,layer_idx, sorted(already_there),
                            sorted(need_to_load), len(already_there_layer) / topk_ids_h.shape[1],
                            already_there_layer, needed, topk_ids_h)
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    logger.info("%s l=%d SHORTFALL: need %d load but only %d slots, "
                                "to_load=%s",
                                flag,layer_idx, len(need_to_load), len(reusable_slots),
                                sorted(need_to_load)[:20])

            dev = layer.w13_weight.device
            n_copies = 0
            for eid in need_to_load:
                if self.cache_policy is not None:
                    victim = self.cache_policy.choose_victim(
                        layer_idx,
                        slot_owner,
                        protected=needed,
                    )
                    slot = int(log2phy_np[victim]) if victim is not None else -1
                elif reusable_slots:
                    slot = reusable_slots.pop()
                    victim = slot_owner[slot]
                else:
                    slot = -1
                    victim = None

                if slot < 0:
                    if self._debug:
                        logger.info(
                            "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, "
                            "missed=%s",
                            layer_idx, len(need_to_load) - n_copies,
                            sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage
                
                self._load_expert_weights_into_slot(layer, layer_idx, eid, slot)
                # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
                if hasattr(layer, 'w13_weight_scale_fp32'):
                    layer.w13_weight_scale_fp32[slot].copy_(
                        layer.w13_weight_scale.data[slot].to(torch.float32))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                on_device.discard(victim)
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                on_device.add(eid)
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            self.load_stream.synchronize()

    def predict_next_layer_experts_npu(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Predict which experts layer layer_idx+1 will need, on NPU.

        Runs entirely on the NPU (softmax + topk) so it can be captured in
        a CUDA/NPU graph. The returned tensors live on NPU.

        Args:
            layer_idx: Current layer index.
            hidden_states: [num_tokens, hidden_dim] NPU tensor.

        Returns:
            (topk_weights, topk_ids) for the first token only, both
            [1, topk] NPU tensors, or None if prediction is not possible.
        """
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers):
            return None  # last layer — nothing to prefetch

        if next_idx >= len(self._gate_weights_npu):
            return None
        gate_w = self._gate_weights_npu[next_idx]
        if gate_w is None:
            return None

        # Predict from the first token only — one representative token's
        # experts is enough for prefetch; others are handled reactively by
        # update_weights(). prefetch_topk (= min(topk, expert_prefetch_num))
        # caps how many experts are prefetched per layer — single-card uses 1
        # (cheap, conservative); raise expert_prefetch_num for more coverage.
        # On-device prediction: [1, hidden_dim] x [n_experts, hidden_dim]^T
        router_logits = F.linear(hidden_states[:1].float(), gate_w)
        probs = router_logits.softmax(dim=-1)
        topk_weights, topk_ids = probs.topk(self.prefetch_topk, dim=-1)
        return topk_weights, topk_ids

    def trigger_next_layer_prefetch(self, layer,
                        hidden_states: torch.Tensor | None = None) -> int:
        """Trigger next-layer expert prefetch after the GMM kernel submits.

        Graph-compatible (mirrors the reactive update_weights path — NO stream
        switch, which would break NPU capture_end): record ready_to_load_event
        on the compute stream, then _launch_host_func registers the prefetch as
        a host callback (re-run every replay). The callback runs the planner+H2D
        inner (load_stream inside gives overlap with subsequent compute) and
        records load_done_event on load_stream for the next layer's reactive to
        stream-join. Eager mode keeps the prefetch-stream overlap path.
        """
        if not self.offload_config.expert_prefetch_enabled:
            return
        if self._skip_prefill:
            return
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return

        staged = self._stage_predicted_topk(layer_idx, hidden_states)
        if staged is None:
            return
        topk_ids_h, topk_weights_h, log2phy_h, log2phy_np, next_layer, next_idx = staged

        # Graph-compatible dispatch (NO stream switch — see docstring).
        current_compute_stream = torch_npu.npu.current_stream()
        ready_to_load_event = torch_npu.npu.Event()
        current_compute_stream.record_event(ready_to_load_event)
        subscribed = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed.add(current_compute_stream)

        prefetch_fn, prefetch_args = self._build_prefetch_call(
            topk_ids_h, topk_weights_h, log2phy_h, log2phy_np, next_layer, next_idx)
        nxt = next_idx

        def _prefetch_host_cb(_args):
            prefetch_fn(_args)
            # prefetch_fn ends with load_stream.synchronize() -> H2D complete;
            # record on load_stream so the next reactive (compute stream) can
            # stream-join it (capture invariant — do not remove).
            ev = torch_npu.npu.Event()
            self.load_stream.record_event(ev)
            with self._prefetch_state_lock:
                self._prefetch_layer_npu_event[nxt] = ev

        if _EXTRA_CTX.capturing:
            torch_npu.npu._launch_host_func(
                current_compute_stream, _prefetch_host_cb, prefetch_args)
        else:
            with torch_npu.npu.stream(self._prefetch_stream):
                self._prefetch_stream.wait_event(ready_to_load_event)
                _prefetch_host_cb(prefetch_args)

        next_layer.log2phy.copy_(log2phy_h, non_blocking=_EXTRA_CTX.capturing)

    def _stage_predicted_topk(self, layer_idx, hidden_states):
        """Resolve the next layer, predict its experts on-device, and D2H-stage
        them into pinned buffers. Returns (topk_ids_h, topk_weights_h, log2phy_h,
        log2phy_np, next_layer, next_idx), or None if prefetch isn't possible
        (last layer / missing gate weights / prediction failed)."""
        next_idx = layer_idx + 1
        if next_idx >= len(self.moe_layers) - 1:
            return None
        predicted = self.predict_next_layer_experts_npu(layer_idx, hidden_states)
        if predicted is None:
            return None
        topk_weights, topk_ids = predicted
        next_layer = self.moe_layers[next_idx]
        num_tokens = topk_ids.size(0)
        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_ids_h.copy_(topk_ids.to(torch.int32), non_blocking=_EXTRA_CTX.capturing)
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32),
                                 non_blocking=_EXTRA_CTX.capturing)
        log2phy_h = self._prefetch_log2phy_h
        log2phy_h.copy_(next_layer.log2phy, non_blocking=_EXTRA_CTX.capturing)
        return (topk_ids_h, topk_weights_h, log2phy_h, self._prefetch_log2phy_np,
                next_layer, next_idx)

    def _build_prefetch_call(self, topk_ids_h, topk_weights_h, log2phy_h,
                             log2phy_np, next_layer, next_idx):
        """Pick the single-card vs multi-card planner+H2D inner and its args."""
        self._is_prefetch = True
        if self.enable_multi_card:
            per_rank_slots = self.num_device_experts // self.ep_size
            # mc2_mask_h=None: prefetch predicts the NEXT layer whose
            # active-token mask isn't known yet, so don't filter. Prefetch
            # placement is corrected by the next layer's reactive update
            # (which does filter), and prefetch calls are excluded from
            # hit-rate stats via is_prefetch=True.
            return self._update_weights_multi_card, (
                topk_ids_h, log2phy_h, next_layer, next_idx, per_rank_slots,
                True, None)
        return self._update_weights, (
            topk_ids_h, log2phy_np, next_layer, next_idx, topk_weights_h,
            self._is_prefetch)


    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_cache_stats(
        self,
        layer_idx: int,
        hit_experts: set[int],
        miss_experts: set[int],
        needed: set[int],
        on_device: set[int],
    ):
        self.cache_calls[layer_idx] += 1
        self.cache_requests[layer_idx] += len(needed)
        self.cache_hits[layer_idx] += len(hit_experts)
        self.cache_misses[layer_idx] += len(miss_experts)
        self.last_hit_experts[layer_idx] = sorted(hit_experts)
        self.last_miss_experts[layer_idx] = sorted(miss_experts)

        interval = self.offload_config.cache_stats_log_interval
        if interval == 0 or self.cache_calls[layer_idx] % interval != 0:
            return

        requests = self.cache_requests[layer_idx]
        hit_rate = self.cache_hits[layer_idx] / requests if requests else 0.0
        policy_step = -1
        if self.cache_policy is not None:
            policy_step = self.cache_policy.layer_step(layer_idx)
        logger.info(
            "[EXPERT-OFFLOAD-CACHE] layer=%d cache_step=%d calls=%d policy_step=%d "
            "hit_rate=%.4f hits=%d misses=%d last_hit=%s last_miss=%s resident=%s",
            layer_idx,
            self.cache_calls[layer_idx],
            self.cache_calls[layer_idx],
            policy_step,
            hit_rate,
            self.cache_hits[layer_idx],
            self.cache_misses[layer_idx],
            self.last_hit_experts[layer_idx],
            self.last_miss_experts[layer_idx],
            sorted(on_device),
        )



_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
    # if no need to init offload manager:
    #     return
    global _EXPERT_OFFLOAD_MANAGER
    if _EXPERT_OFFLOAD_MANAGER is None:
        _EXPERT_OFFLOAD_MANAGER = ExpertOffloadManager(vllm_config)


def has_expert_offload_manager():
    return _EXPERT_OFFLOAD_MANAGER is not None


def get_expert_offload_manager():
    assert _EXPERT_OFFLOAD_MANAGER is not None, (
        "Expert Offload Manager is not initialized"
    )
    return _EXPERT_OFFLOAD_MANAGER
