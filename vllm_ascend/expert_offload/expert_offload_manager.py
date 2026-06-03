"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import torch
import torch_npu
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

        # CPU weight buffers (post-transpose format, matching device after
        # process_weights_after_loading):
        #   w13 per expert: [hidden_size, w13_up_dim]
        #   w2 per expert:  [intermediate_size_per_partition, hidden_size]
        self.w13_weights_cpu: list[list[torch.Tensor]] = []
        self.w2_weights_cpu: list[list[torch.Tensor]] = []

        # Registered AscendFusedMoE layers, indexed by moe_instance_id order
        self.moe_layers: list = []

        # Temporary storage for weights loaded before create_weights()
        self._pending_weights: dict = {}

        # CPU buffers for quantized model scale/offset parameters.
        # Keyed by attr_name (e.g. "w13_weight_scale", "w2_weight_offset").
        # Each value is a list of layers, each layer is a list of expert tensors.
        self.scale_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}
        self.offset_cpu_buffers: dict[str, list[list[torch.Tensor]]] = {}

        # Temporary storage for scale/offset weights loaded before
        # maybe_create_scale_buffers runs.
        self._pending_scales: dict[tuple, dict[str, torch.Tensor]] = {}

        self.num_device_layers = self.offload_config.num_device_layers
        self.num_total_experts = None  # set in create_weights
        self.cache_policy: LRCExpertCachePolicy | None = None
        self.cache_requests: list[int] = []
        self.cache_hits: list[int] = []
        self.cache_misses: list[int] = []
        self.cache_calls: list[int] = []
        self.last_hit_experts: list[list[int]] = []
        self.last_miss_experts: list[list[int]] = []
        self._debug_update_weights = self.offload_config.cache_debug_log_updates

        ExpertOffloadManager._instance = self

        self.load_stream = torch_npu.npu.Stream()

        # Prefill pool: ndl layers × all experts on NPU, shared round-robin
        self._prefill_w13: list[torch.Tensor] = []
        self._prefill_w2: list[torch.Tensor] = []
        self._prefill_w13_scale: list[torch.Tensor] = []       # W8A8
        self._prefill_w13_scale_fp32: list[torch.Tensor] = []   # W8A8
        self._prefill_w13_offset: list[torch.Tensor] = []       # W8A8
        self._prefill_w2_scale: list[torch.Tensor] = []         # W8A8
        self._prefill_w2_offset: list[torch.Tensor] = []        # W8A8
        self._prefill_log2phy: torch.Tensor = None              # identity [0..127]
        self._prefill_initialized: bool = False
        self._skip_prefill: bool = False  # set during profile runs

    # ------------------------------------------------------------------ #
    #  Lifecycle: called from NPUModelRunner during model loading         #
    # ------------------------------------------------------------------ #

    def create_weights(
        self,
        num_moe_layers: int,
        num_total_experts: int,
        w13_up_dim: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
    ):
        """Allocate CPU buffers for all MoE layers."""
        for _ in range(num_moe_layers):
            w13_list = [
                torch.empty(hidden_size, w13_up_dim, dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(num_total_experts)
            ]
            w2_list = [
                torch.empty(intermediate_size_per_partition, hidden_size,
                            dtype=params_dtype, device="cpu", pin_memory=True)
                for _ in range(num_total_experts)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        self._drain_pending_weights()
        self.process_weights_after_loading()

        self.num_total_experts = num_total_experts
        if self.offload_config.cache_policy_enabled:
            self.cache_requests = [0 for _ in range(num_moe_layers)]
            self.cache_hits = [0 for _ in range(num_moe_layers)]
            self.cache_misses = [0 for _ in range(num_moe_layers)]
            self.cache_calls = [0 for _ in range(num_moe_layers)]
            self.last_hit_experts = [[] for _ in range(num_moe_layers)]
            self.last_miss_experts = [[] for _ in range(num_moe_layers)]
            self.cache_policy = LRCExpertCachePolicy(
                num_layers=num_moe_layers,
                num_experts=num_total_experts,
                cache_size=self.num_device_experts,
                topk=self.topk,
                recent_window=self.offload_config.cache_recent_window,
                ema_beta=self.offload_config.cache_ema_beta,
                recent_weight=self.offload_config.cache_recent_weight,
                ema_weight=self.offload_config.cache_ema_weight,
                router_weight=self.offload_config.cache_router_weight,
                age_weight=self.offload_config.cache_age_weight,
            )

        # update weights related buffers
        self.topk_ids_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.topk_weights_h = torch.zeros(
            [self.offload_threshold, self.topk],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        self.log2phy_h = torch.zeros(num_total_experts, dtype=torch.int32, device='cpu', pin_memory=True)
        self.log2phy_np = self.log2phy_h.numpy()

    def process_weights_after_loading(self):
        first_w13 = self.w13_weights_cpu[0][0]
        first_w2 = self.w2_weights_cpu[0][0]
        if first_w13.dtype != torch.int8:
            return
        # for w8a8, npu weight tensor is cast to NZ format,
        # so we also store NZ format weight in weights_cpu,
        # and copy_ tensor's underlying storage instead of tensor itself
        # to avoid implicit format conversion during h2d.
        num_moe_layers = len(self.w13_weights_cpu)
        num_experts = len(self.w13_weights_cpu[0])
        self.w13_element_num = first_w13.nelement()
        self.w2_element_num = first_w2.nelement()
        for layer_id in range(num_moe_layers):
            w13 = torch.stack(self.w13_weights_cpu[layer_id]).to('npu')
            w13_nz = torch_npu.npu_format_cast(w13, ACL_FORMAT_FRACTAL_NZ)
            w13_nz_storage = w13_nz.untyped_storage()
            w2 = torch.stack(self.w2_weights_cpu[layer_id]).to('npu')
            w2_nz = torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ)
            w2_nz_storage = w2_nz.untyped_storage()
            for expert_id in range(num_experts):
                self.w13_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w13_nz_storage[expert_id * self.w13_element_num : (expert_id + 1) * self.w13_element_num]
                )
                self.w2_weights_cpu[layer_id][expert_id].untyped_storage().copy_(
                    w2_nz_storage[expert_id * self.w2_element_num : (expert_id + 1) * self.w2_element_num]
                )

    def register_moe_layer(self, layer):
        self.moe_layers.append(layer)

    def load_w13(self, layer_moe_idx: int, expert_id: int,
                 loaded_weight: torch.Tensor, shard_id: str):
        """Store w1/w3 shard to CPU buffer (with transpose to post format)."""
        if not self.w13_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})[f"w13_{shard_id}"] = \
                loaded_weight.cpu().clone()
            return
        cpu = self.w13_weights_cpu[layer_moe_idx][expert_id]
        intermed = cpu.shape[1] // 2
        w = loaded_weight.cpu()
        if shard_id == "w1":
            cpu[:, :intermed].copy_(w.t())
        elif shard_id == "w3":
            cpu[:, intermed: intermed + w.shape[0]].copy_(w.t())

    def load_w2(self, layer_moe_idx: int, expert_id: int,
                loaded_weight: torch.Tensor):
        """Store w2 weight to CPU buffer (with transpose to post format)."""
        if not self.w2_weights_cpu:
            key = (layer_moe_idx, expert_id)
            self._pending_weights.setdefault(key, {})["w2"] = \
                loaded_weight.cpu().clone()
            return
        self.w2_weights_cpu[layer_moe_idx][expert_id].copy_(loaded_weight.cpu().t())

    # ------------------------------------------------------------------ #
    #  Scale / offset helpers (quantized models only)                     #
    # ------------------------------------------------------------------ #

    def _add_pending_scale(self, layer_moe_idx: int, expert_id: int,
                           attr_name: str, shard_id: str,
                           loaded_weight: torch.Tensor):
        """Store a scale/offset weight that arrived before CPU buffers exist."""
        key = (layer_moe_idx, expert_id)
        sub_key = f"{attr_name}_{shard_id}"
        self._pending_scales.setdefault(key, {})[sub_key] = \
            loaded_weight.cpu().clone()

    def maybe_create_scale_buffers(self, layer, layer_moe_idx: int):
        """Inspect layer for scale/offset params and allocate CPU buffers.

        Called from _register_offload_layers AFTER process_weights_after_loading
        has transformed device tensor shapes, so we detect the final per-expert
        shape from the device tensor.
        """
        attr_names = [
            ("scale_cpu_buffers", "w13_weight_scale"),
            ("scale_cpu_buffers", "w2_weight_scale"),
            ("offset_cpu_buffers", "w13_weight_offset"),
            ("offset_cpu_buffers", "w2_weight_offset"),
        ]
        created_any = False
        global_num_experts = len(self.w13_weights_cpu[layer_moe_idx])

        for buffer_dict_name, attr_name in attr_names:
            if not hasattr(layer, attr_name):
                continue
            dev_tensor = getattr(layer, attr_name)
            per_expert_shape = dev_tensor.shape[1:]
            dtype = dev_tensor.dtype
            buffer_dict: dict = getattr(self, buffer_dict_name)
            if attr_name not in buffer_dict:
                buffer_dict[attr_name] = []
            buffers = buffer_dict[attr_name]
            while len(buffers) <= layer_moe_idx:
                buffers.append([])
            for _ in range(global_num_experts):
                buffers[layer_moe_idx].append(
                    torch.empty(per_expert_shape, dtype=dtype, device="cpu", pin_memory=True))
            created_any = True

        if created_any:
            self._drain_pending_scales()

    def _drain_pending_scales(self):
        """Drain _pending_scales into CPU buffers, assembling w1/w3 shards.

        Only removes entries that were successfully copied to CPU buffers.
        Entries for layers whose buffers haven't been created yet are left
        in _pending_scales for the next call.
        """
        if not self._pending_scales:
            return
        processed_keys: list[tuple] = []
        for (layer_idx, eid), items in self._pending_scales.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            # Group shards by attr_name
            attr_shards: dict[str, dict[str, torch.Tensor]] = {}
            for sub_key, w in items.items():
                # sub_key format: "{attr_name}_{shard_id}"
                # attr_name may contain underscores (e.g. "w13_weight_scale")
                # shard_id is always "w1", "w2", or "w3" (no underscores)
                parts = sub_key.rsplit("_", 1)
                if len(parts) == 2 and parts[1] in ("w1", "w2", "w3"):
                    attr_name, shard = parts[0], parts[1]
                else:
                    attr_name, shard = parts[0], parts[1] if len(parts) > 1 else ""
                attr_shards.setdefault(attr_name, {})[shard] = w

            copied_any = False
            for attr_name, shards in attr_shards.items():
                target_dict = None
                if "scale" in attr_name:
                    target_dict = self.scale_cpu_buffers
                elif "offset" in attr_name:
                    target_dict = self.offset_cpu_buffers
                if target_dict is None or attr_name not in target_dict:
                    continue
                buffers = target_dict[attr_name]
                if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                    continue
                target = buffers[layer_idx][eid]

                if attr_name.startswith("w13_"):
                    # w13 scale/offset: assemble w1 + w3 shards along dim 0
                    if "w1" in shards and "w3" in shards:
                        assembled = torch.cat(
                            [shards["w1"].cpu(), shards["w3"].cpu()], dim=0)
                        # squeeze trailing dim-1 if present (W8A8_DYNAMIC)
                        assembled = assembled.reshape(target.shape)
                        target.copy_(assembled)
                        copied_any = True
                elif attr_name.startswith("w2_"):
                    # w2 scale/offset: single shard
                    if "w2" in shards:
                        w_cpu = shards["w2"]
                        if w_cpu.device.type != "cpu":
                            w_cpu = w_cpu.cpu()
                        w_cpu = w_cpu.reshape(target.shape)
                        target.copy_(w_cpu)
                        copied_any = True
            if copied_any:
                processed_keys.append((layer_idx, eid))
        # Only remove successfully processed entries
        for key in processed_keys:
            del self._pending_scales[key]

    def init_device_experts(self):
        """Refresh derived fp32 scale after weight loading.

        Device experts are already loaded by the weight loader and
        process_weights_after_loading. Only refresh w13_weight_scale_fp32.
        """
        for i, layer in enumerate(self.moe_layers):
            ndev = min(self.num_device_experts, layer.w13_weight.shape[0])
            if hasattr(layer, 'w13_weight_scale_fp32'):
                for j in range(ndev):
                    layer.w13_weight_scale_fp32[j].copy_(
                        layer.w13_weight_scale.data[j].to(torch.float32))

    def create_prefill_pool(self):
        """Allocate prefill pool tensors on NPU with full expert count.

        Called from _register_offload_layers after decode buffers are set up.
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
        ntotal = self.num_total_experts

        for _ in range(ndl):
            # w13: [ntotal, hidden_size, w13_up_dim] — match decode layer shape
            w13_shape = (ntotal,) + tuple(pool_layer.w13_weight.shape[1:])
            self._prefill_w13.append(
                torch.empty(w13_shape, dtype=dt, device=dev))

            # w2: [ntotal, hidden_size, intermediate_size_per_partition]
            w2_shape = (ntotal,) + tuple(pool_layer.w2_weight.shape[1:])
            self._prefill_w2.append(
                torch.empty(w2_shape, dtype=dt, device=dev))

            # W8A8 scale/offset (optional)
            if hasattr(pool_layer, 'w13_weight_scale'):
                s13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale.shape[1:])
                self._prefill_w13_scale.append(
                    torch.empty(s13_shape, dtype=pool_layer.w13_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w13_weight_scale_fp32'):
                fp32_13_shape = (ntotal,) + tuple(pool_layer.w13_weight_scale_fp32.shape[1:])
                self._prefill_w13_scale_fp32.append(
                    torch.empty(fp32_13_shape, dtype=torch.float32, device=dev))
            if hasattr(pool_layer, 'w13_weight_offset'):
                o13_shape = (ntotal,) + tuple(pool_layer.w13_weight_offset.shape[1:])
                self._prefill_w13_offset.append(
                    torch.empty(o13_shape, dtype=pool_layer.w13_weight_offset.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_scale'):
                s2_shape = (ntotal,) + tuple(pool_layer.w2_weight_scale.shape[1:])
                self._prefill_w2_scale.append(
                    torch.empty(s2_shape, dtype=pool_layer.w2_weight_scale.dtype, device=dev))
            if hasattr(pool_layer, 'w2_weight_offset'):
                o2_shape = (ntotal,) + tuple(pool_layer.w2_weight_offset.shape[1:])
                self._prefill_w2_offset.append(
                    torch.empty(o2_shape, dtype=pool_layer.w2_weight_offset.dtype, device=dev))

        # Cast prefill pool weight tensors to NZ format (W8A8 kernel requires it).
        # Must happen BEFORE loading data — same order as decode path:
        # create → NZ-cast → copy_(cpu → npu)
        if dt == torch.int8:
            from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ
            for i in range(ndl):
                self._prefill_w13[i] = torch_npu.npu_format_cast(
                    self._prefill_w13[i], ACL_FORMAT_FRACTAL_NZ)
                self._prefill_w2[i] = torch_npu.npu_format_cast(
                    self._prefill_w2[i], ACL_FORMAT_FRACTAL_NZ)

        # Prefill log2phy: identity — all experts mapped to their slots
        self._prefill_log2phy = torch.arange(ntotal, dtype=torch.int32, device=dev)

        # Pre-initialize all pool slots with layer 0 weights so that
        # profile_run / _dummy_run (which may use prefill path) has
        # valid data.  Subsequent _prefill_load_layer calls will
        # overwrite with the correct per-layer weights.
        self._init_prefill_pool_data(dev, ntotal, ndl)
        self._prefill_initialized = True
        logger.warning("[PREFILL_POOL] allocated %d layers × %d experts, "
                       "w13[0].shape=%s w2[0].shape=%s",
                       ndl, ntotal,
                       tuple(self._prefill_w13[0].shape),
                       tuple(self._prefill_w2[0].shape))

    def _init_prefill_pool_data(self, dev, ntotal: int, ndl: int):
        """Load layer 0 weights into all prefill pool slots.

        Prefill pool tensors are already NZ-cast at this point (done in
        create_prefill_pool). Use simple per-expert copy_() — same pattern
        as the decode path's _update_weights.
        """
        has_scales = bool(self._prefill_w13_scale)
        has_offsets = bool(self._prefill_w13_offset)

        for slot in range(ndl):
            for eid in range(min(ntotal, len(self.w13_weights_cpu[0]))):
                self._prefill_w13[slot].untyped_storage()[eid * self.w13_element_num : (eid + 1) * self.w13_element_num].copy_(
                    self.w13_weights_cpu[0][eid].untyped_storage()
                )
                self._prefill_w2[slot].untyped_storage()[eid * self.w2_element_num : (eid + 1) * self.w2_element_num].copy_(
                    self.w2_weights_cpu[0][eid].untyped_storage()
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
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[scale_name][0][eid])
            if has_offsets:
                for offset_name, prefill_list, cpu_buffers in [
                    ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                    ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
                ]:
                    if (offset_name in cpu_buffers and
                            0 < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][0]))):
                            prefill_list[slot][eid].copy_(
                                cpu_buffers[offset_name][0][eid])
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

        import logging
        _dbg = logging.getLogger(__name__)
        _dbg.warning("[PREFILL_LOAD] layer=%d pool_slot=%d ntotal=%d is_w8a8=%s",
                     layer_idx, pool_slot, ntotal, is_w8a8)

        from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

        with torch_npu.npu.stream(self.load_stream):
            for eid in range(ntotal):
                self._prefill_w13[pool_slot].untyped_storage()[eid * self.w13_element_num : (eid + 1) * self.w13_element_num].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                self._prefill_w2[pool_slot].untyped_storage()[eid * self.w2_element_num : (eid + 1) * self.w2_element_num].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
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
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[scale_name][layer_idx][eid])
            for offset_name, prefill_list, cpu_buffers in [
                ("w13_weight_offset", self._prefill_w13_offset, self.offset_cpu_buffers),
                ("w2_weight_offset", self._prefill_w2_offset, self.offset_cpu_buffers),
            ]:
                if pool_slot < len(prefill_list):
                    if (offset_name in cpu_buffers and
                            layer_idx < len(cpu_buffers[offset_name])):
                        for eid in range(min(ntotal, len(cpu_buffers[offset_name][layer_idx]))):
                            prefill_list[pool_slot][eid].copy_(
                                cpu_buffers[offset_name][layer_idx][eid])

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
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor,
                        topk_weights: torch.Tensor | None = None) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Routes to prefill pool (full-overwrite) when num_tokens exceeds
        offload_threshold, otherwise uses per-expert paging (decode path).

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.

        Returns: number of CPU→NPU copies performed (decode path),
                 0 for prefill path (full-overwrite via pool).
        """
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

        topk_ids_h = self.topk_ids_h[:num_tokens]
        topk_weights_h = None
        if (self.cache_policy is not None and topk_weights is not None and not _EXTRA_CTX.capturing
                and self.offload_config.cache_router_weight != 0):
            topk_weights_h = self.topk_weights_h[:num_tokens]
            topk_weights_h.copy_(topk_weights.to(dtype=torch.float32), non_blocking=False)
        log2phy_h = self.log2phy_h
        log2phy_np = self.log2phy_np
        topk_ids_h.copy_(topk_ids, non_blocking=_EXTRA_CTX.capturing)
        log2phy_h.copy_(log2phy, non_blocking=_EXTRA_CTX.capturing)

        current_compute_stream = torch_npu.npu.current_stream()
        subscribed_compute_streams = get_subscribed_compute_streams()
        if current_compute_stream not in subscribed_compute_streams:
            torch_npu.npu._subscribe_report(current_compute_stream)
            subscribed_compute_streams.add(current_compute_stream)

        args = (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
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
    
    def _update_weights(self, args):
        (
            topk_ids_h,
            log2phy_np,
            layer,
            layer_idx,
            topk_weights_h,
        ) = args
        with torch_npu.npu.stream(self.load_stream):
            if self.cache_policy is not None:
                router_scores = topk_weights_h.tolist() if topk_weights_h is not None else None
                needed = self.cache_policy.observe(
                    layer_idx,
                    topk_ids_h.tolist(),
                    router_scores=router_scores,
                )
            else:
                needed = set(topk_ids_h.unique().tolist())

            # Build reverse map: slot → expert_id currently occupying it
            slot_owner: dict[int, int] = {}
            for eid, slot in enumerate(log2phy_np):
                if slot >= 0:
                    slot_owner[slot] = eid

            on_device = set(slot_owner.values())
            already_there = needed & on_device           # no-op
            need_to_load = needed - already_there          # CPU→NPU copy
            if self.cache_policy is not None:
                self._record_cache_stats(layer_idx, already_there, need_to_load, needed, on_device)
            reusable_slots = [s for s, e in slot_owner.items()
                            if e not in needed]          # slots to recycle

            if self.cache_policy is not None and self._debug_update_weights:
                import logging
                _dbg = logging.getLogger(__name__)
                _dbg.warning(
                    "[UPDATE-W] l=%d call=%d topk_shape=%s |needed|=%d |on_dev|=%d "
                    "|to_load|=%d reusable=%d needed=%s",
                    layer_idx, self.cache_calls[layer_idx], tuple(topk_ids_h.shape),
                    len(needed), len(on_device),
                    len(need_to_load), len(reusable_slots),
                    sorted(needed)[:30],
                )
                _dbg.warning("[UPDATE-W] l=%d cache_hit=%s cache_miss=%s",
                             layer_idx, sorted(already_there), sorted(need_to_load))
                if need_to_load and len(need_to_load) > len(reusable_slots):
                    _dbg.warning("[UPDATE-W] l=%d SHORTFALL: need %d load but only %d slots, to_load=%s",
                                layer_idx, len(need_to_load), len(reusable_slots),
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
                    import logging
                    logging.getLogger(__name__).warning(
                        "[UPDATE-W] l=%d NO SLOTS: %d experts could not be loaded, missed=%s",
                        layer_idx, len(need_to_load) - n_copies,
                        sorted(list(need_to_load))[n_copies:][:20])
                    break  # no free slots — should not happen in normal usage
                # Copy weights from CPU to NPU
                layer.w13_weight.data.untyped_storage()[slot * self.w13_element_num : (slot + 1) * self.w13_element_num].copy_(
                    self.w13_weights_cpu[layer_idx][eid].untyped_storage()
                )
                layer.w2_weight.data.untyped_storage()[slot * self.w2_element_num : (slot + 1) * self.w2_element_num].copy_(
                    self.w2_weights_cpu[layer_idx][eid].untyped_storage()
                )
                # Copy scales/offsets from CPU to NPU
                for attr_name, buffers in self.scale_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                for attr_name, buffers in self.offset_cpu_buffers.items():
                    if layer_idx >= len(buffers) or eid >= len(buffers[layer_idx]):
                        continue
                    dev_tensor = getattr(layer, attr_name, None)
                    if dev_tensor is None:
                        continue
                    dev_tensor.data[slot].copy_(buffers[layer_idx][eid])
                # Refresh derived fp32 scale if present (W8A8_DYNAMIC)
                if hasattr(layer, 'w13_weight_scale_fp32'):
                    layer.w13_weight_scale_fp32[slot].copy_(
                        layer.w13_weight_scale.data[slot].to(torch.float32))
                # Update mapping
                if victim is None:
                    victim = slot_owner[slot]
                log2phy_np[victim] = -1             # evict old occupant
                log2phy_np[eid] = slot               # assign slot to new expert
                slot_owner[slot] = eid
                if slot in reusable_slots:
                    reusable_slots.remove(slot)
                n_copies += 1

            self.load_stream.synchronize()

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

    def _drain_pending_weights(self):
        if not self._pending_weights:
            return
        for (layer_idx, eid), weights in self._pending_weights.items():
            if layer_idx >= len(self.w13_weights_cpu):
                continue
            if eid >= len(self.w13_weights_cpu[layer_idx]):
                continue
            cpu_w13 = self.w13_weights_cpu[layer_idx][eid]
            intermed = cpu_w13.shape[1] // 2
            for key, w in weights.items():
                w_cpu = w if w.device.type == "cpu" else w.cpu()
                if key.startswith("w13_"):
                    shard = key.split("_")[1]
                    if shard == "w1":
                        cpu_w13[:, :intermed].copy_(w_cpu.t())
                    elif shard == "w3":
                        cpu_w13[:, intermed: intermed + w_cpu.shape[0]].copy_(w_cpu.t())
                elif key == "w2":
                    self.w2_weights_cpu[layer_idx][eid].copy_(w_cpu.t())
        self._pending_weights.clear()


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
