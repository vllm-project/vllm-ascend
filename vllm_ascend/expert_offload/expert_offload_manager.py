"""Expert Offload Manager — manages CPU-side expert weights and NPU paging."""

import queue
import threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.expert_offload.hotness_tracker import ExpertHotnessTracker
from vllm_ascend.expert_offload.sliding_window_counter import SlidingWindowCounter

if TYPE_CHECKING:
    pass


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

    def __init__(self, vllm_config=None):
        from vllm_ascend.ascend_config import get_ascend_config

        self.offload_config = get_ascend_config().expert_offload_config
        self.num_device_experts = self.offload_config.num_device_experts

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

        # --- Prefetch components ---
        self._hotness_counter: SlidingWindowCounter | None = None
        self._hotness_tracker: ExpertHotnessTracker | None = None
        self._transfer_thread: ExpertOffloadThread | None = None
        self._prefetch_config = {
            "enabled": True,
            "hotness_top_k": self.offload_config.prefetch_hotness_top_k,
            "min_hotness_threshold": self.offload_config.prefetch_min_threshold,
        }

        ExpertOffloadManager._instance = self

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
                torch.empty(hidden_size, w13_up_dim, dtype=params_dtype, device="cpu")
                for _ in range(num_total_experts)
            ]
            w2_list = [
                torch.empty(intermediate_size_per_partition, hidden_size,
                            dtype=params_dtype, device="cpu")
                for _ in range(num_total_experts)
            ]
            self.w13_weights_cpu.append(w13_list)
            self.w2_weights_cpu.append(w2_list)
        self._drain_pending_weights()

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

    def init_device_experts(self):
        """Copy the first num_device_experts experts from CPU to NPU."""
        for i, layer in enumerate(self.moe_layers):
            dev = layer.w13_weight.device
            dt = layer.w13_weight.dtype
            for j in range(min(self.num_device_experts,
                               layer.w13_weight.shape[0])):
                layer.w13_weight.data[j].copy_(
                    self.w13_weights_cpu[i][j].to(dev).to(dt))
                layer.w2_weight.data[j].copy_(
                    self.w2_weights_cpu[i][j].to(dev).to(dt))

    def init_async_offload(self, num_layers: int, window_size: int = 200):
        """Initialize async offload components (prefetch thread, hotness tracker).

        Args:
            num_layers: MoE layer count
            window_size: Sliding window size for hotness tracking
        """
        if not self._prefetch_config["enabled"]:
            logger.info("Expert prefetch is disabled")
            return

        self._hotness_counter = SlidingWindowCounter(num_layers, window_size)
        self._hotness_tracker = ExpertHotnessTracker(
            self._hotness_counter,
            top_k_default=self._prefetch_config["hotness_top_k"]
        )
        self._transfer_thread = ExpertOffloadThread(self)
        self._transfer_thread.start()
        logger.info(
            f"Expert async offload initialized: num_layers={num_layers}, "
            f"window_size={window_size}"
        )

    def shutdown_async_offload(self):
        """Shutdown async offload thread."""
        if self._transfer_thread:
            self._transfer_thread.stop()
            self._transfer_thread = None
            logger.info("Expert async offload shutdown")

    # ------------------------------------------------------------------ #
    #  Forward path: page in experts based on topk_ids                    #
    # ------------------------------------------------------------------ #

    def update_weights(self, layer, topk_ids: torch.Tensor,
                        log2phy: torch.Tensor) -> int:
        """Incrementally page in needed experts, overwriting unused slots.

        Only copies experts that are NOT already on device.  Experts
        already mapped to a device slot (log2phy[eid] >= 0) are left
        untouched.  Reusable slots come from experts not in the current
        topk_ids set.

        Args:
            layer: AscendFusedMoE instance.
            topk_ids: [num_tokens, top_k] routed expert indices.
            log2phy: [global_num_experts] CPU tensor, modified in-place.

        Returns: number of CPU→NPU copies performed.
        """
        try:
            layer_idx = self.moe_layers.index(layer)
        except ValueError:
            return 0

        unique_experts = topk_ids.unique().cpu().tolist()
        needed = set(unique_experts)

        # Build reverse map: slot → expert_id currently occupying it
        slot_owner: dict[int, int] = {}
        for eid in range(len(log2phy)):
            s = log2phy[eid].item()
            if s >= 0:
                slot_owner[s] = eid

        on_device = set(slot_owner.values())
        already_there = needed & on_device           # no-op
        need_to_load = needed - already_there          # CPU→NPU copy
        reusable_slots = [s for s, e in slot_owner.items()
                          if e not in needed]          # slots to recycle

        if not need_to_load:
            return 0

        dev = layer.w13_weight.device
        dt = layer.w13_weight.dtype
        n_copies = 0

        for eid in need_to_load:
            if not reusable_slots:
                break  # no free slots — should not happen in normal usage
            slot = reusable_slots.pop()
            # Copy from CPU to NPU
            layer.w13_weight.data[slot].copy_(
                self.w13_weights_cpu[layer_idx][eid].to(dev).to(dt))
            layer.w2_weight.data[slot].copy_(
                self.w2_weights_cpu[layer_idx][eid].to(dev).to(dt))
            # Update mapping
            log2phy[slot_owner[slot]] = -1   # evict old occupant
            log2phy[eid] = slot               # assign slot to new expert
            slot_owner[slot] = eid
            n_copies += 1

        return n_copies

    def trigger_prefetch_for_next_layer(
        self,
        current_layer_idx: int,
        current_expert_ids: list[int],
    ) -> None:
        """Layer N 计算完成后，触发 Layer N+1 的 prefetch。

        预取并集来源:
        1. Layer N 激活的 expert 编号
        2. Layer N+1 上个 step 的 expert 编号
        3. Layer N+1 的热点 expert 编号

        Args:
            current_layer_idx: Layer N 的索引
            current_expert_ids: Layer N 激活的 expert ID 列表
        """
        if not self._prefetch_config["enabled"]:
            return

        next_layer_idx = current_layer_idx + 1
        if next_layer_idx >= len(self.moe_layers):
            return

        if not self._transfer_thread or not self._hotness_tracker:
            return

        self._hotness_tracker.record_step_experts(
            next_layer_idx, current_expert_ids
        )

        self._hotness_counter.record(current_layer_idx, current_expert_ids)

        union_experts = self._hotness_tracker.get_union_experts(
            layer_idx=current_layer_idx,
            source1_experts=current_expert_ids,
            hotness_top_k=self._prefetch_config["hotness_top_k"]
        )

        if not union_experts:
            return

        layer = self.moe_layers[next_layer_idx]
        log2phy = layer.log2phy

        slot_owner: dict[int, int] = {}
        for eid in range(len(log2phy)):
            s = log2phy[eid].item()
            if s >= 0:
                slot_owner[s] = eid

        on_device = set(slot_owner.values())
        need_prefetch = union_experts - on_device

        if not need_prefetch:
            return

        logger.debug(
            f"[Prefetch] Layer {next_layer_idx}: prefetching "
            f"{len(need_prefetch)} experts from union of {len(union_experts)}"
        )

        for eid in need_prefetch:
            self._transfer_thread.add_prefetch_task(
                layer_idx=next_layer_idx,
                expert_ids=[eid],
                priority=2
            )

    def reset_request_scope(self) -> None:
        """请求结束时重置滑动窗口计数器和历史记录。"""
        if self._hotness_counter:
            self._hotness_counter.reset()
        if self._hotness_tracker:
            self._hotness_tracker.reset()

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

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


class ExpertOffloadThread(threading.Thread):
    """异步 H2D 传输线程，处理 expert 权重的异步预取。"""

    def __init__(self, manager: ExpertOffloadManager):
        super().__init__(daemon=True, name="ExpertOffloadThread")
        self._manager = manager
        self._task_queue: queue.Queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._running = True
        self._lock = threading.Lock()
        self._pending_tasks: dict[int, dict] = {}
        self._task_id_counter = 0

    def add_prefetch_task(
        self,
        layer_idx: int,
        expert_ids: list[int],
        priority: int = 1,
        task_id: int | None = None,
    ) -> int:
        """添加预取任务到队列。

        Args:
            layer_idx: MoE 层索引
            expert_ids: 需要预取的 expert ID 列表
            priority: 优先级，数值越小优先级越高
            task_id: 任务 ID，如果为 None 则自动生成

        Returns:
            任务 ID
        """
        if task_id is None:
            task_id = self._task_id_counter
            self._task_id_counter += 1

        task = {
            "layer_idx": layer_idx,
            "expert_ids": expert_ids,
            "priority": priority,
        }

        with self._lock:
            self._pending_tasks[task_id] = task

        self._task_queue.put((priority, task_id, task))
        return task_id

    def run(self):
        """传输线程主循环。"""
        torch.npu.set_device()
        while self._running:
            try:
                _, task_id, task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                self._execute_transfer(task)
            except Exception as e:
                logger.error(f"Expert transfer failed: {e}")

            with self._lock:
                self._pending_tasks.pop(task_id, None)

            self._task_queue.task_done()

    def _execute_transfer(self, task: dict):
        """执行实际的 H2D 传输。"""
        layer_idx = task["layer_idx"]
        expert_ids = task["expert_ids"]

        layer = self._manager.moe_layers[layer_idx]
        dev = layer.w13_weight.device
        dt = layer.w13_weight.dtype
        log2phy = layer.log2phy

        slot_owner: dict[int, int] = {}
        for eid in range(len(log2phy)):
            s = log2phy[eid].item()
            if s >= 0:
                slot_owner[s] = eid

        on_device = set(slot_owner.values())
        need_load = [eid for eid in expert_ids if eid not in on_device]

        if not need_load:
            return

        all_slots = set(range(len(log2phy)))
        occupied = set(slot_owner.keys())
        free_slots = list(all_slots - occupied)

        if not free_slots:
            return

        for eid in need_load:
            if not free_slots:
                break

            slot = free_slots.pop(0)

            layer.w13_weight.data[slot].copy_(
                self._manager.w13_weights_cpu[layer_idx][eid].to(dev).to(dt)
            )
            layer.w2_weight.data[slot].copy_(
                self._manager.w2_weights_cpu[layer_idx][eid].to(dev).to(dt)
            )

            log2phy[eid] = slot
            slot_owner[slot] = eid

            logger.debug(
                f"[Transfer] Layer {layer_idx}: loaded expert {eid} to slot {slot}"
            )

    def stop(self):
        """停止传输线程。"""
        self._running = False
        self._executor.shutdown(wait=False)


_EXPERT_OFFLOAD_MANAGER: ExpertOffloadManager = None


def maybe_init_expert_offload_manager(vllm_config: VllmConfig):
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