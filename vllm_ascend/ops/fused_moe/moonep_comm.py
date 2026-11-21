# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""MoonEP communication backend for expert-parallel MoE.

Planning assigns routed rows to local Home experts or redundant Slots.
The SDK dispatches activations over SHMEM and the weight manager prefetches
remote experts into the selected Slots. Grouped MLP consumes Home rows
followed by Slot rows. Weighted combine uses the source rank's FP32 Top-K
weights and group counts to gather valid expert results.

Model Runner V2 initializes the SHMEM buffer pools and publishes weights
after model loading. Each step selects a capacity shared by all EP ranks,
and kernels use the real token count to exclude unused capacity.

MoonEP requires eager execution, contiguous expert sharding, and more than
one EP rank. Dynamic EPLB, expert maps, mixed placement, and LoRA are
unsupported. The SDK is loaded lazily when the backend is used.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401
from vllm.distributed.parallel_state import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, get_moonep_tokens_capacity
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import (
    MoETokenDispatchInput,
    MoETokenDispatchOutput,
    TMoECombineMetadata,
)
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult, MoECommMethod
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAll2All
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatcher


@dataclass
class MoonEPCombineMetadata:
    """Plan, weights, and row counts shared by one dispatch, MLP, and combine."""

    plan: object  # ascend_moonep MoonEPCommPlan
    buffer: object  # ascend_moonep Buffer used for this call
    num_tokens: int  # actual (pre-pad) token count of this MoE call
    hidden_shape: torch.Size  # original hidden_states shape (for restore)
    group_list: torch.Tensor  # Home and Slot row counts for fused publication
    source_topk_weights_sk: torch.Tensor  # FP32 Top-K weights on the source rank
    state: object | None = None  # weight state for the current MoE layer


# AscendRoutedExperts registers each layer when MoonEP is enabled.
# V2 load_model then initializes the pools and publishes weights after loading.
# Collective SHMEM initialization needs the full set of descriptors up front.
_moonep_layer_registry: list[dict] = []
_registry_lock = threading.Lock()
_SLOT_TLS = threading.local()  # current layer weight state during dispatch

# Planning, dispatch and combine always use the real token count. Buffer
# capacity remains independent of the rows included in routing and computation.


def iter_prefetch_layers(model) -> Iterator[torch.nn.Module]:
    """Defer decoder traversal until prefetch hook installation is enabled."""
    for path in ("model.layers", "layers", "transformer.layers", "model.model.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and len(obj):
            yield from obj
            return


# Step-adaptive dispatch capacity (upstream MoonEPBufferPool equivalent).
# Each Buffer fixes its receive capacity at S*K rows. Real token bounds and
# group_list exclude unused rows from expert load and GEMM work; smaller
# rungs still reduce capacity-sized copies and tail-processing overhead.
# MOONEP_MIN_DISPATCH_TOKENS is the per-step selection floor (default 128),
# not the first allocated rung.
def _min_dispatch_tokens() -> int:
    value = int(os.environ.get("MOONEP_MIN_DISPATCH_TOKENS", "128"))
    if value < 1:
        # The capacity selection floor must be positive.
        raise ValueError(f"MOONEP_MIN_DISPATCH_TOKENS must be >= 1, got {value}")
    return value


MOONEP_MIN_DISPATCH_TOKENS = _min_dispatch_tokens()


def _moonep_pool_capacities(s_max: int, cost_per_token: int = 32_768) -> list[int]:
    """Capacity ladder for the buffer pool (MOONEP_POOL_TOKEN_CAPACITY_TIERS overrides).

    Each rung fixes receive capacity at S*K rows, independently of the real
    token count used for planning, routing and expert computation. Powers of
    two cover small steps through 512; uniformly spaced rungs cover larger
    steps. Spacing is chosen using MOONEP_POOL_MAX_TIER_COUNT (default 32) and
    MOONEP_POOL_MEMORY_BUDGET_MB (default 2048 MiB).

    ``cost_per_token`` estimates hidden-buffer bytes per token of capacity
    (K*H*2). The budget guides rung spacing, not a hard cap on total memory:
    operator workspaces and other buffers also occupy device memory. Every
    rung is created before collective SHMEM initialization.
    """
    override = envs.MOONEP_POOL_TOKEN_CAPACITY_TIERS
    if override:
        caps = sorted({int(c) for c in override.split(",") if c.strip()})
        if not caps or caps[-1] != int(s_max):
            caps = sorted(set(caps) | {int(s_max)})
        return [c for c in caps if c <= int(s_max)]
    s_max = int(s_max)
    caps: list[int] = []

    def _add(c: int) -> None:
        c = int(c)
        if 0 < c <= s_max and c not in caps:
            caps.append(c)

    # Powers of two up to 512, then uniform spacing up to s_max. This keeps
    # the gap between selected capacity and the real step size bounded by
    # the uniform step; unused capacity does not become expert token load.
    #
    # Two caps bind, and the count one dominates: every Buffer also allocates
    # its op workspaces as separate VMM physical chunks (~10 per buffer, 16 MiB
    # scale each), and the driver starts failing with
    # "aclrtMallocPhysical(...) failed with 207001" past a few dozen buffers
    # (measured: 51 rungs -> pool init dies; 31 rungs -> fine).
    max_rungs = envs.MOONEP_POOL_MAX_TIER_COUNT
    budget = envs.MOONEP_POOL_MEMORY_BUDGET_MB * 2**20
    base = [c for c in (16, 32, 64, 128, 256, 512) if c <= s_max]
    for c in base:
        _add(c)
    top = base[-1] if base else s_max
    if s_max > top:
        # smallest uniform step that fits both caps
        step = 128
        while step < s_max:
            count = (s_max - top) // step
            span = (top + s_max) / 2.0 * count * cost_per_token
            if count + len(base) <= max_rungs and (budget <= 0 or span <= budget):
                break
            step *= 2
        c = top + step
        while c <= s_max:
            _add(c)
            c += step
    _add(s_max)
    return sorted(caps)


def _step_token_count(tokens: int):
    """Rank-symmetric token count for this step, or None when unknown.

    The forward context takes the maximum from DP token-count metadata,
    or the local count without DP. Missing context or DP metadata returns
    None so every EP rank falls back to the full capacity.
    """
    try:
        forward_context = get_forward_context()
    except Exception:  # noqa: BLE001
        return None
    if get_dp_group().world_size > 1 and forward_context.dp_metadata is None:
        return None
    max_across_dp = getattr(forward_context, "max_tokens_across_dp", None)
    if max_across_dp is None:
        return None
    return max(int(max_across_dp), tokens)


def _moonep_slot_count(epn: int) -> int:
    """Per-rank redundant expert slots; unset defaults to epn = E/R."""
    configured_slot_count = envs.MOONEP_REDUNDANT_EXPERT_SLOT_COUNT
    b = int(epn if configured_slot_count is None else configured_slot_count)
    if b < 1 or b > epn:
        raise ValueError(f"MOONEP_REDUNDANT_EXPERT_SLOT_COUNT must be in [1, {epn}], got {b}")
    return b


def register_moonep_layer(*, E: int, K: int, H: int, nle: int, layer=None) -> None:
    """Record one MoE layer for MoonEP. Cheap; only called when enabled."""
    with _registry_lock:
        _moonep_layer_registry.append({"E": int(E), "K": int(K), "H": int(H), "nle": int(nle), "layer": layer})


def _registry_keys() -> set[tuple]:
    with _registry_lock:
        return {(e["E"], e["K"], e["H"], e["nle"]) for e in _moonep_layer_registry}


# ---------------------------------------------------------------------------
# AICPU custom 调度器预热：必须排在"建立跨设备映射"之前，否则整个 DP wave 挂死。
#
# 驱动约束（2026-09-22 定位，2 卡最小复现脚本 scripts/repro_aicpu_vs_moonep_order.py）：
# 本进程 VA 里一旦已经存在**跨设备**的 task group 映射——MoonEP 的对称堆正是在
# collective init 时把其它 rank 的物理内存映射进本地 VA 空间——之后再启动本进程的
# AICPU custom 调度器，驱动就会拒绝：
#   dmesg  : [svm] um_smm_mmap_pre_handle Task group map dst_devid should be consistent
#            with src_devid. (dst_udevid=0; src_udevid=1) → event_sched ret=3
#   deviceN: [svm_attach_task_group] Notice master failed. (ret=3) →
#            Failed to halMemBindSibling ... ret[3] →
#            Aicpu custom schedule start failed, ret[22201]
# 同一条约束对"跨设备映射之后才新建/首次使用的 HCCL 通信域"一样成立（表现为
# ret=3 / HCCL error 19）。
# 而 vllm-ascend 的 A5 attention metadata 算子（csrc/attention/*_metadata，按
# add_modules_sources_aicpu 编译）正是 AICPU custom 算子：调度器起不来 → 算子永不
# 执行 → host 卡在紧随其后的 .item()，该步在 DP 上永远等不齐 → 全体停住
# （2026-09-22 14:32 的 DSv4 slot 运行实测：device-0 aicpu 日志 ret[22201]、8 卡
#  再无输出、14:41 强杀）。
# 调度器是在第一个 AICPU custom 任务**提交**时同步起来的，所以这里随便跑一个最小的
# metadata 算子就够；起来之后即使已有 peer 映射，再跑 aicpu 算子也正常（已实测）。
# 失败只告警不抛：真没有它最多是回到旧行为，不该把 MoonEP 起链整块拖垮。
# ---------------------------------------------------------------------------
_AICPU_WARMUP_OP = "npu_vllm_quant_lightning_indexer_metadata"


def _warmup_aicpu_custom_scheduler() -> bool:
    """在建立跨设备映射之前，先把本进程的 AICPU custom 调度器拉起来。

    固定在 MoonEP 初始化时尝试预热，返回是否成功；原有跳过和失败策略保持不变。
    """
    rank = get_ep_group().rank_in_group
    try:
        # 本函数在 load_model 末尾被调，此时通常**没有** forward context
        # （vllm 的 get_forward_context() 会直接 assert），所以只能防御性读取。
        capturing = bool(_EXTRA_CTX.capturing)
    except Exception:  # noqa: BLE001
        capturing = False
    if capturing:
        # 图捕获期间不能插新算子、更不能 host 同步。
        logger.info("[moonep-init r%s] stage=aicpu-warmup skipped (capturing)", rank)
        return False
    try:
        op = getattr(torch.ops._C_ascend, _AICPU_WARMUP_OP)
    except Exception:  # noqa: BLE001 - 别的模型/版本可能没有这个算子
        logger.info("[moonep-init r%s] stage=aicpu-warmup skipped (op %s not registered)", rank, _AICPU_WARMUP_OP)
        return False
    dev = f"npu:{torch.npu.current_device()}"
    lens = torch.ones(1, dtype=torch.int32, device=dev)
    try:
        # 1 token 的最小调用：只为了触发调度器启动，忽略输出。
        op(
            num_heads_q=64,
            num_heads_k=1,
            head_dim=128,
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=lens.clone(),
            actual_seq_lengths_key=lens.clone(),
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=512,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            device=dev,
        )
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - 预热失败不许改变原有行为
        logger.warning(
            "[moonep-init r%s] stage=aicpu-warmup FAILED (%s: %s) —— "
            "AICPU custom 调度器可能没起来，后续 aicpu 算子（A5 attention metadata）有挂住的风险",
            rank,
            type(exc).__name__,
            exc,
        )
        return False
    logger.info("[moonep-init r%s] stage=aicpu-warmup done op=%s", rank, _AICPU_WARMUP_OP)
    return True


# ---------------------------------------------------------------------------
# Runtime pool: capacity-specific Buffers per unique (E, K, H) layer shape.
# Construction registers CANN VMM descriptors only; ShmemRuntime is initialized
# once, after every Buffer has been created (two-phase, collective).
# ---------------------------------------------------------------------------
class _MoonEPRuntime:
    def __init__(self):
        self._lock = threading.Lock()
        self._ready = False
        self._fail: Exception | None = None
        self._buffers: dict[tuple, object] = {}
        self._pads: dict[tuple, dict] = {}
        self._state_by_w13: dict = {}

    def lookup_slot_state(self, weight_id: int):
        return self._state_by_w13.get(weight_id)

    def _ensure_initialized(self):
        with self._lock:
            if self._ready:
                return
            if self._fail is not None:
                raise RuntimeError(f"MoonEP runtime failed to initialize: {self._fail}") from self._fail
            try:
                self._initialize_locked()
            except Exception as exc:  # noqa: BLE001 - keep the first failure
                self._fail = exc
                raise
            self._ready = True

    def _initialize_locked(self):
        # Lazy imports: the ascend-moonep package must not be required on
        # configurations where MoonEP is never selected.
        from ascend_moonep import Buffer, ShmemRuntime

        if ShmemRuntime.is_initialized():
            raise RuntimeError("MoonEP: ascend_moonep ShmemRuntime already initialized outside the MoonEP runtime.")
        ep_group = get_ep_group()
        group = ep_group.device_group
        R = ep_group.world_size
        if R <= 1:
            raise RuntimeError("MoonEP requires expert parallelism (EP world size > 1).")
        capacity = get_moonep_tokens_capacity()
        if capacity is None or capacity <= 0:
            raise RuntimeError("MoonEP: token capacity S was not configured (set_moonep_tokens_capacity).")
        # Fix the transport once before creating Buffers and SHMEM contexts.
        # Planning remains native; the other three stages must match this choice.
        use_udma = envs.MOONEP_TRANSPORT_TYPE == "udma"

        keys = sorted(_registry_keys())
        if not keys:
            raise RuntimeError(
                "MoonEP: no MoE layers registered (register_moonep_layer). "
                "MoE layers must be created before the first forward."
            )
        logger.info("[moonep-init r%s] stage=creating-buffers nkeys=%s", get_ep_group().rank_in_group, len(keys))
        # Validate + create every Buffer *before* the single collective init.
        for E, K, H, nle in keys:
            if E % R != 0:
                raise ValueError(f"MoonEP: num_experts E={E} must be divisible by EP size R={R}.")
            if nle != E // R:
                raise NotImplementedError(
                    f"MoonEP: local experts {nle} != E/R = {E // R}; only plain contiguous "
                    "expert sharding is supported (no dynamic EPLB / expert maps)."
                )
            if K > 16:
                raise ValueError(f"MoonEP: combine supports K <= 16, got K={K}.")
            if H % 16 != 0:
                raise ValueError(f"MoonEP: combine requires H % 16 == 0, got H={H}.")
            if K * H * capacity >= 2**31:
                raise ValueError("MoonEP: S*K exceeds the planning int32 range.")
            # The per-rank Slot count is bounded by the local expert count.
            capacities = _moonep_pool_capacities(int(capacity), cost_per_token=K * H * 2)
            self._buffers[(E, K, H)] = {
                cap: Buffer(
                    int(cap),
                    H,
                    K,
                    E,
                    R,
                    token_padding=1,
                    B=_moonep_slot_count(E // R),
                    group=group,
                    explicitly_destroy=True,
                    # Keep the SDK's num_sms default. Planning selects its own
                    # block count and sizes its workspace consistently; only
                    # MOONEP_PLANNING_NUM_SMS overrides that selection.
                    **({"use_udma": True} if use_udma else {}),
                )
                for cap in capacities
            }
            logger.info(
                "[moonep-init r%s] stage=pool-ready E=%s K=%s H=%s capacities=%s",
                get_ep_group().rank_in_group,
                E,
                K,
                H,
                capacities,
            )
            # Static [S_max, *] workspace shared across layers of one shape.
            # Each call overwrites the real prefix; kernels ignore the tail.
            dev = f"npu:{torch.npu.current_device()}"
            self._pads[(E, K, H)] = {
                "hidden": torch.zeros(capacity, H, dtype=torch.bfloat16, device=dev),
                "route": torch.zeros(capacity, K, dtype=torch.float32, device=dev),
                "topk": torch.zeros(capacity, K, dtype=torch.int32, device=dev),
                "combine_route": torch.zeros(capacity, ((K + 7) // 8) * 8, dtype=torch.float32, device=dev),
            }
        from vllm_ascend.ops.fused_moe.moonep_weight_manager import _build_slot_states

        dev = int(torch.npu.current_device())
        ep_grp = get_ep_group()
        rank = ep_grp.rank_in_group
        with _registry_lock:
            entries = list(_moonep_layer_registry)
        # All loaded payload layouts and per-pool counts are known before
        # allocating caches; no new descriptors may be added after SHMEM init.
        self._states = states = _build_slot_states(entries, rank, dev)
        self._state_by_w13 = {id(st._w13_src): st for st in states}

        from vllm_ascend.ops.fused_moe.moonep_weight_manager import udma_prefetch_overlap_requested

        prefetch_overlap = udma_prefetch_overlap_requested(use_udma, True)
        if prefetch_overlap:
            ShmemRuntime.validate_udma_prefetch_overlap()
        if not ShmemRuntime.is_initialized():
            # 必须在下面的 collective init 之前：它会把 peer 的物理内存映射进本地
            # VA，之后就再也起不了 AICPU custom 调度器了（见上方
            # _warmup_aicpu_custom_scheduler 的驱动约束注释）。
            _warmup_aicpu_custom_scheduler()
            # Engine 环境默认走 ipport/config-store bootstrap：uid 路径要在
            # 每个 worker 的 HCCL 组上做 UID 广播（dist.broadcast over HCCL），
            # 在引擎里与刚创建的 UDMA-flagged VMM 上下文及模型加载后的设备态
            # 冲突（实测 UB_INLINE_WRITE/HD Put 任务异常 → 全体自旋）。ipport
            # 只在本机 rank0 起 config-store，无 HCCL 广播，引擎内实测通过
            # （2026-09-09：uid 卡 init_with_buffer:182，ipport 可到首层 MoE
            # dispatch）。standalone torchrun 无此问题，可用 MOONEP_SHMEM_BOOTSTRAP_TYPE=uid 覆盖。
            bootstrap = envs.MOONEP_SHMEM_BOOTSTRAP_TYPE
            logger.info(
                "[moonep-init r%s] stage=shmem-init-enter bootstrap_type=%s",
                get_ep_group().rank_in_group,
                bootstrap,
            )
            if bootstrap == "ipport":
                # Config-store bootstrap (the path ascend-moonep's own multi-rank
                # tests validated) instead of the UID-over-HCCL broadcast.
                from ascend_moonep.buffer_c import symmetric_descriptors

                port = envs.MOONEP_SHMEM_BOOTSTRAP_PORT
                # All nodes must connect to the same rank-0 config store.
                # Use vLLM's existing cluster address; keep loopback valid for
                # single-node deployments, but reject it for a multi-node group.
                import ipaddress

                from vllm.config import get_current_vllm_config

                parallel = get_current_vllm_config().parallel_config
                bootstrap_ip = parallel.data_parallel_master_ip
                local_dp = parallel.data_parallel_size_local or parallel.data_parallel_size
                if not bootstrap_ip:
                    raise RuntimeError("MoonEP ipport bootstrap requires a DP master address")
                if parallel.data_parallel_size > local_dp:
                    try:
                        loopback = ipaddress.ip_address(bootstrap_ip).is_loopback
                    except ValueError:
                        loopback = bootstrap_ip.lower() == "localhost"
                    if loopback:
                        raise RuntimeError(
                            "Multi-node MoonEP requires --data-parallel-address "
                            "to be reachable from every node, not loopback"
                        )
                ShmemRuntime.init_with_ip_port(
                    group=group,
                    ip_port=f"tcp://{bootstrap_ip}:{port}",
                    descriptors=symmetric_descriptors(),
                    enable_udma=use_udma,
                    qp_count=0,
                    prefetch_overlap=prefetch_overlap,
                )
            else:
                ShmemRuntime.init_with_buffer(group=group, enable_udma=use_udma, prefetch_overlap=prefetch_overlap)
            logger.info("[moonep-init r%s] stage=shmem-init-done", get_ep_group().rank_in_group)
        qp_layout = ShmemRuntime.require_udma_prefetch_overlap() if prefetch_overlap else None
        if qp_layout is not None:
            logger.info(
                "[moonep-udma-qp] allocated=%s dispatch_count=%s prefetch_qp=%s",
                qp_layout[0],
                qp_layout[1],
                qp_layout[2],
            )
        # Check effective context flags once at initialization. Planning is
        # native in this SDK; the other three stages support UDMA.
        expected_backends = {
            "planning": False,
            "dispatch": use_udma,
            "combine": use_udma,
            "prefetch": use_udma,
        }
        for (experts, topk, hidden), pool in self._buffers.items():
            for cap, buffer in pool.items():
                ctx = buffer._require_ctx()
                ctx["udma_prefetch_overlap_ready"] = qp_layout is not None
                actual = {name: bool(ctx[f"_{name}_use_udma"]) for name in expected_backends}
                if actual != expected_backends:
                    raise RuntimeError(f"MoonEP backend mismatch: requested={expected_backends}, effective={actual}")
                backends = " ".join(f"{name}={'udma' if enabled else 'mte'}" for name, enabled in actual.items())
                logger.info(
                    "[moonep-backend r%s] E=%s K=%s H=%s S=%s %s",
                    get_ep_group().rank_in_group,
                    experts,
                    topk,
                    hidden,
                    cap,
                    backends,
                )
        dev = torch.npu.current_device()
        torch.npu.synchronize(dev)
        torch.npu.empty_cache()
        allocated_before = torch.npu.memory_allocated(dev)
        free_before, _ = torch.npu.mem_get_info(dev)
        for st in self._states:
            st.publish()
        torch.npu.synchronize(dev)
        torch.npu.empty_cache()
        free_after, _ = torch.npu.mem_get_info(dev)
        released_layers = sum(bool(getattr(st, "_released", False)) for st in self._states)
        logger.info(
            "[moonep-release r%s] home_backed_layers=%s/%s torch_allocated_released_bytes=%s "
            "device_free_delta_bytes=%s device_free_bytes=%s",
            ep_group.rank_in_group,
            released_layers,
            len(self._states),
            allocated_before - torch.npu.memory_allocated(dev),
            free_after - free_before,
            free_after,
        )


_MOONEP_RUNTIME = _MoonEPRuntime()


def ensure_runtime_initialized() -> None:
    """MindSpeed-style wiring-time init for the MoonEP runtime.

    Mirrors MindSpeed-MM (init_moonep_layer_states): as soon as the model graph
    exists (end of worker load_model — every AscendFusedMoE layer is then
    registered), construct all symmetric Buffers and run the single collective
    ShmemRuntime.init_with_buffer, *before any forward*. Doing it lazily inside
    the first forward worked for the MTE path but the UDMA path stalls in the
    engine; early collective init on an otherwise-quiet device state is the
    MindSpeed pattern that keeps the UDMA/URMA kernels healthy.

    一并把 AICPU custom 调度器预热提前到这里（collective init 之前）：
    ``_initialize_locked`` 里若先建了跨设备映射，A5 attention metadata 这类
    AICPU custom 算子就再也起不来了，整个 DP wave 会挂死——见
    ``_warmup_aicpu_custom_scheduler``。

    Idempotent; no-op when MoonEP is not enabled or EP world size <= 1.
    """
    from vllm_ascend.ascend_config import get_ascend_config

    if not get_ascend_config().enable_moonep:
        return
    if get_ep_group().world_size <= 1:
        return
    # [env-audit] 一次性诊断：worker 进程运行时实际看到的 MoonEP 开关集合
    # （/proc/environ 只反映 exec 时刻，运行时可能有重注入，故在初始化点直接打印）。
    import os as _os_audit

    logger.info(
        "[moonep-env-audit] %s",
        " ".join(
            f"{k}={_os_audit.environ.get(k, '<unset>')}"
            for k in (
                "MOONEP_PLANNING_SOURCE_DEDUPLICATION_ENABLED",
                "MOONEP_PLANNING_LOAD_STATS_BARRIER_MODE",
                "MOONEP_MIN_DISPATCH_TOKENS",
            )
        ),
    )
    _MOONEP_RUNTIME._ensure_initialized()


# Cache the capacity ladder for each initialized buffer pool.
_GEO_CACHE: dict = {}


def _geometry(E: int, K: int, H: int) -> dict:
    """Cache the sorted capacities and upper bound of an (E, K, H) pool."""
    geo = _GEO_CACHE.get((E, K, H))
    if geo is None:
        caps = sorted(_MOONEP_RUNTIME._buffers[(E, K, H)])
        geo = {"caps": caps, "s_max": caps[-1]}
        _GEO_CACHE[(E, K, H)] = geo
    return geo


# Reuse device-side row-count scratch across layers with the same geometry.
_SCRATCH: dict = {}


def _scratch_for(E: int, K: int, H: int, extra: int, device) -> dict:
    """``extra`` = B, the per-rank prefetch-slot count: the library's cu_seqlens
    has E + B entries, which can exceed E + K."""
    key = (int(E), int(K), int(H))
    sc = _SCRATCH.get(key)
    if sc is not None:
        return sc
    nrows = int(E) + int(extra)
    # cu_pad[0] is zero for the whole run (nothing ever writes index 0), so the
    # per-group row counts come out of a single subtract -- counts[i] =
    # cu_pad[i+1] - cu_pad[i] == cu[i] - cu[i-1] with the implicit cu[-1] = 0.
    # That replaces the old `counts[0] = cu64[0]` scalar splice, which cost a
    # kernel launch of its own every layer.
    cu_pad = torch.zeros(nrows + 1, dtype=torch.int64, device=device)
    sc = {
        "cu_pad": cu_pad,
        # Planning validates this vector but derives counts from topk itself.
        "tpe_compat": torch.zeros(int(E), dtype=torch.int32, device=device),
        "counts": torch.empty(nrows, dtype=torch.int64, device=device),
        "own": torch.empty(int(E), dtype=torch.int64, device=device),
    }
    _SCRATCH[key] = sc
    return sc


def _select_buffer_for_step(E: int, K: int, H: int, tokens: int):
    """Smallest pooled capacity that fits this step -> ``(buffer, capacity)``."""
    _MOONEP_RUNTIME._ensure_initialized()
    pool = _MOONEP_RUNTIME._buffers[(E, K, H)]
    geo = _geometry(E, K, H)
    s_max = geo["s_max"]
    step_tokens = _step_token_count(tokens)
    if step_tokens is None or step_tokens > s_max:
        capacity = s_max
    else:
        # smallest rung that still fits the step: with the dense ladder this is
        # the "S as close to T as the pool allows" choice
        need = max(int(step_tokens), MOONEP_MIN_DISPATCH_TOKENS)
        capacity = next((c for c in geo["caps"] if c >= need), s_max)
    if capacity < tokens:
        raise RuntimeError(f"MoonEP: step has {tokens} tokens but the selected capacity is {capacity} (s_max={s_max}).")
    return pool[capacity], capacity


def _pads_for(E: int, K: int, H: int):
    _MOONEP_RUNTIME._ensure_initialized()
    return _MOONEP_RUNTIME._pads[(E, K, H)]


# ---------------------------------------------------------------------------
# Token dispatcher
# ---------------------------------------------------------------------------
def _moonep_dispatch_transport(quant, hidden_states) -> str:
    """Select from per-layer quant metadata; never reinterpret quantized input."""
    from vllm_ascend.quantization.quant_type import QuantType

    if quant.quant_type != QuantType.W4A8MXFP:
        return "bf16"
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        raise ValueError("MXFP8 transport expects contiguous original BF16 activations")
    mxfp = quant.mxfp
    if mxfp is None:
        raise ValueError("W4A8MXFP dispatch requires MXFP quantization parameters")
    if mxfp.act_quant_type not in (None, torch.float8_e4m3fn):
        raise NotImplementedError("W4A8MXFP dispatch requires E4M3 activations")

    # These dependencies stay lazy: native Off and BF16-only models do not
    # need to load the SDK wire module or MXFP dtype compatibility helpers.
    FLOAT8_E8M0FNU_DTYPE = getattr(torch, "float8_e8m0fnu", None)
    from ascend_moonep.mxfp8_transport import wire_geometry

    if FLOAT8_E8M0FNU_DTYPE is None:
        raise NotImplementedError("MXFP8 dispatch requires E8M0 scale dtype support")
    # torch-npu quant methods pass the backend's format identifier, whereas
    # tensor-facing callers may use the native torch dtype. Both denote E8M0;
    # an arbitrary uint8 carrier dtype alone does not declare a scale format.

    npu_e8m0_format = getattr(torch_npu, "float8_e8m0fnu", None)
    if mxfp.per_token_scale_dtype not in (None, FLOAT8_E8M0FNU_DTYPE, npu_e8m0_format):
        raise NotImplementedError("W4A8MXFP dispatch requires group32 E8M0 activation scales")
    # Keep the SDK as the authority for wire geometry. Unsupported W4A8MXFP
    # shapes fail here; there is no BF16 retry or exception-driven fallback.
    wire_geometry(int(hidden_states.shape[1]))
    return "mxfp8"


class TokenDispatcherWithMoonEP(MoETokenDispatcher[TMoECombineMetadata]):
    """Dispatch home/Slot rows and combine using source-rank FP32 route weights."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.top_k = int(kwargs.get("top_k", 0))
        self.num_experts = int(kwargs.get("num_experts", 0))
        self.num_local_experts = int(kwargs.get("num_local_experts", 0))
        self._logged_transports: set[tuple] = set()

    def token_dispatch(
        self,
        token_dispatch_input: MoETokenDispatchInput,
    ) -> MoETokenDispatchOutput[TMoECombineMetadata]:
        if _EXTRA_CTX.capturing:
            raise RuntimeError(
                "MoonEP dispatch cannot run inside captured graphs; "
                "run with --enforce-eager (or disable graph capture) when enable_moonep is on."
            )
        hidden_states = token_dispatch_input.hidden_states
        topk_weights = token_dispatch_input.topk_weights
        topk_ids = token_dispatch_input.topk_ids
        if hidden_states.dim() != 2:
            raise ValueError(f"MoonEP dispatch expects 2D hidden_states, got shape {tuple(hidden_states.shape)}")
        T, H = hidden_states.shape
        E, K = self.num_experts, self.top_k
        transport = _moonep_dispatch_transport(token_dispatch_input.quant, hidden_states)
        use_mxfp8 = transport == "mxfp8"
        if K <= 0 or E <= 0:
            raise ValueError(f"MoonEP: invalid geometry K={K}, E={E}")

        s_max = get_moonep_tokens_capacity()
        if s_max is None or T > s_max:  # noqa: SIM300 - retain the original token-bound check
            raise RuntimeError(
                f"MoonEP: step has {T} tokens > static capacity S={s_max}. "
                "Raise --max-num-batched-tokens or disable enable_moonep."
            )
        # Every EP rank picks the same pooled capacity (see _step_token_count).
        # Kernels bound their row loops by the real T (num_valid_tokens), so
        # planning, balancing and computation exclude the unused capacity.
        buffer, capacity = _select_buffer_for_step(E, K, H, T)
        pads = _pads_for(E, K, H)
        hidden_c = pads["hidden"][:capacity]
        route_c = pads["route"][:capacity]
        topk_c = pads["topk"][:capacity]
        if hidden_states.dtype != torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)
        # copy_ casts on the device: no per-call .to() temporaries.
        if not use_mxfp8:
            hidden_c[:T].copy_(hidden_states)
        route_c[:T].copy_(topk_weights)
        topk_c[:T].copy_(topk_ids)
        source_topk_weights_sk = pads["combine_route"][:capacity]
        source_topk_weights_sk[:T, :K].copy_(topk_weights)
        # SDK validates this compatibility tensor; Planning rebuilds real load
        # from topk_c, independently of these unused histogram values.
        sc = _scratch_for(E, K, H, buffer._require_ctx()["B"], topk_c.device)
        tpe = sc["tpe_compat"]

        dispatch_scale = None
        if use_mxfp8:
            h_nvs, dispatch_scale, _, cu, plan = buffer.dispatch_mxfp8(
                hidden_states,
                route_c,
                topk_c,
                tpe,
                num_tokens=T,
                wire_storage=hidden_c,
            )
        else:
            # zero_copy=True returns a view of the symmetric shard the dispatch
            # kernel already wrote, so the MLP reads it in place and the
            # library's full-shard copy-out (402 MB/layer) disappears.  The
            # legacy combine later publishes the MLP output back into that same
            # shard; both are on the current stream, so the GMM has finished
            # reading before that copy is enqueued.
            h_nvs, _, cu, plan = buffer.dispatch(
                hidden_c,
                route_c,
                topk_c,
                tpe,
                zero_copy=True,
                inter_rank_sync=True,
                num_tokens=T,
                reuse_outputs=True,
            )
        quant_type = token_dispatch_input.quant.quant_type
        log_key = (quant_type, H)
        if log_key not in self._logged_transports:
            reason = "W4A8MXFP_E4M3_group32" if use_mxfp8 else "existing_BF16_dispatch_path"
            logger.info(
                "[moonep-dispatch-auto] quant_type=%s transport=%s H=%s reason=%s",
                quant_type,
                transport,
                H,
                reason,
            )
            self._logged_transports.add(log_key)

        # Per-local-expert row counts = cu diff over this rank's owned global
        # expert range [rank*epn, (rank+1)*epn). Shard rows are prefix-packed
        # in ascending owned-expert order, which is exactly the row/group order
        # the grouped-matmul MLP expects. Slot groups follow the home groups.
        ep_group = get_ep_group()
        epn = E // ep_group.world_size
        rank = ep_group.rank_in_group
        # counts[i] = cu[i] - cu[i-1] with an implicit cu[-1] = 0 (the leading
        # zero term is why this is shifted: slicing per rank would be off by one
        # for rank 0). All three buffers are pre-allocated scratch: the old
        # cast+cat+sub+contiguous chain allocated four tensors per layer.
        counts = sc["counts"]
        cu_pad = sc["cu_pad"]
        n_cu = int(cu.numel())
        cu_pad[1 : n_cu + 1].copy_(cu)  # int32 -> int64 cast
        torch.sub(cu_pad[1 : n_cu + 1], cu_pad[:n_cu], out=counts[:n_cu])
        own_counts = sc["own"]
        own_counts[:epn].copy_(counts[rank * epn : (rank + 1) * epn])
        own_counts = own_counts[:epn]
        # Rows for prefetched Slot experts stay local and follow home rows.
        slot_counts = counts[E:].contiguous()
        local_counts = torch.cat((own_counts, slot_counts))
        # Keep the row count on the device. The MLP takes the full [NvS, H]
        # shard plus the count-mode `group_list`; grouped matmul walks exactly
        # sum(group_list) rows. Combine masks the untouched tail on the device.
        # This avoids the per-layer row-count `.item()` and host-side slicing;
        # it does not replace communication barriers or stream event waits.
        #
        # Return a disposable view: dispose_tensor() calls Tensor.set_, which
        # must not mutate the SDK's cached dispatch output tensor object.
        hidden_trim = h_nvs[:]
        state = getattr(_SLOT_TLS, "state", None)
        return MoETokenDispatchOutput(
            hidden_states=hidden_trim,
            dynamic_scale=dispatch_scale[: hidden_trim.shape[0]] if dispatch_scale is not None else None,
            group_list=local_counts,
            group_list_type=1,  # count mode
            combine_metadata=MoonEPCombineMetadata(
                plan=plan,
                buffer=buffer,
                num_tokens=T,
                hidden_shape=hidden_states.shape,
                state=state,
                group_list=local_counts,
                source_topk_weights_sk=source_topk_weights_sk,
            ),
        )

    def token_combine(self, hidden_states, combine_metadata, bias=None):
        assert bias is None, "Bias is not supported in TokenDispatcherWithMoonEP."
        assert isinstance(combine_metadata, MoonEPCombineMetadata)
        T = combine_metadata.num_tokens
        # Slot applies source-rank FP32 route weights inside combine, retaining
        # FP32 products until the final BF16 output cast. Grouped matmul keeps
        # the MLP output rows in the dispatched shard order.
        if combine_metadata.source_topk_weights_sk is None:
            raise ValueError("MoonEP Slot combine requires source-rank route weights")
        buffer = combine_metadata.buffer
        row_counts = combine_metadata.group_list
        if row_counts is None:
            raise ValueError("Fused publish requires counts for a masked tail")
        # Native publication writes valid rows and zeros the SHMEM tail.
        # Do not materialize a mask, a sum scalar, or a BF16 clean tensor.
        combine_input = hidden_states
        out, _, _ = buffer.combine_weighted(
            plan=combine_metadata.plan,
            hidden_nvsh=combine_input,
            source_topk_weights_sk=combine_metadata.source_topk_weights_sk,
            num_tokens=T,
            inter_rank_sync=True,
            fused_publish=True,
            row_counts=row_counts,
        )
        return out[:T].view(combine_metadata.hidden_shape)


class MoonEPCommImpl(MoECommMethod):
    """Expert-parallel MoE over MoonEP SHMEM dispatch, reusing the all-to-all
    prepare/finalize (TP slicing/unpadding semantics) and the standard MLP."""

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMoonEP(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def fused_experts(self, fused_experts_input, quant_method=None):
        from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import build_token_dispatch_input
        from vllm_ascend.ops.fused_moe.moonep_weight_manager import run_slot_forward

        layer = fused_experts_input.layer
        if layer is None or quant_method is None:
            raise RuntimeError("MoonEP v0.30 requires the routed-expert layer and quant method")
        if fused_experts_input.lora_context is not None:
            raise NotImplementedError("MoonEP Slot does not support MoE LoRA")
        lookup_slot_state = _MOONEP_RUNTIME.lookup_slot_state
        state = lookup_slot_state(id(layer.w13_weight))
        if state is None:
            raise RuntimeError("MoonEP balance state missing: initialize after expert weight loading")
        from vllm_ascend.ops.fused_moe.moonep_weight_manager import select_prefetch_layer

        select_prefetch_layer(layer)
        record_stage_events = bool(getattr(layer, "return_with_event", False))
        if record_stage_events:
            from vllm_ascend.ascend_config import get_ascend_config

            record_stage_events = bool(get_ascend_config().multistream_overlap_shared_expert)
        before_dispatch_evt = torch.npu.current_stream().record_event() if record_stage_events else None
        _SLOT_TLS.state = state
        try:
            disp = self.token_dispatcher.token_dispatch(
                build_token_dispatch_input(fused_experts_input=fused_experts_input)
            )
            meta = disp.combine_metadata
            mlp_rows, before_gmm2_evt = run_slot_forward(
                state=meta.state,
                lookup_slot_state=lookup_slot_state,
                buffer=meta.buffer,
                plan=meta.plan,
                token_dispatch_output=disp,
                fused_experts_input=fused_experts_input,
                use_fusion_ops=None,
                moe_config=self.moe_config,
                quant_method=quant_method,
                record_before_gmm2=record_stage_events,
            )
            before_combine_evt = torch.npu.current_stream().record_event() if record_stage_events else None
            routed_out = self.token_dispatcher.token_combine(mlp_rows, meta)
        finally:
            _SLOT_TLS.state = None
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            shared_activation_overlap_start_evt=before_gmm2_evt,
            group_list_type=disp.group_list_type,
            expert_tokens=disp.group_list,
        )
