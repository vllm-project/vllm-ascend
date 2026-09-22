# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""apply_top_k_top_p: top-k/top-p 过滤 (+ 可选融合 softmax) 的 Triton 实现。

语义与 CANN ``torch_npu.npu_top_k_top_p`` 严格一致（升序稳定排序 → 第 k 大
阈值 → 严格小于阈值置 -inf → softmax 求概率 → 自最小概率累加 →
cumsum <= 1-p 过滤且末位强制保留 → 按排序下标散回原序）：

    sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)
    topKValue[b]  = sortedValue[b][V - k[b]]
    topKMask      = sortedValue < topKValue
    sortedValue   = where(topKMask, -inf, sortedValue)
    probsSum      = cumsum(softmax(sortedValue, dim=-1), dim=-1)
    topPMask[b][v]= probsSum[b][v] <= 1 - p[b];  topPMask[b][-1] = false
    sortedValue   = where(topPMask, -inf, sortedValue)
    out[b][sortedIndices[b][v]] = sortedValue[b][v]

两组输出契约（同一对 kernel，编译期 ``FUSED_SOFTMAX`` 开关）：

- :func:`apply_top_k_top_p` / :func:`apply_top_k_top_p_with_sorted`
  输出原序 masked logits（保留位 = 原始 logit，过滤位 = -inf），与
  ``torch_npu.npu_top_k_top_p`` 位精确一致（随机用例 torch.equal 对拍通过）。
- :func:`fused_topk_topp_softmax` / :func:`fused_topk_topp_softmax_with_sorted`
  kernel 内融合最终 softmax，直接输出 fp32 概率（masked 位 = 0，行和 = 1），
  等价于生产链路 ``npu_top_k_top_p + softmax(dim=-1, dtype=float32)``。

设计要点：

  - k / p 直接接受逐请求 [B] 张量（vllm SamplingMetadata 原生形态），
    不需要 ``k[0].item()`` 这类 CPU 同步，也不要求全 batch 配置一致。
  - 被过滤位置由 ``torch.full``/``torch.zeros`` 整块预填，kernel 只散写
    存活元素（≈ 保留数），避免全量 V 的随机散写。

实现要点：

  - kept 集 = ``s >= topKValue AND (probsSum > 1-p OR v == V-1)``，两个条件
    都是后缀单调，等价于 ``v >= m``（标量边界）；scatter 的 mask 用
    ``offs >= m`` 比较形式——cumsum 派生的布尔向量直接做 scatter mask
    在 triton-ascend 3.2 下会误编译（实测 store 塌缩），切勿改回。
  - 两条路径：``V <= 4096`` 单块一次载入；``V > 4096`` 分块多遍
    （求 top-k 门限下分母与边界 → 定位 top-p 边界 → 仅处理存活后缀）。
  - k[b] <= 0 时下标钳位到 V-1（只保留最大值）；k[b] >= V（vllm 的禁用
    约定）时钳位到 0（不过滤）。
"""

import torch
from vllm.triton_utils import tl, triton

__all__ = [
    "apply_top_k_top_p",
    "apply_top_k_top_p_with_sorted",
    "fused_topk_topp_softmax",
    "fused_topk_topp_softmax_with_sorted",
]

# 单块路径上界：BLOCK=8192 的 fp32 中间量超过 192KB UB。
_SINGLE_PASS_MAX_V = 4096
# 分块路径 tile 大小（单 tile fp32 约 8KB，远小于 UB 预算）。
_TILED_BLOCK = 2048
_MIN_BLOCK = 16
_NUM_VECTORCORE = -1


def _get_npu_vectorcore_num() -> int:
    """当前设备的 vector core 数（910B3 为 40）。"""
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE <= 0:
        device = torch.npu.current_device()
        _NUM_VECTORCORE = int(triton.runtime.driver.active.utils.get_device_properties(device)["num_vectorcore"])
    return _NUM_VECTORCORE


@triton.jit
def _apply_topk_topp_single_pass_kernel(
    sv_ptr,  # *sortedValue [B, V]，升序
    si_ptr,  # *sortedIndices [B, V]，int32/int64 → 原始词表下标
    k_ptr,  # *k [B]，int32
    p_ptr,  # *p [B]，fp32
    out_ptr,  # *out [B, V]，原序，预填 -inf（FUSED_SOFTMAX 时预填 0）
    B,
    V,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,  # B*V < 2^31 时用 int32 行基址
    FUSED_SOFTMAX: tl.constexpr,  # True: 散写概率 exp/E_kept；False: 散写原始 logit
):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, B)
    offs = tl.arange(0, BLOCK)

    for row in range(row_start, row_end):
        mask = offs < V
        if USE_I32:
            row_base = row * V
        else:
            row_base = row.to(tl.int64) * V

        # topKValue[b] = sortedValue[b][V - k[b]]（下标钳位到 [0, V-1]）
        k_b = tl.load(k_ptr + row)
        k_idx = tl.minimum(tl.maximum(V - k_b, 0), V - 1)
        topk_value = tl.load(sv_ptr + row_base + k_idx).to(tl.float32)

        s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        # 升序排列的全局最大值在末尾，作为 softmax 的数值稳定项
        s_max = tl.load(sv_ptr + row_base + V - 1).to(tl.float32)

        # topKMask 置 -inf 后的 softmax（fp32），概率只用于求 top-p 门限
        topk_mask = s < topk_value
        e = tl.where(mask & ~topk_mask, tl.exp(s - s_max), 0.0)
        probs = e / tl.sum(e, axis=0)

        # kept(v) = ~topKMask(v) AND (probsSum(v) > 1-p[b] OR v == V-1)。
        # 两个条件都是后缀单调，等价于 kept(v) ⟺ v >= m；用标量边界 + 比较型
        # mask（与 tiled kernel 同构），规避 cumsum 派生布尔向量直接做
        # scatter mask 在 triton-ascend 下的误编译。
        p_b = tl.load(p_ptr + row)
        cum = tl.cumsum(probs, axis=0)
        m_p = tl.min(tl.where(mask & (cum > 1.0 - p_b), offs, V), axis=0)
        m_k = tl.min(tl.where(mask & ~topk_mask, offs, V), axis=0)
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        # 仅散写存活元素；非存活位置由 out 的预填（-inf 或 0）覆盖
        kept = (offs >= m) & mask
        idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
        if FUSED_SOFTMAX:
            # 融合最终 softmax：分母 = kept 集合的 exp 和（masked 位预填 0）
            e_kept = tl.sum(tl.where(kept, e, 0.0), axis=0)
            val = tl.where(kept, e / e_kept, 0.0)
        else:
            val = s
        tl.store(out_ptr + row_base + idx, val, mask=kept)


@triton.jit
def _apply_topk_topp_tiled_kernel(
    sv_ptr,
    si_ptr,
    k_ptr,
    p_ptr,
    out_ptr,
    B,
    V,
    rows_per_prog,
    BLOCK: tl.constexpr,
    USE_I32: tl.constexpr,
    FUSED_SOFTMAX: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * rows_per_prog
    row_end = tl.minimum(row_start + rows_per_prog, B)
    offs_base = tl.arange(0, BLOCK)

    for row in range(row_start, row_end):
        if USE_I32:
            row_base = row * V
        else:
            row_base = row.to(tl.int64) * V

        k_b = tl.load(k_ptr + row)
        p_b = tl.load(p_ptr + row)
        thr_p = 1.0 - p_b
        k_idx = tl.minimum(tl.maximum(V - k_b, 0), V - 1)
        topk_value = tl.load(sv_ptr + row_base + k_idx).to(tl.float32)
        s_max = tl.load(sv_ptr + row_base + V - 1).to(tl.float32)

        # pass 1: softmax 分母 E = sum(exp(s - s_max))（top-k 存活位），
        # 以及 top-k 存活下界 m_k（升序 + 严格小于阈值 ⇒ 存活集是后缀）
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        m_k = V
        for t in range(0, V, BLOCK):
            offs = t + offs_base
            mask = offs < V
            s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            alive = mask & (s >= topk_value)
            acc += tl.where(alive, tl.exp(s - s_max), 0.0)
            m_k = tl.minimum(m_k, tl.min(tl.where(alive, offs, V), axis=0))
        E = tl.sum(acc, axis=0)

        # pass 2: m_p = 升序 cumsum 首次 > 1-p[b] 的下标（不存在则为 V）
        running = 0.0
        m_p = V
        for t in range(0, V, BLOCK):
            if m_p == V:
                offs = t + offs_base
                mask = offs < V
                s = tl.load(sv_ptr + row_base + offs, mask=mask, other=-float("inf")).to(tl.float32)
                probs = tl.where(mask & (s >= topk_value), tl.exp(s - s_max), 0.0) / E
                tile_sum = tl.sum(probs, axis=0)
                if running + tile_sum > thr_p:
                    c = running + tl.cumsum(probs, axis=0)
                    hit = tl.min(tl.where(c > thr_p, offs_base, BLOCK), axis=0)
                    m_p = tl.where(hit < BLOCK, t + hit, V)
                running += tile_sum

        # kept(v) = [s>=topk_value] AND [probsSum > 1-p OR v == V-1] 两个条件
        # 都是后缀单调，等价于 kept(v) ⟺ v >= m
        m = tl.minimum(tl.maximum(m_p, m_k), V - 1)

        # pass 3: 仅处理存活后缀，其余位置保持预填（-inf 或 0）
        t_start = (m // BLOCK) * BLOCK

        if FUSED_SOFTMAX:
            # pass 3a: 融合 softmax 的分母 = kept 后缀的 exp 和
            e_acc = tl.zeros([BLOCK], dtype=tl.float32)
            for t in range(t_start, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                e_acc += tl.where(kept, tl.exp(s - s_max), 0.0)
            e_inv = 1.0 / tl.sum(e_acc, axis=0)

            # pass 3b: 散写概率 exp(s - s_max) / E_kept
            for t in range(t_start, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=-float("inf")).to(tl.float32)
                idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
                tl.store(out_ptr + row_base + idx, tl.exp(s - s_max) * e_inv, mask=kept)
        else:
            # 散写原始 logit（masked logits，CANN 语义）
            for t in range(t_start, V, BLOCK):
                offs = t + offs_base
                kept = (offs >= m) & (offs < V)
                s = tl.load(sv_ptr + row_base + offs, mask=kept, other=0.0).to(tl.float32)
                idx = tl.load(si_ptr + row_base + offs, mask=kept, other=0)
                tl.store(out_ptr + row_base + idx, s, mask=kept)


def _validate_and_prepare(sorted_values: torch.Tensor, sorted_indices: torch.Tensor):
    """校验并规整 sorted 输入，返回 (sv, si, B, V)。"""
    if sorted_values.dim() != 2:
        raise ValueError(f"sorted_values must be 2D [B, V], got {sorted_values.dim()}D")
    if sorted_values.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"sorted_values only supports float32/bfloat16/float16, got {sorted_values.dtype}")
    if sorted_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"sorted_indices only supports int32/int64, got {sorted_indices.dtype}")
    if sorted_values.shape != sorted_indices.shape:
        raise ValueError(f"shape mismatch: {sorted_values.shape} vs {sorted_indices.shape}")
    if sorted_values.numel() == 0:
        raise ValueError("input tensor must not be empty")
    if sorted_values.device != sorted_indices.device:
        raise ValueError(f"device mismatch: {sorted_values.device} vs {sorted_indices.device}")
    if sorted_values.device.type != "npu":
        raise ValueError(f"input must be on npu, got {sorted_values.device}")

    sv = sorted_values if sorted_values.is_contiguous() else sorted_values.contiguous()
    si = sorted_indices if sorted_indices.is_contiguous() else sorted_indices.contiguous()
    B, V = sv.shape
    if V >= 2**31:
        raise ValueError(f"vocab size {V} too large (>= 2^31)")
    return sv, si, B, V


def _normalize_k(k, B: int, V: int, device) -> torch.Tensor:
    """k 规整为 [B] int32 设备张量；None → V（禁用 top-k）。

    vllm 传入的 [B] int32 张量走 no-op（不产生额外算子调用）。
    """
    if k is None:
        return torch.full((B,), V, dtype=torch.int32, device=device)
    if isinstance(k, bool) or not isinstance(k, (int, float, torch.Tensor)):
        raise TypeError(f"k must be None or int or [B] tensor, got {type(k).__name__}")
    if isinstance(k, float):
        if not k.is_integer():
            raise ValueError(f"k must be an integer, got {k}")
        k = int(k)
    if isinstance(k, int):
        return torch.full((B,), k, dtype=torch.int32, device=device)
    if k.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"k tensor only supports int32/int64, got {k.dtype}")
    if k.numel() == 1:
        k = k.reshape(1).expand(B)
    if k.shape != (B,):
        raise ValueError(f"k tensor must have shape [{B}], got {tuple(k.shape)}")
    return k.to(device=device, dtype=torch.int32).contiguous()


def _normalize_p(p, B: int, device) -> torch.Tensor:
    """p 规整为 [B] fp32 设备张量；None → 1.0（禁用 top-p）。

    vllm 传入的 [B] fp32 张量走 no-op（不产生额外算子调用）。
    """
    if p is None:
        return torch.ones((B,), dtype=torch.float32, device=device)
    if isinstance(p, bool) or not isinstance(p, (int, float, torch.Tensor)):
        raise TypeError(f"p must be None or float or [B] tensor, got {type(p).__name__}")
    if isinstance(p, (int, float)):
        return torch.full((B,), float(p), dtype=torch.float32, device=device)
    if not p.is_floating_point():
        raise ValueError(f"p tensor must be floating point, got {p.dtype}")
    if p.numel() == 1:
        p = p.reshape(1).expand(B)
    if p.shape != (B,):
        raise ValueError(f"p tensor must have shape [{B}], got {tuple(p.shape)}")
    return p.to(device=device, dtype=torch.float32).contiguous()


def _launch(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k,
    p,
    fused_softmax: bool,
) -> torch.Tensor:
    """校验输入、规整 k/p、预填 out 并下发 kernel。"""
    sv, si, B, V = _validate_and_prepare(sorted_values, sorted_indices)
    k_t = _normalize_k(k, B, V, sv.device)
    p_t = _normalize_p(p, B, sv.device)

    if fused_softmax:
        # 概率输出固定 fp32（对齐 softmax(dtype=fp32)），masked 位预填 0
        out = torch.zeros((B, V), device=sv.device, dtype=torch.float32)
    else:
        # 过滤位整块预填 -inf（连续大块写），kernel 只散写存活元素
        out = torch.full_like(sv, float("-inf"))

    core_num = _get_npu_vectorcore_num()
    if core_num >= B:
        grid, rows_per_prog = (B,), 1
    else:
        grid, rows_per_prog = (core_num,), triton.cdiv(B, core_num)

    use_i32 = B * V < 2**31
    # si 不做 host 侧 int32 cast（省一次全词表拷贝）；kernel 内加载后由指针
    # 运算自动提升（int64 下标值 < V < 2^31，值域安全）。

    if V <= _SINGLE_PASS_MAX_V:
        block = max(triton.next_power_of_2(V), _MIN_BLOCK)
        _apply_topk_topp_single_pass_kernel[grid](
            sv,
            si,
            k_t,
            p_t,
            out,
            B,
            V,
            rows_per_prog,
            BLOCK=block,
            USE_I32=use_i32,
            FUSED_SOFTMAX=fused_softmax,
        )
    else:
        _apply_topk_topp_tiled_kernel[grid](
            sv,
            si,
            k_t,
            p_t,
            out,
            B,
            V,
            rows_per_prog,
            BLOCK=_TILED_BLOCK,
            USE_I32=use_i32,
            FUSED_SOFTMAX=fused_softmax,
        )
    return out


def _sort_ascending(logits: torch.Tensor):
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [B, V], got {logits.dim()}D")
    if logits.device.type != "npu":
        raise ValueError(f"input must be on npu, got {logits.device}")
    return torch.sort(logits, dim=-1, descending=False, stable=True)


def apply_top_k_top_p_with_sorted(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """在升序 sorted 输入上做 top-k/top-p 过滤，返回原序 masked logits。

    Args:
        sorted_values: [B, V] 升序 sorted logits（torch.sort 的 values）。
        sorted_indices: [B, V] 对应的 sort indices（int32/int64）。
        k: top-k 保留数，标量或 [B] 张量；k[b] >= V 不过滤，None 禁用。
        p: top-p 阈值，标量或 [B] 张量，取值 [0, 1]；None 禁用。
    Returns:
        [B, V] 原序 masked logits：保留位为原始 logit，过滤位为 -inf。
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=False)


def apply_top_k_top_p(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """端到端接口：升序稳定排序 + top-k/top-p 过滤，返回原序 masked logits。

    Args:
        logits: [B, V] 原始 logits（float32/bfloat16/float16）。
        k / p: 同 :func:`apply_top_k_top_p_with_sorted`。
    Returns:
        [B, V] 原序 masked logits：保留位为原始 logit，过滤位为 -inf。
    """
    return apply_top_k_top_p_with_sorted(*_sort_ascending(logits), k, p)


def fused_topk_topp_softmax_with_sorted(
    sorted_values: torch.Tensor,
    sorted_indices: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """sorted 输入 + top-k/top-p 过滤 + 融合 softmax，返回原序概率分布。

    等价于 ``apply_top_k_top_p_with_sorted(...).softmax(dim=-1, dtype=float32)``
    （masked 位 -inf → 概率 0），但 softmax 直接在 kernel 内完成：
    存活元素按 exp(s - s_max) / E_kept 散写，非存活位预填 0。

    Returns:
        [B, V] fp32 概率分布：masked 位 = 0，每行和 = 1。
    """
    return _launch(sorted_values, sorted_indices, k, p, fused_softmax=True)


def fused_topk_topp_softmax(
    logits: torch.Tensor,
    k=None,
    p=None,
) -> torch.Tensor:
    """端到端接口：升序稳定排序 + top-k/top-p 过滤 + 融合 softmax。

    与生产链路 ``torch_npu.npu_top_k_top_p(logits, k, p).softmax(dim=-1,
    dtype=float32)`` 契约一致。

    Args:
        logits: [B, V] 原始 logits（float32/bfloat16/float16）。
        k / p: 同 :func:`apply_top_k_top_p_with_sorted`。
    Returns:
        [B, V] fp32 概率分布：masked 位 = 0，每行和 = 1。
    """
    return fused_topk_topp_softmax_with_sorted(*_sort_ascending(logits), k, p)
