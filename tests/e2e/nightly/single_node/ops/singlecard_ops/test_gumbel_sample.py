"""
GumbelSample 算子 e2e 测试
测试 torch.ops._C_ascend.npu_gumbel_sample 与 CPU golden 的精度对印。

输入签名（正确语义）：
  logits      [num_tokens, vocab_size]  FP32  — 每个 token slot 一行，按 batch_idx 直接索引
  temperature [num_req_states]          FP32  — 每 req_state 温度系数
  seeds       [num_req_states]          INT64 — 每 req_state 随机种子
  pos         [num_tokens]              INT64 — 每 token slot 位置，按 batch_idx 直接索引
  idx_mapping [num_tokens]              INT32 — idx_mapping[batch_idx] = req_state_idx，
                                                仅用于索引 temperature/seeds

运行方式：
    pytest -v -s tests/e2e/nightly/single_node/ops/singlecard_ops/test_gumbel_sample.py

每个 case 会打印 NPU 侧延迟（warmup=3, iters=20）。
"""

import time

import numpy as np
import pytest
import torch
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# ============================================================
# 性能测量常量
# ============================================================
PERF_WARMUP = 3
PERF_ITERS = 20

# ============================================================
# Philox4x32-10 CPU golden
#   与 op_kernel/gumbel_sample.h 中的 Philox4x32 实现字节级对齐：
#   - mulhi 使用有符号乘法（triton-ascend NPU 后端将 tt.mulhiui 编译为 smulhi）
#   - uint_to_uniform_float：abs_signed(raw) * 4.6566127342e-10
# ============================================================
PHILOX_KEY_A   = np.uint32(0x9E3779B9)
PHILOX_KEY_B   = np.uint32(0xBB67AE85)
PHILOX_ROUND_A = np.uint32(0xD2511F53)
PHILOX_ROUND_B = np.uint32(0xCD9E8D57)
PHILOX_ROUNDS  = 10
PHILOX_FLOAT_SCALE = np.float32(4.6566127342e-10)
GUMBEL_EPS     = np.float32(1e-20)


def _smulhi(a: np.uint32, b: np.uint32) -> np.uint32:
    """有符号 32×32→64 乘法，取高 32 位（与 triton-ascend NPU smulhi 对齐）。"""
    prod = int(np.int32(a)) * int(np.int32(b))
    return np.uint32((prod >> 32) & 0xFFFFFFFF)


def _philox4x32(c0: int, k0: int, k1: int) -> np.uint32:
    """Philox4x32-10 单元素实现，返回 c0 输出（与 kernel Philox4x32 完全对齐）。"""
    c0 = np.uint32(c0)
    c1 = np.uint32(0)
    c2 = np.uint32(0)
    c3 = np.uint32(0)
    k0 = np.uint32(k0)
    k1 = np.uint32(k1)
    for _ in range(PHILOX_ROUNDS):
        hi_b = _smulhi(PHILOX_ROUND_B, c2)
        hi_a = _smulhi(PHILOX_ROUND_A, c0)
        new_c0 = hi_b ^ c1 ^ k0
        new_c2 = hi_a ^ c3 ^ k1
        new_c1 = np.uint32(np.int32(PHILOX_ROUND_B) * np.int32(c2))
        new_c3 = np.uint32(np.int32(PHILOX_ROUND_A) * np.int32(c0))
        c0, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3
        k0 = np.uint32(k0 + PHILOX_KEY_A)
        k1 = np.uint32(k1 + PHILOX_KEY_B)
    return c0


def _philox_to_uniform(raw: np.uint32) -> np.float32:
    """uint32 → uniform float，与 kernel 中 abs_signed * PHILOX_FLOAT_SCALE 对齐。"""
    sx = int(np.int32(raw))
    if sx < 0:
        sx = -sx - 1
    return np.float32(sx) * PHILOX_FLOAT_SCALE


def golden_gumbel_sample(
    logits: np.ndarray,
    temperature: np.ndarray,
    seeds: np.ndarray,
    pos: np.ndarray,
    idx_mapping: np.ndarray,
    apply_temperature: bool = True,
) -> np.ndarray:
    """
    CPU golden：返回 sampled[num_tokens]（int64）。
    与 op_kernel/gumbel_sample.h ProcessOneRow 完全对齐：
      logits[num_tokens, vocab_size]，pos[num_tokens]，idx_mapping[num_tokens]
      temperature[num_req_states]，seeds[num_req_states]

    对每个 batch_idx（token slot）：
      1. req_state_idx = idx_mapping[batch_idx]  — 仅用于 temp/seed 索引
      2. temp = temperature[req_state_idx]
      3. seed64 = seeds[req_state_idx]
      4. pos_i32 = pos[batch_idx]  — 直接索引
      5. gumbelSeed = Philox4x32(c0=pos_i32, k0=seedLo, k1=seedHi)
      6. 对每个 vocabIdx: raw = Philox4x32(c0=vocabIdx, k0=gumbelSeed, k1=0)
      7. u = abs_signed(raw) * PHILOX_FLOAT_SCALE
      8. g = -ln(-ln(u + eps) + eps)
      9. sampled = argmax(logits[batch_idx]/temp + g)  — logits 直接按 batch_idx 索引
    """
    num_tokens = logits.shape[0]
    vocab_size = logits.shape[1]
    sampled = np.zeros(num_tokens, dtype=np.int64)
    for batch_idx in range(num_tokens):
        req_state_idx = int(idx_mapping[batch_idx])
        temp = float(temperature[req_state_idx])
        seed64 = int(seeds[req_state_idx]) & 0xFFFFFFFFFFFFFFFF
        pos_i32 = int(pos[batch_idx]) & 0xFFFFFFFF  # pos 直接按 batch_idx 索引

        is_greedy = (temp == 0.0)
        logits_row = logits[batch_idx].astype(np.float32).copy()  # logits 直接按 batch_idx 索引

        if not is_greedy:
            if apply_temperature:
                logits_row = logits_row / np.float32(temp)

            seed_lo = np.uint32(seed64 & 0xFFFFFFFF)
            seed_hi = np.uint32((seed64 >> 32) & 0xFFFFFFFF)
            gumbel_seed = _philox4x32(pos_i32, int(seed_lo), int(seed_hi))

            g = np.empty(vocab_size, dtype=np.float32)
            for i in range(vocab_size):
                raw = _philox4x32(i, int(gumbel_seed), 0)
                u = _philox_to_uniform(raw)
                u = u + GUMBEL_EPS
                g[i] = -np.log(-np.log(u) + GUMBEL_EPS)

            logits_row = logits_row + g

        sampled[batch_idx] = int(np.argmax(logits_row))
    return sampled


# ============================================================
# 辅助：构造输入 + 调用 NPU 算子 + 打印延迟
# ============================================================
def _make_inputs(num_tokens: int, vocab_size: int, num_req_states: int = None,
                 temp_val: float = 1.0, seed_base: int = 42, pos_base: int = 0):
    """
    构造 GumbelSample 输入张量（CPU numpy）。
    num_tokens:     logits/pos/idx_mapping 的行数（= grid size）
    num_req_states: temperature/seeds 的大小（默认 = num_tokens，即每 token 一个 req_state）
    idx_mapping:    默认为 [0, 1, ..., num_tokens-1] % num_req_states（循环映射）

    参数自洽检查：num_tokens >= 1, vocab_size >= 1, num_req_states >= 1
    """
    assert num_tokens >= 1, f"num_tokens must be >= 1, got {num_tokens}"
    assert vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}"
    if num_req_states is None:
        num_req_states = num_tokens
    assert num_req_states >= 1, f"num_req_states must be >= 1, got {num_req_states}"

    rng = np.random.default_rng(seed_base + 1000)
    logits_np      = rng.standard_normal((num_tokens, vocab_size)).astype(np.float32)
    temp_np        = np.full(num_req_states, temp_val, dtype=np.float32)
    seeds_np       = np.arange(seed_base, seed_base + num_req_states, dtype=np.int64)
    pos_np         = np.arange(pos_base, pos_base + num_tokens, dtype=np.int64)
    # idx_mapping[batch_idx] = batch_idx % num_req_states（循环映射，保证在 [0, num_req_states) 内）
    idx_mapping_np = np.arange(num_tokens, dtype=np.int32) % num_req_states
    return logits_np, temp_np, seeds_np, pos_np, idx_mapping_np


def _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
             apply_temperature: bool, label: str = ""):
    """调用 NPU 算子，含 warmup + 多次平均延迟打印。"""
    logits_npu      = torch.from_numpy(logits_np).npu()
    temp_npu        = torch.from_numpy(temp_np).npu()
    seeds_npu       = torch.from_numpy(seeds_np).npu()
    pos_npu         = torch.from_numpy(pos_np).npu()
    idx_mapping_npu = torch.from_numpy(idx_mapping_np).npu()

    for _ in range(PERF_WARMUP):
        _ = torch.ops._C_ascend.npu_gumbel_sample(
            logits_npu, temp_npu, seeds_npu, pos_npu, idx_mapping_npu, apply_temperature)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        out = torch.ops._C_ascend.npu_gumbel_sample(
            logits_npu, temp_npu, seeds_npu, pos_npu, idx_mapping_npu, apply_temperature)
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / PERF_ITERS

    num_tokens     = logits_np.shape[0]
    num_req_states = temp_np.shape[0]
    vocab          = logits_np.shape[1]
    tag = (f"num_tokens={num_tokens}, num_req_states={num_req_states}, "
           f"vocab={vocab}, apply_temp={apply_temperature}")
    if label:
        tag = f"{label} | {tag}"
    print(f"  [perf] {tag} → {elapsed_us:.1f} μs/call", flush=True)

    return out.cpu().numpy()


# ============================================================
# Group 1: basic — 基本功能（num_tokens = num_req_states，1:1 映射）
# ============================================================
@pytest.mark.parametrize("num_tokens,vocab_size", [
    (1,   32000),   # 最小 batch
    (4,   32000),   # 小 batch
    (16,  32000),   # 中 batch
    (32,  128256),  # 大 vocab（Llama-3 规模）
])
@torch.inference_mode()
def test_gumbel_sample_basic(num_tokens, vocab_size):
    logits_np, temp_np, seeds_np, pos_np, idx_mapping_np = _make_inputs(num_tokens, vocab_size)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True, label="basic")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 2: idx_mapping 间接寻址 — num_tokens > num_req_states
#   多个 token slot 共享同一 req_state（典型 prefill 场景）
# ============================================================
@pytest.mark.parametrize("num_tokens,num_req_states,vocab_size", [
    (8,   4,   32000),   # 每 req_state 平均 2 个 token
    (16,  8,   32000),   # 同上，更大 batch
    (16,  4,   32000),   # 稀疏映射（4 个 req_state 覆盖 16 个 token）
    (32,  8,   128256),  # 大 vocab + 多 token
    (4,   1,   32000),   # 所有 token 共享同一 req_state
])
@torch.inference_mode()
def test_gumbel_sample_idx_mapping(num_tokens, num_req_states, vocab_size):
    rng = np.random.default_rng(777)
    logits_np      = rng.standard_normal((num_tokens, vocab_size)).astype(np.float32)
    temp_np        = np.full(num_req_states, 1.0, dtype=np.float32)
    seeds_np       = np.arange(num_req_states, dtype=np.int64)
    pos_np         = np.arange(num_tokens, dtype=np.int64)
    # idx_mapping: 每个 token slot 随机映射到一个 req_state（在 [0, num_req_states) 内）
    idx_mapping_np = rng.integers(0, num_req_states, size=num_tokens).astype(np.int32)

    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True, label="idx_mapping")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 3: apply_temperature=False（跳过缩放，TilingKey=0 分支）
# ============================================================
@pytest.mark.parametrize("num_tokens,vocab_size", [
    (1,   32000),
    (8,   32000),
    (16,  128256),
    (32,  32000),
    (64,  32000),
])
@torch.inference_mode()
def test_gumbel_sample_no_temperature(num_tokens, vocab_size):
    logits_np, temp_np, seeds_np, pos_np, idx_mapping_np = _make_inputs(
        num_tokens, vocab_size, temp_val=1.0)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=False)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=False, label="no_temp")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 4: 属性组合 — 不同 temperature 值 × apply_temperature 组合
# ============================================================
@pytest.mark.parametrize("temp_val,apply_temperature", [
    (0.5,  True),   # 低温（更尖锐分布）
    (1.0,  True),   # 标准温度
    (2.0,  True),   # 高温（更平坦分布）
    (0.1,  True),   # 极低温（接近 greedy）
    (10.0, True),   # 极高温（接近均匀）
    (0.5,  False),  # 低温但跳过缩放
    (2.0,  False),  # 高温但跳过缩放
])
@torch.inference_mode()
def test_gumbel_sample_attr_combinations(temp_val, apply_temperature):
    num_tokens, vocab_size = 8, 32000
    logits_np, _, seeds_np, pos_np, idx_mapping_np = _make_inputs(
        num_tokens, vocab_size, temp_val=temp_val)
    temp_np = np.full(num_tokens, temp_val, dtype=np.float32)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature,
                      label=f"attr temp={temp_val} apply={apply_temperature}")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 5: large — vLLM 真实负载规模
# ============================================================
@pytest.mark.parametrize("num_tokens,num_req_states,vocab_size,label", [
    (1,   1,   32000,  "prefill_single_req"),    # prefill：单请求
    (256, 256, 32000,  "decode_large_batch"),    # decode：大 batch
    (512, 512, 32000,  "decode_max_batch"),      # decode：最大 batch
    (1,   1,   128256, "prefill_llama3_vocab"),  # prefill：Llama-3 vocab
    # prefill 场景：多 token 共享少量 req_state（典型 continuous batching）
    (128, 8,   32000,  "prefill_multi_token"),
])
@torch.inference_mode()
def test_gumbel_sample_large(num_tokens, num_req_states, vocab_size, label):
    logits_np, temp_np, seeds_np, pos_np, idx_mapping_np = _make_inputs(
        num_tokens, vocab_size, num_req_states=num_req_states)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True, label=label)
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 6: boundary — 数值 / shape 边界
# ============================================================
@pytest.mark.parametrize("case_name,num_tokens,vocab_size,temp_val,seed_base,pos_base", [
    # 最小规模
    ("min_scale",           1,   1,     1.0,  0,    0),
    # vocab 恰好 = BLOCK_SIZE（4096，整数倍对齐）
    ("vocab_eq_block",      4,   4096,  1.0,  1,    0),
    # vocab 恰好 = 2×BLOCK_SIZE
    ("vocab_2x_block",      4,   8192,  1.0,  2,    0),
    # vocab 非整数倍（4096+1）
    ("vocab_non_aligned",   4,   4097,  1.0,  3,    0),
    # seed=0（边界值）
    ("seed_zero",           8,   32000, 1.0,  0,    0),
    # pos=0（边界值）
    ("pos_zero",            8,   32000, 1.0,  42,   0),
    # 极大 pos（接近 int32 上限）
    ("pos_large",           4,   32000, 1.0,  42,   2147483647 - 4),
    # 极大 seed（接近 int64 上限）
    ("seed_large",          4,   32000, 1.0,  9223372036854775800, 0),
    # 混合 batch：num_tokens 不整除 usedCoreNum（假设 20 核）
    ("batch_non_divisible", 21,  32000, 1.0,  7,    0),
    # num_tokens > num_req_states（多 token 共享 req_state）
    ("multi_token_per_req", 16,  32000, 1.0,  5,    0),
])
@torch.inference_mode()
def test_gumbel_sample_boundary(case_name, num_tokens, vocab_size, temp_val, seed_base, pos_base):
    num_req_states = max(1, num_tokens // 4) if num_tokens > 4 else num_tokens
    logits_np, _, seeds_np, pos_np, idx_mapping_np = _make_inputs(
        num_tokens, vocab_size, num_req_states=num_req_states,
        temp_val=temp_val, seed_base=seed_base, pos_base=pos_base)
    temp_np = np.full(num_req_states, temp_val, dtype=np.float32)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True, label=f"boundary/{case_name}")
    np.testing.assert_array_equal(actual, golden)
