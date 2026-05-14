"""
GumbelSample 算子 e2e 测试
测试 torch.ops._C_ascend.npu_gumbel_sample 与 CPU golden 的精度对比。

输入签名：
  logits      [num_tokens, vocab_size]  FP32
  temperature [num_req_states]          FP32
  seeds       [num_req_states]          INT64
  pos         [num_tokens]              INT64
  idx_mapping [num_tokens]              INT32 — idx_mapping[i] = req_state_idx

测试用例：
  case1: num_tokens=num_req_states=10, vocab=1500, apply_temp=True,  temp=0.6, idx_mapping=全0
  case2: num_tokens=20, num_req_states=2, vocab=1500, apply_temp=True,  temp=0.6, idx_mapping=[0]*10+[1]*10
  case3: num_tokens=num_req_states=10, vocab=1500, apply_temp=False, idx_mapping=全0

运行方式：
    pytest -v -s tests/e2e/nightly/single_node/ops/singlecard_ops/test_gumbel_sample.py

仅验证精度，不测量性能。
"""

import numpy as np
import pytest
import torch
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

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
    c0_u32 = np.uint32(c0)
    c1 = np.uint32(0)
    c2 = np.uint32(0)
    c3 = np.uint32(0)
    k0_u32 = np.uint32(k0)
    k1_u32 = np.uint32(k1)
    for _ in range(PHILOX_ROUNDS):
        hi_b = _smulhi(PHILOX_ROUND_B, c2)
        hi_a = _smulhi(PHILOX_ROUND_A, c0_u32)
        new_c0 = hi_b ^ c1 ^ k0_u32
        new_c2 = hi_a ^ c3 ^ k1_u32
        new_c1 = np.uint32(np.int32(PHILOX_ROUND_B) * np.int32(c2))
        new_c3 = np.uint32(np.int32(PHILOX_ROUND_A) * np.int32(c0_u32))
        c0_u32, c1, c2, c3 = new_c0, new_c1, new_c2, new_c3
        k0_u32 = np.uint32(k0_u32 + PHILOX_KEY_A)
        k1_u32 = np.uint32(k1_u32 + PHILOX_KEY_B)
    return c0_u32


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
def _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
             apply_temperature: bool):
    """调用 NPU 算子，仅验证精度，不测量性能。"""
    logits_npu      = torch.from_numpy(logits_np).npu()
    temp_npu        = torch.from_numpy(temp_np).npu()
    seeds_npu       = torch.from_numpy(seeds_np).npu()
    pos_npu         = torch.from_numpy(pos_np).npu()
    idx_mapping_npu = torch.from_numpy(idx_mapping_np).npu()

    out = torch.ops._C_ascend.npu_gumbel_sample(
        logits_npu, temp_npu, seeds_npu, pos_npu, idx_mapping_npu, apply_temperature)
    torch.npu.synchronize()

    return out.cpu().numpy()


# ============================================================
# Case 1: num_tokens=num_req_states=10, vocab=1500, apply_temp=True, temp=0.6
#   idx_mapping 全 0（10 个 token 全部映射到 req_state 0）
# ============================================================
@torch.inference_mode()
def test_gumbel_sample_case1():
    num_tokens     = 10
    num_req_states = 10
    vocab_size     = 1500
    rng = np.random.default_rng(2025)
    logits_np      = np.random.default_rng(1 * 7919).standard_normal(
                         (num_tokens, vocab_size)).astype(np.float32)
    temp_np        = np.full(num_req_states, 0.6, dtype=np.float32)
    seeds_np       = rng.integers(0, 2**31, size=num_req_states, dtype=np.int64)
    pos_np         = np.arange(num_tokens, dtype=np.int64)
    idx_mapping_np = np.zeros(num_tokens, dtype=np.int32)

    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True)
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Case 2: num_tokens=20, num_req_states=2, vocab=1500, apply_temp=True, temp=0.6
#   idx_mapping 前 10 为 0，后 10 为 1
# ============================================================
@torch.inference_mode()
def test_gumbel_sample_case2():
    num_tokens     = 20
    num_req_states = 2
    vocab_size     = 1500
    rng = np.random.default_rng(2025)
    # 消耗与 case1 相同数量的随机数以保持 seed 序列一致
    rng.integers(0, 2**31, size=10, dtype=np.int64)
    logits_np      = np.random.default_rng(2 * 7919).standard_normal(
                         (num_tokens, vocab_size)).astype(np.float32)
    temp_np        = np.full(num_req_states, 0.6, dtype=np.float32)
    seeds_np       = rng.integers(0, 2**31, size=num_req_states, dtype=np.int64)
    pos_np         = np.arange(num_tokens, dtype=np.int64)
    idx_mapping_np = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=True)
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Case 3: num_tokens=num_req_states=10, vocab=1500, apply_temp=False
#   idx_mapping 全 0
# ============================================================
@torch.inference_mode()
def test_gumbel_sample_case3():
    num_tokens     = 10
    num_req_states = 10
    vocab_size     = 1500
    rng = np.random.default_rng(2025)
    # 消耗与 case1+case2 相同数量的随机数
    rng.integers(0, 2**31, size=10, dtype=np.int64)
    rng.integers(0, 2**31, size=2,  dtype=np.int64)
    logits_np      = np.random.default_rng(3 * 7919).standard_normal(
                         (num_tokens, vocab_size)).astype(np.float32)
    temp_np        = np.ones(num_req_states, dtype=np.float32)
    seeds_np       = rng.integers(0, 2**31, size=num_req_states, dtype=np.int64)
    pos_np         = np.arange(num_tokens, dtype=np.int64)
    idx_mapping_np = np.zeros(num_tokens, dtype=np.int32)

    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                                   apply_temperature=False)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, idx_mapping_np,
                      apply_temperature=False)
    np.testing.assert_array_equal(actual, golden)
