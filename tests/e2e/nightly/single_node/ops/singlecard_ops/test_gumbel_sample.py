"""
GumbelSample 算子 e2e 测试
测试 torch.ops._C_ascend.npu_gumbel_sample 与 CPU golden 的精度对印。

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
    apply_temperature: bool = True,
) -> np.ndarray:
    """
    CPU golden：返回 sampled[num_reqs]（int64）。
    与 op_kernel/gumbel_sample.h ProcessOneRow 完全对齐：
      1. gumbelSeed = Philox4x32(c0=pos_i32, k0=seedLo, k1=seedHi)
      2. 对每个 vocabIdx: raw = Philox4x32(c0=vocabIdx, k0=gumbelSeed, k1=0)
      3. u = abs_signed(raw) * PHILOX_FLOAT_SCALE
      4. g = -ln(-ln(u + eps) + eps)
      5. sampled = argmax(logits/temp + g)
    """
    num_reqs, vocab_size = logits.shape
    sampled = np.zeros(num_reqs, dtype=np.int64)
    for r in range(num_reqs):
        temp = float(temperature[r])
        seed64 = int(seeds[r]) & 0xFFFFFFFFFFFFFFFF
        pos_i32 = int(pos[r]) & 0xFFFFFFFF

        is_greedy = (temp == 0.0)
        logits_r = logits[r].astype(np.float32).copy()

        if not is_greedy:
            if apply_temperature:
                logits_r = logits_r / np.float32(temp)

            seed_lo = np.uint32(seed64 & 0xFFFFFFFF)
            seed_hi = np.uint32((seed64 >> 32) & 0xFFFFFFFF)
            gumbel_seed = _philox4x32(pos_i32, int(seed_lo), int(seed_hi))

            g = np.empty(vocab_size, dtype=np.float32)
            for i in range(vocab_size):
                raw = _philox4x32(i, int(gumbel_seed), 0)
                u = _philox_to_uniform(raw)
                u = u + GUMBEL_EPS
                g[i] = -np.log(-np.log(u) + GUMBEL_EPS)

            logits_r = logits_r + g

        sampled[r] = int(np.argmax(logits_r))
    return sampled


# ============================================================
# 辅助：构造输入 + 调用 NPU 算子 + 打印延迟
# ============================================================
def _make_inputs(num_reqs: int, vocab_size: int, temp_val: float = 1.0,
                 seed_base: int = 42, pos_base: int = 0):
    """
    构造 GumbelSample 输入张量（CPU numpy）。
    参数自洽检查：num_reqs >= 1, vocab_size >= 1。
    """
    assert num_reqs >= 1, f"num_reqs must be >= 1, got {num_reqs}"
    assert vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}"

    rng = np.random.default_rng(seed_base + 1000)
    logits_np = rng.standard_normal((num_reqs, vocab_size)).astype(np.float32)
    temp_np   = np.full(num_reqs, temp_val, dtype=np.float32)
    seeds_np  = np.arange(seed_base, seed_base + num_reqs, dtype=np.int64)
    pos_np    = np.arange(pos_base, pos_base + num_reqs, dtype=np.int64)
    return logits_np, temp_np, seeds_np, pos_np


def _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature: bool,
             label: str = ""):
    """调用 NPU 算子，含 warmup + 多次平均延迟打印。"""
    logits_npu = torch.from_numpy(logits_np).npu()
    temp_npu   = torch.from_numpy(temp_np).npu()
    seeds_npu  = torch.from_numpy(seeds_np).npu()
    pos_npu    = torch.from_numpy(pos_np).npu()

    for _ in range(PERF_WARMUP):
        _ = torch.ops._C_ascend.npu_gumbel_sample(
            logits_npu, temp_npu, seeds_npu, pos_npu, apply_temperature)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        out = torch.ops._C_ascend.npu_gumbel_sample(
            logits_npu, temp_npu, seeds_npu, pos_npu, apply_temperature)
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / PERF_ITERS

    tag = f"num_reqs={logits_np.shape[0]}, vocab={logits_np.shape[1]}, apply_temp={apply_temperature}"
    if label:
        tag = f"{label} | {tag}"
    print(f"  [perf] {tag} → {elapsed_us:.1f} μs/call", flush=True)

    return out.cpu().numpy()


# ============================================================
# Group 1: basic — 基本功能（默认 apply_temperature=True）
# ============================================================
@pytest.mark.parametrize("num_reqs,vocab_size", [
    (1,   32000),   # 最小 batch
    (4,   32000),   # 小 batch
    (16,  32000),   # 中 batch
    (32,  128256),  # 大 vocab（Llama-3 规模）
])
@torch.inference_mode()
def test_gumbel_sample_basic(num_reqs, vocab_size):
    logits_np, temp_np, seeds_np, pos_np = _make_inputs(num_reqs, vocab_size)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True,
                      label="basic")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 2: apply_temperature=False（跳过缩放，TilingKey=0 分支）
# ============================================================
@pytest.mark.parametrize("num_reqs,vocab_size", [
    (1,   32000),
    (8,   32000),
    (16,  128256),
    (32,  32000),
    (64,  32000),
])
@torch.inference_mode()
def test_gumbel_sample_no_temperature(num_reqs, vocab_size):
    logits_np, temp_np, seeds_np, pos_np = _make_inputs(num_reqs, vocab_size, temp_val=1.0)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, apply_temperature=False)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature=False,
                      label="no_temp")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 3: 属性组合 — 不同 temperature 值 × apply_temperature 组合
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
    num_reqs, vocab_size = 8, 32000
    logits_np, _, seeds_np, pos_np = _make_inputs(num_reqs, vocab_size, temp_val=temp_val)
    temp_np = np.full(num_reqs, temp_val, dtype=np.float32)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, apply_temperature)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature,
                      label=f"attr temp={temp_val} apply={apply_temperature}")
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 4: large — vLLM 真实负载规模
# ============================================================
@pytest.mark.parametrize("num_reqs,vocab_size,label", [
    (1,   32000,  "prefill_single_req"),    # prefill：单请求长 seq
    (256, 32000,  "decode_large_batch"),    # decode：大 batch 短 seq
    (512, 32000,  "decode_max_batch"),      # decode：最大 batch
    (1,   128256, "prefill_llama3_vocab"),  # prefill：Llama-3 vocab
])
@torch.inference_mode()
def test_gumbel_sample_large(num_reqs, vocab_size, label):
    logits_np, temp_np, seeds_np, pos_np = _make_inputs(num_reqs, vocab_size)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True,
                      label=label)
    np.testing.assert_array_equal(actual, golden)


# ============================================================
# Group 5: boundary — 数值 / shape 边界
# ============================================================
@pytest.mark.parametrize("case_name,num_reqs,vocab_size,temp_val,seed_base,pos_base", [
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
    # 混合 batch：num_reqs 不整除 usedCoreNum（假设 20 核）
    ("batch_non_divisible", 21,  32000, 1.0,  7,    0),
])
@torch.inference_mode()
def test_gumbel_sample_boundary(case_name, num_reqs, vocab_size, temp_val, seed_base, pos_base):
    logits_np, _, seeds_np, pos_np = _make_inputs(
        num_reqs, vocab_size, temp_val=temp_val,
        seed_base=seed_base, pos_base=pos_base)
    temp_np = np.full(num_reqs, temp_val, dtype=np.float32)
    golden = golden_gumbel_sample(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True)
    actual = _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature=True,
                      label=f"boundary/{case_name}")
    np.testing.assert_array_equal(actual, golden)
