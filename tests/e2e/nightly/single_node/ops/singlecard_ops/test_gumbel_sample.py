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
# CPU golden 实现
#   与 op_kernel 中 float LCG hash 算法完全对齐：
#   splitmix64(seed, pos) → gumbelSeedF → 两轮 LCG → u → g = -ln(-ln(u+ε)+ε)
#   argmax(logits/temp + g)
# ============================================================
LCG_A = np.float32(1664525.0)
LCG_C = np.float32(1013904223.0)
NORM  = np.float32(2.3283064e-10)
EPS   = np.float32(1e-20)


def _splitmix64(seed: int, pos: int) -> np.float32:
    """splitmix64 混合 seed 和 pos，返回 float32 种子（与 kernel 一致）。"""
    h = (int(seed) & 0xFFFFFFFFFFFFFFFF) ^ ((int(pos) & 0xFFFFFFFF) << 32)
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9
    h &= 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB
    h &= 0xFFFFFFFFFFFFFFFF
    h = h ^ (h >> 31)
    h &= 0xFFFFFFFFFFFFFFFF
    # 转为有符号 int64 再转 float32（与 kernel static_cast<float>(int64_t(h)) 一致）
    h_signed = h if h < (1 << 63) else h - (1 << 64)
    return np.float32(h_signed)


def _lcg_gumbel(seed_f: np.float32, vocab_size: int) -> np.ndarray:
    """生成 vocab_size 个 Gumbel 噪声（float32），与 kernel GenerateGumbelTile 对齐。"""
    idx = np.arange(vocab_size, dtype=np.float32)
    state = idx + seed_f
    # 两轮 LCG（float32 运算）
    state = state * LCG_A + LCG_C
    state = state * LCG_A + LCG_C
    u = np.abs(state) * NORM
    u = u + EPS
    g = -np.log(-np.log(u) + EPS)
    return g


def golden_gumbel_sample(
    logits: np.ndarray,
    temperature: np.ndarray,
    seeds: np.ndarray,
    pos: np.ndarray,
    apply_temperature: bool = True,
) -> np.ndarray:
    """
    CPU golden：返回 sampled[num_reqs]（int64）。
    logits: [num_reqs, vocab_size] float32
    temperature: [num_reqs] float32
    seeds: [num_reqs] int64
    pos: [num_reqs] int64
    """
    num_reqs, vocab_size = logits.shape
    sampled = np.zeros(num_reqs, dtype=np.int64)
    for r in range(num_reqs):
        temp = float(temperature[r])
        seed64 = int(seeds[r])
        pos_i  = int(pos[r])
        is_greedy = (temp == 0.0)

        logits_r = logits[r].astype(np.float32).copy()
        if not is_greedy:
            if apply_temperature:
                logits_r = logits_r / np.float32(temp)
            seed_f = _splitmix64(seed64, pos_i)
            g = _lcg_gumbel(seed_f, vocab_size)
            logits_r = logits_r + g

        sampled[r] = int(np.argmax(logits_r))
    return sampled


# ============================================================
# 辅助：构造输入 + 调用 NPU 算子 + 打印延迟
# ============================================================
def _make_inputs(num_reqs: int, vocab_size: int, temp_val: float = 1.0,
                 seed_base: int = 42, pos_base: int = 0):
    """
    构造 GumbelSample 输入张量（CPU numpy + NPU tensor）。
    参数自洽检查：num_reqs >= 1, vocab_size >= 1。
    """
    assert num_reqs >= 1, f"num_reqs must be >= 1, got {num_reqs}"
    assert vocab_size >= 1, f"vocab_size must be >= 1, got {vocab_size}"

    rng = np.random.default_rng(seed_base)
    logits_np   = rng.standard_normal((num_reqs, vocab_size)).astype(np.float32)
    temp_np     = np.full(num_reqs, temp_val, dtype=np.float32)
    seeds_np    = np.arange(seed_base, seed_base + num_reqs, dtype=np.int64)
    pos_np      = np.arange(pos_base, pos_base + num_reqs, dtype=np.int64)
    return logits_np, temp_np, seeds_np, pos_np


def _run_npu(logits_np, temp_np, seeds_np, pos_np, apply_temperature: bool,
             label: str = ""):
    """调用 NPU 算子，含 warmup + 多次平均延迟打印。"""
    logits_npu = torch.from_numpy(logits_np).npu()
    temp_npu   = torch.from_numpy(temp_np).npu()
    seeds_npu  = torch.from_numpy(seeds_np).npu()
    pos_npu    = torch.from_numpy(pos_np).npu()

    # warmup
    for _ in range(PERF_WARMUP):
        _ = torch.ops._C_ascend.npu_gumbel_sample(
            logits_npu, temp_npu, seeds_npu, pos_npu, apply_temperature)
    torch.npu.synchronize()

    # measure
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
    ("min_scale",          1,   1,     1.0,  0,    0),
    # vocab 恰好 = BLOCK_SIZE（4096，整数倍对齐）
    ("vocab_eq_block",     4,   4096,  1.0,  1,    0),
    # vocab 恰好 = 2×BLOCK_SIZE
    ("vocab_2x_block",     4,   8192,  1.0,  2,    0),
    # vocab 非整数倍（4096+1）
    ("vocab_non_aligned",  4,   4097,  1.0,  3,    0),
    # seed=0（边界值）
    ("seed_zero",          8,   32000, 1.0,  0,    0),
    # pos=0（边界值）
    ("pos_zero",           8,   32000, 1.0,  42,   0),
    # 极大 pos（接近 int32 上限）
    ("pos_large",          4,   32000, 1.0,  42,   2147483647 - 4),
    # 极大 seed（接近 int64 上限）
    ("seed_large",         4,   32000, 1.0,  9223372036854775800, 0),
    # 混合 batch：num_reqs 不整除 usedCoreNum（假设 20 核）
    ("batch_non_divisible", 21, 32000, 1.0,  7,    0),
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
