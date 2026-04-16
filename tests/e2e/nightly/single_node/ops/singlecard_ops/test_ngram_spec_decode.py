"""E2E accuracy test for NgramSpecDecode custom operator.

Tests the Ascend C kernel against a CPU golden reference implementation
with parametrized test cases covering various configurations.
"""

import time
import numpy as np
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42
PERF_WARMUP = 3
PERF_ITERS = 20


# ---------------------------------------------------------------------------
# Golden reference (CPU, pure Python/NumPy)
# ---------------------------------------------------------------------------

def golden_ngram_spec_decode(
    token_ids: np.ndarray,           # [B, M], int32, 会被 in-place 修改
    num_tokens_no_spec: np.ndarray,  # [B], int32
    sampled_token_ids: np.ndarray,   # [B, N], int32
    discard_request_mask: np.ndarray, # [B], int32
    vocab_size: int,
    min_n: int,
    max_n: int,
    k: int,
):
    """CPU golden reference for NgramSpecDecode.

    Returns:
        (token_ids_modified, next_token_ids, draft_token_ids, num_valid_draft_tokens)
    """
    B = token_ids.shape[0]
    M = token_ids.shape[1]
    next_token_ids = np.zeros(B, dtype=np.int32)
    draft_token_ids = np.full((B, k), -1, dtype=np.int32)
    num_valid_draft_tokens = np.zeros(B, dtype=np.int32)

    for i in range(B):
        seq_len = int(num_tokens_no_spec[i])
        discard = int(discard_request_mask[i])
        valid_count = 0

        # Stage 1: 采样 token 有效性处理
        backup_pos = max(seq_len - 1, 0)
        backup_token = int(token_ids[i, backup_pos])

        for j in range(sampled_token_ids.shape[1]):
            val = int(sampled_token_ids[i, j])
            if discard != 0:
                sampled_token_ids[i, j] = -1
            elif val != -1 and val < vocab_size:
                valid_count += 1
            else:
                sampled_token_ids[i, j] = -1

        avail_space = M - seq_len
        if avail_space < 0:
            avail_space = 0
        if valid_count > avail_space:
            valid_count = avail_space

        if valid_count > 0:
            next_token_ids[i] = int(sampled_token_ids[i, valid_count - 1])
        else:
            next_token_ids[i] = backup_token

        # Stage 2: scatter 采样 token 到 token_ids 尾部
        nt = seq_len + valid_count
        for j in range(valid_count):
            token_ids[i, seq_len + j] = int(sampled_token_ids[i, j])

        # Stage 3: 后缀 n-gram 匹配
        best_match_pos = -1
        best_ngram_len = 0

        if valid_count > 0 and nt >= min_n:
            for ngram_len in range(min_n, max_n + 1):
                if ngram_len > nt:
                    break
                wc = nt - ngram_len
                if wc <= 0:
                    break

                suffix = token_ids[i, nt - ngram_len: nt].tolist()
                found = False
                for pos in range(wc):
                    window = token_ids[i, pos: pos + ngram_len].tolist()
                    if window == suffix:
                        best_match_pos = pos
                        best_ngram_len = ngram_len
                        found = True
                        break
                if found:
                    break

        # Stage 4: 提取 draft tokens
        if best_match_pos >= 0:
            draft_start = best_match_pos + best_ngram_len
            tokens_available = nt - draft_start
            for j in range(k):
                if j < tokens_available:
                    draft_token_ids[i, j] = int(token_ids[i, draft_start + j])
                else:
                    draft_token_ids[i, j] = -1
        # else: 已初始化为 -1

        # 统计有效 draft token
        valid_draft_count = 0
        for j in range(k):
            if draft_token_ids[i, j] != -1:
                valid_draft_count += 1
            else:
                break
        num_valid_draft_tokens[i] = valid_draft_count

    return token_ids, next_token_ids, draft_token_ids, num_valid_draft_tokens


# ---------------------------------------------------------------------------
# 输入构造 helper
# ---------------------------------------------------------------------------

def _make_inputs(
    batch_size: int,
    seq_len: int,
    max_new_tokens: int,
    k: int,
    vocab_size: int = 32000,
    min_n: int = 3,
    max_n: int = 5,
    discard_rate: float = 0.0,
    invalid_rate: float = 0.0,
    seed: int = SEED,
):
    """构造测试输入，带参数自洽检查。

    Returns:
        (token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
         vocab_size, min_n, max_n, k)
    """
    rng = np.random.RandomState(seed)
    token_ids = rng.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)

    # num_tokens_no_spec: 每条序列实际 token 数，至少为 1
    # 确保留有空间给 sampled tokens: num_tokens + max_new_tokens <= seq_len
    max_valid_tokens = seq_len - max_new_tokens
    if max_valid_tokens < 1:
        max_valid_tokens = 1
    num_tokens_no_spec = rng.randint(1, max_valid_tokens + 1, size=(batch_size,), dtype=np.int32)

    # sampled_token_ids: 有效 token 或 -1
    sampled_token_ids = rng.randint(0, vocab_size, size=(batch_size, max_new_tokens), dtype=np.int32)
    if invalid_rate > 0:
        invalid_mask = rng.rand(batch_size, max_new_tokens) < invalid_rate
        sampled_token_ids[invalid_mask] = -1

    # discard_request_mask
    discard_request_mask = np.zeros(batch_size, dtype=np.int32)
    if discard_rate > 0:
        discard_mask = rng.rand(batch_size) < discard_rate
        discard_request_mask[discard_mask] = 1

    return token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
           vocab_size, min_n, max_n, k


def _run_npu(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
             vocab_size, min_n, max_n, k):
    """调用 NPU 算子并返回结果。"""
    token_ids_t = torch.from_numpy(token_ids).to("npu")
    num_tokens_t = torch.from_numpy(num_tokens_no_spec).to("npu")
    sampled_t = torch.from_numpy(sampled_token_ids).to("npu")
    discard_t = torch.from_numpy(discard_request_mask).to("npu")

    result = torch.ops._C_ascend.npu_ngram_spec_decode(
        token_ids_t, num_tokens_t, sampled_t, discard_t,
        vocab_size, min_n, max_n, k)
    torch.npu.synchronize()

    return result


def _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k):
    """warmup + 多次测量延迟。"""
    token_ids_t = torch.from_numpy(token_ids.copy()).to("npu")
    num_tokens_t = torch.from_numpy(num_tokens_no_spec.copy()).to("npu")
    sampled_t = torch.from_numpy(sampled_token_ids.copy()).to("npu")
    discard_t = torch.from_numpy(discard_request_mask.copy()).to("npu")

    for _ in range(PERF_WARMUP):
        _ = torch.ops._C_ascend.npu_ngram_spec_decode(
            token_ids_t, num_tokens_t, sampled_t, discard_t,
            vocab_size, min_n, max_n, k)
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        _ = torch.ops._C_ascend.npu_ngram_spec_decode(
            token_ids_t, num_tokens_t, sampled_t, discard_t,
            vocab_size, min_n, max_n, k)
    torch.npu.synchronize()
    elapsed_us = (time.perf_counter() - t0) * 1e6 / PERF_ITERS

    print(f"  [perf] B={token_ids.shape[0]} M={token_ids.shape[1]} N={sampled_token_ids.shape[1]} "
          f"k={k} min_n={min_n} max_n={max_n} -> {elapsed_us:.1f} us/call", flush=True)


# ===========================================================================
# Group 1: basic - 基本功能（默认配置）
# ===========================================================================

@pytest.mark.parametrize("batch_size,seq_len,max_new_tokens,k", [
    (1, 16, 4, 3),
    (4, 64, 8, 5),
    (16, 128, 16, 5),
])
@torch.inference_mode()
def test_ngram_spec_decode_basic(batch_size, seq_len, max_new_tokens, k):
    """Group 1: 基本功能测试 - 小/中/大规模 batch。"""
    inputs = _make_inputs(batch_size, seq_len, max_new_tokens, k)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    # CPU golden
    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    # NPU
    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    # 对比
    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist(), \
        f"token_ids mismatch: B={batch_size}"
    assert result_next.cpu().numpy().tolist() == golden_next.tolist(), \
        f"next_token_ids mismatch: B={batch_size}"
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist(), \
        f"draft_token_ids mismatch: B={batch_size}"
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist(), \
        f"num_valid_draft_tokens mismatch: B={batch_size}"

    _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k)


# ===========================================================================
# Group 2: padding / optional - 有 padding 或可选输入
# ===========================================================================

@pytest.mark.parametrize("batch_size,seq_len,max_new_tokens,k,invalid_rate", [
    (4, 64, 8, 5, 0.3),
    (4, 64, 8, 5, 0.7),
    (4, 128, 16, 5, 0.5),
])
@torch.inference_mode()
def test_ngram_spec_decode_padding(batch_size, seq_len, max_new_tokens, k, invalid_rate):
    """Group 2: 含无效采样 token 的场景。"""
    inputs = _make_inputs(batch_size, seq_len, max_new_tokens, k, invalid_rate=invalid_rate)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()

    _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k)


@pytest.mark.parametrize("discard_rate", [0.2, 0.5, 1.0])
@torch.inference_mode()
def test_ngram_spec_decode_discard(discard_rate):
    """Group 2: 丢弃请求掩码场景。"""
    inputs = _make_inputs(4, 64, 8, 5, discard_rate=discard_rate)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()


# ===========================================================================
# Group 3: 属性组合 - 不同的 min_n / max_n / k
# ===========================================================================

@pytest.mark.parametrize("min_n,max_n,k", [
    (1, 1, 3),
    (2, 4, 5),
    (1, 8, 10),
    (3, 3, 1),
    (5, 10, 8),
])
@torch.inference_mode()
def test_ngram_spec_decode_attrs(min_n, max_n, k):
    """Group 3: 不同属性组合。"""
    inputs = _make_inputs(4, 128, 16, k, min_n=min_n, max_n=max_n)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, _, _, _ = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()

    _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k)


# ===========================================================================
# Group 4: large - vLLM 真实负载规模
# ===========================================================================

@torch.inference_mode()
def test_ngram_spec_decode_prefill():
    """Group 4: Prefill 场景 - 单请求长序列。"""
    inputs = _make_inputs(1, 2048, 16, 5, vocab_size=32000)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()

    _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k)


@torch.inference_mode()
def test_ngram_spec_decode_decode():
    """Group 4: Decode 场景 - 大 batch 短序列。"""
    inputs = _make_inputs(64, 32, 5, 3, vocab_size=32000)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()

    _measure_perf(token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
                  vocab_size, min_n, max_n, k)


# ===========================================================================
# Group 5: boundary - 数值 / shape 边界
# ===========================================================================

@torch.inference_mode()
def test_ngram_spec_decode_minimal():
    """Group 5: 最小规模 - 1 token / 1 request。"""
    inputs = _make_inputs(1, 4, 1, 1, vocab_size=100)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()


@torch.inference_mode()
def test_ngram_spec_decode_no_valid_sampled():
    """Group 5: 所有 sampled token 均无效。"""
    inputs = _make_inputs(4, 64, 8, 5)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs
    # 全部设为无效
    sampled_token_ids[:] = -1

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()


@torch.inference_mode()
def test_ngram_spec_decode_exact_match():
    """Group 5: 构造精确匹配场景确保 draft 提取正确。"""
    B, M, N, k = 2, 32, 4, 3
    vocab_size = 1000
    token_ids = np.array([
        [1, 2, 3, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 20, 10, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    num_tokens_no_spec = np.array([6, 5], dtype=np.int32)
    # sampled tokens: 延续序列
    sampled_token_ids = np.array([
        [1, 2, 3, -1],
        [10, 20, -1, -1],
    ], dtype=np.int32)
    discard_request_mask = np.array([0, 0], dtype=np.int32)

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, 3, 5, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, 3, 5, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()


@torch.inference_mode()
def test_ngram_spec_decode_full_capacity():
    """Group 5: token_ids 几乎填满（num_tokens + max_new_tokens ≈ max_seq_len）。"""
    inputs = _make_inputs(4, 48, 16, 5, min_n=2, max_n=4)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs
    # 设置 num_tokens 接近 seq_len，只留刚好够的空间
    num_tokens_no_spec[:] = token_ids.shape[1] - sampled_token_ids.shape[1]

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()


@torch.inference_mode()
def test_ngram_spec_decode_k1():
    """Group 5: k=1 极端场景。"""
    inputs = _make_inputs(4, 64, 8, 1, min_n=2, max_n=3)
    token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask, \
        vocab_size, min_n, max_n, k = inputs

    golden_ids, golden_next, golden_draft, golden_valid = golden_ngram_spec_decode(
        token_ids.copy(), num_tokens_no_spec.copy(), sampled_token_ids.copy(),
        discard_request_mask.copy(), vocab_size, min_n, max_n, k)

    result_ids, result_next, result_draft, result_valid = _run_npu(
        token_ids, num_tokens_no_spec, sampled_token_ids, discard_request_mask,
        vocab_size, min_n, max_n, k)

    assert result_ids.cpu().numpy().tolist() == golden_ids.tolist()
    assert result_next.cpu().numpy().tolist() == golden_next.tolist()
    assert result_draft.cpu().numpy().tolist() == golden_draft.tolist()
    assert result_valid.cpu().numpy().tolist() == golden_valid.tolist()
