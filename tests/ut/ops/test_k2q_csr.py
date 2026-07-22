# SPDX-License-Identifier: Apache-2.0
"""NPU correctness tests for AscendC ``npu_k2q_csr``.

Validates ``torch.ops._C_ascend.npu_k2q_csr`` against a pure-PyTorch golden
reference for Concat (order_method=0) and Round-robin (order_method=1).

Prerequisite: AscendC kernels compiled and installed via::

  bash csrc/build_aclnn.sh <ROOT_DIR> ascend910b

Run::

  pytest tests/ut/ops/test_k2q_csr.py -v
"""

from __future__ import annotations

import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

SEED = 42

# Small fixed shapes for fast CI coverage.
FIXED_CASES = [
    # single batch, Concat-friendly
    {
        "q2k": torch.tensor(
            [[[2, 5], [5, 7], [1, 6]], [[0, 4], [3, 7], [6, -1]]],
            dtype=torch.int32,
        ),
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        "cu_block_lens": torch.tensor([0, 8], dtype=torch.int32),
    },
    # multi-batch variable block lens
    {
        "q2k": torch.tensor(
            [
                [[2, -1], [1, 4], [1, 0], [0, 4], [0, 3]],
                [[3, 4], [0, 4], [1, 2], [0, -1], [2, 1]],
            ],
            dtype=torch.int32,
        ),
        "cu_seqlens": torch.tensor([0, 5], dtype=torch.int32),
        "cu_block_lens": torch.tensor([0, 5], dtype=torch.int32),
    },
]


def _require_npu_k2q_csr() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is not available")
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "npu_k2q_csr"):
        pytest.skip("torch.ops._C_ascend.npu_k2q_csr is not registered")


# ---------------------------------------------------------------------------
# Golden reference (CPU / any device, pure PyTorch)
# ---------------------------------------------------------------------------


def _build_round_robin_row_map(block_lens: torch.Tensor) -> torch.Tensor:
    """Match AscendC Meta order_method=1 row packing."""
    device = block_lens.device
    B = int(block_lens.numel())
    max_kv = int(block_lens.max().item()) if B > 0 else 0
    levels = torch.arange(max_kv, device=device, dtype=torch.int64)
    valid = block_lens.unsqueeze(1) > levels.unsqueeze(0)
    rows_before = torch.minimum(block_lens.unsqueeze(1), levels.unsqueeze(0)).sum(dim=0)
    active_rank = valid.to(torch.int64).cumsum(dim=0) - 1
    return torch.where(
        valid,
        rows_before.unsqueeze(0) + active_rank,
        torch.full((B, max_kv), -1, dtype=torch.int64, device=device),
    )


def golden_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU golden for q2k[H,T,topk] + cu_seqlens + cu_block_lens -> CSR."""
    if q2k.dtype != torch.int32:
        raise TypeError(f"q2k must be int32, got {q2k.dtype}")
    if q2k.ndim != 3:
        raise ValueError(f"q2k must be [H,T,topk], got {tuple(q2k.shape)}")
    if order_method not in (0, 1):
        raise ValueError(f"order_method must be 0 or 1, got {order_method}")

    H, T, topk = (int(x) for x in q2k.shape)
    device = q2k.device
    cu_q = cu_seqlens.to(device=device, dtype=torch.int64).reshape(-1)
    cu_b = cu_block_lens.to(device=device, dtype=torch.int64).reshape(-1)
    B = int(cu_q.numel()) - 1
    seq_lens = cu_q[1:] - cu_q[:-1]
    block_lens = cu_b[1:] - cu_b[:-1]
    total_rows = int(cu_b[-1].item())

    token_batch = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int64),
        seq_lens,
    )
    local = q2k.long()
    masked_valid = local >= 0

    if order_method == 0:
        b_row_off = cu_b.to(dtype=local.dtype)[token_batch]
        global_blk = torch.where(masked_valid, local + b_row_off.view(1, T, 1), local)
    else:
        row_map = _build_round_robin_row_map(block_lens)
        max_kv = int(block_lens.max().item()) if B > 0 else 0
        batch_idx = token_batch.view(1, T, 1).expand_as(local)
        flat_idx = batch_idx * max_kv + local.clamp(min=0)
        mapped = row_map.reshape(-1)[flat_idx]
        global_blk = torch.where(masked_valid, mapped, local)

    flat = global_blk.reshape(H, T * topk)
    flat_valid = masked_valid.reshape(H, T * topk)

    # Invalid edges (flat==-1) land in bin 0 via (flat+1); clear afterwards.
    counts = torch.zeros(H, 1 + total_rows, dtype=torch.int32, device=device)
    counts.scatter_add_(1, flat + 1, torch.ones_like(flat, dtype=torch.int32))
    counts[:, 0] = 0
    row_ptr = counts.cumsum(1).to(torch.int32)

    sort_key = flat.masked_fill(~flat_valid, total_rows)
    order = torch.argsort(sort_key, dim=-1, stable=True)
    invalid = torch.sort(~flat_valid, dim=-1, stable=True)[0]

    token_pos = order // topk
    local_q = token_pos - cu_q.to(dtype=token_pos.dtype)[token_batch][token_pos]
    q_ind = local_q.masked_fill(invalid, -1).to(torch.int32)
    slot = (order % topk).masked_fill(invalid, -1).to(torch.int32)
    return row_ptr, q_ind, slot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_q2k_tnd(
    H: int,
    topk: int,
    list_seqlens: list[int],
    list_blocklens: list[int],
    *,
    seed: int = SEED,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate valid TND q2k with local block indices in [-1, block_len)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    chunks: list[torch.Tensor] = []
    for s, nb in zip(list_seqlens, list_blocklens):
        if nb <= 0 or s <= 0:
            chunks.append(torch.empty(H, 0, topk, dtype=torch.int32))
            continue
        vals = torch.randint(0, nb, (H, s, topk), generator=g, dtype=torch.int32)
        # Inject some padding (-1).
        mask = torch.rand((H, s, topk), generator=g) < 0.15
        vals = vals.masked_fill(mask, -1)
        chunks.append(vals)
    q2k = torch.cat(chunks, dim=1) if chunks else torch.empty(H, 0, topk, dtype=torch.int32)
    cu_seqlens = torch.tensor([0] + list_seqlens, dtype=torch.int32).cumsum(0).to(torch.int32)
    cu_block_lens = torch.tensor([0] + list_blocklens, dtype=torch.int32).cumsum(0).to(torch.int32)
    return q2k.to(device), cu_seqlens.to(device), cu_block_lens.to(device)


def _npu_k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    order_method: int = 0,
    use_simt: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_rows = int(cu_block_lens[-1].item())
    max_kv = int((cu_block_lens[1:] - cu_block_lens[:-1]).max().item()) if cu_block_lens.numel() > 1 else 0
    row_ptr, q_ind, slot = torch.ops._C_ascend.npu_k2q_csr(
        q2k.npu().contiguous(),
        cu_seqlens.npu().contiguous(),
        cu_block_lens.npu().contiguous(),
        int(order_method),
        total_rows,
        max_kv,
        int(use_simt),
    )
    return row_ptr.cpu(), q_ind.cpu(), slot.cpu()


def _assert_csr_equal(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    a_row, a_q, a_slot = actual
    r_row, r_q, r_slot = reference
    assert a_row.shape == r_row.shape, f"row_ptr shape {a_row.shape} != {r_row.shape}"
    assert a_q.shape == r_q.shape, f"q_ind shape {a_q.shape} != {r_q.shape}"
    assert a_slot.shape == r_slot.shape, f"slot shape {a_slot.shape} != {r_slot.shape}"
    assert torch.equal(a_row, r_row), "row_ptr mismatch vs golden"
    assert torch.equal(a_q, r_q), "q_ind mismatch vs golden"
    assert torch.equal(a_slot, r_slot), "slot mismatch vs golden"


def _cleanup_npu() -> None:
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_method", [0, 1], ids=["concat", "round_robin"])
@pytest.mark.parametrize("case_idx", range(len(FIXED_CASES)))
def test_k2q_csr_fixed_samples(order_method: int, case_idx: int):
    _require_npu_k2q_csr()
    case = FIXED_CASES[case_idx]
    q2k = case["q2k"]
    cu_seqlens = case["cu_seqlens"]
    cu_block_lens = case["cu_block_lens"]

    ref = golden_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=order_method)
    out = _npu_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=order_method)
    _assert_csr_equal(out, ref)
    _cleanup_npu()


@pytest.mark.parametrize("order_method", [0, 1], ids=["concat", "round_robin"])
@pytest.mark.parametrize("H", [1, 2, 4])
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize(
    "list_seqlens,list_blocklens",
    [
        ([4], [5]),
        ([4, 6, 3], [5, 7, 6]),
        ([8, 16], [4, 8]),
        ([32, 16, 8], [8, 8, 4]),
    ],
    ids=["single", "simple", "equalish", "varlen"],
)
def test_k2q_csr_random_tnd(
    order_method: int,
    H: int,
    topk: int,
    list_seqlens: list[int],
    list_blocklens: list[int],
):
    _require_npu_k2q_csr()
    q2k, cu_seqlens, cu_block_lens = _fake_q2k_tnd(
        H,
        topk,
        list_seqlens,
        list_blocklens,
        seed=SEED + H + topk + order_method,
    )

    ref = golden_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=order_method)
    out = _npu_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=order_method)
    _assert_csr_equal(out, ref)

    # Structural invariants.
    nnz = int((q2k >= 0).sum().item())
    assert int(out[0][:, -1].sum().item()) == nnz
    valid_slots = out[2][out[2] >= 0]
    if valid_slots.numel() > 0:
        assert int(valid_slots.max().item()) < topk
    assert out[0][:, 0].eq(0).all()
    _cleanup_npu()


def test_k2q_csr_explicit_total_rows_max_kv():
    """Host-side total_rows/max_kv path (avoid D2H in adapter)."""
    _require_npu_k2q_csr()
    q2k, cu_seqlens, cu_block_lens = _fake_q2k_tnd(
        2,
        4,
        [8, 12],
        [6, 10],
        seed=7,
    )
    total_rows = int(cu_block_lens[-1].item())
    max_kv = int((cu_block_lens[1:] - cu_block_lens[:-1]).max().item())

    ref = golden_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=0)
    row_ptr, q_ind, slot = torch.ops._C_ascend.npu_k2q_csr(
        q2k.npu(),
        cu_seqlens.npu(),
        cu_block_lens.npu(),
        0,
        total_rows,
        max_kv,
        0,
    )
    _assert_csr_equal((row_ptr.cpu(), q_ind.cpu(), slot.cpu()), ref)
    _cleanup_npu()


def test_k2q_csr_meta_shapes_match_inputs():
    """Output shapes follow [H, total_rows+1] / [H, T*topk]."""
    _require_npu_k2q_csr()
    H, T, topk = 3, 10, 4
    q2k, cu_seqlens, cu_block_lens = _fake_q2k_tnd(H, topk, [10], [7], seed=3)
    total_rows = int(cu_block_lens[-1].item())
    row_ptr, q_ind, slot = _npu_k2q_csr(q2k, cu_seqlens, cu_block_lens, order_method=1)
    assert row_ptr.shape == (H, total_rows + 1)
    assert q_ind.shape == (H, T * topk)
    assert slot.shape == (H, T * topk)
    _cleanup_npu()
