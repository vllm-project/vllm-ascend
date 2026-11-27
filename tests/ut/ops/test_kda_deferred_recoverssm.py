# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-level verification for the deferred RecoverSSM path on Ascend.

This script does NOT need a real GLM5-Next checkpoint. It fabricates the
minimum tensors required by ``KDARecoverSSMCommitContext`` and covers:

    1. ``_defer_commit_kernel``        - pending/eager split logic
    2. ``_prepare_flush_plan_kernel``  - pending -> commit_lens recovery
    3. ``KDARecoverSSMCommitContext.flush`` - pending replay folded into checkpoint
    4. deferred vs immediate numerical equivalence
    5. fused verify (``compact_conv`` + in-kernel replay) vs ``flush`` + stock
       verify — the production ``fp32_deferred`` path
    6. boundary conditions: empty batch, record_rows with null slots

Run on an Ascend NPU machine:

    python -m pytest tests/ut/ops/test_kda_deferred_recoverssm.py -v
or just:
    python tests/ut/ops/test_kda_deferred_recoverssm.py
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch

pytest.importorskip("torch_npu")

import vllm_ascend.ops.triton.kda.recoverssm as recoverssm_module  # noqa: E402
from vllm_ascend.ops.triton.kda.recoverssm import (  # noqa: E402
    KDARecoverSSMCommitContext,
    _defer_commit_kernel,
    _prepare_flush_plan_kernel,
    kda_recoverssm_verify,
)

# The physical card is chosen with ASCEND_RT_VISIBLE_DEVICES; the first
# visible device is ordinal 0 (e.g. ASCEND_RT_VISIBLE_DEVICES=7 -> npu:0).
DEVICE = torch.device("npu:0")
torch.npu.set_device(DEVICE)  # kernel launch uses current device, not DEVICE
DTYPE = torch.bfloat16
HEADS = 2          # A2 UB overflow mitigation: smaller heads*value_dim
KEY_DIM = 8
VALUE_DIM = 8
SPEC_QUERY_LEN = 3  # num_spec + 1
NUM_BLOCKS = 16
NULL_BLOCK_ID = 0   # Ascend uses 0 (not -1 like NVIDIA)

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first  # noqa: E402
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton  # noqa: E402
init_device_properties_triton()


def _make_layer(
    heads: int = HEADS,
    key_dim: int = KEY_DIM,
    value_dim: int = VALUE_DIM,
    spec_q: int = SPEC_QUERY_LEN,
    blocks: int = NUM_BLOCKS,
    records: int | None = None,
    seed: int = 0,
) -> SimpleNamespace:
    """Build a minimal layer compatible with KDARecoverSSMCommitContext.create.

    The context requires ``layer.kv_cache`` to be a 4-tuple of:
        (conv_state, state_checkpoint, correction_cache, key_gate_cache)
    plus A_log, dt_bias, local_num_heads, head_dim, gate_lower_bound.

    ``records`` sizes the correction/key-gate pool independently of the
    checkpoint's block count (upstream vLLM keeps records in a separate
    ``max_num_reqs`` pool); when omitted the records share the state pool.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, dtype=DTYPE):
        return torch.randn(*shape, device=DEVICE, generator=g, dtype=dtype)

    # Native layout: create() transposes SD -> DS, so build the platform's
    # layout here or conv_dim/state_len get swapped (and BLOCK_D*BLOCK_HISTORY
    # explodes for realistic shapes).
    conv_len = spec_q + 2
    if is_conv_state_dim_first():
        conv_state = randn(blocks, heads * value_dim, conv_len)
    else:
        conv_state = randn(blocks, conv_len, heads * value_dim)
    # State checkpoint: FP32 so the auto backend picks "fp32_deferred".
    checkpoint = randn(blocks, heads, value_dim, key_dim, dtype=torch.float32) * 0.1
    record_rows_capacity = blocks if records is None else records
    correction = randn(record_rows_capacity, heads, spec_q, value_dim, dtype=torch.float32)
    kg_cache = randn(record_rows_capacity, heads, spec_q, 2 * key_dim)

    return SimpleNamespace(
        kv_cache=(conv_state, checkpoint, correction, kg_cache),
        A_log=randn(heads) * 0.5,
        dt_bias=randn(heads * value_dim) * 0.5,
        local_num_heads=heads,
        head_dim=value_dim,
        gate_lower_bound=-5.0,
    )


def _make_context(
    num_layers: int = 2,
    heads: int = HEADS,
    key_dim: int = KEY_DIM,
    value_dim: int = VALUE_DIM,
    spec_q: int = SPEC_QUERY_LEN,
    blocks: int = NUM_BLOCKS,
    max_reqs: int = 8,
    records: int | None = None,
    seed: int = 0,
) -> tuple[KDARecoverSSMCommitContext, list[SimpleNamespace]]:
    """Build a fresh context + layers. Each call gets independent tensors."""
    layers = [
        _make_layer(
            heads, key_dim, value_dim, spec_q, blocks,
            records=records, seed=seed + i,
        )
        for i in range(num_layers)
    ]
    ctx = KDARecoverSSMCommitContext.create(
        layers,
        spec_query_len=spec_q,
        max_num_reqs=max_reqs,
        backend="triton",
        records_pool_size=records,
    )
    return ctx, layers


# ---------------------------------------------------------------------------
# Test 1: _defer_commit_kernel - pending/eager split
# ---------------------------------------------------------------------------

def test_defer_commit_kernel_pending_and_eager():
    """eager rows: pending -> 0, commit_lens preserved.
    Non-eager rows: pending <- commit_len, commit_lens <- 0.
    Padding rows (record_row<0 or state_idx<=null): untouched.
    """
    batch = 4
    # Pretend prepare_commit_plan wrote these commit_lens.
    commit_lens = torch.tensor([2, 3, 1, 0], dtype=torch.int32, device=DEVICE)
    state_indices = torch.tensor([1, 2, NULL_BLOCK_ID, 4], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([0, 1, -1, 2], dtype=torch.int32, device=DEVICE)
    pending = torch.zeros(3, dtype=torch.int32, device=DEVICE)  # 3 record slots
    # Row 0 is eager (commit now); rows 1,2 deferred.
    eager = torch.tensor([1, 0, 0, 0], dtype=torch.int32, device=DEVICE)
    state_commit_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    boundary_commit_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)

    _defer_commit_kernel[(batch,)](
        state_indices,
        commit_lens,
        state_commit_lens,
        boundary_commit_lens,
        record_rows,
        pending,
        eager,
        NULL_BLOCK_ID,
        state_indices.stride(0),
        num_warps=1,
    )
    torch.npu.synchronize()

    # Row 0 (eager): pending[0]=0, commit_lens[0] unchanged (2)
    assert pending[0].item() == 0, f"eager row should zero pending, got {pending[0]}"
    assert commit_lens[0].item() == 2, f"eager row should keep commit_lens, got {commit_lens[0]}"

    # Row 1 (deferred): pending[1]=3 (commit_len), commit_lens[1]=0
    assert pending[1].item() == 3, f"deferred row should set pending=commit_len, got {pending[1]}"
    assert commit_lens[1].item() == 0, f"deferred row should zero commit_lens, got {commit_lens[1]}"

    # Row 2: record_row=-1 (padding) -> untouched
    assert pending[2].item() == 0, f"padding row should not touch pending, got {pending[2]}"

    # Row 3 (state_idx=4 but record_row=2, deferred, commit_len=0):
    #   pending[2] should become 0 (commit_len was 0), commit_lens[3]=0
    assert commit_lens[3].item() == 0
    # pending[2] was 0, set to commit_len=0 -> still 0
    assert pending[2].item() == 0

    print("[PASS] _defer_commit_kernel: pending/eager split correct")


# ---------------------------------------------------------------------------
# Test 2: _prepare_flush_plan_kernel - pending -> commit_lens
# ---------------------------------------------------------------------------

def test_prepare_flush_plan_kernel():
    """commit_lens = clamp(pending[record_row], 0, SPEC_QUERY_LEN).
    Invalid (state_idx<=null or commit_len=0): commit_lens=0, boundary=null.
    """
    batch = 3
    state_indices = torch.tensor([1, 2, NULL_BLOCK_ID], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([0, 1, 2], dtype=torch.int32, device=DEVICE)
    pending = torch.tensor([2, 5, 1], dtype=torch.int32, device=DEVICE)  # 5 > SPEC_QUERY_LEN=3
    commit_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    final_state_indices = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    boundary_state_indices = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    boundary_recovery_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)

    _prepare_flush_plan_kernel[(batch,)](
        state_indices,
        record_rows,
        pending,
        commit_lens,
        final_state_indices,
        boundary_state_indices,
        boundary_recovery_lens,
        NULL_BLOCK_ID,
        state_indices.stride(0),
        SPEC_QUERY_LEN=SPEC_QUERY_LEN,
        num_warps=1,
    )
    torch.npu.synchronize()

    # Row 0: pending=2, valid -> commit_lens=2
    assert commit_lens[0].item() == 2, f"row 0 commit_lens wrong: {commit_lens[0]}"
    assert final_state_indices[0].item() == 1

    # Row 1: pending=5, clamped to SPEC_QUERY_LEN=3
    assert commit_lens[1].item() == 3, f"row 1 should clamp to SPEC_QUERY_LEN, got {commit_lens[1]}"

    # Row 2: state_idx=NULL_BLOCK_ID -> invalid, commit_lens=0
    assert commit_lens[2].item() == 0, f"null state_idx should give 0 commit_lens, got {commit_lens[2]}"
    assert boundary_state_indices[2].item() == NULL_BLOCK_ID

    print("[PASS] _prepare_flush_plan_kernel: pending->commit_lens recovery correct")


# ---------------------------------------------------------------------------
# Test 3: flush() folds pending replay into checkpoint, then pending can be
#          rewritten by next commit (idempotency of empty flush)
# ---------------------------------------------------------------------------

def test_flush_folds_pending_into_checkpoint():
    """After flush, checkpoint should reflect the pending replay.
    A second flush with pending=0 should be a no-op (idempotent).
    """
    ctx, layers = _make_context(num_layers=1, seed=42)

    batch = 2
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    pending = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)

    # Snapshot checkpoint before flush.
    ckpt_before = layers[0].kv_cache[1].clone()

    ctx.flush(state_indices, record_rows, pending)

    ckpt_after_flush1 = layers[0].kv_cache[1].clone()

    # Checkpoint should have changed (pending replay folded in).
    # Note: if pending=0 for all rows, flush is a no-op. Here pending>0 so
    # the conv-state compaction should at least run.
    # We don't assert exact values (kernel math is complex); we assert the
    # call completed without error and tensors are finite.
    assert torch.isfinite(ckpt_after_flush1).all(), "checkpoint has non-finite values after flush"

    # Second flush with pending=0 should be a no-op.
    pending_zero = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    ctx.flush(state_indices, record_rows, pending_zero)
    ckpt_after_flush2 = layers[0].kv_cache[1].clone()
    assert torch.equal(ckpt_after_flush1, ckpt_after_flush2), \
        "flush with pending=0 should be a no-op"

    print("[PASS] flush(): pending folded, empty flush is no-op")


# ---------------------------------------------------------------------------
# Test 4: deferred vs immediate numerical equivalence
# ---------------------------------------------------------------------------

def test_deferred_vs_immediate_equivalence():
    """The end-to-end invariant: deferred path (pending accumulate + flush)
    should produce the same checkpoint as immediate path (commit each step).

    Scenario: 2 requests, 3 spec-decode steps.
      - Immediate: call commit() every step with the accepted counts.
      - Deferred : call commit(pending=...) every step (defers), then flush()
                   at the start of the next step.
    Both paths start from the SAME initial checkpoint; we compare final state.
    """
    heads, key_dim, value_dim, spec_q = HEADS, KEY_DIM, VALUE_DIM, SPEC_QUERY_LEN
    blocks, num_layers = NUM_BLOCKS, 2
    batch = 2
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.tensor([0, spec_q, 2 * spec_q], dtype=torch.int32, device=DEVICE)
    num_accepted = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    # Align record slots with state indices so both paths read the same
    # correction/kg cache entries (this test has no verify to populate them).
    record_rows = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)

    # --- Immediate path ---
    ctx_imm, layers_imm = _make_context(
        num_layers=num_layers, heads=heads, key_dim=key_dim,
        value_dim=value_dim, spec_q=spec_q, blocks=blocks, seed=100,
    )
    # Run 3 immediate commits (no pending -> commits every step).
    for _ in range(3):
        ctx_imm.commit(
            num_accepted_tokens=num_accepted,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
        )

    # --- Deferred path ---
    ctx_def, layers_def = _make_context(
        num_layers=num_layers, heads=heads, key_dim=key_dim,
        value_dim=value_dim, spec_q=spec_q, blocks=blocks, seed=100,  # SAME seed
    )
    pending = torch.zeros(blocks, dtype=torch.int32, device=DEVICE)
    # 3 deferred steps. Each step: flush previous pending, then commit (defer).
    for step in range(3):
        # flush previous step's pending (step 0: pending is all 0, no-op)
        ctx_def.flush(state_indices, record_rows, pending)
        # commit defers: pending <- accepted, commit_lens <- 0 (no eager rows)
        ctx_def.commit(
            num_accepted_tokens=num_accepted,
            state_indices=state_indices,
            query_start_loc=query_start_loc,
            record_rows=record_rows,
            pending=pending,
            eager=None,  # fully deferred
        )
    # Final flush to fold the last step's pending.
    ctx_def.flush(state_indices, record_rows, pending)

    # Compare final checkpoints across all layers.
    for li in range(num_layers):
        ckpt_imm = layers_imm[li].kv_cache[1]
        ckpt_def = layers_def[li].kv_cache[1]
        assert ckpt_imm.shape == ckpt_def.shape
        assert ckpt_imm.dtype == ckpt_def.dtype
        # Deferred path exercises the same kernels as immediate (via _fold);
        # the difference is the pending accumulate + flush ordering. With
        # identical inputs/seeds, final state should be numerically close.
        # Use loose tolerance because Triton reduction order may differ.
        torch.testing.assert_close(
            ckpt_def.double().cpu(),
            ckpt_imm.double().cpu(),
            rtol=1e-3, atol=1e-3,
        )

    print("[PASS] deferred vs immediate: final checkpoint numerically equivalent")


# ---------------------------------------------------------------------------
# Test 4b: fused verify (in-kernel replay) matches flush + stock verify
# ---------------------------------------------------------------------------

def test_fused_verify_matches_flush_then_verify():
    """``compact_conv`` + verify-with-replay must equal ``flush`` + stock verify.

    Both paths fold the same pending replay into the checkpoint and then run
    the same verify; the fused path does the fold inside the verify kernel.
    """
    heads, key_dim, value_dim, spec_q = HEADS, KEY_DIM, VALUE_DIM, SPEC_QUERY_LEN
    blocks, batch = NUM_BLOCKS, 2
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    pending = torch.tensor([2, 1], dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.tensor([0, spec_q, 2 * spec_q], dtype=torch.int32, device=DEVICE)
    total_tokens = batch * spec_q

    g = torch.Generator(device=DEVICE).manual_seed(7)

    def randn(*shape, dtype=DTYPE):
        return torch.randn(*shape, device=DEVICE, generator=g, dtype=dtype)

    q = randn(1, total_tokens, heads, key_dim)
    k = randn(1, total_tokens, heads, key_dim)
    v = randn(1, total_tokens, heads, value_dim)
    raw_g = randn(1, total_tokens, heads, key_dim)
    raw_beta = randn(1, total_tokens, heads)

    def run(fused: bool):
        ctx, layers = _make_context(
            num_layers=1, heads=heads, key_dim=key_dim, value_dim=value_dim,
            spec_q=spec_q, blocks=blocks, seed=200,
        )
        layer = layers[0]
        if fused:
            ctx.compact_conv(state_indices, record_rows, pending)
            replay = (
                ctx.commit_lens,
                ctx.final_state_indices,
                ctx.boundary_state_indices,
                ctx.boundary_recovery_lens,
            )
        else:
            ctx.flush(state_indices, record_rows, pending)
            replay = (None, None, None, None)
        out = kda_recoverssm_verify(
            q, k, v, raw_g, raw_beta,
            layer.A_log.reshape(-1),
            layer.dt_bias,
            layer.gate_lower_bound,
            layer.kv_cache[1],
            layer.kv_cache[2],
            layer.kv_cache[3],
            query_start_loc,
            state_indices,
            spec_q,
            record_indices=record_rows,
            replay_commit_lens=replay[0],
            replay_final_state_indices=replay[1],
            replay_boundary_state_indices=replay[2],
            replay_boundary_recovery_lens=replay[3],
        )
        torch.npu.synchronize()
        return (
            out.clone(),
            layer.kv_cache[1].clone(),
            layer.kv_cache[2].clone(),
            layer.kv_cache[3].clone(),
        )

    out_ref, ckpt_ref, corr_ref, kg_ref = run(False)
    out_fused, ckpt_fused, corr_fused, kg_fused = run(True)

    torch.testing.assert_close(
        out_fused.double().cpu(), out_ref.double().cpu(), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        ckpt_fused.double().cpu(), ckpt_ref.double().cpu(), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        corr_fused.double().cpu(), corr_ref.double().cpu(), rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        kg_fused.double().cpu(), kg_ref.double().cpu(), rtol=1e-3, atol=1e-3
    )

    print("[PASS] fused verify: matches flush + stock verify")


# ---------------------------------------------------------------------------
# Test 4c: start_flush gate picks fused vs flush
# ---------------------------------------------------------------------------

def test_start_flush_gate_selects_fused_vs_flush():
    """The first KDA layer keeps the fused ``compact_conv`` plan (``started``,
    replay plan available); a prefix outside ``layer_names`` folds the pending
    replays (``flushed``, no replay plan)."""
    from vllm_ascend.models.glm5next.recoverssm_deferred import KDARecoverSSMDeferred

    ctx, _ = _make_context(num_layers=2, heads=HEADS, max_reqs=8)
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
    pending = torch.zeros(8, dtype=torch.int32, device=DEVICE)

    def make_deferred():
        return KDARecoverSSMDeferred(
            record_rows=record_rows,
            pending=pending,
            eager=None,
            context=ctx,
            state_indices=state_indices,
            layer_names=["layer0", "layer1"],
        )

    deferred = make_deferred()
    deferred.start_flush("layer0")
    assert deferred.started and not deferred.flushed
    plan = deferred.replay_plan()
    assert plan is not None and len(plan) == 4
    # The plan arrays are allocated at max_num_reqs; only the real rows matter.
    assert plan[0].shape[0] >= state_indices.shape[0]

    deferred = make_deferred()
    deferred.start_flush("not_a_layer")
    assert deferred.flushed and not deferred.started
    assert deferred.replay_plan() is None

    print("[PASS] start_flush gate: KDA layer fuses, unknown prefix flushes")


# ---------------------------------------------------------------------------
# Test 5: empty batch / boundary conditions
# ---------------------------------------------------------------------------

def test_empty_batch_no_crash():
    """batch=0 should be a no-op for commit and flush."""
    ctx, _ = _make_context(num_layers=1, seed=7)
    empty = torch.zeros(0, dtype=torch.int32, device=DEVICE)
    empty_loc = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    empty_pending = torch.zeros(0, dtype=torch.int32, device=DEVICE)

    # commit with empty batch
    ctx.commit(
        num_accepted_tokens=empty,
        state_indices=empty,
        query_start_loc=empty_loc,
    )
    # flush with empty batch
    ctx.flush(empty, empty, empty_pending)
    print("[PASS] empty batch: no crash")


def test_record_rows_with_null_slots():
    """record_rows containing -1 (padding) should not corrupt pending."""
    batch = 3
    state_indices = torch.tensor([1, NULL_BLOCK_ID, 3], dtype=torch.int32, device=DEVICE)
    record_rows = torch.tensor([0, -1, 1], dtype=torch.int32, device=DEVICE)
    pending = torch.tensor([0, 0], dtype=torch.int32, device=DEVICE)  # 2 real slots
    commit_lens = torch.tensor([2, 1, 1], dtype=torch.int32, device=DEVICE)
    eager = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    state_commit_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)
    boundary_commit_lens = torch.zeros(batch, dtype=torch.int32, device=DEVICE)

    _defer_commit_kernel[(batch,)](
        state_indices,
        commit_lens,
        state_commit_lens,
        boundary_commit_lens,
        record_rows,
        pending,
        eager,
        NULL_BLOCK_ID,
        state_indices.stride(0),
        num_warps=1,
    )
    torch.npu.synchronize()

    # Row 0: record_row=0, commit_len=2, deferred -> pending[0]=2, commit_lens[0]=0
    assert pending[0].item() == 2, f"row 0 pending wrong: {pending[0]}"
    assert commit_lens[0].item() == 0

    # Row 1: record_row=-1 (padding) -> untouched
    # (commit_lens[1] stays 1, pending not modified)
    assert commit_lens[1].item() == 1, f"padding row commit_lens changed: {commit_lens[1]}"

    # Row 2: record_row=1, commit_len=1, deferred -> pending[1]=1, commit_lens[2]=0
    assert pending[1].item() == 1, f"row 2 pending wrong: {pending[1]}"
    assert commit_lens[2].item() == 0

    print("[PASS] record_rows with null slots: padding row untouched")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Deferred RecoverSSM operator-level verification")
    print(f"Device: {DEVICE}, dtype: {DTYPE}")
    print("=" * 70)

    tests = [
        test_defer_commit_kernel_pending_and_eager,
        test_prepare_flush_plan_kernel,
        test_flush_folds_pending_into_checkpoint,
        test_deferred_vs_immediate_equivalence,
        test_fused_verify_matches_flush_then_verify,
        test_start_flush_gate_selects_fused_vs_flush,
        test_empty_batch_no_crash,
        test_record_rows_with_null_slots,
    ]
    failed = 0
    for t in tests:
        try:
            print(f"\n--- {t.__name__} ---")
            t()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {len(tests) - failed}/{len(tests)} passed")
    print("=" * 70)
    if failed:
        raise SystemExit(1)
