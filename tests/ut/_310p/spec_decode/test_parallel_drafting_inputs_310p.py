#
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

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch as mock_patch

import numpy as np
import torch

from tests.ut.base import TestBase
from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import expand_parallel_drafting_inputs

MASK_ID = 151666
BLOCK_SIZE = 128
GUARD = -99  # sentinel written past the used region to catch overruns

# Profile-run fixture sizes. These must differ so a fix that only covers the
# context forward is distinguishable from one that covers both.
NUM_REQS = 2
CONTEXT_TOKENS = 12


def _flag_now():
    from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

    return AscendRotaryEmbedding310._is_drafting_update_enabled


def _set_flag(state):
    from vllm_ascend._310p.ops.rotary_embedding import AscendRotaryEmbedding310

    AscendRotaryEmbedding310.set_rope_position_flag_310p(state)


@contextlib.contextmanager
def _null_context(*args, **kwargs):
    yield

# `_expand_drafting_inputs` is defined in dflash_proposer, so every name it looks
# up -- is_310p, the Triton launcher -- resolves from that module even when the
# caller is a DSpark proposer. Patching dspark_proposer.* would silently miss.
DFLASH_MOD = "vllm_ascend.spec_decode.dflash_proposer"
HELPER_MOD = "vllm_ascend._310p.spec_decode.parallel_drafting_inputs"


def reference_expand(
    *,
    next_token_ids,
    target_positions,
    context_slot_mapping,
    out_input_ids,
    out_context_positions,
    out_query_positions,
    out_context_slot_mapping,
    out_query_slot_mapping,
    out_token_indices,
    block_table,
    query_start_loc,
    seq_lens,
    num_rejected_tokens,
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_tokens,
    total_input_tokens,
    batch_size,
    sample_from_anchor,
):
    """Line-by-line transcription of
    ops/triton/spec_decode/utils.py::copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid.

    This is the golden, not a second implementation. Do not vectorize it: its whole
    value is that a mismatch points at one specific line of the original kernel.
    """
    for req_idx in range(batch_size):
        ctx_start = int(query_start_loc[req_idx])
        ctx_end = int(query_start_loc[req_idx + 1])

        for j in range(ctx_start, ctx_end):
            out_context_positions[j] = int(target_positions[j])
            out_context_slot_mapping[j] = int(context_slot_mapping[j])

        num_rejected = int(num_rejected_tokens[req_idx]) if num_rejected_tokens is not None else 0
        valid_ctx_end = ctx_end - num_rejected
        effective_seq_len = int(seq_lens[req_idx]) - num_rejected
        last_pos = int(target_positions[valid_ctx_end - 1])

        for q_idx in range(num_query_per_req):
            out_idx = req_idx * num_query_per_req + q_idx
            out_query_positions[out_idx] = last_pos + 1 + q_idx

            cache_pos = effective_seq_len + q_idx
            block_id = int(block_table[req_idx, cache_pos // block_size])
            out_query_slot_mapping[out_idx] = block_id * block_size + cache_pos % block_size

            if q_idx == 0:
                out_input_ids[out_idx] = int(next_token_ids[req_idx])
            else:
                out_input_ids[out_idx] = parallel_drafting_token_id

            if sample_from_anchor:
                out_token_indices[req_idx * num_speculative_tokens + q_idx] = out_idx
            elif q_idx > 0:
                out_token_indices[req_idx * num_speculative_tokens + q_idx - 1] = out_idx


def make_case(*, ctx_lens, seq_lens, rejected, num_query_per_req, num_spec):
    """Build one input set.

    ctx_lens is this round's scheduled segment per request; seq_lens is the total KV
    length per request. They are NOT the same thing -- conflating them is the easiest
    way to write a test that passes against a broken helper. Positions are absolute:
    a request whose total KV is 257 and which scheduled 4 tokens this round carries
    positions [253, 254, 255, 256].
    """
    batch_size = len(ctx_lens)
    assert len(seq_lens) == batch_size
    for n_ctx, n_seq in zip(ctx_lens, seq_lens):
        assert n_seq >= n_ctx, "total KV length cannot be shorter than this round's segment"
    total = sum(ctx_lens)

    qsl = torch.zeros(batch_size + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)

    positions = torch.cat(
        [torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32) for n_ctx, n_seq in zip(ctx_lens, seq_lens)]
    )
    ctx_slots = torch.arange(total, dtype=torch.int32) * 3 + 7

    max_blocks = (max(seq_lens) + num_query_per_req) // BLOCK_SIZE + 2
    # Descending, non-contiguous physical page ids, shifted past the logical index
    # range so no entry can equal its own column.
    #
    # That invariant is what makes these fixtures discriminating: a query slot is
    # physical * block_size + offset while the raw cache position is
    # logical * block_size + offset, with the same offset. So physical != logical
    # is exactly the condition under which a slot differs from the position, and a
    # helper that ignored the block table entirely would be caught.
    #
    # The plain reversal used before had a fixed point at the middle column for odd
    # max_blocks (arange(3).flip(0) == [2, 1, 0], so column 1 maps to page 1). With
    # seq_lens=[129] the whole query block landed on that column and every slot
    # equalled its cache position.
    block_table = (
        torch.arange(batch_size * max_blocks, dtype=torch.int32).flip(0).reshape(batch_size, max_blocks) + max_blocks
    )
    assert bool(
        (block_table != torch.arange(max_blocks, dtype=torch.int32)).all()
    ), "block table maps some logical page to the same physical page; slots would be degenerate"

    return dict(
        next_token_ids=torch.arange(batch_size, dtype=torch.int32) + 1000,
        target_positions=positions,
        context_slot_mapping=ctx_slots,
        block_table=block_table,
        query_start_loc=qsl,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_rejected_tokens=(torch.tensor(rejected, dtype=torch.int32) if rejected is not None else None),
        parallel_drafting_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        num_query_per_req=num_query_per_req,
        num_speculative_tokens=num_spec,
        total_input_tokens=total,
        batch_size=batch_size,
    )


def run_both(case, sample_from_anchor):
    b, q, k = case["batch_size"], case["num_query_per_req"], case["num_speculative_tokens"]
    total = case["total_input_tokens"]
    slack = 8  # extra room so an overrun lands on GUARD instead of out of bounds

    def fresh():
        return dict(
            out_input_ids=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_positions=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_positions=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_context_slot_mapping=torch.full((total + slack,), GUARD, dtype=torch.int32),
            out_query_slot_mapping=torch.full((b * q + slack,), GUARD, dtype=torch.int32),
            out_token_indices=torch.full((b * k + slack,), GUARD, dtype=torch.int32),
        )

    got, want = fresh(), fresh()
    expand_parallel_drafting_inputs(**case, **got, sample_from_anchor=sample_from_anchor)
    reference_expand(**case, **want, sample_from_anchor=sample_from_anchor)
    return got, want, slack


class TestExpandParallelDraftingInputs(TestBase):
    def _assert_matches(self, case, sample_from_anchor):
        got, want, slack = run_both(case, sample_from_anchor)
        for name in want:
            torch.testing.assert_close(got[name], want[name], msg=f"{name} mismatch")
            tail = got[name][-slack:]
            self.assertTrue(
                bool((tail == GUARD).all()),
                f"{name}: helper wrote past its used region (tail={tail.tolist()})",
            )

    def test_dflash_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )

    def test_dspark_single_request(self):
        self._assert_matches(
            make_case(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=7, num_spec=7),
            sample_from_anchor=True,
        )

    def test_ragged_segment_with_distinct_seq_lens(self):
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )

    def test_rejected_tail_excluded_from_query_but_kept_in_context(self):
        # Every request keeps at least one valid context token after rejection.
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=9, num_spec=8),
            sample_from_anchor=False,
        )
        self._assert_matches(
            make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=7, num_spec=7),
            sample_from_anchor=True,
        )

    def test_query_slots_cross_page_boundary(self):
        # effective_seq_len lands at 127/128/129 so the K+1 query slots straddle a
        # kernel page and must follow the block table into a new physical page.
        for seq_len in (127, 128, 129):
            self._assert_matches(
                make_case(ctx_lens=[2], seq_lens=[seq_len], rejected=None, num_query_per_req=9, num_spec=8),
                sample_from_anchor=False,
            )

    def test_context_buffer_untouched_beyond_scheduled_segment(self):
        case = make_case(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=None, num_query_per_req=9, num_spec=8)
        got, _, _ = run_both(case, sample_from_anchor=False)
        total = case["total_input_tokens"]
        self.assertTrue(bool((got["out_context_positions"][total:] == GUARD).all()))
        self.assertTrue(bool((got["out_context_slot_mapping"][total:] == GUARD).all()))

    def test_query_slots_never_equal_raw_cache_positions(self):
        """The block table must be load-bearing for every shape the suite uses.

        This is implied by the no-fixed-point invariant asserted in make_case, but
        checking the end result across all shapes keeps the two from drifting apart
        if the fixture construction is ever changed.
        """
        shapes = [
            dict(ctx_lens=[2], seq_lens=[127], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[2], seq_lens=[128], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[2], seq_lens=[129], rejected=None, num_query_per_req=9, num_spec=8),
            dict(ctx_lens=[4], seq_lens=[257], rejected=None, num_query_per_req=7, num_spec=7),
            dict(ctx_lens=[1, 4, 2], seq_lens=[257, 134, 66], rejected=[0, 3, 1], num_query_per_req=9, num_spec=8),
        ]
        for shape in shapes:
            case = make_case(**shape)
            _, want, _ = run_both(case, sample_from_anchor=False)
            b, q = case["batch_size"], case["num_query_per_req"]
            rejected = shape["rejected"] or [0] * b
            for req in range(b):
                effective_seq_len = shape["seq_lens"][req] - rejected[req]
                slots = want["out_query_slot_mapping"][req * q : (req + 1) * q]
                raw = torch.arange(effective_seq_len, effective_seq_len + q, dtype=torch.int32)
                self.assertFalse(
                    bool(torch.equal(slots, raw)),
                    f"{shape}: request {req} slots equal raw cache positions, "
                    f"so the block table is not exercised",
                )


class _ExplodingLauncher:
    """Stand-in for the Triton kernel: subscripting it (kernel[grid]) raises.

    Used in both directions -- the 310P path must never reach it, and the Triton
    path must always reach it.
    """

    def __getitem__(self, _grid):
        raise AssertionError("Triton kernel was launched")


def make_dflash_proposer(*, num_spec=8, selected_block_size=BLOCK_SIZE, kernel_block_size=BLOCK_SIZE, candidates=None):
    """Build an AscendDflashProposer without running __init__.

    Only the attributes set_inputs_first_pass actually touches are stubbed.
    """
    from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

    p = AscendDflashProposer.__new__(AscendDflashProposer)
    p.num_speculative_tokens = num_spec
    p.device = torch.device("cpu")
    p.parallel_drafting_token_id = MASK_ID
    p.kernel_block_size = kernel_block_size
    p.kv_cache_gid = 0
    p.input_ids = torch.zeros(64, dtype=torch.int32)
    p.positions = torch.zeros(64, dtype=torch.int32)
    p._context_positions_buffer = torch.zeros(64, dtype=torch.int32)
    p._context_slot_mapping_buffers = torch.zeros(64, dtype=torch.int32)
    p._slot_mapping_buffer = torch.zeros(64, dtype=torch.int32)
    p._dflash_hidden_states = torch.zeros(64, 8)
    p.arange_dflash = torch.arange(65, dtype=torch.int32)
    p.token_arange_np = np.arange(65, dtype=np.int32)
    p.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table={0: SimpleNamespace(block_size=selected_block_size)}),
        kernel_block_sizes={0: list(candidates if candidates is not None else [128, 64])},
    )
    return p


def make_dflash_cad(ctx_lens, seq_lens):
    batch_size = len(ctx_lens)
    total = sum(ctx_lens)
    qsl = torch.zeros(batch_size + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(ctx_lens, dtype=torch.int32).cumsum(0)
    return SimpleNamespace(
        num_reqs=batch_size,
        slot_mapping=torch.arange(total, dtype=torch.int32) * 3 + 7,
        block_table_tensor=torch.arange(batch_size * 6, dtype=torch.int32).flip(0).reshape(batch_size, 6) + 6,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.clone(),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_seq_len=max(seq_lens),
        num_actual_tokens=total,
        max_query_len=0,
        causal=True,
        attn_mask=object(),
        attn_state=None,
        actual_seq_lengths_q=[],
        decode_token_per_req=0,
    )


class TestDflashDispatch(TestBase):
    CTX_LENS = [2, 4]
    SEQ_LENS = [257, 134]

    def _call(self, proposer, cad):
        total = sum(self.CTX_LENS)
        return proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(total, dtype=torch.int32),
            next_token_ids=torch.tensor([11, 22], dtype=torch.int32),
            target_positions=torch.cat(
                [
                    torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32)
                    for n_ctx, n_seq in zip(self.CTX_LENS, self.SEQ_LENS)
                ]
            ),
            target_hidden_states=torch.zeros(total, 8),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )

    def test_310p_uses_helper_and_never_launches_triton(self):
        seen = {}

        def spy(**kwargs):
            seen.update(kwargs)

        proposer = make_dflash_proposer()
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)
        with (
            mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True),
            mock_patch(f"{HELPER_MOD}.expand_parallel_drafting_inputs", spy),
            mock_patch(f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                       _ExplodingLauncher()),
        ):
            self._call(proposer, cad)

        self.assertEqual(seen["num_query_per_req"], 9)
        self.assertEqual(seen["num_speculative_tokens"], 8)
        self.assertIs(seen["sample_from_anchor"], False)
        self.assertEqual(seen["block_size"], BLOCK_SIZE)
        self.assertEqual(seen["batch_size"], 2)
        self.assertEqual(seen["total_input_tokens"], sum(self.CTX_LENS))
        self.assertIsNone(seen["num_rejected_tokens"])
        # Buffer identity: catches wiring the helper to the wrong destination,
        # which a value comparison would not.
        self.assertIs(seen["out_input_ids"], proposer.input_ids)
        self.assertIs(seen["out_query_positions"], proposer.positions)
        self.assertIs(seen["out_context_positions"], proposer._context_positions_buffer)
        self.assertIs(seen["out_context_slot_mapping"], proposer._context_slot_mapping_buffers)
        self.assertIs(seen["out_query_slot_mapping"], proposer._slot_mapping_buffer)
        self.assertIs(seen["block_table"], cad.block_table_tensor)

    def test_non_310p_still_launches_triton(self):
        proposer = make_dflash_proposer()
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)

        def must_not_run(_proposer):
            raise AssertionError("resolve_310p_block_size ran on the non-310P path")

        with (
            mock_patch(f"{DFLASH_MOD}.is_310p", return_value=False),
            mock_patch(f"{HELPER_MOD}.resolve_310p_block_size", must_not_run),
            mock_patch(f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                       _ExplodingLauncher()),
        ):
            with self.assertRaisesRegex(AssertionError, "Triton kernel was launched"):
                self._call(proposer, cad)

    def test_310p_slot_mapping_matches_the_verified_helper(self):
        """End-to-end through the real helper, not a spy: proves the wiring works."""
        proposer = make_dflash_proposer()
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)
        with (
            mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True),
            mock_patch(f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                       _ExplodingLauncher()),
        ):
            num_query_total, _, _, _ = self._call(proposer, cad)

        want = torch.zeros(64, dtype=torch.int32)
        for req, seq_len in enumerate(self.SEQ_LENS):
            for q_idx in range(9):
                cache_pos = seq_len + q_idx
                physical = int(cad.block_table_tensor[req, cache_pos // BLOCK_SIZE])
                want[req * 9 + q_idx] = physical * BLOCK_SIZE + cache_pos % BLOCK_SIZE
        torch.testing.assert_close(proposer._slot_mapping_buffer[:num_query_total], want[:num_query_total])

    def test_310p_metadata_is_switched_to_non_causal_parallel_draft(self):
        proposer = make_dflash_proposer()
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)
        with (
            mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True),
            mock_patch(f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                       _ExplodingLauncher()),
        ):
            num_query_total, token_indices, out_cad, _ = self._call(proposer, cad)

        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        self.assertEqual(num_query_total, 2 * 9)
        self.assertEqual(token_indices.shape[0], 2 * 8)
        self.assertEqual(out_cad.num_actual_tokens, 2 * 9)
        self.assertEqual(out_cad.max_query_len, 9)
        self.assertIs(out_cad.causal, False)
        self.assertIsNone(out_cad.attn_mask)
        self.assertEqual(out_cad.attn_state, AscendAttentionState.ChunkedPrefill)
        self.assertEqual(out_cad.actual_seq_lengths_q, [9, 9])
        # seq_lens must become effective history + this round's query block.
        torch.testing.assert_close(out_cad.seq_lens, torch.tensor([257 + 9, 134 + 9], dtype=torch.int32))


class TestResolve310pBlockSize(TestBase):
    def _resolve(self, **kwargs):
        from vllm_ascend._310p.spec_decode.parallel_drafting_inputs import resolve_310p_block_size

        return resolve_310p_block_size(make_dflash_proposer(**kwargs))

    def test_accepts_the_supported_configuration(self):
        self.assertEqual(self._resolve(), BLOCK_SIZE)

    def test_rejects_a_selected_size_outside_scope(self):
        with self.assertRaisesRegex(RuntimeError, "only covers kernel block size 128"):
            self._resolve(selected_block_size=64)

    def test_rejects_drafter_disagreeing_with_the_block_table(self):
        with self.assertRaisesRegex(RuntimeError, "kernel_block_size is 64"):
            self._resolve(kernel_block_size=64)

    def test_rejects_a_candidate_list_without_the_supported_size(self):
        with self.assertRaisesRegex(RuntimeError, "not in the runner's candidate list"):
            self._resolve(candidates=[64])


def make_dspark_proposer(*, num_spec=7, num_groups=1, kv_spec_block_size=BLOCK_SIZE):
    """Build an AscendDSparkProposer without running __init__.

    DSpark drives the expansion per KV cache group, so it needs the per-group
    buffers on top of what DFlash uses.
    """
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

    p = AscendDSparkProposer.__new__(AscendDSparkProposer)
    p.num_speculative_tokens = num_spec
    # Anchor-first layout: the only one 310P covers, so q == K rather than 1 + K.
    p.sample_from_anchor = True
    p.num_query_per_req = num_spec
    p.device = torch.device("cpu")
    p.parallel_drafting_token_id = MASK_ID
    p.kernel_block_size = BLOCK_SIZE
    p.kv_cache_gid = 0
    p.input_ids = torch.zeros(64, dtype=torch.int32)
    p.positions = torch.zeros(64, dtype=torch.int32)
    p._context_positions_buffer = torch.zeros(64, dtype=torch.int32)
    p._dflash_hidden_states = torch.zeros(64, 8)
    p._dspark_seed_buffer = torch.zeros(64, dtype=torch.int64)
    p.arange_dflash = torch.arange(65, dtype=torch.int32)
    p.token_arange_np = np.arange(65, dtype=np.int32)
    p.runner = SimpleNamespace(
        input_batch=SimpleNamespace(block_table={0: SimpleNamespace(block_size=BLOCK_SIZE)}),
        kernel_block_sizes={0: [128, 64]},
    )

    gids = list(range(num_groups))
    p.draft_attn_groups = [
        SimpleNamespace(kv_cache_group_id=gid, kv_cache_spec=SimpleNamespace(block_size=kv_spec_block_size))
        for gid in gids
    ]
    # Stub the source, not the derived dict: set_inputs_first_pass rebuilds
    # _per_group_block_table_buffers from _per_group_block_tables on every call,
    # so anything written to the former is discarded before it is read.
    p._per_group_block_tables = {
        gid: torch.arange(2 * 6, dtype=torch.int32).flip(0).reshape(2, 6) + 6 for gid in gids
    }
    p._per_group_block_table_buffers = {}
    p._per_group_slot_mappings = {gid: torch.arange(64, dtype=torch.int32) * 3 + 7 for gid in gids}
    p._per_group_context_slot_mapping_buffers = {gid: torch.zeros(64, dtype=torch.int32) for gid in gids}
    p._per_group_query_slot_mapping_buffers = {gid: torch.zeros(64, dtype=torch.int32) for gid in gids}
    p._layer_group_idx = gids
    return p


class TestDSparkDispatch(TestBase):
    CTX_LENS = [2, 4]
    SEQ_LENS = [257, 134]
    Q_PER_REQ = 7  # DSpark queries K positions, not K + 1

    def _call(self, proposer, cad):
        total = sum(self.CTX_LENS)
        return proposer.set_inputs_first_pass(
            target_token_ids=torch.zeros(total, dtype=torch.int32),
            next_token_ids=torch.tensor([11, 22], dtype=torch.int32),
            target_positions=torch.cat(
                [
                    torch.arange(n_seq - n_ctx, n_seq, dtype=torch.int32)
                    for n_ctx, n_seq in zip(self.CTX_LENS, self.SEQ_LENS)
                ]
            ),
            target_hidden_states=torch.zeros(total, 8),
            token_indices_to_sample=None,
            cad=cad,
            num_rejected_tokens_gpu=None,
        )

    def _run_310p(self, proposer, spy=None):
        """Drive set_inputs_first_pass as if on 310P.

        `is_310p` is patched in both modules: dspark_proposer reads it for the
        single-group scope guard, dflash_proposer for the dispatch itself.
        With spy=None the real helper runs, so the wiring is exercised end to end.
        """
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True))
            stack.enter_context(
                mock_patch("vllm_ascend.spec_decode.dspark_proposer.is_310p", return_value=True)
            )
            stack.enter_context(
                mock_patch(
                    f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                    _ExplodingLauncher(),
                )
            )
            if spy is not None:
                stack.enter_context(mock_patch(f"{HELPER_MOD}.expand_parallel_drafting_inputs", spy))
            return self._call(proposer, cad), cad

    def test_single_group_calls_helper_once_with_dspark_semantics(self):
        calls = []

        def spy(**kwargs):
            calls.append(kwargs)

        proposer = make_dspark_proposer()
        self._run_310p(proposer, spy=spy)

        self.assertEqual(len(calls), 1, "Qwen3-8B DSpark must drive exactly one draft KV group")
        seen = calls[0]
        # DSpark queries K positions and samples from the anchor; DFlash does K + 1
        # and skips it. Getting these wrong does not raise, it only degrades
        # acceptance, so assert them explicitly.
        self.assertEqual(seen["num_query_per_req"], 7)
        self.assertEqual(seen["num_speculative_tokens"], 7)
        self.assertIs(seen["sample_from_anchor"], True)
        # 310P substitutes the pinned kernel block size for the caller's allocation
        # block size, so this is 128 (the cache block), not 7 (the algorithm block).
        self.assertEqual(seen["block_size"], BLOCK_SIZE)
        # Per-group buffers, not the DFlash flat ones.
        self.assertIs(seen["out_context_slot_mapping"], proposer._per_group_context_slot_mapping_buffers[0])
        self.assertIs(seen["out_query_slot_mapping"], proposer._per_group_query_slot_mapping_buffers[0])
        self.assertIs(seen["context_slot_mapping"], proposer._per_group_slot_mappings[0])
        self.assertIs(seen["block_table"], proposer._per_group_block_tables[0])

    def test_multi_group_is_refused_on_310p(self):
        proposer = make_dspark_proposer(num_groups=2)
        with self.assertRaisesRegex(RuntimeError, "single draft KV cache group"):
            self._run_310p(proposer, spy=lambda **kw: None)

    def test_metadata_matches_dspark_layout(self):
        proposer = make_dspark_proposer()
        (result, _) = self._run_310p(proposer)
        num_query_total, token_indices, out_cad, _ = result

        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        self.assertEqual(num_query_total, 2 * self.Q_PER_REQ)
        self.assertEqual(token_indices.shape[0], 2 * self.Q_PER_REQ)
        self.assertEqual(out_cad.num_actual_tokens, 2 * self.Q_PER_REQ)
        self.assertEqual(out_cad.num_input_tokens, 2 * self.Q_PER_REQ)
        self.assertEqual(out_cad.max_query_len, self.Q_PER_REQ)
        # positions is handed over whole, not sliced to num_query_total: the
        # attention backend slices it to num_input_tokens, which is DP-padded and
        # can exceed the local query count. A pre-slice here reads out of bounds
        # under multi-DP, so assert identity rather than length.
        self.assertIs(out_cad.positions, proposer.positions)
        self.assertGreaterEqual(out_cad.positions.shape[0], num_query_total)
        self.assertIs(out_cad.causal, False)
        self.assertIsNone(out_cad.attn_mask)
        self.assertEqual(out_cad.attn_state, AscendAttentionState.ChunkedPrefill)
        self.assertEqual(out_cad.actual_seq_lengths_q, [self.Q_PER_REQ] * 2)
        torch.testing.assert_close(
            out_cad.seq_lens,
            torch.tensor([257 + self.Q_PER_REQ, 134 + self.Q_PER_REQ], dtype=torch.int32),
        )
        # slot_mapping must come from the primary group's query buffer.
        expected = proposer._per_group_query_slot_mapping_buffers[0][: 2 * self.Q_PER_REQ]
        self.assertEqual(out_cad.slot_mapping.data_ptr(), expected.data_ptr())

    def test_slot_mapping_matches_the_verified_helper(self):
        """Runs the real helper end to end, so the per-group wiring is checked."""
        proposer = make_dspark_proposer()
        (result, _) = self._run_310p(proposer)
        num_query_total = result[0]

        block_table = proposer._per_group_block_tables[0]
        want = torch.zeros(64, dtype=torch.int32)
        for req, seq_len in enumerate(self.SEQ_LENS):
            for q_idx in range(self.Q_PER_REQ):
                cache_pos = seq_len + q_idx
                physical = int(block_table[req, cache_pos // BLOCK_SIZE])
                want[req * self.Q_PER_REQ + q_idx] = physical * BLOCK_SIZE + cache_pos % BLOCK_SIZE
        torch.testing.assert_close(
            proposer._per_group_query_slot_mapping_buffers[0][:num_query_total],
            want[:num_query_total],
        )

    def test_non_310p_still_launches_triton(self):
        proposer = make_dspark_proposer()
        cad = make_dflash_cad(self.CTX_LENS, self.SEQ_LENS)
        with (
            mock_patch(f"{DFLASH_MOD}.is_310p", return_value=False),
            mock_patch("vllm_ascend.spec_decode.dspark_proposer.is_310p", return_value=False),
            mock_patch(f"{DFLASH_MOD}.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid",
                       _ExplodingLauncher()),
        ):
            with self.assertRaisesRegex(AssertionError, "Triton kernel was launched"):
                self._call(proposer, cad)


class TestProfileRopeFlagCoverage(TestBase):
    """dummy_run(is_profile=True) must keep the 310P drafting RoPE flag on for both
    the context KV precompute and the query forward that follows it.

    The flag is what makes each rotary call refresh the global cos/sin slice with
    its own positions. Covering only the first forward leaves the query forward
    reading the context slice, which produces wrong numbers rather than an error.
    """

    def _fake_model(self, seen):
        class FakeModel:
            def precompute_and_store_context_kv(_self, states, positions):
                seen.append(("context", _flag_now(), int(positions.shape[0])))

            def __call__(_self, *, input_ids, positions, inputs_embeds):
                seen.append(("query", _flag_now(), int(positions.shape[0])))

        return FakeModel()

    def _make_proposer(self, cls, seen, num_spec):
        p = cls.__new__(cls)
        p.model = self._fake_model(seen)
        p.num_speculative_tokens = num_spec
        # Only DSpark reads this attribute; DFlash derives 1 + K locally. The
        # anchor-first value is the one 310P covers.
        p.sample_from_anchor = True
        p.num_query_per_req = num_spec
        p.max_query_tokens = 64
        p.use_cuda_graph = False
        p.vllm_config = SimpleNamespace()
        p.input_ids = torch.zeros(64, dtype=torch.int32)
        p.hidden_states = torch.zeros(64, 8)
        p.token_indices_to_sample = torch.zeros(64, dtype=torch.int32)
        p._context_positions_buffer = torch.arange(64, dtype=torch.int32)
        p._slot_mapping_buffer = torch.zeros(64, dtype=torch.int32)
        p._dflash_hidden_states = torch.zeros(64, 8)
        p.arange_dflash = torch.arange(65, dtype=torch.int32)
        p.token_arange_np = np.arange(65, dtype=np.int32)
        p.attn_layer_names = []
        p.draft_attn_groups = []
        p.kv_cache_gid = 0
        p._get_positions = lambda n: torch.arange(n, dtype=torch.int32)
        p._pad_draft_buffers = lambda *a, **kw: None
        p.runner = MagicMock()
        p.runner._sync_metadata_across_dp.return_value = (CONTEXT_TOKENS, None, None)
        p.runner.attn_groups = []
        return p

    def _run_profile(self, cls, num_spec, caller_module):
        import importlib

        seen = []
        proposer = self._make_proposer(cls, seen, num_spec)
        module = importlib.import_module(caller_module)

        # set_ascend_forward_context / get_forward_context are imported directly
        # into each proposer module, so they must be patched where dummy_run reads
        # them; patching llm_base_proposer.* would not be seen. dspark_proposer
        # does not import get_forward_context at all, so patch it only where it
        # exists rather than creating an attribute that production code lacks.
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True))
            stack.enter_context(mock_patch(f"{caller_module}.set_ascend_forward_context", _null_context))
            if hasattr(module, "get_forward_context"):
                stack.enter_context(mock_patch(f"{caller_module}.get_forward_context", MagicMock()))
            proposer.dummy_run(num_tokens=CONTEXT_TOKENS, num_reqs=NUM_REQS, is_profile=True)
        return seen

    def _assert_both_forwards_covered(self, seen):
        self.assertEqual([kind for kind, _, _ in seen], ["context", "query"])
        for kind, flag, _ in seen:
            self.assertTrue(flag, f"{kind} forward ran with the drafting RoPE flag off")
        ctx_len, q_len = seen[0][2], seen[1][2]
        self.assertNotEqual(
            ctx_len,
            q_len,
            "context and query lengths coincide, so this cannot tell a half fix "
            "(flag on for only the first forward) from a full one",
        )

    def test_dflash_profile_covers_context_and_query(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        self._assert_both_forwards_covered(
            self._run_profile(AscendDflashProposer, 8, "vllm_ascend.spec_decode.dflash_proposer")
        )

    def test_dspark_profile_covers_context_and_query(self):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        self._assert_both_forwards_covered(
            self._run_profile(AscendDSparkProposer, 7, "vllm_ascend.spec_decode.dspark_proposer")
        )

    def test_flag_is_restored_after_the_profile_branch(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        _set_flag(False)
        self._run_profile(AscendDflashProposer, 8, "vllm_ascend.spec_decode.dflash_proposer")
        self.assertFalse(_flag_now(), "flag leaked out of the profile branch")

    def test_flag_is_restored_even_when_the_forward_raises(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        proposer = AscendDflashProposer.__new__(AscendDflashProposer)
        # Forcing is_310p matters: without it the context manager yields without
        # touching the flag and the restore assertion below passes vacuously.
        with mock_patch(f"{DFLASH_MOD}.is_310p", return_value=True):
            _set_flag(False)
            with self.assertRaises(ValueError):
                with proposer._profile_rope_context():
                    self.assertTrue(_flag_now(), "flag was never set, so the check below is vacuous")
                    raise ValueError("boom")
        self.assertFalse(_flag_now())

    def test_non_310p_leaves_the_flag_untouched(self):
        from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

        proposer = AscendDflashProposer.__new__(AscendDflashProposer)
        _set_flag(False)
        with mock_patch(f"{DFLASH_MOD}.is_310p", return_value=False):
            with proposer._profile_rope_context():
                self.assertFalse(_flag_now(), "non-310P must not enable the 310P drafting flag")
        self.assertFalse(_flag_now())


class TestBonusAnchorRefusedOn310p(TestBase):
    """310P covers only the anchor-first DSpark layout.

    With ``dspark_bonus_anchor=True`` a request queries 1 + K positions instead
    of K. The 310P expansion fallback and the FIA route both assume q == K, and
    the mismatch does not raise anywhere -- it shifts every slot and token index
    by one and silently returns wrong tokens. So construction must refuse it.
    """

    NUM_SPEC = 7
    MAX_BATCH = 4

    def _init_proposer(self, *, bonus_anchor, on_310p):
        from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer

        hf_config = SimpleNamespace(dspark_bonus_anchor=True) if bonus_anchor else SimpleNamespace()
        draft_model_config = SimpleNamespace(hf_config=hf_config, get_hidden_size=lambda: 8)
        vllm_config = SimpleNamespace(
            speculative_config=SimpleNamespace(
                draft_sample_method="greedy",
                draft_model_config=draft_model_config,
            )
        )

        def parent_init(proposer, cfg, device, runner=None):
            del runner
            proposer.draft_model_config = cfg.speculative_config.draft_model_config
            proposer.num_speculative_tokens = self.NUM_SPEC
            proposer.max_batch_size = self.MAX_BATCH
            proposer.max_num_tokens = 64
            proposer.dtype = torch.float32
            proposer.device = device
            proposer.hidden_size = 8
            proposer.hidden_states = torch.empty(0)
            proposer._dflash_hidden_states = torch.empty(0)

        with (
            mock_patch.object(AscendDSparkProposer.__base__, "__init__", parent_init),
            mock_patch("vllm_ascend.spec_decode.dspark_proposer.is_310p", return_value=on_310p),
        ):
            return AscendDSparkProposer(vllm_config, torch.device("cpu"))

    def test_bonus_anchor_is_refused_on_310p(self):
        with self.assertRaisesRegex(NotImplementedError, "dspark_bonus_anchor"):
            self._init_proposer(bonus_anchor=True, on_310p=True)

    def test_bonus_anchor_is_allowed_off_310p(self):
        """The guard is device-scoped: other devices keep upstream's layout."""
        proposer = self._init_proposer(bonus_anchor=True, on_310p=False)
        self.assertFalse(proposer.sample_from_anchor)
        self.assertEqual(proposer.num_query_per_req, 1 + self.NUM_SPEC)

    def test_anchor_first_is_accepted_on_310p(self):
        """The guard is layout-scoped: it must not reject the supported case."""
        proposer = self._init_proposer(bonus_anchor=False, on_310p=True)
        self.assertTrue(proposer.sample_from_anchor)
        self.assertEqual(proposer.num_query_per_req, self.NUM_SPEC)
