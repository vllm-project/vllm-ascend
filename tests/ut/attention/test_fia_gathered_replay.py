# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the gathered HND full-graph FIA replay contract.

PR #22649 review (Liuchenbing-2026): under full_graph_fia the head-major
(HND/LBHNC) KV-cache read gathers the referenced blocks into compact
buffers and captures FIA against them plus a sequential remapped_table
(constant content: 0 .. num_rows * table_width - 1, plus one padding row).
Replay used to rebind the FIA block_table input to the raw physical
metadata.block_tables, which indexes out of the compact gathered buffer
(wrong blocks, or OOB once a physical id >= the gathered row count).

These tests pin the fixed contract:

1. FIAParamProvider.resolve emits no block_table override for gathered
   captures, so the captured remapped table stays bound at replay (while
   the default NHD layout keeps rebinding the physical table).
2. full_graph_fia registers the provider with the remapped flag and
   retains the remapped table strongly: UpdatableGraph.register_task wraps
   the capture kwargs with weak_ref_tensors, which do not keep the storage
   alive.
3. The captured gather covers the replay-time synthetic padding request
   via one extra gathered slot holding physical block 0 and one extra
   remapped row mapping to that slot.
4. The eager path is unchanged: no padding row outside graph capture.
5. The eager gather is bounded by the live KV lengths (the P2 review of
   zhaochuang001): the host-side seq_lens_list batch maximum limits the
   gathered width to the pages the batch actually reads, while graph
   capture keeps the full static width.

The tests run on CPU tensors only (no NPU device or torch_npu ops are
executed; the torch_npu symbols referenced by full_graph_fia are stubbed).
"""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.attention.attention_v1 as attention_v1
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackendImpl,
    AscendAttentionState,
    FIAParamProvider,
)

NUM_BLOCKS = 7
NUM_KV_HEADS = 2
BLOCK_SIZE = 4
HEAD_DIM = 8
NUM_HEADS = 4

# Physical ids deliberately include values >= rows * table_width (the
# gathered buffer holds rows * table_width + 1 slots): with the pre-fix
# raw-table rebinding these ids would read out of the compact buffer.
PHYSICAL_TABLE = torch.tensor([[3, 6], [0, 2]], dtype=torch.int32)
NUM_ROWS, TABLE_WIDTH = PHYSICAL_TABLE.shape
NUM_GATHERED = NUM_ROWS * TABLE_WIDTH

# A wide block table (the P2 review scenario): only the first two columns
# are live; the remaining columns are unfilled (-1) or repeated block-0
# padding that a full-width eager gather would copy for nothing.
WIDE_TABLE = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [2, 3, -1, -1, -1, -1, -1, -1],
        [4, 5, -1, -1, -1, -1, -1, -1],
        [6, 0, -1, -1, -1, -1, -1, -1],
    ],
    dtype=torch.int32,
)
WIDE_ROWS, WIDE_WIDTH = WIDE_TABLE.shape
# Live KV lengths: the batch maximum is 8 tokens = 2 blocks at BLOCK_SIZE.
WIDE_SEQ_LENS = [8, 5, 7, 4]
WIDE_BOUND_WIDTH = 2


def _make_backend():
    backend = AscendAttentionBackendImpl.__new__(AscendAttentionBackendImpl)
    backend.key_cache = torch.arange(NUM_BLOCKS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM, dtype=torch.float32).reshape(
        NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM
    )
    backend.value_cache = torch.full((NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM), 7.0)
    backend.use_bnsd_kv_cache = True
    backend.sliding_window = None
    backend.num_heads = NUM_HEADS
    backend.num_kv_heads = NUM_KV_HEADS
    backend.head_size = HEAD_DIM
    backend.scale = 0.5
    backend.enable_c8_quant = False
    backend.sinks = None
    backend.kv_cache_dtype = torch.float32
    backend._layer_name = "layer0"
    backend._use_layer_aware_fia_graph_replay = False
    backend._use_max_workspace_for_fia_graph = False
    # Production __init__ allocates this; __new__ bypasses __init__.
    backend._gathered_pa_read_strong_refs = []
    return backend


def _make_metadata(block_tables):
    return SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        block_tables=block_tables,
        seq_lens_list=[3, 2],
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        actual_seq_lengths_q=[1, 1],
        num_actual_tokens=2,
        causal=True,
        attn_mask=None,
        max_query_len=1,
    )


def _expected_token_major(cache, flat_ids):
    return cache.index_select(0, flat_ids).permute(0, 2, 1, 3).reshape(flat_ids.numel(), BLOCK_SIZE, -1)


def _run_full_graph_fia(backend, block_tables, capturing=True):
    """Drive the real full_graph_fia with stubbed NPU symbols.

    Returns the recorded (operation, weakly-ref'd kwargs, provider) triple
    exactly as UpdatableGraph.register_task would have stored it.
    """
    stub = SimpleNamespace(
        _npu_fused_infer_attention_score_get_max_workspace=Mock(return_value=torch.empty(8, dtype=torch.int8)),
        npu_fused_infer_attention_score=SimpleNamespace(out=Mock()),
    )
    recorder = {}

    def fake_register_task(operation, kwargs, provider):
        from vllm_ascend.utils import weak_ref_tensors

        recorder["operation"] = operation
        recorder["kwargs"] = weak_ref_tensors(kwargs)
        recorder["provider"] = provider

    extra_ctx = SimpleNamespace(capturing=capturing, is_draft_model=False)
    query = torch.zeros(2, NUM_HEADS, HEAD_DIM)
    output = torch.empty(2, NUM_HEADS * HEAD_DIM)
    metadata = _make_metadata(block_tables)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(attention_v1, "torch_npu", stub)
        mp.setattr(attention_v1, "register_task", fake_register_task)
        mp.setattr(attention_v1, "_EXTRA_CTX", extra_ctx)
        backend.full_graph_fia(query, None, None, metadata, output)
    assert recorder, "full_graph_fia must register an FIA task under capture"
    return recorder


class TestProviderResolveContract:
    def test_gathered_capture_does_not_rebind_block_table(self):
        """The review's core complaint: replay keeps the remapped table."""
        provider = FIAParamProvider(
            "layer0",
            None,
            is_draft_model=False,
            is_non_pa=False,
            block_table_is_remapped=True,
        )
        raw_table = torch.full((2, 2), 99, dtype=torch.int32)
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1, 1],
                    seq_lens_list=[3, 2],
                    block_tables=raw_table,
                    max_query_len=1,
                )
            }
        )
        assert "block_table" not in params, (
            "gathered HND captures must not rebind the FIA block table to raw physical ids at replay"
        )
        assert params["actual_seq_lengths"] == [1, 1]
        assert params["actual_seq_lengths_kv"] == [3, 2]

    def test_gathered_draft_capture_does_not_rebind_block_table(self):
        provider = FIAParamProvider(
            "layer0",
            None,
            is_draft_model=True,
            is_non_pa=False,
            block_table_is_remapped=True,
        )
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1],
                    seq_lens_list=[4],
                    block_tables=torch.zeros(1, 2, dtype=torch.int32),
                    max_query_len=1,
                )
            }
        )
        assert "block_table" not in params

    def test_default_layout_still_rebinds_physical_block_table(self):
        provider = FIAParamProvider("layer0", None, is_non_pa=False)
        raw_table = torch.zeros(2, 2, dtype=torch.int32)
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1, 1],
                    seq_lens_list=[3, 2],
                    block_tables=raw_table,
                    max_query_len=1,
                )
            }
        )
        assert params["block_table"] is raw_table

    def test_non_pa_capture_still_rebinds_to_none(self):
        provider = FIAParamProvider("layer0", None, is_non_pa=True)
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1],
                    seq_lens_list=[1],
                    block_tables=None,
                    max_query_len=1,
                )
            }
        )
        assert params["block_table"] is None

    def test_sliding_window_default_layout_still_omits_block_table(self):
        provider = FIAParamProvider("layer0", 512)
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1],
                    seq_lens_list=[9],
                    block_tables=torch.zeros(1, 2, dtype=torch.int32),
                    max_query_len=1,
                )
            }
        )
        assert "block_table" not in params


class TestFullGraphCapture:
    def test_capture_registers_remapped_provider_and_table(self):
        backend = _make_backend()
        recorder = _run_full_graph_fia(backend, PHYSICAL_TABLE.clone())

        provider = recorder["provider"]
        assert provider.block_table_is_remapped is True

        remapped = recorder["kwargs"]["block_table"]
        # Sequential mapping over the compact gather buffers plus one
        # padding row whose entries point at the extra gathered slot.
        assert remapped.shape == (NUM_ROWS + 1, TABLE_WIDTH)
        assert remapped.dtype == PHYSICAL_TABLE.dtype
        assert remapped[:NUM_ROWS].flatten().tolist() == list(range(NUM_GATHERED))
        assert remapped[NUM_ROWS].tolist() == [NUM_GATHERED] * TABLE_WIDTH

        # The gathered buffers hold the physical blocks referenced by the
        # captured table (plus physical block 0 for the padding request):
        # FIA reading gathered[remapped[r, c]] must observe cache[table[r, c]].
        flat_ids = PHYSICAL_TABLE.reshape(-1).to(torch.int64).clamp(min=0)
        extended_ids = torch.cat([flat_ids, flat_ids.new_zeros(1)])
        assert torch.equal(recorder["kwargs"]["key"], _expected_token_major(backend.key_cache, extended_ids))
        assert torch.equal(recorder["kwargs"]["value"], _expected_token_major(backend.value_cache, extended_ids))
        gathered_key = recorder["kwargs"]["key"]
        for r in range(NUM_ROWS):
            for c in range(TABLE_WIDTH):
                slot = int(remapped[r, c])
                # FIA reads gathered[remapped[r, c]]: it must observe the
                # token-major view of the physical block table[r, c].
                physical_slot = torch.tensor([int(PHYSICAL_TABLE[r, c])], dtype=torch.int64)
                assert torch.equal(
                    gathered_key[slot],
                    _expected_token_major(backend.key_cache, physical_slot)[0],
                )

    def test_replay_keeps_remapped_table_bound(self):
        """End-to-end replay contract: resolved params never replace the
        captured remapped table with raw physical ids."""
        backend = _make_backend()
        recorder = _run_full_graph_fia(backend, PHYSICAL_TABLE.clone())
        provider = recorder["provider"]
        captured_block_table = recorder["kwargs"]["block_table"]

        # A later step with different, raw physical ids.
        next_table = torch.tensor([[5, 4], [1, 6]], dtype=torch.int32)
        params = provider.resolve(
            {
                "layer0": SimpleNamespace(
                    actual_seq_lengths_q=[1, 1],
                    seq_lens_list=[6, 5],
                    block_tables=next_table,
                    max_query_len=1,
                )
            }
        )
        # GraphUpdateTask.bind semantics: {**captured_kwargs, **params}.
        runtime_kwargs = {**recorder["kwargs"], **params}
        assert runtime_kwargs["block_table"] is captured_block_table
        # ... and the (constant) remapped content still addresses the
        # compact buffer sequentially, not the physical pool.
        assert runtime_kwargs["block_table"].flatten().tolist()[:NUM_GATHERED] == list(range(NUM_GATHERED))

    def test_remapped_table_strong_ref_outlives_capture_scope(self):
        """register_task weak-refs the capture kwargs; the backend must own
        the remapped table's storage for the graph's lifetime."""
        backend = _make_backend()
        recorder = _run_full_graph_fia(backend, PHYSICAL_TABLE.clone())
        remapped = recorder["kwargs"]["block_table"]
        keeper = weakref.ref(remapped)

        strong_refs = getattr(backend, "_gathered_pa_read_strong_refs", None)
        assert strong_refs, (
            "backend must retain the captured remapped table strongly "
            "(UpdatableGraph.register_task wraps capture kwargs with "
            "weak_ref_tensors)"
        )
        assert any(ref is remapped for ref in strong_refs)

        del recorder, remapped  # the capture frame (and its locals) goes away
        assert keeper() is not None, "backend strong ref must keep the table alive"
        backend._gathered_pa_read_strong_refs.clear()
        assert keeper() is None

    def test_v2_path_retains_remapped_table(self):
        """FIAV2ParamProvider never resolves a block_table override, so the
        strong reference is the only replay-safety net there."""
        backend = _make_backend()
        query = torch.zeros(2, NUM_HEADS, HEAD_DIM)
        output = torch.empty(2, NUM_HEADS * HEAD_DIM)
        stub = SimpleNamespace(
            _npu_fused_infer_attention_score_v2_get_max_workspace=Mock(return_value=torch.empty(8, dtype=torch.int8)),
            npu_fused_infer_attention_score_v2=SimpleNamespace(out=Mock()),
        )
        recorder = {}

        def fake_register_task(operation, kwargs, provider):
            recorder["kwargs"] = kwargs
            recorder["provider"] = provider

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "torch_npu", stub)
            mp.setattr(attention_v1, "register_task", fake_register_task)
            mp.setattr(
                attention_v1,
                "_EXTRA_CTX",
                SimpleNamespace(capturing=True, is_draft_model=False),
            )
            backend.full_graph_fia_v2(query, None, None, _make_metadata(PHYSICAL_TABLE.clone()), output)
        strong_refs = getattr(backend, "_gathered_pa_read_strong_refs", None)
        assert strong_refs and strong_refs[-1] is recorder["kwargs"]["block_table"]


class TestMaterializePaddingRow:
    def test_capture_adds_padding_row_for_synthetic_request(self):
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                attention_v1,
                "_EXTRA_CTX",
                SimpleNamespace(capturing=True),
            )
            key, value, remapped = backend._materialize_bnsd_pa_read(PHYSICAL_TABLE.clone())
        assert remapped.shape == (NUM_ROWS + 1, TABLE_WIDTH)
        assert key.shape[0] == NUM_GATHERED + 1
        assert value.shape[0] == NUM_GATHERED + 1
        # The extra gathered slot is physical block 0 (the padding request
        # points at the null block, mirroring the physical-side padding).
        zero_ids = torch.zeros(1, dtype=torch.int64)
        assert torch.equal(
            key[NUM_GATHERED],
            _expected_token_major(backend.key_cache, zero_ids)[0],
        )

    def test_eager_path_unchanged_no_padding_row(self):
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                attention_v1,
                "_EXTRA_CTX",
                SimpleNamespace(capturing=False),
            )
            key, value, remapped = backend._materialize_bnsd_pa_read(PHYSICAL_TABLE.clone())
        assert remapped.shape == (NUM_ROWS, TABLE_WIDTH)
        assert key.shape[0] == NUM_GATHERED
        assert remapped.flatten().tolist() == list(range(NUM_GATHERED))


class TestCapturedGatherUsesPersistentTable:
    """The builder's synthetic padding cat must not feed the captured gather.

    build() pads metadata.block_tables with torch.cat when the FIA batch
    outgrows the table rows; torch.cat COPIES, and ACL-graph replay
    re-executes only captured device work, so a captured gather that
    consumed the padded copy would read capture-time physical ids forever
    (the runner refreshes the persistent buffer in place, not the copy).
    The captured HND gather must source the runner-persistent
    pre-padding view instead; the synthetic request is already mirrored by
    the gathered world's extra block-0 slot.
    """

    def _capture_with_builder_padding(self, backend, persistent):
        """Emulate a capture whose metadata went through build()'s cat."""
        num_reqs, width = persistent.shape
        padded = torch.cat([persistent, persistent.new_zeros((1, width))])
        metadata = _make_metadata(padded)
        metadata.actual_seq_lengths_q = [1] * (num_reqs + 1)
        metadata.seq_lens_list = [3] * num_reqs + [1]
        # The patched build() retains the pre-cat view on the metadata.
        metadata.block_tables_pre_padding = persistent
        seen = {}

        orig = backend._materialize_bnsd_pa_read

        def spy(block_table, max_kv_len=None):
            seen["table"] = block_table
            return orig(block_table, max_kv_len=max_kv_len)

        backend._materialize_bnsd_pa_read = spy
        try:
            import sys as _sys

            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(_sys.modules[__name__], "_make_metadata", lambda bt: metadata)
                recorder = _run_full_graph_fia(backend, padded, capturing=True)
        finally:
            backend._materialize_bnsd_pa_read = orig
        return recorder, seen

    def test_captured_gather_consumes_persistent_storage_not_cat_copy(self):
        backend = _make_backend()
        persistent = PHYSICAL_TABLE.clone()
        recorder, seen = self._capture_with_builder_padding(backend, persistent)
        # The captured gather's input must share storage with the runner's
        # persistent block table (refreshed in place each step), not the
        # builder's torch.cat snapshot.
        assert seen["table"].data_ptr() == persistent.data_ptr(), (
            "captured HND gather consumed the builder's padded cat copy "
            "(a stale snapshot) instead of the runner-persistent table"
        )

    def test_captured_gather_follows_inplace_refresh_across_replay(self):
        backend = _make_backend()
        persistent = PHYSICAL_TABLE.clone()
        recorder, seen = self._capture_with_builder_padding(backend, persistent)
        remapped = recorder["kwargs"]["block_table"]
        # The synthetic request grew the FIA batch by one; the remapped
        # table must cover it exactly (rows == batch), not double-pad.
        assert remapped.shape[0] == NUM_ROWS + 1
        # Replay: the runner rewrites the persistent buffer in place.
        persistent.copy_(torch.tensor([[6, 3], [2, 0]], dtype=torch.int32))
        # The captured gather re-executes reading the SAME storage (the
        # persistent view), so re-running it yields the refreshed ids.
        flat = seen["table"].reshape(-1).to(torch.int64).clamp(min=0)
        extended = torch.cat([flat, flat.new_zeros(1)])
        gathered_now = _expected_token_major(backend.key_cache, extended)
        # The decisive contract: every remapped slot observes the NEW
        # physical block now stored at that table position, and the
        # synthetic padding row still resolves to physical block 0.
        for r in range(NUM_ROWS):
            for c in range(TABLE_WIDTH):
                slot = int(remapped[r, c])
                physical = int(persistent[r, c])
                assert torch.equal(
                    _expected_token_major(backend.key_cache, torch.tensor([physical], dtype=torch.int64))[0],
                    gathered_now[slot],
                )
        padding_slot = int(remapped[NUM_ROWS, 0])
        block_zero = _expected_token_major(backend.key_cache, torch.tensor([0], dtype=torch.int64))[0]
        assert torch.equal(gathered_now[padding_slot], block_zero)


class TestGatheredLayoutHelper:
    @pytest.mark.parametrize(
        ("layout", "expected"),
        [("NHD", False), ("HND", True), ("LBHNC", True)],
    )
    def test_helper_mirrors_backend_layout_choice(self, layout, expected):
        helper = getattr(attention_v1, "full_graph_fia_uses_gathered_kv_read", None)
        assert helper is not None, (
            "draft speculators need the shared layout helper to mirror the "
            "backend's capture-time FIA block-table binding"
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1.envs_vllm, "VLLM_KV_CACHE_LAYOUT", layout)
            assert helper() is expected


class TestBoundedEagerGather:
    """Bound the HND gather by the live KV lengths (P2, zhaochuang001).

    PR #22649 review: cached-read eager paths passed the full block-table
    width, so index_select copied unused entries too - repeated block 0
    included. At 128 requests x 1,024 table columns, 128 tokens per block,
    8 KV heads and head dim 128, the BF16 K and V results alone require
    64 GiB even if every request has a short prefix, which can OOM on the
    first cached read. The fix uses the eager caller's host-side
    seq_lens_list batch maximum to bound the gather to the pages this
    batch actually needs, avoiding a device-to-host read.
    """

    def test_eager_gather_bounded_by_live_kv_lengths(self):
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, remapped = backend._materialize_bnsd_pa_read(WIDE_TABLE.clone(), max_kv_len=max(WIDE_SEQ_LENS))
        gathered = WIDE_ROWS * WIDE_BOUND_WIDTH
        # The bound applies: only the live columns are gathered.
        assert key.shape == (gathered, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)
        assert value.shape == (gathered, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)
        # The FIA contract is unchanged: the remapped table stays the
        # sequential 0..N-1 mapping, with no padding row outside capture.
        assert remapped.shape == (WIDE_ROWS, WIDE_BOUND_WIDTH)
        assert remapped.flatten().tolist() == list(range(gathered))
        # FIA reads gathered[remapped[r, c]]: it must observe the
        # token-major view of the live physical block WIDE_TABLE[r, c].
        for r in range(WIDE_ROWS):
            for c in range(WIDE_BOUND_WIDTH):
                physical_slot = torch.tensor([int(WIDE_TABLE[r, c])], dtype=torch.int64)
                assert torch.equal(
                    key[int(remapped[r, c])],
                    _expected_token_major(backend.key_cache, physical_slot)[0],
                )

    @pytest.mark.parametrize(
        "state",
        [
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.PrefillCacheHit,
            AscendAttentionState.ChunkedPrefill,
        ],
    )
    def test_cached_read_states_derive_bound_from_seq_lens_list(self, state):
        """Every eager cached-read state gets the bound from the host-side
        seq_lens_list (PrefillNoCache is untouched: block_table is None)."""
        backend = _make_backend()
        metadata = SimpleNamespace(
            attn_state=state,
            block_tables=WIDE_TABLE.clone(),
            seq_lens_list=list(WIDE_SEQ_LENS),
            seq_lens=torch.tensor(WIDE_SEQ_LENS, dtype=torch.int32),
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, block_size, block_table, actual_seq_lengths_kv = backend._get_fia_params(None, None, metadata)
        assert actual_seq_lengths_kv == WIDE_SEQ_LENS
        assert block_size == BLOCK_SIZE
        assert block_table.shape == (WIDE_ROWS, WIDE_BOUND_WIDTH)
        assert key.shape == (WIDE_ROWS * WIDE_BOUND_WIDTH, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)

    def test_missing_seq_lens_list_falls_back_to_full_width(self):
        backend = _make_backend()
        metadata = SimpleNamespace(
            attn_state=AscendAttentionState.DecodeOnly,
            block_tables=WIDE_TABLE.clone(),
            seq_lens_list=None,
        )
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, block_size, block_table, actual_seq_lengths_kv = backend._get_fia_params(None, None, metadata)
        assert block_table.shape == (WIDE_ROWS, WIDE_WIDTH)
        assert key.shape == (WIDE_ROWS * WIDE_WIDTH, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)

    def test_eager_max_kv_len_none_gathers_full_width(self):
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, remapped = backend._materialize_bnsd_pa_read(WIDE_TABLE.clone())
        assert key.shape == (WIDE_ROWS * WIDE_WIDTH, BLOCK_SIZE, NUM_KV_HEADS * HEAD_DIM)
        assert remapped.shape == (WIDE_ROWS, WIDE_WIDTH)
        assert remapped.flatten().tolist() == list(range(WIDE_ROWS * WIDE_WIDTH))

    def test_capture_ignores_bound_keeps_full_width_and_padding_row(self):
        """Graph capture keeps the full static width even when the live
        lengths are known: the captured shape must stay static and the
        padding row for the synthetic replay request is appended at the
        full width."""
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=True))
            key, value, remapped = backend._materialize_bnsd_pa_read(WIDE_TABLE.clone(), max_kv_len=max(WIDE_SEQ_LENS))
        assert remapped.shape == (WIDE_ROWS + 1, WIDE_WIDTH)
        assert key.shape[0] == WIDE_ROWS * WIDE_WIDTH + 1
        assert value.shape[0] == WIDE_ROWS * WIDE_WIDTH + 1
        assert remapped[WIDE_ROWS].tolist() == [WIDE_ROWS * WIDE_WIDTH] * WIDE_WIDTH

    def test_bound_saturates_at_table_width(self):
        # A bound needing more blocks than the table has columns keeps
        # the full width (the bound never grows the table).
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, remapped = backend._materialize_bnsd_pa_read(
                WIDE_TABLE.clone(), max_kv_len=(WIDE_WIDTH + 1) * BLOCK_SIZE
            )
        assert remapped.shape == (WIDE_ROWS, WIDE_WIDTH)
        assert key.shape[0] == WIDE_ROWS * WIDE_WIDTH

    def test_bound_minimum_is_one_column(self):
        backend = _make_backend()
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(attention_v1, "_EXTRA_CTX", SimpleNamespace(capturing=False))
            key, value, remapped = backend._materialize_bnsd_pa_read(WIDE_TABLE.clone(), max_kv_len=0)
        assert remapped.shape == (WIDE_ROWS, 1)
        assert key.shape[0] == WIDE_ROWS
