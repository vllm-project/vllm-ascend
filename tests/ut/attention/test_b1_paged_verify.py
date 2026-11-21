#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import AttentionSelectorConfig

from vllm_ascend.attention import attention_v2_b1
from vllm_ascend.attention.attention_v2_b1 import build_paged_verify_mask
from vllm_ascend.attention.utils import (
    expand_pa_per_token_rows,
    in_full_cudagraph_mode,
)

BLOCK_SIZE = 4


def _paged_metadata(block_tables, query_ends, seq_lens_list):
    return SimpleNamespace(
        pa_rows_expanded=False,
        pa_block_tables=None,
        pa_seq_lens=None,
        block_tables=block_tables,
        seq_lens=torch.tensor(seq_lens_list, dtype=torch.int32),
        actual_seq_lengths_q=query_ends,
        seq_lens_list=seq_lens_list,
    )


def test_build_paged_verify_mask_is_causal_over_the_paged_width():
    # Two requests, three speculative tokens (k + 1 = 3 queries each) and a
    # block table wide enough for 2 * 4 = 8 KV positions.
    block_table = torch.zeros((2, 2), dtype=torch.int32)
    seq_lens = torch.tensor([3, 8], dtype=torch.int32)

    mask = build_paged_verify_mask(
        seq_lens,
        block_table,
        num_speculative_tokens=2,
        block_size=BLOCK_SIZE,
        device=torch.device("cpu"),
    )

    assert mask.dtype == torch.bool
    assert mask.shape == (2, 1, 3, 8)
    for batch, seq_len in enumerate(seq_lens.tolist()):
        for query in range(3):
            # Query j may read prefix + j + 1 positions, where the prefix is
            # seq_len - (k + 1).
            allowed = seq_len - 3 + query + 1
            assert not mask[batch, 0, query, :allowed].any()
            assert mask[batch, 0, query, allowed:].all()


def test_build_paged_verify_mask_reuses_the_kpos_cache():
    attention_v2_b1._VERIFY_MASK_KPOS_CACHE.clear()
    block_table = torch.zeros((1, 2), dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    for _ in range(2):
        build_paged_verify_mask(
            seq_lens,
            block_table,
            num_speculative_tokens=2,
            block_size=BLOCK_SIZE,
            device=torch.device("cpu"),
        )

    assert len(attention_v2_b1._VERIFY_MASK_KPOS_CACHE) == 1


def test_expand_pa_per_token_rows_splits_multi_query_rows():
    block_tables = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    # The runner appends a zero-query pad row, which contributes nothing.
    metadata = _paged_metadata(block_tables, [2, 5, 5], [2, 5, 5])

    assert expand_pa_per_token_rows(metadata) is True

    assert metadata.pa_block_tables.tolist() == [
        [10, 11],
        [10, 11],
        [20, 21],
        [20, 21],
        [20, 21],
    ]
    # Request 0 holds the whole prefix (2 tokens), request 1 attends 2 computed
    # tokens plus its own causal position.
    assert metadata.pa_seq_lens.tolist() == [1, 2, 3, 4, 5]
    assert metadata.pa_rows_expanded is True
    # The stock per-request view must survive: an uncaptured FULL batch falls
    # back to FIA, which binds one block-table row per request.
    assert metadata.block_tables is block_tables
    assert metadata.seq_lens.tolist() == [2, 5, 5]
    assert metadata.seq_lens_list == [2, 5, 5]


def test_expand_pa_per_token_rows_keeps_single_query_metadata():
    block_tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    metadata = _paged_metadata(block_tables, [1, 2], [4, 5])

    assert expand_pa_per_token_rows(metadata) is True
    # One query per row already: the PA view aliases the per-request fields.
    assert metadata.pa_block_tables is block_tables
    assert metadata.pa_seq_lens is metadata.seq_lens
    assert metadata.block_tables is block_tables
    assert metadata.seq_lens_list == [4, 5]
    assert metadata.pa_rows_expanded is True


def test_expand_pa_per_token_rows_rejects_incomplete_metadata():
    assert not expand_pa_per_token_rows(SimpleNamespace(actual_seq_lengths_q=[1], seq_lens_list=[1]))
    assert not expand_pa_per_token_rows(_paged_metadata(torch.zeros((1, 2)), [2, 4], [4]))
    assert not expand_pa_per_token_rows(_paged_metadata(torch.zeros((2, 2)), [2, 1], [2, 1]))


def test_expand_pa_per_token_rows_is_idempotent():
    metadata = _paged_metadata(torch.tensor([[1, 2]], dtype=torch.int32), [2], [2])
    assert expand_pa_per_token_rows(metadata) is True
    pa_block_tables = metadata.pa_block_tables

    assert expand_pa_per_token_rows(metadata) is True
    assert metadata.pa_block_tables is pa_block_tables
    assert metadata.block_tables.tolist() == [[1, 2]]


def test_shared_metadata_builder_keeps_pa_row_expansion_out():
    # Regression for the stock FIA path: with B1 disabled the 512-dim layer
    # dispatches to FIA, and using_paged_attention() is False whenever
    # speculative decoding is on, so every request must keep exactly one
    # block-table row. The expansion belongs to the B1 builder only.
    from vllm_ascend.attention import attention_v1

    assert not hasattr(attention_v1, "expand_pa_per_token_rows")


def test_b1_builder_expands_pa_rows_in_full_mode(monkeypatch):
    builder = object.__new__(attention_v2_b1.AscendB1AttentionMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(head_size=512)
    builder.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL))
    # Two requests with two queries each: the decode-only PA op needs one row
    # per query, which the B1 builder must produce for FULL capture/replay.
    metadata = _paged_metadata(torch.tensor([[10, 11], [20, 21]], dtype=torch.int32), [2, 4], [2, 4])
    monkeypatch.setattr(
        attention_v2_b1.AscendAttentionMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )

    built = builder.build(0, SimpleNamespace())

    assert built.b1_pa_graph is True
    assert built.pa_block_tables.tolist() == [[10, 11], [10, 11], [20, 21], [20, 21]]
    assert built.pa_seq_lens.tolist() == [1, 2, 3, 4]
    # The per-request view stays intact for steps that are not replayed.
    assert built.block_tables.tolist() == [[10, 11], [20, 21]]
    assert built.seq_lens.tolist() == [2, 4]
    assert built.seq_lens_list == [2, 4]


def test_uncaptured_full_batch_falls_back_to_the_per_request_table(monkeypatch):
    # FULL mode does not guarantee a graph hit: a batch that misses the capture
    # sizes runs eagerly through the stock FIA path, which binds one block-table
    # row per request. The expanded PA rows must therefore never leak into
    # block_tables/seq_lens (two requests with two queries each here).
    builder = object.__new__(attention_v2_b1.AscendB1AttentionMetadataBuilder)
    builder.kv_cache_spec = SimpleNamespace(head_size=512)
    builder.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL))
    block_tables = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    metadata = _paged_metadata(block_tables, [2, 4], [2, 4])
    monkeypatch.setattr(
        attention_v2_b1.AscendAttentionMetadataBuilder,
        "build",
        lambda *args, **kwargs: metadata,
    )
    built = builder.build(0, SimpleNamespace())
    assert built.pa_block_tables.tolist() == [[10, 11], [10, 11], [20, 21], [20, 21]]

    impl = object.__new__(attention_v2_b1.AscendB1AttentionBackendImpl)
    impl.full_cudagraph = True
    seen = {}

    def stock_fia(self, query, key, value, attn_metadata, output, kv_cache=None):
        seen["block_tables"] = attn_metadata.block_tables
        seen["seq_lens"] = attn_metadata.seq_lens
        seen["seq_lens_list"] = attn_metadata.seq_lens_list
        return output

    monkeypatch.setattr(attention_v2_b1.AscendAttentionBackendImpl, "forward_fused_infer_attention", stock_fia)
    monkeypatch.setattr(attention_v2_b1, "_EXTRA_CTX", SimpleNamespace(capturing=False, is_draft_model=False))

    output = torch.zeros(4, 3)
    query = torch.zeros(4, 2, 3)
    assert impl.forward_fused_infer_attention(query, None, None, built, output) is output

    # The fallback sees the original, per-request layout.
    assert seen["block_tables"].tolist() == [[10, 11], [20, 21]]
    assert seen["seq_lens"].tolist() == [2, 4]
    assert seen["seq_lens_list"] == [2, 4]


def test_b1_pa_param_provider_rebinds_the_expanded_view():
    metadata = _paged_metadata(torch.tensor([[10, 11], [20, 21]], dtype=torch.int32), [2, 4], [2, 4])
    assert expand_pa_per_token_rows(metadata) is True

    layer = "model.layers.0.self_attn.attn"
    params = attention_v2_b1.B1PAParamProvider(layer).resolve({layer: metadata})

    assert params["block_table"].tolist() == [[10, 11], [10, 11], [20, 21], [20, 21]]
    assert params["context_lens"].tolist() == [1, 2, 3, 4]


def test_in_full_cudagraph_mode_only_matches_full():
    def config(mode):
        return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=mode))

    assert in_full_cudagraph_mode(config(CUDAGraphMode.FULL)) is True
    assert in_full_cudagraph_mode(config(CUDAGraphMode.FULL_DECODE_ONLY)) is False
    assert in_full_cudagraph_mode(SimpleNamespace(compilation_config=None)) is False


def _gemma4_selector(**overrides):
    return AttentionSelectorConfig(
        head_size=overrides.pop("head_size", 512),
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=128,
        attn_type=overrides.pop("attn_type", AttentionType.DECODER),
        **overrides,
    )


def _speculative_config():
    return SimpleNamespace(speculative_config=SimpleNamespace(num_speculative_tokens=3))


@pytest.fixture
def b1_enabled(monkeypatch):
    # Drive the switch through os.environ: monkeypatching the module attribute
    # writes a plain global on undo, which then shadows envs.__getattr__ for
    # the rest of the session and breaks tests/ut/test_envs.py.
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_B1_PAGED_VERIFY", "1")


def test_want_b1_paged_verify_accepts_gemma4_mtp_geometry(b1_enabled):
    from vllm_ascend import platform

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_speculative_config()),
        patch("vllm_ascend.utils.is_950", return_value=False),
    ):
        assert platform._want_b1_paged_verify(_gemma4_selector()) is True


def test_want_b1_paged_verify_requires_explicit_opt_in(monkeypatch):
    import vllm_ascend.platform as platform

    monkeypatch.delenv("VLLM_ASCEND_ENABLE_B1_PAGED_VERIFY", raising=False)
    with patch("vllm.config.get_current_vllm_config_or_none", return_value=_speculative_config()):
        assert platform._want_b1_paged_verify(_gemma4_selector()) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"head_size": 128},
        {"has_sliding_window": True},
        {"attn_type": AttentionType.ENCODER},
        {"use_mla": True},
        {"has_sink": True},
        {"use_non_causal": True},
        {"use_adaptive_verification": True},
        {"use_rswa": True},
    ],
)
def test_want_b1_paged_verify_rejects_other_geometries(b1_enabled, overrides):
    from vllm_ascend import platform

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_speculative_config()),
        patch("vllm_ascend.utils.is_950", return_value=False),
    ):
        assert platform._want_b1_paged_verify(_gemma4_selector(**overrides)) is False


def test_want_b1_paged_verify_rejects_hisparse(b1_enabled):
    if "use_hisparse" not in AttentionSelectorConfig._fields:
        pytest.skip("this vLLM revision does not expose use_hisparse")

    from vllm_ascend import platform

    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_speculative_config()),
        patch("vllm_ascend.utils.is_950", return_value=False),
    ):
        assert platform._want_b1_paged_verify(_gemma4_selector(use_hisparse=True)) is False


def test_want_b1_paged_verify_rejects_missing_or_a5_context(b1_enabled):
    from vllm_ascend import platform

    selector = _gemma4_selector()
    with patch("vllm.config.get_current_vllm_config_or_none", return_value=None):
        assert platform._want_b1_paged_verify(selector) is False
    with patch(
        "vllm.config.get_current_vllm_config_or_none",
        return_value=SimpleNamespace(speculative_config=None),
    ):
        assert platform._want_b1_paged_verify(selector) is False
    with (
        patch("vllm.config.get_current_vllm_config_or_none", return_value=_speculative_config()),
        patch("vllm_ascend.utils.is_950", return_value=True),
    ):
        assert platform._want_b1_paged_verify(selector) is False
