# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_adapter():
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen4_exp/input_adapter.py"
    module = ast.parse(source.read_text())
    module.body = [node for node in module.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = {"li": 0, "gbsa": 0}

    def lightning_metadata(*args, **kwargs):
        del args
        calls["li"] += 1
        return torch.tensor(
            [kwargs["batch_size"], int(kwargs["seqused_k"].sum())],
            dtype=torch.int32,
        )

    def gbsa_metadata(*args, **kwargs):
        calls["gbsa"] += 1
        return torch.tensor(
            [args[0].shape[1], int(args[1].sum()), int(kwargs["seqused_kv"].sum())],
            dtype=torch.int32,
        )

    namespace = {
        "torch": torch,
        "os": os,
        "Any": object,
        "GBSA_COMPRESS_RATIO": 4,
        "GBSA_PAGE_SIZE": 128,
        "generic_block_sparse_attention_metadata": gbsa_metadata,
        "lightning_indexer_v2_metadata": lightning_metadata,
    }
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["Qwen4ExpPreparedInputAdapter"], calls


class AscendQwen4ExpQSAAttention:
    def __init__(self):
        self.layer_name = "model.layers.0.self_attn"
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 256
        self.rotary_emb = SimpleNamespace(cos_sin_cache=torch.arange(64 * 8).reshape(64, 8).float())
        self.q_norm = SimpleNamespace(weight=torch.empty(1))
        self.indexer = SimpleNamespace(
            index_n_heads=4,
            index_head_dim=128,
            token_topk=2048,
            compress_ratio=4,
            raw_key_cache=SimpleNamespace(prefix="raw"),
            compressed_key_cache=SimpleNamespace(prefix="compressed"),
        )


class _Model:
    def __init__(self, layer):
        self.layer = layer

    def modules(self):
        return iter((self.layer,))


class _TwoLayerModel:
    def __init__(self, first, second):
        self.layers = (first, second)

    def modules(self):
        return iter(self.layers)


@pytest.mark.parametrize("tokens", [1, 2, 7])
def test_mrope_slices_token_axis_and_keeps_fixed_address(tokens: int):
    adapter_cls, _ = _load_adapter()
    layer = AscendQwen4ExpQSAAttention()
    adapter = adapter_cls(_Model(layer))
    positions = torch.stack((torch.arange(9), torch.arange(9) + 1, torch.arange(9) + 2))

    adapter.prepare_positions(positions, tokens)
    first = layer._qsa_prepared_cos_sin
    first_ptr = first.data_ptr()
    assert first.shape == (3, tokens, 8)

    adapter.prepare_positions(positions + 3, tokens)
    assert layer._qsa_prepared_cos_sin.data_ptr() == first_ptr
    torch.testing.assert_close(layer._qsa_prepared_cos_sin, layer.rotary_emb.cos_sin_cache[(positions + 3)[:, :tokens]])


def test_one_dimensional_positions_keep_fixed_address():
    adapter_cls, _ = _load_adapter()
    layer = AscendQwen4ExpQSAAttention()
    adapter = adapter_cls(_Model(layer))

    adapter.prepare_positions(torch.arange(8), 4)
    pointer = layer._qsa_prepared_cos_sin.data_ptr()
    adapter.prepare_positions(torch.arange(8) + 1, 4)

    assert layer._qsa_prepared_cos_sin.shape == (4, 8)
    assert layer._qsa_prepared_cos_sin.data_ptr() == pointer


def test_full_graph_metadata_uses_padded_geometry_and_fixed_address():
    adapter_cls, calls = _load_adapter()
    layer = AscendQwen4ExpQSAAttention()
    adapter = adapter_cls(_Model(layer))

    def make_metadata(lengths):
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        raw = SimpleNamespace(
            token_to_req=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
            logical_positions=torch.tensor([3, 4, 5, 6], dtype=torch.int32),
            seq_lens=seq_lens,
        )
        compressed = SimpleNamespace(
            token_to_req=raw.token_to_req,
            seq_lens=seq_lens,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            qsa_num_live_requests=1,
        )
        main = SimpleNamespace(
            num_actual_tokens=1,
            block_table=torch.zeros((2, 8), dtype=torch.int32),
            qsa_request_plan=None,
        )
        return {layer.layer_name: main, "raw": raw, "compressed": compressed}, main, compressed

    metadata, main, compressed = make_metadata([4, 8])
    adapter.prepare_attention_metadata(metadata, full_graph=True)
    li_ptr = compressed.qsa_lightning_v2_metadata.data_ptr()
    gbsa_ptr = main.qsa_gbsa_metadata.data_ptr()
    first_li = compressed.qsa_lightning_v2_metadata.clone()
    first_gbsa = main.qsa_gbsa_metadata.clone()

    # MRV2 refreshes persistent metadata tensor values in place. Reusing the
    # same views must therefore keep the prepared schedule addresses stable.
    raw = metadata["raw"]
    raw.seq_lens.copy_(torch.tensor([12, 16], dtype=torch.int32))
    adapter.prepare_attention_metadata(metadata, full_graph=True)

    assert compressed.qsa_lightning_v2_metadata.data_ptr() == li_ptr
    assert main.qsa_gbsa_metadata.data_ptr() == gbsa_ptr
    assert not torch.equal(compressed.qsa_lightning_v2_metadata, first_li)
    assert not torch.equal(main.qsa_gbsa_metadata, first_gbsa)
    assert compressed.qsa_lightning_v2_metadata[0].item() == 2
    assert main.qsa_gbsa_metadata[0].item() == 4
    assert calls == {"li": 2, "gbsa": 2}


def test_full_graph_negative_padding_request_uses_last_sequence():
    adapter_cls, _ = _load_adapter()
    layer = AscendQwen4ExpQSAAttention()
    adapter = adapter_cls(_Model(layer))
    seq_lens = torch.tensor([4, 20], dtype=torch.int32)
    raw = SimpleNamespace(
        token_to_req=torch.tensor([0, -1], dtype=torch.int32),
        logical_positions=torch.tensor([3, 15], dtype=torch.int32),
        seq_lens=seq_lens,
    )
    compressed = SimpleNamespace(
        token_to_req=raw.token_to_req,
        seq_lens=seq_lens,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        qsa_num_live_requests=1,
    )
    main = SimpleNamespace(
        num_actual_tokens=1,
        block_table=torch.zeros((2, 8), dtype=torch.int32),
        qsa_request_plan=None,
    )

    adapter.prepare_attention_metadata(
        {layer.layer_name: main, "raw": raw, "compressed": compressed},
        full_graph=True,
    )

    # Visible lengths are [4, 16], not [4, 4]. Counts are [1, 4] and
    # expanded over two KV heads, so the mocked metadata reports 10.
    torch.testing.assert_close(
        main.qsa_gbsa_metadata,
        torch.tensor([2, 10, 20], dtype=torch.int32),
    )


def test_compatible_layers_share_one_schedule_per_step():
    adapter_cls, calls = _load_adapter()
    first = AscendQwen4ExpQSAAttention()
    second = AscendQwen4ExpQSAAttention()
    second.layer_name = "model.layers.1.self_attn"
    second.indexer.raw_key_cache = SimpleNamespace(prefix="raw2")
    second.indexer.compressed_key_cache = SimpleNamespace(prefix="compressed2")
    # Real QSA layers share the same rotary cache; model it explicitly.
    second.rotary_emb = first.rotary_emb
    adapter = adapter_cls(_TwoLayerModel(first, second))

    raw = SimpleNamespace(
        token_to_req=torch.tensor([0, 0], dtype=torch.int32),
        logical_positions=torch.tensor([7, 8], dtype=torch.int32),
        seq_lens=torch.tensor([9], dtype=torch.int32),
    )
    compressed = SimpleNamespace(
        token_to_req=raw.token_to_req,
        seq_lens=raw.seq_lens,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        qsa_num_live_requests=1,
    )
    main1 = SimpleNamespace(
        num_actual_tokens=2,
        block_table=torch.zeros((1, 8), dtype=torch.int32),
        qsa_request_plan=None,
    )
    main2 = SimpleNamespace(
        num_actual_tokens=2,
        block_table=torch.zeros((1, 8), dtype=torch.int32),
        qsa_request_plan=None,
    )
    metadata = {
        first.layer_name: main1,
        second.layer_name: main2,
        "raw": raw,
        "raw2": raw,
        "compressed": compressed,
        "compressed2": compressed,
    }

    adapter.prepare_attention_metadata(metadata, full_graph=False)
    adapter.prepare_positions(torch.arange(4), 2)

    assert calls == {"li": 1, "gbsa": 1}
    assert main1.qsa_gbsa_metadata.data_ptr() == main2.qsa_gbsa_metadata.data_ptr()
    assert first._qsa_prepared_cos_sin.data_ptr() == second._qsa_prepared_cos_sin.data_ptr()


def test_same_geometry_with_different_dynamic_tensors_does_not_share():
    adapter_cls, calls = _load_adapter()
    first = AscendQwen4ExpQSAAttention()
    second = AscendQwen4ExpQSAAttention()
    second.layer_name = "model.layers.1.self_attn"
    second.indexer.raw_key_cache = SimpleNamespace(prefix="raw2")
    second.indexer.compressed_key_cache = SimpleNamespace(prefix="compressed2")
    adapter = adapter_cls(_TwoLayerModel(first, second))

    def make_side_metadata():
        seq_lens = torch.tensor([9], dtype=torch.int32)
        raw = SimpleNamespace(
            token_to_req=torch.tensor([0, 0], dtype=torch.int32),
            logical_positions=torch.tensor([7, 8], dtype=torch.int32),
            seq_lens=seq_lens,
        )
        compressed = SimpleNamespace(
            token_to_req=raw.token_to_req,
            seq_lens=seq_lens,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            qsa_num_live_requests=1,
        )
        return raw, compressed

    raw1, compressed1 = make_side_metadata()
    raw2, compressed2 = make_side_metadata()
    main1 = SimpleNamespace(
        num_actual_tokens=2,
        block_table=torch.zeros((1, 8), dtype=torch.int32),
        qsa_request_plan=None,
    )
    main2 = SimpleNamespace(
        num_actual_tokens=2,
        block_table=torch.zeros((1, 8), dtype=torch.int32),
        qsa_request_plan=None,
    )

    adapter.prepare_attention_metadata(
        {
            first.layer_name: main1,
            second.layer_name: main2,
            "raw": raw1,
            "raw2": raw2,
            "compressed": compressed1,
            "compressed2": compressed2,
        },
        full_graph=False,
    )

    assert calls == {"li": 2, "gbsa": 2}
    assert compressed1.qsa_lightning_v2_metadata.data_ptr() != compressed2.qsa_lightning_v2_metadata.data_ptr()
    assert main1.qsa_gbsa_metadata.data_ptr() != main2.qsa_gbsa_metadata.data_ptr()


def test_persistent_metadata_shape_error_names_operator_kind():
    adapter_cls, _ = _load_adapter()
    adapter = adapter_cls(_Model(AscendQwen4ExpQSAAttention()))

    adapter._persistent_metadata(("gbsa", "signature"), torch.zeros(2, dtype=torch.int32))
    with pytest.raises(RuntimeError, match="QSA gbsa metadata shape changed"):
        adapter._persistent_metadata(("gbsa", "signature"), torch.zeros(3, dtype=torch.int32))
