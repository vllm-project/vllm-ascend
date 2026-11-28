# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
# ruff: noqa: E402

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "vllm.transformers_utils.configs.deepseek_v41",
    reason="DeepSeek V4.1 is unavailable on this vLLM release",
)

import torch
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.attention.dsa_v41 import AscendDSAV41Impl, H8GroupedInt8KV
from vllm_ascend.models.deepseek_v41 import model as model_module
from vllm_ascend.models.deepseek_v41.model import (
    DeepseekV41Model,
    DeepseekV41SharedAttentionState,
    build_layer_plan,
)


@pytest.fixture
def text_config() -> DeepseekV41Config:
    return DeepseekV41Config(
        text_config={
            "num_hidden_layers": 40,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "candidate_source_layer_id": 20,
            "candidate_topk_blocks": 2048,
            "candidate_block_size": 8,
            "index_topk": 512,
            "engram_layer_ids": [1, 14],
        }
    )


def test_builds_expected_source_groups(text_config: DeepseekV41Config):
    topology = build_layer_plan(text_config)

    assert topology.kv_consumers(2) == tuple(range(2, 8))
    assert topology.kv_consumers(8) == tuple(range(8, 14))
    assert topology.kv_consumers(14) == tuple(range(14, 20))
    assert topology.kv_consumers(20) == tuple(range(20, 40))

    assert topology.index_consumers(20) == tuple(range(20, 24))
    assert topology.index_consumers(24) == tuple(range(24, 28))
    assert topology.index_consumers(36) == tuple(range(36, 40))


def test_layer_26_resolves_layer_20_kv_and_layer_24_index(text_config: DeepseekV41Config):
    topology = build_layer_plan(text_config)
    role = topology.layer(26)
    assert role.kv_source_layer == 20
    assert role.index_source_layer == 24
    assert topology.candidate_source_layer_id == 20
    assert role.compress_ratio == 1
    assert role.uses_candidate_filter


def test_source_roles_and_engram_slots(text_config: DeepseekV41Config):
    topology = build_layer_plan(text_config)

    assert topology.layer(2).is_kv_source
    assert topology.layer(2).is_index_source
    assert topology.layer(20).is_candidate_source
    assert topology.layer(1).engram_slot == 0
    assert topology.layer(14).engram_slot == 1
    assert topology.layer(0).kv_source_layer is None


def test_shared_state_resets_sparse_attention_metadata():
    topk_indices = torch.zeros((4, 1, 512), dtype=torch.int32)
    candidates = torch.zeros((4, 1, 16), dtype=torch.int32)
    state = DeepseekV41SharedAttentionState(topk_indices, candidates)

    state.reset()

    assert state.topk_indices is topk_indices
    assert state.candidates is candidates


@pytest.mark.parametrize(
    ("enabled", "heads", "topk", "ubatching", "grouped"),
    [
        (True, 8, 512, False, True),
        (False, 8, 512, False, False),
        (True, 64, 512, False, False),
        (True, 32, 512, False, False),  # PCP4/TP2: PCP distributes tokens, TP distributes heads.
        (True, 8, 256, False, False),
        (True, 8, 512, True, False),
    ],
)
def test_serving_grouped_int8_layer_modes(text_config, monkeypatch, enabled, heads, topk, ubatching, grouped):
    monkeypatch.setattr(model_module, "get_ascend_config", lambda: SimpleNamespace(dsv41_int8_long_kv=enabled))
    text_config.index_topk = topk
    topology = build_layer_plan(text_config)
    layers = []
    for role in topology.layers:
        impl = SimpleNamespace(grouped_int8_kv=None, topk_value_mode=1)
        attn = SimpleNamespace(v41_impl=impl, topology=topology, role=role, n_local_heads=heads)
        layers.append(SimpleNamespace(self_attn=attn))
    model = SimpleNamespace(
        layers=layers,
        start_layer=4,
        end_layer=len(layers),
        decoder_replay_start=21,
        topk_indices_buffer=torch.empty(3, topk, dtype=torch.int32),
    )
    DeepseekV41Model._configure_grouped_int8_kv(
        model, SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=ubatching))
    )
    writers = {4, 8, 14, 20, 21, 24, 28, 32, 36}
    for idx, layer in enumerate(layers):
        impl = layer.self_attn.v41_impl
        if grouped and idx >= model.start_layer and layer.self_attn.role.has_long_context:
            assert impl.topk_value_mode == (71 if idx in writers else 72)
            assert impl.grouped_int8_kv is layers[4].self_attn.v41_impl.grouped_int8_kv
        else:
            assert impl.topk_value_mode == 1 and impl.grouped_int8_kv is None


def test_grouped_int8_service_refreshes_header_and_uses_active_rows(monkeypatch):
    capped = H8GroupedInt8KV(torch.empty(512, 512, dtype=torch.int32))
    assert capped.max_tokens == H8GroupedInt8KV.MAX_TOKENS == 256
    assert capped.buffer.numel() == 2 * 1024 + 256 * H8GroupedInt8KV.WORDS_PER_QUERY

    ws = H8GroupedInt8KV(torch.empty(5, 512, dtype=torch.int32))
    header = torch.arange(1024, dtype=torch.int32)
    desc = SimpleNamespace(
        attention=SimpleNamespace(
            block_table=torch.zeros(1, 1), cache_seq_lens=torch.tensor([1024]), cmp_residual=None, smla_metadata=header
        ),
        swa=SimpleNamespace(
            num_reqs=1,
            query_start_loc=torch.tensor([0, 2]),
            seq_lens=torch.tensor([2048]),
            block_table=torch.zeros(1, 1),
            replay_block_table=None,
            ori_sparse_indices=None,
            ori_topk_length=None,
            ori_mask_mode=4,
            ori_win_left=127,
            ori_win_right=0,
        ),
    )
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(swa_cache_layer=SimpleNamespace(kv_cache=torch.empty(0))),
        attn_sink=torch.zeros(8),
        softmax_scale=0.044,
    )
    impl = SimpleNamespace(
        role=SimpleNamespace(compress_ratio=2, has_long_context=True),
        topology=SimpleNamespace(index_topk=512),
        grouped_int8_kv=ws,
        topk_value_mode=71,
    )
    calls = []

    def record(q, **kwargs):
        calls.append(kwargs)
        return q, None

    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", record, raising=False)
    for ratio, tokens, mode in ((2, 2, 71), (2, 2, 72), (1, 1, 71), (1, 1, 72), (2, 3, 71)):
        impl.role.compress_ratio, impl.topk_value_mode = ratio, mode
        if mode == 71:
            header.add_(1)
        q = torch.empty(tokens, 8, 512)
        result = AscendDSAV41Impl._forward_attention(
            impl,
            attn,
            q,
            desc,
            torch.zeros(tokens, 512, dtype=torch.int32),
            source_cache=torch.empty(0, dtype=torch.int8),
        )
        assert result is q
        view = calls[-1]["metadata"]
        assert calls[-1]["topk_value_mode"] == mode
        assert view.numel() == ratio * 1024 + tokens * 512 * 544 // 4
        assert torch.equal(view[:1024], header)
        assert view[ratio * 1024 :].data_ptr() == ws.buffer[2048:].data_ptr()
        if mode == 71:
            view[ratio * 1024] = tokens + 10
        else:
            assert view[ratio * 1024].item() == tokens + 10

    for mode in (71, 72):
        impl.role.compress_ratio, impl.topk_value_mode = 2, mode
        q = torch.empty(6, 8, 512)
        AscendDSAV41Impl._forward_attention(
            impl,
            attn,
            q,
            desc,
            torch.zeros(6, 512, dtype=torch.int32),
            source_cache=torch.empty(0, dtype=torch.int8),
        )
        assert calls[-1]["topk_value_mode"] == 1
        assert calls[-1]["metadata"] is header

    impl.grouped_int8_kv = capped
    for tokens, expected_mode in ((256, 71), (257, 1)):
        impl.topk_value_mode = 71
        AscendDSAV41Impl._forward_attention(
            impl,
            attn,
            torch.empty(tokens, 8, 512),
            desc,
            torch.zeros(tokens, 512, dtype=torch.int32),
            source_cache=torch.empty(0, dtype=torch.int8),
        )
        assert calls[-1]["topk_value_mode"] == expected_mode
        assert (calls[-1]["metadata"] is header) == (tokens > 256)
