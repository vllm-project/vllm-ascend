# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention import attention_v1 as attention


@pytest.fixture
def graph_env(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        kv_transfer_config=None,
        quant_config=None,
        use_v2_model_runner=True,
    )
    monkeypatch.setattr(attention, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attention, "needs_layer_aware_fia_graph_replay", lambda: False)
    monkeypatch.setattr(attention, "using_paged_attention", lambda *args: False)
    monkeypatch.setattr(attention, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, sinks=False))
    monkeypatch.setattr(attention, "_ATTN_KEYS_BUFFER", [])
    monkeypatch.setattr(torch, "npu", MagicMock())
    monkeypatch.setattr(attention.torch_npu, "npu", MagicMock())
    fia = MagicMock()
    monkeypatch.setattr(attention.torch_npu, "npu_fused_infer_attention_score", fia, raising=False)
    monkeypatch.setattr(attention, "weak_ref_tensors", lambda value: value)
    params = SimpleNamespace(attn_params={4: []}, handles={4: []}, events={4: []}, workspaces={4: torch.empty(0)})
    monkeypatch.setattr(attention, "get_graph_params", lambda: params)
    return config, params, fia


def make_impl():
    return attention.AscendAttentionBackendImpl(
        num_heads=2,
        head_size=8,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
    )


@pytest.mark.parametrize("use_v2", [False, True])
@pytest.mark.parametrize("existing_layer_aware", [False, True])
def test_v2_layer_identity_preserves_workspace_policy(graph_env, monkeypatch, use_v2, existing_layer_aware):
    config, _, _ = graph_env
    config.use_v2_model_runner = use_v2
    monkeypatch.setattr(attention, "needs_layer_aware_fia_graph_replay", lambda: existing_layer_aware)
    impl = make_impl()
    assert impl._use_layer_aware_fia_graph_replay == (existing_layer_aware or use_v2)
    assert impl._use_max_workspace_for_fia_graph == existing_layer_aware


def test_v2_replay_keeps_target_metadata_with_local_mtp_layer(graph_env):
    config, params, fia = graph_env
    metadata = {}
    target_keys = [f"language_model.model.layers.{index}.self_attn.attn" for index in (35, 39)]
    for index, name in enumerate(target_keys):
        impl = make_impl()
        query = torch.empty(4, 2, 8)
        key = value = torch.empty(4, 1, 8)
        output = torch.empty_like(query)
        layer = SimpleNamespace(layer_name=name, _k_scale_float=1.0, _v_scale_float=1.0)
        impl.forward(layer, query, key, value, None, None, output)
        table = torch.tensor([[10 + index]], dtype=torch.int32)
        meta = SimpleNamespace(
            actual_seq_lengths_q=[4],
            seq_lens_list=[33 + index],
            block_tables=table,
            attn_mask=None,
            causal=True,
        )
        metadata[name] = meta
        impl._get_fia_params = MagicMock(return_value=(key, value, 128, table, meta.seq_lens_list))
        impl.full_graph_fia(query, key, value, meta, output)

    assert [param[-1] for param in params.attn_params[4]] == target_keys
    # A local draft layer must not shift the captured target layers' metadata.
    metadata = {
        "mtp.layers.0.self_attn.attn": SimpleNamespace(
            actual_seq_lengths_q=[1],
            seq_lens_list=[999],
            block_tables=torch.tensor([[99]]),
        ),
        **metadata,
    }
    fia.reset_mock()
    attention.AscendAttentionBackendImpl.update_graph_params(
        MagicMock(), SimpleNamespace(attn_metadata=metadata), 4, config, SimpleNamespace(method="mtp")
    )
    assert fia.out.call_count == 2
    assert attention._ATTN_KEYS_BUFFER == []
    for call, name in zip(fia.out.call_args_list, target_keys):
        assert call.kwargs["block_table"] is metadata[name].block_tables
        assert call.kwargs["actual_seq_lengths_kv"] == metadata[name].seq_lens_list
