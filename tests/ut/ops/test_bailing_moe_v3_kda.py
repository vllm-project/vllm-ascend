# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_ascend.ops.bailing_moe_v3_kda as bailing


@pytest.mark.parametrize("decodes,prefills", [(0, 1), (1, 0), (1, 1)])
def test_kda_dispatch_preserves_token_order_and_state_rows(monkeypatch, decodes, prefills):
    count = decodes + 3 * prefills
    state = torch.zeros(8, 1, 2, 2)
    meta = SimpleNamespace(
        num_actual_tokens=count,
        num_prefills=prefills,
        num_decodes=decodes,
        num_decode_tokens=decodes,
        non_spec_state_indices_tensor=torch.tensor(([2] * decodes) + ([5] * prefills)),
        non_spec_query_start_loc=torch.tensor([0] + ([1] * decodes) + ([count] * prefills)),
        prefill_state_indices=torch.tensor([5]),
        prefill_has_initial_state=torch.tensor([False]),
        non_spec_prefill_metadata=SimpleNamespace(
            chunk=SimpleNamespace(
                cu_seqlens_host=(0, 3), cu_seqlens_kern=None, chunk_indices_chunk64_host=(0, 0), keep_meta=None
            )
        ),
    )
    layer = SimpleNamespace(
        prefix="layer",
        kv_cache=(None, state),
        safe_gate=True,
        lower_bound=-5.0,
        A_log=torch.ones(1),
        dt_bias=torch.zeros(2),
    )
    data = torch.arange(count * 2).reshape(1, count, 1, 2).float()
    layer._convolve = lambda projections, metadata: [x.reshape_as(data) for x in projections]
    monkeypatch.setattr(bailing, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"layer": meta}))

    def recurrent(q, k, v, g, beta, history, starts, indices, a_log, bias, **kw):
        assert history is state and indices.tolist() == [2] and starts.tolist() == [0, 1]
        assert torch.equal(g, q) and kw["lower_bound"] == -5.0
        return q + 10

    def chunk(q, k, v, g, beta, initial, starts, chunk_indices, a_log, bias, **kw):
        assert q.shape[1] == 3 and torch.equal(g, q) and starts == (0, 3)
        assert not initial.any() and chunk_indices == (0, 0)
        return q + 20, torch.ones_like(initial)

    monkeypatch.setattr(bailing, "run_recurrent_kda", recurrent)
    monkeypatch.setattr(bailing, "run_chunk_kda", chunk)
    monkeypatch.setattr(bailing, "gather_ssm_states", lambda cache, idx, keep: cache[idx] * keep[:, None, None, None])

    def scatter(cache, idx, rows, active_rows):
        cache[idx[active_rows]] = rows[active_rows]

    monkeypatch.setattr(bailing, "scatter_ssm_states_", scatter)
    output = torch.full_like(data, torch.nan)
    # The upstream registered custom op dispatches by these keyword names.
    bailing.AscendBailingMoeV3KimiDeltaAttention._forward(
        layer,
        q_proj_states=data[0],
        k_proj_states=data[0],
        v_proj_states=data[0],
        g1=data,
        beta=torch.ones(1, count, 1),
        core_attn_out=output,
    )
    torch.testing.assert_close(output[:, :decodes], data[:, :decodes] + 10)
    torch.testing.assert_close(output[:, decodes:], data[:, decodes:] + 20)
    assert state[5].sum().item() == 4 * prefills and not state[:5].any()


def test_profile_without_layer_metadata(monkeypatch):
    monkeypatch.setattr(bailing, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    layer = SimpleNamespace(prefix="layer", _convolve=Mock(side_effect=AssertionError))
    bailing.AscendBailingMoeV3KimiDeltaAttention._forward(layer, None, None, None, None, None, None)
    layer._convolve.assert_not_called()


@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
def test_compact_conv_writes_only_live_rows(monkeypatch, dim_first, prefill):
    # Real slot 5, null slot 0 and pad slot -1 share one padded backing tensor.
    shape = (6, 2) if dim_first else (2, 6)
    raw = torch.full((7, 20), 37, dtype=torch.float16)
    state = raw[:, :12].view(7, *shape)
    expected = raw.clone()
    inputs = [torch.ones((3, 2), dtype=torch.bfloat16) for _ in range(3)]
    conv = SimpleNamespace(weight=torch.ones(2, 1, 3), bias=None)
    layer = SimpleNamespace(
        kv_cache=(state, None), q_conv1d=conv, k_conv1d=conv, v_conv1d=conv, local_num_heads=1, head_dim=2
    )
    ids = torch.tensor([5, 0, -1])
    starts = torch.tensor([0, 1, 2, 3])
    meta = SimpleNamespace(
        non_spec_state_indices_tensor=ids,
        non_spec_query_start_loc=starts,
        num_prefills=int(prefill),
        has_initial_state=torch.tensor([False, False, False]),
    )
    monkeypatch.setattr(bailing, "is_conv_state_dim_first", lambda: dim_first)

    def gather(cache, ids, keep, output_dtype):
        return torch.where(keep[:, None, None], cache[ids.clamp_min(0)], 0).to(output_dtype)

    def scatter(cache, ids, rows, active_rows):
        cache[ids[active_rows]] = rows[active_rows].to(cache.dtype)

    def update(x, rows, weights, **kwargs):
        assert rows.dtype == torch.bfloat16 and rows.shape == (3, 2, 6) and rows.is_contiguous()
        assert kwargs["conv_state_indices"].tolist() == [0, -1, -1]
        assert rows[0].eq(0 if prefill else 37).all()
        rows.fill_(11)
        return x

    def chunk(x, weights, bias, *, conv_states, cache_indices, **kwargs):
        return update(x, conv_states, weights, conv_state_indices=cache_indices)

    monkeypatch.setattr(bailing, "gather_ssm_states", gather)
    monkeypatch.setattr(bailing, "scatter_ssm_states_", scatter)
    monkeypatch.setattr(bailing, "causal_conv1d_update", update)
    monkeypatch.setattr(bailing, "causal_conv1d_fn", chunk)
    outputs = bailing.AscendBailingMoeV3KimiDeltaAttention._convolve(layer, inputs, meta)
    expected[5, :12] = 11
    torch.testing.assert_close(raw, expected)
    for x in outputs:
        torch.testing.assert_close(x.reshape_as(inputs[0]), inputs[0])
