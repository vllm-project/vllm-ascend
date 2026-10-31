# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DSpine speculative-decoding proposer and model helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.models.qwen3_dspine import (
    DSpineCodeBridge,
    DSpineQwen3DecoderLayer,
    DSpineQwen3ForCausalLM,
    DSpineQwen3Model,
    build_serial_tables,
    dspine_serial_chain,
    dspine_serial_table,
    resolve_dspine_head_topk,
)
from vllm_ascend.worker.v1.spec_decode.dspine_proposer import (
    AscendDSpineProposer,
    is_dspine_draft,
)


def _dspine_hf_config(**overrides) -> SimpleNamespace:
    dflash_config = {
        "draft_block_causal": True,
        "selected_prefix_code_bridge": {"code_size": 8, "eps": 1e-6, "rank": 4},
        "selected_prefix_feature_layers": [1, 2, 3],
        "selected_prefix_feature_write_mode": "independent",
        "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {"head_topk": 4, "kind": "exact_sender"},
        "selected_prefix_terminal": False,
    }
    dflash_config.update(overrides.pop("dflash_config", {}))
    return SimpleNamespace(
        hidden_size=16,
        num_hidden_layers=3,
        block_size=4,
        vocab_size=32,
        dflash_config=dflash_config,
        **overrides,
    )


def _dspine_spec_config(hf_config=None) -> SimpleNamespace:
    return SimpleNamespace(
        method="dflash",
        draft_model_config=SimpleNamespace(
            architectures=["DFlashDraftModel"],
            hf_config=hf_config or _dspine_hf_config(),
        ),
    )


def test_is_dspine_draft_requires_bridge_and_dflash_method():
    assert is_dspine_draft(_dspine_spec_config())
    # Plain DFlash draft: no code bridge.
    assert not is_dspine_draft(
        _dspine_spec_config(_dspine_hf_config(dflash_config={"selected_prefix_code_bridge": None}))
    )
    # Wrong method.
    cfg = _dspine_spec_config()
    cfg.method = "eagle"
    assert not is_dspine_draft(cfg)
    # Missing hf_config / dflash_config entirely.
    assert not is_dspine_draft(SimpleNamespace(method="dflash", draft_model_config=SimpleNamespace(hf_config=None)))


def test_get_spec_decode_method_dispatches_dspine():
    from vllm_ascend.worker.v1.spec_decode import get_spec_decode_method

    vllm_config = SimpleNamespace(speculative_config=_dspine_spec_config())
    with (
        patch("vllm_ascend.worker.v1.spec_decode.AscendDSpineProposer", return_value="dspine") as dspine,
        patch("vllm_ascend.worker.v1.spec_decode.AscendDflashProposer", return_value="dflash1") as d1,
    ):
        assert get_spec_decode_method("dflash", vllm_config, "cpu", None) == "dspine"
        dspine.assert_called_once()
        d1.assert_not_called()


def test_dspine_declares_factory_classes():
    """DSpine subclasses route through the decoder-layer and model factories."""
    assert DSpineQwen3Model.decoder_layer_cls is DSpineQwen3DecoderLayer
    assert DSpineQwen3ForCausalLM.model_cls is DSpineQwen3Model
    # The checkpoint ships no lm_head; the target's head is shared.
    assert DSpineQwen3ForCausalLM.has_own_lm_head is False


def test_resolve_head_topk_uses_refine_config():
    config = _dspine_hf_config()
    assert resolve_dspine_head_topk(config, num_layers=3) == 4


def test_resolve_head_topk_cross_checks_second_pass():
    config = _dspine_hf_config(code_bridge_second_pass={"mode": "serial_table", "first_exact": 2, "head_topk": 4})
    assert resolve_dspine_head_topk(config, num_layers=3) == 4
    # Disagreeing widths must fail loudly.
    config.code_bridge_second_pass["head_topk"] = 8
    with pytest.raises(ValueError, match="disagree"):
        resolve_dspine_head_topk(config, num_layers=3)


def test_resolve_head_topk_rejects_bad_config():
    config = _dspine_hf_config()
    config.dflash_config["selected_prefix_last_write_refine"] = None
    with pytest.raises(ValueError, match="no refinement width"):
        resolve_dspine_head_topk(config, num_layers=3)

    config = _dspine_hf_config()
    config.dflash_config["selected_prefix_last_write_refine"]["kind"] = "rowwise"
    with pytest.raises(ValueError, match="exact_sender"):
        resolve_dspine_head_topk(config, num_layers=3)

    config = _dspine_hf_config()
    config.dflash_config["selected_prefix_last_write_refine"]["head_topk"] = 6
    with pytest.raises(ValueError, match="power of two"):
        resolve_dspine_head_topk(config, num_layers=3)


def test_bridge_config_validation():
    # Terminal-enabled checkpoints are unsupported.
    with pytest.raises(ValueError, match="terminal"):
        DSpineCodeBridge(_dspine_hf_config(dflash_config={"selected_prefix_terminal": True}))
    # The last layer must be an active writer.
    with pytest.raises(ValueError, match="last layer must be active"):
        DSpineCodeBridge(_dspine_hf_config(dflash_config={"selected_prefix_feature_layers": [1, 2]}))
    # Non-causal drafts are rejected.
    with pytest.raises(ValueError, match="draft_block_causal"):
        DSpineCodeBridge(_dspine_hf_config(dflash_config={"draft_block_causal": False}))
    # A complete config constructs cleanly and reports rank/block size.
    bridge = DSpineCodeBridge(_dspine_hf_config())
    assert bridge.rank == 4
    assert bridge.block_size == 4
    assert bridge.active_layers == frozenset({0, 1, 2})


def test_build_serial_tables_shape_contract():
    v, h, r, c = 32, 16, 4, 8
    lm_head_w = torch.randn(v, h, dtype=torch.bfloat16)
    norm_w = torch.randn(h)
    write_w = torch.randn(h, r)
    pv_w = torch.randn(2 * r, c)
    codebook = torch.randn(v, c)
    tg, tsv = build_serial_tables(lm_head_w, norm_w, write_w, pv_w, codebook)
    assert tg.shape == (v, r) and tg.dtype == torch.bfloat16
    assert tsv.shape == (v, 2 * r) and tsv.dtype == torch.bfloat16
    # TG = bf16((lm_head * norm) @ write_w), chunked identically to rows.
    expected_tg = (lm_head_w.float() * norm_w.float()).to(torch.bfloat16) @ write_w.to(torch.bfloat16)
    torch.testing.assert_close(tg, expected_tg)


def _reference_serial_table(raw, norm_w, cand, input_ids, lm_head_w, tg, tsv, packed, rms, gate_b, block_size):
    """Slow Python mirror of ``dspine_serial_table``."""
    t = block_size
    bs = raw.shape[0] // t
    n, k = cand.shape
    r = tg.shape[1]
    raw_rows = raw.view(bs, t, -1)[:, 1:, :].reshape(n, -1).float()
    rms_rows = rms.view(bs, t)[:, 1:].reshape(n).float()
    q = packed.view(bs, t, -1)[:, 1:, :r].reshape(n, r).float()
    anchors = input_ids.view(bs, t)[:, 0].long()
    idx = torch.empty(n, k, dtype=torch.int64)
    for row in range(n):
        req, sub = divmod(row, t - 1)
        for j in range(k):
            sender = anchors[req] if sub == 0 else cand.view(bs, t - 1, k)[req, sub - 1, j]
            gp, a = tsv[sender].float().split(r)
            msg = torch.sigmoid(q[row] + gp + gate_b.float()) * a
            u = raw_rows[row] * norm_w.float()
            scores = [
                u @ lm_head_w[cand[row, c]].float() + rms_rows[row] * (msg @ tg[cand[row, c]].float()) for c in range(k)
            ]
            idx[row, j] = int(torch.tensor(scores).argmax())
    return idx


def test_serial_table_and_chain_match_reference():
    torch.manual_seed(0)
    bs, t, k, h, r, c, v = 2, 4, 4, 16, 4, 8, 64
    raw = torch.randn(bs * t, h, dtype=torch.bfloat16)
    norm_w = torch.randn(h)
    input_ids = torch.randint(v, (bs, t))
    cand = torch.randint(v, (bs * (t - 1), k))
    lm_head_w = torch.randn(v, h, dtype=torch.bfloat16)
    tg, tsv = build_serial_tables(
        lm_head_w,
        norm_w,
        torch.randn(h, r),
        torch.randn(2 * r, c),
        torch.randn(v, c),
    )
    packed = torch.randn(bs * t, 3 * r, dtype=torch.bfloat16)
    rms = torch.rand(bs * t) + 0.5
    gate_b = torch.randn(r)

    idx = dspine_serial_table(raw, norm_w, cand, input_ids.view(-1), lm_head_w, tg, tsv, packed, rms, gate_b, t)
    expected = _reference_serial_table(raw, norm_w, cand, input_ids, lm_head_w, tg, tsv, packed, rms, gate_b, t)
    torch.testing.assert_close(idx, expected)

    # Chain: start from the anchor-conditioned lane (j=0 selects the anchor
    # sender for row 0), then follow the table.
    out = dspine_serial_chain(idx, cand, t)
    cand3 = cand.view(bs, t - 1, k)
    idx3 = idx.view(bs, t - 1, k)
    expected_tokens = []
    for req in range(bs):
        j = int(idx3[req, 0, 0])
        for row in range(t - 1):
            expected_tokens.append(int(cand3[req, row, j]))
            if row + 1 < t - 1:
                j = int(idx3[req, row + 1, j])
    torch.testing.assert_close(out, torch.tensor(expected_tokens))


def test_compute_draft_token_ids_runs_serial_refinement(monkeypatch):
    """The proposer slices the stash to the live rows, top-Ks the lm_head logits,
    and returns the greedy chain."""
    bs, t, k, h, r, v = 2, 4, 4, 16, 4, 64
    proposer = AscendDSpineProposer.__new__(AscendDSpineProposer)
    proposer.input_ids = torch.randint(v, (bs * t,))

    model = MagicMock()
    bridge = SimpleNamespace(block_size=t, gate=SimpleNamespace(bias=torch.randn(r)))
    model.model.selected_prefix_bridge = bridge
    model.model.head_topk = k
    model.model.stash_raw = torch.randn(64, h, dtype=torch.bfloat16)
    model.model.stash_rms = torch.rand(64) + 0.5
    model.model.stash_packed = torch.randn(64, 3 * r, dtype=torch.bfloat16)
    model.model.norm = SimpleNamespace(weight=torch.randn(h))
    model.lm_head = SimpleNamespace(weight=torch.randn(v, h, dtype=torch.bfloat16))
    model._dspine_tg = torch.randn(v, r)
    model._dspine_tsv = torch.randn(v, 2 * r)
    logits = torch.randn(bs * (t - 1), v)
    model.logits_processor.return_value = logits
    proposer.model = model

    captured = {}

    def fake_table(raw, norm_w, cand, input_ids, lm_head_w, tg, tsv, packed, rms, gate_b, block_size):
        captured.update(raw=raw, cand=cand, input_ids=input_ids, rms=rms)
        return torch.zeros(cand.shape, dtype=torch.int64)

    def fake_chain(idx, cand, block_size):
        return cand.view(-1, block_size - 1, cand.shape[-1])[:, :, 0].reshape(-1)

    monkeypatch.setattr("vllm_ascend.worker.v1.spec_decode.dspine_proposer.dspine_serial_table", fake_table)
    monkeypatch.setattr("vllm_ascend.worker.v1.spec_decode.dspine_proposer.dspine_serial_chain", fake_chain)

    hidden = torch.randn(bs * (t - 1), h)
    tokens, probs = proposer.compute_draft_token_ids(hidden)

    assert probs is None
    n = bs * (t - 1)
    assert captured["cand"].shape == (n, k)
    # top-K over the lm_head logits.
    torch.testing.assert_close(captured["cand"], logits.topk(k, dim=-1).indices)
    # Stash sliced to the live m = bs * t rows.
    assert captured["raw"].shape == (bs * t, h)
    assert captured["rms"].shape == (bs * t,)
    torch.testing.assert_close(captured["input_ids"], proposer.input_ids[: bs * t])
    assert tokens.shape == (n,)
