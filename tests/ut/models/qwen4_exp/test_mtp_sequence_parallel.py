# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.qwen4_exp import mtp


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("mrope", [False, True])
def test_mtp_preserves_decoder_token_sharding_and_gathers_outputs(monkeypatch, sequence_parallel, mrope):
    rows, hidden_size, hc_count = 4, 3, 2
    local_rows = rows // 2 if sequence_parallel else rows
    calls = []
    hidden = torch.arange(rows * hidden_size * hc_count, dtype=torch.float32).reshape(rows, -1)
    embeds = torch.ones(rows, hidden_size)

    def fc_hidden(x):
        # Ascend quantized linears require a two-dimensional input.
        assert x.ndim == 2
        return x

    def shard(x):
        calls.append("shard")
        return x[:local_rows].clone()

    def decoder(**kwargs):
        x = kwargs["hidden_states"]
        assert x.shape == (local_rows, hidden_size * hc_count)
        assert kwargs["positions"].shape == (rows,)
        return x, x[:, :hidden_size].contiguous(), torch.zeros(local_rows, hc_count)

    def mix(x, block, injection):
        assert x.shape[0] == block.shape[0] == injection.shape[0] == local_rows
        return x, block, None

    def gather(x):
        calls.append("gather")
        return torch.cat([x, x], dim=0)

    monkeypatch.setattr(mtp, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True))
    monkeypatch.setattr(mtp.upstream_mtp, "sp_shard", shard)
    monkeypatch.setattr(mtp.upstream_mtp, "sp_all_gather", gather)
    layer = SimpleNamespace(
        hc_count=hc_count,
        hidden_size=hidden_size,
        num_mtp_layers=1,
        use_hc_sequence_parallel=sequence_parallel,
        pre_fc_norm_embedding=lambda x: x,
        fc_embedding=lambda x: x,
        pre_fc_norm_hidden=lambda x: x,
        fc_hidden=fc_hidden,
        layers=[decoder],
        hyper_connection_mixer=SimpleNamespace(combine_and_mix=mix),
    )
    positions = torch.arange(rows)
    if mrope:
        positions = positions.expand(3, -1)
    sample, multi = mtp.AscendQwen4ExpMultiTokenPredictor.forward(
        layer, None, positions, hidden_states=hidden, inputs_embeds=embeds
    )
    expected = hidden + 1
    if sequence_parallel:
        expected = expected[:local_rows].repeat(2, 1)
    torch.testing.assert_close(multi, expected)
    torch.testing.assert_close(sample, expected[:, :hidden_size])
    assert calls == (["shard", "gather"] if sequence_parallel else [])
    assert sample.is_contiguous() and multi.is_contiguous()
