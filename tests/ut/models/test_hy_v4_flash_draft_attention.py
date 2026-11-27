# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft attention must replace target-cache queries without mutating history."""

import pytest
import torch

from vllm_ascend.models.hy_v4_flash.dfly import DFlyAttention


@pytest.mark.parametrize("batch", [1, 10])
@pytest.mark.parametrize("width", [1, 5])
def test_draft_attention_substitutes_causal_query_rows(batch, width):
    generator = torch.Generator().manual_seed(1024)
    heads, latent, nope, rope, value, page_size = 2, 4, 3, 2, 3, 4
    history_lengths = [3 + request % 3 for request in range(batch)]
    max_history = max(history_lengths)
    pages_per_request = (max_history + width + page_size - 1) // page_size
    pages = torch.randperm(batch * pages_per_request, generator=generator).view(batch, -1)
    cache = torch.randn(batch * pages_per_request * page_size, latent + rope, generator=generator)
    original_cache = cache.clone()
    positions = torch.tensor(history_lengths)[:, None] + torch.arange(width)
    selected = torch.full((batch, width, 1 + max_history + width), -1, dtype=torch.long)
    for request, history in enumerate(history_lengths):
        candidates = torch.arange(history + width)
        selected[request, :, 1 : 1 + candidates.numel()] = candidates
    selected.masked_fill_(selected > positions[..., None], -1)

    layer = DFlyAttention.__new__(DFlyAttention)
    torch.nn.Module.__init__(layer)
    layer.context = (page_size, pages, selected, positions)
    target = torch.nn.Module()
    target.kv_cache = cache
    target.cache_width = target.kv_storage_width = latent + rope
    target.use_pd_decode = False
    layer.bind_target(target)
    layer._decode_index_constants = {}
    layer._decode_key_projection = torch.randn(heads, nope, latent, generator=generator)
    layer._decode_value_projection = torch.randn(heads, latent, value, generator=generator)
    layer.use_fp8_cache = layer.native_decode_attention = False
    layer.latent_dim = latent
    layer.scale = (latent + rope) ** -0.5
    layer.learnable_sink_param = torch.randn(heads, generator=generator)
    q_nope = torch.randn(batch * width, heads, nope, generator=generator)
    q_pe = torch.randn(batch * width, heads, rope, generator=generator)
    kv_c = torch.randn(batch * width, latent, generator=generator)
    k_pe = torch.randn(batch * width, rope, generator=generator)
    output = torch.empty(batch * width, heads * value)
    layer._forward_paged(None, None, q_nope, q_pe, kv_c, k_pe, positions, output)

    # Build dense causal histories independently of the paged reader and draft
    # substitution: stale target-cache rows at query positions must be ignored.
    expected = []
    queries = torch.cat((torch.einsum("qhi,hij->qhj", q_nope, layer._decode_key_projection), q_pe), -1)
    draft_kv = torch.cat((kv_c, k_pe), -1).view(batch, width, -1)
    for request, history in enumerate(history_lengths):
        offsets = torch.arange(history)
        slots = pages[request, offsets // page_size] * page_size + offsets % page_size
        for step in range(width):
            rows = torch.cat((cache[slots], draft_kv[request, : step + 1]))
            scores = queries[request * width + step] @ rows.T * layer.scale
            probabilities = torch.cat((scores, layer.learnable_sink_param[:, None]), -1).softmax(-1)[:, :-1]
            context = probabilities @ rows[:, :latent]
            expected.append(torch.einsum("hl,hlv->hv", context, layer._decode_value_projection).flatten())
    torch.testing.assert_close(output, torch.stack(expected), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(cache, original_cache, rtol=0, atol=0)
