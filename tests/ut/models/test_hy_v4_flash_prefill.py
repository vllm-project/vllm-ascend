# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.hisparse.c8 import pack_c8_components
from vllm_ascend.models.hy_v4_flash import attention


@pytest.mark.parametrize("other_layer_first", [False, True])
def test_prefill_attention_uses_current_layer_metadata(monkeypatch, other_layer_first):
    layer_name = "layers.8.attn"
    metadata = attention.FlashAttentionMetadata((0, 2), (2,), torch.empty(0), torch.empty(0), prefill_reqs=(True,))
    entries = [(layer_name, metadata), ("other.attn", SimpleNamespace())]
    if other_layer_first:
        entries.reverse()
    context = SimpleNamespace(attn_metadata=dict(entries))
    monkeypatch.setattr(attention, "get_forward_context", lambda: context)
    monkeypatch.setattr(attention, "is_forward_context_available", lambda: True)
    layer = attention.AscendHYV4FlashAttention.__new__(attention.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(layer)
    layer.layer_name = layer_name
    layer.native_decode_attention = False
    layer.use_cann_prefill = True
    layer.use_cannbot_prolog = False
    layer.use_eager_prefill_rotary = False
    layer.fuse_attn_output_gate = False
    layer.overlap_attn_output_gate = False
    layer.sequence_parallel_output = False
    layer.post_proj_norm = None
    layer.local_heads = 1
    layer.latent_dim = layer.nope_dim = layer.rope_dim = layer.v_dim = 2
    layer._project_query = lambda hidden: (hidden, hidden)
    layer.kv_a_proj_with_mqa = lambda hidden: (hidden,)
    layer.kv_a_layernorm = lambda kv: kv
    layer._rotate_qk = lambda positions, q, k: (q, k)
    layer._project_output_gate = lambda hidden, fuse: hidden[:, :2]
    layer.o_proj = lambda output: (output,)

    def forward_paged(*args, absorbed, reuse_prefill_output=False):
        assert reuse_prefill_output
        return torch.full_like(args[-1], 3)

    layer._forward_paged = forward_paged
    output = layer(torch.arange(2), torch.ones(2, 4))
    torch.testing.assert_close(output, torch.full((2, 2), 3.0))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize(
    ("query_lengths", "is_prefilling", "precise_decode", "expect_reuse"),
    [
        ((33, 35), (True, True), False, True),
        ((16, 7), (True, True), False, False),
        ((6, 6), (False, False), True, False),
        ((33, 35), (True, False), False, True),
        ((33, 35), (False, False), False, False),
        ((33, 35), None, False, False),
    ],
    ids=["prefill", "short-prefill", "verify", "mixed", "decode", "unknown"],
)
def test_prefill_history_reuse_preserves_candidates_and_outputs(
    sparse, query_lengths, is_prefilling, precise_decode, expect_reuse
):
    generator = torch.Generator().manual_seed(1024)
    heads, latent, rope, block_size = 2, 512, 64, 8
    seq_lens = (33, 79)
    starts = (0, query_lengths[0], sum(query_lengths))
    model = attention.AscendHYV4FlashAttention.__new__(attention.AscendHYV4FlashAttention)
    torch.nn.Module.__init__(model)
    model.window = 8
    model.ops = attention.triton_ops
    model.use_cann_prefill = False
    model.fuse_c8_prefill_window = False
    model.latent_dim = latent
    model.kv_width = latent + rope
    model.kv_storage_width = 656
    model.use_fp8_cache = True
    model.batch_verify_attention = False
    model.use_triton_mla_logits = False
    model.incremental_index_pool = False
    model.scale = 576**-0.5
    model.learnable_sink_param = torch.randn(heads, generator=generator)
    model.indexer = None
    model.use_pd_decode = False
    model.bulk_c8_decode_reads = False
    if sparse:
        keys = torch.randn(20, 128, generator=generator)
        index_query = torch.randn(starts[-1], heads, 128, generator=generator)
        index_weight = torch.randn(starts[-1], heads, generator=generator)
        model.indexer = SimpleNamespace(
            ratio=4,
            budget=2,
            encode_queries=lambda *args: (index_query, index_weight),
            encode_keys=lambda raw, pos: keys[: raw.shape[0] // 4],
        )
        model._read_index_rows = lambda cache, table, size, history, ndim: (
            torch.zeros(history.numel(), 128),
            history,
        )
    # Each request has its own permuted physical pages. Include unused sidecar
    # bytes so a read cannot accidentally assume the 656-byte payload stride.
    pages = torch.randperm(20, generator=generator).reshape(2, 10)
    rows = pages.numel() * block_size
    payload = torch.randn(rows, latent, generator=generator).to(torch.float8_e4m3fn)
    rotary = torch.randn(rows, rope, generator=generator).bfloat16()
    scales = torch.rand(rows, 4, generator=generator) + 0.01
    packed = pack_c8_components(payload, rotary, scales)
    cache = torch.zeros(rows, 1024, dtype=torch.uint8)
    cache[:, :656] = packed
    query = torch.randn(starts[-1], heads, latent + rope, generator=generator).bfloat16()
    weight = torch.randn(heads, latent, 4, generator=generator).bfloat16()
    builder = attention.FlashMetadataBuilder.__new__(attention.FlashMetadataBuilder)
    builder.precise_decode = precise_decode
    metadata = builder.build(
        0,
        SimpleNamespace(
            num_reqs=len(seq_lens),
            seq_lens_cpu=torch.tensor(seq_lens),
            query_start_loc_cpu=torch.tensor(starts),
            is_prefilling=torch.tensor(is_prefilling) if is_prefilling is not None else None,
            block_table_tensor=pages,
            slot_mapping=torch.empty(0, dtype=torch.long),
        ),
    )
    outputs, snapshots, reads = [], [], []
    gather = model._gather_attention_rows
    for prefill_metadata in (replace(metadata, prefill_reqs=()), metadata):
        model.topk_snapshot = torch.full((starts[-1], 80), -1, dtype=torch.int64)
        count = []

        def recorded_gather(*args, recorded=count, **kwargs):
            recorded.append(args[3].numel())
            return gather(*args, **kwargs)

        model._gather_attention_rows = recorded_gather
        out = torch.empty(starts[-1], heads * 4, dtype=torch.bfloat16)
        model._attend_eager(
            prefill_metadata, cache, block_size, query, query, torch.arange(starts[-1]), query, weight, out
        )
        outputs.append(out)
        snapshots.append(model.topk_snapshot.clone())
        reads.append(count)
    torch.testing.assert_close(snapshots[0], snapshots[1], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
    if expect_reuse and not sparse:
        if is_prefilling and all(is_prefilling):
            assert len(reads[1]) == len(seq_lens)
        assert len(reads[1]) < len(reads[0])
        assert sum(reads[1]) < sum(reads[0])
    else:
        assert reads[0] == reads[1]


@pytest.mark.parametrize(
    ("role", "prefill"),
    [(None, False), ("kv_both", False), ("kv_consumer", False), ("kv_producer", True)],
)
def test_hy4_operator_role(role, prefill):
    transfer = None if role is None else SimpleNamespace(kv_role=role)
    config = SimpleNamespace(kv_transfer_config=transfer)
    assert attention.uses_prefill_ops(config) is prefill
