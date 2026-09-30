# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression tests for the MLA-NoPE chunked-context paged gather."""

from types import SimpleNamespace

import torch

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.device_op import DeviceOperator

NUM_TOKENS = 6
NUM_HEADS = 2
QK_NOPE_HEAD_DIM = 4
V_HEAD_DIM = 3
LATENT_KV_DIM = 8
CHUNK_TOKS = 5


def _build_impl(rope_dim: int) -> AscendMLAImpl:
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.num_heads = NUM_HEADS
    impl.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    impl.qk_rope_head_dim = rope_dim
    impl.v_head_dim = V_HEAD_DIM
    impl.head_padding = 0
    impl.scale = 1.0
    impl.fa_quant_layer = False
    impl.kv_b_proj = lambda x: (torch.zeros(x.size(0), NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)),)
    impl.get_context_seq_len_npu = lambda index, metadata: torch.tensor([CHUNK_TOKS], dtype=torch.int32)
    return impl


def _build_metadata():
    chunked_context = SimpleNamespace(
        seq_tot=[CHUNK_TOKS],
        starts=[torch.zeros(1, dtype=torch.int32)],
        chunk_actual_seq_lengths_kv_list=[[CHUNK_TOKS]],
    )
    prefill = SimpleNamespace(
        chunked_context=chunked_context,
        actual_seq_lengths_q=[NUM_TOKENS],
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    return SimpleNamespace(prefill=prefill)


def _install_fakes(monkeypatch, captured):
    def fake_kv_cache_load(cache_kv_c, cache_k_pe, block_table, context_seq_len_npu, seq_starts, key, value):
        captured["pe_cache"] = cache_k_pe
        captured["key"] = key
        captured["value"] = value

    monkeypatch.setattr(DeviceOperator, "kv_cache_load", staticmethod(fake_kv_cache_load))
    monkeypatch.setattr(
        "vllm_ascend.attention.mla_v1.torch_npu",
        SimpleNamespace(
            npu_fused_infer_attention_score=lambda *args, **kwargs: (
                torch.zeros(NUM_TOKENS, NUM_HEADS, V_HEAD_DIM),
                torch.zeros(NUM_TOKENS * NUM_HEADS),
            ),
            npu_attention_update=lambda lse, out, dim: (
                torch.zeros(NUM_TOKENS * NUM_HEADS, V_HEAD_DIM),
                None,
            ),
        ),
    )


def _run(impl, rope_dim, cache_k_pe):
    return impl._compute_prefill_context(
        torch.zeros(NUM_TOKENS, NUM_HEADS, QK_NOPE_HEAD_DIM),
        torch.zeros(NUM_TOKENS, NUM_HEADS, rope_dim),
        (torch.zeros(2, 3, 1, LATENT_KV_DIM), cache_k_pe),
        rope_dim,
        _build_metadata(),
        torch.zeros(NUM_TOKENS, NUM_HEADS, V_HEAD_DIM),
        torch.zeros(NUM_HEADS, NUM_TOKENS),
    )


def test_chunked_gather_avoids_zero_width_operands_without_rope(monkeypatch):
    captured: dict = {}
    _install_fakes(monkeypatch, captured)

    _run(_build_impl(rope_dim=0), 0, torch.zeros(2, 3, 1, 0))

    # npu_gather_pa_kv_cache returns without filling the latent output when the
    # rope cache and its destination are both zero-width, which fed
    # uninitialised KV into every prefill longer than max_num_batched_tokens.
    assert captured["pe_cache"].size(-1) != 0
    assert captured["value"].size(-1) != 0
    assert captured["key"].shape == (CHUNK_TOKS, 1, LATENT_KV_DIM)


def test_chunked_gather_keeps_the_rope_cache_when_rope_is_present(monkeypatch):
    captured: dict = {}
    _install_fakes(monkeypatch, captured)
    cache_k_pe = torch.zeros(2, 3, 1, 64)

    _run(_build_impl(rope_dim=64), 64, cache_k_pe)

    assert captured["pe_cache"] is cache_k_pe
    assert captured["value"].shape == (CHUNK_TOKS, 1, 64)
