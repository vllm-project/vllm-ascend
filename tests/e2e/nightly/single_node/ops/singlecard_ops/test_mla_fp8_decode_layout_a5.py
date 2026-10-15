# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real A5 FIA output-to-V-projection regressions, without model weights.

Run on Ascend 950 with the same torch_npu/CANN bundle as the serving test::

    pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_fp8_decode_layout_a5.py

These tests cover the backend and its updatable attention graph, rather than
claiming full-model Eagle3 generation or FULL-graph dispatch coverage.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, DecodeMLAPreprocessResult
from vllm_ascend.compilation import acl_graph
from vllm_ascend.device.device_config import check_ascend_device_type, is_950

NUM_HEADS = 16
LATENT_DIM = 512
ROPE_DIM = 64
VALUE_DIM = 128
BLOCK_SIZE = 128
INITIAL_KV_LEN = 31
UPDATED_KV_LEN = 47


@pytest.fixture(scope="module", autouse=True)
def require_a5():
    if not torch.npu.is_available():
        pytest.skip("FP8 MLA decode layout requires Ascend 950")
    check_ascend_device_type()
    if not is_950():
        pytest.skip("FP8 MLA decode layout requires Ascend 950")


def _inputs(num_tokens, fp8, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16

    def tensor(shape, tensor_dtype):
        return (torch.randn(shape, generator=generator) * 0.5).to(tensor_dtype).npu()

    return DecodeMLAPreprocessResult(
        ql_nope=tensor((num_tokens, NUM_HEADS, LATENT_DIM), dtype),
        q_pe=tensor((num_tokens, NUM_HEADS, ROPE_DIM), torch.bfloat16),
        k_nope=tensor((num_tokens, BLOCK_SIZE, 1, LATENT_DIM), dtype),
        k_pe=tensor((num_tokens, BLOCK_SIZE, 1, ROPE_DIM), torch.bfloat16),
        dequant_scale_q_nope=torch.ones(num_tokens, NUM_HEADS, device="npu") if fp8 else None,
    )


def _case(monkeypatch, num_tokens, speculative, fp8):
    # Bypass model construction only. FIA, _forward_decode, _v_up_proj and
    # graph_task_update all remain the production implementations.
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.num_heads = impl.num_heads_padded = NUM_HEADS
    impl.num_kv_heads = 1
    impl.head_padding = 0
    impl.kv_lora_rank = LATENT_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.v_head_dim = VALUE_DIM
    impl.scale = (LATENT_DIM + ROPE_DIM) ** -0.5
    impl.enable_kv_nz = False
    impl.support_fp8_attention = True
    impl.fa_quant_layer = fp8
    impl.fak_descale_float = torch.ones(1, device="npu")
    impl.speculative_config = (
        SimpleNamespace(use_eagle=lambda: True, num_speculative_tokens=num_tokens - 1) if speculative else None
    )
    generator = torch.Generator(device="cpu").manual_seed(19)
    impl.W_UV = (torch.randn(NUM_HEADS, LATENT_DIM, VALUE_DIM, generator=generator) / LATENT_DIM**0.5).bfloat16().npu()

    num_seqs = 1 if speculative else num_tokens
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding if speculative else AscendAttentionState.DecodeOnly,
        causal=True,
        decode=SimpleNamespace(
            block_table=torch.arange(num_seqs, dtype=torch.int32, device="npu").view(num_seqs, 1),
            seq_lens_list=[INITIAL_KV_LEN + i for i in range(num_seqs)],
            actual_seq_lengths_q=[num_tokens] if speculative else None,
            attn_mask=(
                torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device="npu"), diagonal=1) if speculative else None
            ),
        ),
    )
    extra_context = SimpleNamespace(capturing=False, is_draft_model=False, is_draft_model_prefill=False)
    monkeypatch.setattr(mla_v1, "_EXTRA_CTX", extra_context)
    params = acl_graph.GraphParams(
        events={num_tokens: []},
        workspaces={num_tokens: None},
        handles={num_tokens: []},
        attn_params={num_tokens: []},
    )
    monkeypatch.setattr(acl_graph, "_graph_params", params)
    assert mla_v1.get_graph_params() is params
    return impl, metadata, extra_context, params


def _reference(impl, inputs, metadata, speculative):
    """Native token-major FIA plus independent FP32 CPU V projection.

    TND_NTD is deliberately never a reference: the affected FP8 bundle can
    report an NTD shape while retaining token-major flattened values.
    """
    num_tokens = inputs.ql_nope.shape[0]
    if speculative:
        query = inputs.ql_nope
        query_rope = inputs.q_pe
        layout = "TND"
        query_scale = inputs.dequant_scale_q_nope
    else:
        query = inputs.ql_nope.view(num_tokens, NUM_HEADS, 1, LATENT_DIM)
        query_rope = inputs.q_pe.view(num_tokens, NUM_HEADS, 1, ROPE_DIM)
        layout = "BNSD"
        query_scale = inputs.dequant_scale_q_nope.view(num_tokens, NUM_HEADS, 1) if impl.fa_quant_layer else None
    quant_args = {}
    if impl.fa_quant_layer:
        quant_args = dict(
            query_quant_mode=3,
            key_quant_mode=0,
            value_quant_mode=0,
            dequant_scale_query=query_scale,
            dequant_scale_key=impl.fak_descale_float,
            dequant_scale_value=impl.fak_descale_float,
        )
    key = inputs.k_nope.view(-1, 1, BLOCK_SIZE, LATENT_DIM)
    key_rope = inputs.k_pe.view(-1, 1, BLOCK_SIZE, ROPE_DIM)
    token_major, _ = torch_npu.npu_fused_infer_attention_score_v2(
        query,
        key,
        key,
        query_rope=query_rope,
        key_rope=key_rope,
        num_query_heads=NUM_HEADS,
        num_key_value_heads=1,
        input_layout=layout,
        sparse_mode=3 if speculative else 0,
        atten_mask=metadata.decode.attn_mask,
        softmax_scale=impl.scale,
        block_table=metadata.decode.block_table,
        block_size=BLOCK_SIZE,
        actual_seq_qlen=metadata.decode.actual_seq_lengths_q,
        actual_seq_kvlen=metadata.decode.seq_lens_list,
        **quant_args,
    )
    token_major = token_major.reshape(num_tokens, NUM_HEADS, LATENT_DIM).float().cpu()
    projected = torch.einsum("thl,hlv->thv", token_major, impl.W_UV.float().cpu())
    return projected.reshape(num_tokens, NUM_HEADS * VALUE_DIM).bfloat16()


def _assert_output(actual, expected):
    # The reference accumulates in FP32; the real fused projection returns BF16.
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=3e-3)


@pytest.mark.e2e_coverage(
    arch="", feature="fia_comparison", parallel="", deploy="", hardware="A5", quantization="", graph_mode="eager"
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4])
@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16_control"])
@torch.inference_mode()
def test_mla_decode_output_to_v_projection_a5(monkeypatch, num_tokens, speculative, fp8):
    impl, metadata, _, _ = _case(monkeypatch, num_tokens, speculative, fp8)
    inputs = _inputs(num_tokens, fp8, seed=41)
    expected = _reference(impl, inputs, metadata, speculative)
    actual = impl._forward_decode(inputs, BLOCK_SIZE, metadata)
    _assert_output(actual, expected)


@pytest.mark.e2e_coverage(
    arch="", feature="fia_comparison", parallel="", deploy="", hardware="A5", quantization="", graph_mode="full_graph"
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4])
@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16_control"])
@torch.inference_mode()
def test_mla_decode_graph_replay_changed_inputs_a5(monkeypatch, num_tokens, speculative, fp8):
    impl, metadata, extra_context, params = _case(monkeypatch, num_tokens, speculative, fp8)
    inputs = _inputs(num_tokens, fp8, seed=41)
    expected = _reference(impl, inputs, metadata, speculative)
    for _ in range(2):
        impl._forward_decode(inputs, BLOCK_SIZE, metadata)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    extra_context.capturing = True
    try:
        with torch.npu.graph(graph):
            actual = impl._forward_decode(inputs, BLOCK_SIZE, metadata)
    finally:
        extra_context.capturing = False
    assert len(params.attn_params[num_tokens]) == 1
    saved_layout = params.attn_params[num_tokens][0][6]
    if speculative:
        expected_layout = "TND" if fp8 else "TND_NTD"
    else:
        expected_layout = "BNSD" if fp8 else "BNSD_NBSD"
    assert saved_layout == expected_layout
    graph.replay()
    _assert_output(actual, expected)

    update_stream = torch.npu.Stream()
    for seed in (73, 109):
        fresh = _inputs(num_tokens, fp8, seed)
        for persistent, replacement in zip(inputs, fresh):
            if persistent is not None:
                persistent.copy_(replacement)
        # Changing CPU sequence-length arguments also exercises the production
        # FIA task update, rather than only replaying in-place tensor copies.
        metadata.decode.seq_lens_list = [UPDATED_KV_LEN + i for i in range(len(metadata.decode.seq_lens_list))]
        changed_expected = _reference(impl, inputs, metadata, speculative)
        assert not torch.allclose(changed_expected.float(), expected.float(), rtol=1e-3, atol=1e-3)
        torch.npu.synchronize()
        AscendMLAImpl.update_graph_params(
            update_stream,
            SimpleNamespace(attn_metadata={"attn": metadata}),
            num_tokens,
            speculative_config=impl.speculative_config,
        )
        torch.npu.current_stream().wait_stream(update_stream)
        graph.replay()
        _assert_output(actual, changed_expected)
