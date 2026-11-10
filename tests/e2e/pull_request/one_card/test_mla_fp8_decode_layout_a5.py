# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real A5 FIA output-to-V-projection regressions, without model weights.

Run on Ascend 950 with the same torch_npu/CANN bundle as the serving test::

    pytest -sv tests/e2e/pull_request/one_card/test_mla_fp8_decode_layout_a5.py

These tests cover the backend and its updatable attention graph, rather than
claiming full-model Eagle3 generation or FULL-graph dispatch coverage.
The projection benchmark reports device time against the correct explicit-copy
implementation; it does not measure generation throughput.
"""

from statistics import median
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
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32)
FP8_MAX_QUERY_TOKENS_PER_SEQ = 16
# Keep speculative query lengths valid, including the 32-token case, and
# leave room for different per-sequence lengths inside one cache block.
INITIAL_KV_LEN = 47
UPDATED_KV_LEN = 79
PROJECTION_WARMUP_ITERS = 5
PROJECTION_BENCHMARK_ITERS = 50
PROJECTION_BENCHMARK_REPEATS = 5


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
    # Bypass model construction only. FIA, _forward_decode, both V projection
    # helpers and graph_task_update remain the production implementations.
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
    query_tokens_per_seq = num_tokens if speculative else 1
    if speculative and fp8:
        # The tested CANN 9.2 FP8 path supports at most 16 queries per sequence.
        query_tokens_per_seq = min(query_tokens_per_seq, FP8_MAX_QUERY_TOKENS_PER_SEQ)
    num_seqs = num_tokens // query_tokens_per_seq
    impl.speculative_config = (
        SimpleNamespace(use_eagle=lambda: True, num_speculative_tokens=query_tokens_per_seq - 1)
        if speculative
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(19)
    impl.W_UV = (torch.randn(NUM_HEADS, LATENT_DIM, VALUE_DIM, generator=generator) / LATENT_DIM**0.5).bfloat16().npu()

    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.SpecDecoding if speculative else AscendAttentionState.DecodeOnly,
        causal=True,
        decode=SimpleNamespace(
            block_table=torch.arange(num_seqs, dtype=torch.int32, device="npu").view(num_seqs, 1),
            seq_lens_list=[INITIAL_KV_LEN + i for i in range(num_seqs)],
            actual_seq_lengths_q=[query_tokens_per_seq * (i + 1) for i in range(num_seqs)] if speculative else None,
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
    """Native FIA with independent FP32 CPU and explicit-copy projections.

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
    token_major = token_major.reshape(num_tokens, NUM_HEADS, LATENT_DIM)
    return _cpu_projection_reference(impl, token_major), _explicit_copy_projection(impl, token_major)


def _cpu_projection_reference(impl, token_major):
    projected = torch.einsum("thl,hlv->thv", token_major.float().cpu(), impl.W_UV.float().cpu())
    return projected.reshape(token_major.shape[0], NUM_HEADS * VALUE_DIM).bfloat16()


def _explicit_copy_projection(impl, token_major):
    # This is the correct unfused layout normalization, not the original
    # head-major view of token-major data that caused this regression.
    return impl._v_up_proj(token_major.transpose(0, 1).contiguous())


def _assert_output(actual, expected):
    # Both NPU projection variants return BF16. Compare numerically rather
    # than requiring bitwise identity across different accumulation paths.
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=3e-3)


@pytest.mark.e2e_coverage(
    arch="", feature="fia_comparison", parallel="", deploy="", hardware="A5", quantization="", graph_mode="eager"
)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16_control"])
@torch.inference_mode()
def test_mla_decode_output_to_v_projection_a5(monkeypatch, num_tokens, speculative, fp8):
    impl, metadata, _, _ = _case(monkeypatch, num_tokens, speculative, fp8)
    inputs = _inputs(num_tokens, fp8, seed=41)
    expected, explicit_copy = _reference(impl, inputs, metadata, speculative)
    actual = impl._forward_decode(inputs, BLOCK_SIZE, metadata)
    _assert_output(actual, expected)
    _assert_output(explicit_copy, expected)
    _assert_output(actual, explicit_copy.cpu())


@pytest.mark.e2e_coverage(
    arch="", feature="fia_comparison", parallel="", deploy="", hardware="A5", quantization="", graph_mode="full_graph"
)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16_control"])
@torch.inference_mode()
def test_mla_decode_graph_replay_changed_inputs_a5(monkeypatch, num_tokens, speculative, fp8):
    impl, metadata, extra_context, params = _case(monkeypatch, num_tokens, speculative, fp8)
    inputs = _inputs(num_tokens, fp8, seed=41)
    expected, explicit_copy = _reference(impl, inputs, metadata, speculative)
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
    update_stream = torch.npu.Stream()
    # The captured FIA task waits for its ExternalEvent on the first replay too.
    AscendMLAImpl.update_graph_params(
        update_stream,
        SimpleNamespace(attn_metadata={"attn": metadata}),
        num_tokens,
        speculative_config=impl.speculative_config,
    )
    torch.npu.current_stream().wait_stream(update_stream)
    graph.replay()
    _assert_output(actual, expected)
    _assert_output(explicit_copy, expected)
    _assert_output(actual, explicit_copy.cpu())

    for update_index, seed in enumerate((73, 109)):
        fresh = _inputs(num_tokens, fp8, seed)
        for persistent, replacement in zip(inputs, fresh):
            if persistent is not None:
                persistent.copy_(replacement)
        # Changing CPU sequence-length arguments also exercises the production
        # FIA task update, rather than only replaying in-place tensor copies.
        metadata.decode.seq_lens_list = [
            UPDATED_KV_LEN + update_index + i for i in range(len(metadata.decode.seq_lens_list))
        ]
        changed_expected, changed_copy = _reference(impl, inputs, metadata, speculative)
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
        _assert_output(changed_copy, changed_expected)
        _assert_output(actual, changed_copy.cpu())
        expected = changed_expected


def _projection_time_us(project):
    for _ in range(PROJECTION_WARMUP_ITERS):
        project()
    torch.npu.synchronize()
    timings = []
    for _ in range(PROJECTION_BENCHMARK_REPEATS):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(PROJECTION_BENCHMARK_ITERS):
            project()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / PROJECTION_BENCHMARK_ITERS)
    return median(timings)


@pytest.mark.e2e_coverage(
    arch="", feature="fia_comparison", parallel="", deploy="", hardware="A5", quantization="", graph_mode="eager"
)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS[1:])
@torch.inference_mode()
def test_mla_fp8_v_projection_fusion_a5(monkeypatch, record_property, num_tokens):
    impl, _, _, _ = _case(monkeypatch, num_tokens, speculative=False, fp8=True)
    generator = torch.Generator(device="cpu").manual_seed(151)
    token_major = torch.randn(num_tokens, NUM_HEADS, LATENT_DIM, generator=generator).bfloat16().npu()
    expected = _cpu_projection_reference(impl, token_major)
    fused = impl._v_up_proj_batch_major(token_major)
    explicit_copy = _explicit_copy_projection(impl, token_major)
    _assert_output(fused, expected)
    _assert_output(explicit_copy, expected)
    _assert_output(fused, explicit_copy.cpu())

    # Measure only the projection, including the baseline's standalone copy.
    # Record the ratio for regression comparisons on the same NPU/CANN bundle;
    # a fixed cross-device latency threshold would make this test unstable.
    fused_us = _projection_time_us(lambda: impl._v_up_proj_batch_major(token_major))
    copy_us = _projection_time_us(lambda: _explicit_copy_projection(impl, token_major))
    assert fused_us > 0 and copy_us > 0
    record_property("fused_projection_us", fused_us)
    record_property("explicit_copy_projection_us", copy_us)
    record_property("projection_speedup", copy_us / fused_us)
    record_property("fused_projection_tokens_per_second", num_tokens * 1e6 / fused_us)
    print(
        f"tokens={num_tokens}: fused={fused_us:.2f} us, "
        f"explicit_copy={copy_us:.2f} us, speedup={copy_us / fused_us:.3f}"
    )
