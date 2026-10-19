# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 no-RoPE MLAPO + real scatter checks and complete-preprocess timings.

DCP cases exercise current-buffer reuse, without emulating distributed attention.
Run with ``pytest -s`` to display device timings; no speed threshold is imposed.
"""

import math
from statistics import median
from types import MethodType, SimpleNamespace

import pytest

torch_npu = pytest.importorskip("torch_npu")
torch = pytest.importorskip("torch")
pytest.importorskip("cann_ops_transformer")

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402

from vllm_ascend.attention.mla_v1 import AscendMLAImpl  # noqa: E402
from vllm_ascend.device.device_config import check_ascend_device_type  # noqa: E402
from vllm_ascend.device.device_op import DeviceOperator  # noqa: E402
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile  # noqa: E402

BLOCK, HIDDEN, Q_RANK, KV_RANK, Q_DIM, ROPE_DIM = 128, 7168, 1536, 512, 128, 64
SENTINEL = -17.0


def _cache(blocks):
    block_stride = BLOCK * (KV_RANK + ROPE_DIM) + 32
    raw = torch.full((32 + blocks * block_stride,), SENTINEL, dtype=torch.bfloat16, device="npu")
    fused = raw.as_strided(
        (blocks, BLOCK, 1, KV_RANK + ROPE_DIM),
        (block_stride, KV_RANK + ROPE_DIM, KV_RANK + ROPE_DIM, 1),
        32,
    )
    return fused, raw


def _scatter(k, kr, cache, slots):
    DeviceOperator.reshape_and_cache(
        key=k.unsqueeze(1), value=kr.unsqueeze(1), key_cache=cache[0], value_cache=cache[1], slot_mapping=slots
    )


def _unfused(x, weights, gamma, eps, heads, *, native_norm):
    qkv = x @ weights[0]
    q, kv = qkv.split((Q_RANK, KV_RANK + ROPE_DIM), dim=-1)
    k, kr = kv.split((KV_RANK, ROPE_DIM), dim=-1)

    def norm(tensor, scale, epsilon):
        if native_norm:
            return torch_npu.npu_rms_norm(tensor.contiguous(), scale, epsilon)[0]
        tensor_fp32 = tensor.float()
        return (tensor_fp32 * torch.rsqrt(tensor_fp32.square().mean(-1, keepdim=True) + epsilon) * scale.float()).to(
            tensor.dtype
        )

    q = norm(q, gamma[0], eps[0]) @ weights[1]
    q, qr = q.view(-1, heads, Q_DIM + ROPE_DIM).split((Q_DIM, ROPE_DIM), dim=-1)
    q = torch.bmm(q.transpose(0, 1), weights[2]).transpose(0, 1)
    return q, qr, norm(k, gamma[1], eps[1]), kr


def _time_ms(operation):
    for _ in range(2):
        operation()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    begin.record()
    for _ in range(5):
        operation()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 5


@pytest.mark.parametrize("tokens", [1, 9, 129, 1024])
@pytest.mark.parametrize("heads", [8, 12])
@pytest.mark.parametrize("dcp_current", [False, True])
@torch.inference_mode()
def test_flashmla_mlapo_no_rope(tokens, heads, dcp_current, record_property):
    """Compare complete preprocessing and in-place history writes over multiple rounds."""
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    check_ascend_device_type()
    if not get_current_hardware_profile().supports(HardwareCapability.MLAPO_NATIVE_WEIGHTS):
        pytest.skip("requires A5 native BF16 MLAPO")
    torch.manual_seed(17)
    device = torch.device("npu")

    def weight(shape, fan_in):
        return (torch.randn(shape, dtype=torch.float32, device=device) / math.sqrt(fan_in)).bfloat16()

    fused_weight = weight((HIDDEN, Q_RANK + KV_RANK + ROPE_DIM), HIDDEN)
    q_up = weight((Q_RANK, heads * (Q_DIM + ROPE_DIM)), Q_RANK)
    uk = weight((heads, Q_DIM, KV_RANK), Q_DIM)
    gamma = (
        torch.ones(Q_RANK, dtype=torch.bfloat16, device=device),
        torch.ones(KV_RANK, dtype=torch.bfloat16, device=device),
    )
    eps = (1e-5, 1e-4)
    padded_heads = 1 << (heads - 1).bit_length()
    padded_q_up = torch.nn.functional.pad(
        q_up.view(Q_RANK, heads, Q_DIM + ROPE_DIM), (0, 0, 0, padded_heads - heads)
    ).reshape(Q_RANK, -1)
    padded_uk = torch.nn.functional.pad(uk, (0, 0, 0, 0, 0, padded_heads - heads))

    # Retain the production preprocess and scatter, replacing only model construction.
    impl = object.__new__(AscendMLAImpl)
    for name, value in {
        "num_heads": heads,
        "mlapo_num_heads": padded_heads,
        "num_kv_heads": 1,
        "kv_lora_rank": KV_RANK,
        "qk_rope_head_dim": ROPE_DIM,
        "use_mla_rope": False,
        "use_flashmla": True,
        "dcp_enabled": dcp_current,
        "support_fp8_attention": True,
        "mlapo_weight_quant_mode": 0,
        "fa_quant_layer": False,
        "q_a_layernorm": SimpleNamespace(weight=gamma[0], variance_epsilon=eps[0]),
        "kv_a_layernorm": SimpleNamespace(weight=gamma[1], variance_epsilon=eps[1]),
        "weight_dq": torch_npu.npu_format_cast(fused_weight[:, :Q_RANK].contiguous(), 29),
        "weight_dkv_kr": torch_npu.npu_format_cast(fused_weight[:, Q_RANK:].contiguous(), 29),
        "weight_uq_qr": torch_npu.npu_format_cast(padded_q_up.contiguous(), 29),
        "mlapo_W_UK_T": padded_uk.contiguous(),
    }.items():
        setattr(impl, name, value)
    scratch_shape = ((tokens + BLOCK - 1) // BLOCK, BLOCK, 1)
    buffers = (
        torch.empty((*scratch_shape, KV_RANK), dtype=torch.bfloat16, device=device),
        torch.empty((*scratch_shape, ROPE_DIM), dtype=torch.bfloat16, device=device),
        torch.arange(tokens, dtype=torch.int64, device=device),
    )
    impl._flashmla_writer_buffers = buffers if not dcp_current else None
    impl._dcp_current_kv_buffers = buffers if dcp_current else None
    impl._decode_requires_current_kv = MethodType(lambda self, metadata: dcp_current, impl)
    addresses = tuple(t.data_ptr() for t in buffers)
    fused, raw = _cache((tokens + BLOCK - 1) // BLOCK + 2)
    baseline, _ = _cache(fused.shape[0])
    cache = (fused[..., :KV_RANK], fused[..., KV_RANK:])
    baseline_cache = (baseline[..., :KV_RANK], baseline[..., KV_RANK:])
    expected_raw = raw.cpu().clone()
    expected = expected_raw.as_strided(fused.shape, fused.stride(), fused.storage_offset())
    times: tuple[list[float], list[float]] = [], []
    for iteration in range(3):
        x = torch.randn((tokens, HIDDEN), device=device, dtype=torch.bfloat16) * 0.03
        slots_cpu = torch.randperm(fused.shape[0] * BLOCK)[:tokens].long()
        if tokens > 1:
            slots_cpu[1::4] = -1
        slots = slots_cpu.to(device)
        metadata = SimpleNamespace(
            num_decode_tokens=tokens,
            slot_mapping=torch.full_like(slots, -1),
            external_flashmla=SimpleNamespace(slots=slots),
            decode=SimpleNamespace(cos=None, sin=None),
        )
        result = impl.mla_preprocess_only_decode(x, cache, metadata)[0]
        q, qr, k, kr = _unfused(x, (fused_weight, q_up, uk), gamma, eps, heads, native_norm=False)
        for actual, reference in ((result.ql_nope, q), (result.q_pe, qr)):
            torch.testing.assert_close(actual.float(), reference.float(), rtol=0.04, atol=0.015)
        assert torch.count_nonzero(result.q_pe).item() > 0, "no-RoPE must preserve the 64 projected dimensions"
        if dcp_current:
            for actual, reference in ((result.current_k_nope, k), (result.current_k_pe, kr)):
                torch.testing.assert_close(actual.squeeze(1).float(), reference.float(), rtol=0.04, atol=0.015)
        else:
            assert result.current_k_nope is None and result.current_k_pe is None
        assert result.k_nope is cache[0] and result.k_pe is cache[1]
        packed = torch.cat((k, kr), dim=-1).cpu()
        valid = slots_cpu >= 0
        expected[slots_cpu[valid] // BLOCK, slots_cpu[valid] % BLOCK, 0] = packed[valid]
        actual_raw = raw.cpu()
        torch.testing.assert_close(actual_raw.float(), expected_raw.float(), rtol=0.04, atol=0.015)
        untouched = expected_raw == SENTINEL
        assert torch.equal(actual_raw[untouched], expected_raw[untouched])
        assert tuple(t.data_ptr() for t in buffers) == addresses

        def baseline_preprocess(x=x, slots=slots):
            _, _, key, key_rope = _unfused(x, (fused_weight, q_up, uk), gamma, eps, heads, native_norm=True)
            _scatter(key, key_rope, baseline_cache, slots)

        times[0].append(_time_ms(lambda x=x, metadata=metadata: impl.mla_preprocess_only_decode(x, cache, metadata)))
        times[1].append(_time_ms(baseline_preprocess))
        record_property(f"mlapo_ms_round_{iteration}", times[0][-1])
        record_property(f"unfused_ms_round_{iteration}", times[1][-1])
    print(
        f"T={tokens} heads={heads} dcp_current={dcp_current}: "
        f"MLAPO+scatter={median(times[0]):.4f}ms, unfused+scatter={median(times[1]):.4f}ms"
    )


@torch.inference_mode()
def test_flashmla_mlapo_graph_replay_after_native_weight_reload(monkeypatch):
    """Replay the original graph after production postload refreshes padded NZ packs."""
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    check_ascend_device_type()
    if not get_current_hardware_profile().supports(HardwareCapability.MLAPO_NATIVE_WEIGHTS):
        pytest.skip("requires A5 native BF16 MLAPO")
    torch.manual_seed(23)
    tokens, heads, padded_heads = 9, 12, 16

    def linear(rows, columns):
        value = torch.randn(rows, columns, dtype=torch.bfloat16, device="npu") / math.sqrt(columns)
        return SimpleNamespace(
            weight=torch.nn.Parameter(value, requires_grad=False),
            quant_method=UnquantizedLinearMethod(),
            input_size=columns,
        )

    impl = object.__new__(AscendMLAImpl)
    for name, value in {
        "num_heads": heads,
        "num_heads_padded": padded_heads,
        "head_padding": padded_heads - heads,
        "num_kv_heads": 1,
        "q_lora_rank": Q_RANK,
        "kv_lora_rank": KV_RANK,
        "qk_nope_head_dim": Q_DIM,
        "qk_rope_head_dim": ROPE_DIM,
        "qk_head_dim": Q_DIM + ROPE_DIM,
        "v_head_dim": Q_DIM,
        "dtype": torch.bfloat16,
        "use_mla_rope": False,
        "use_flashmla": True,
        "enable_mlapo": True,
        "is_draft_model": False,
        "dcp_enabled": False,
        "pcp_enabled": False,
        "is_pcp_decode_sharded": False,
        "support_fp8_attention": True,
        "fa_quant_layer": False,
        "enable_kv_nz": False,
        "fused_qkv_a_proj": linear(Q_RANK + KV_RANK + ROPE_DIM, HIDDEN),
        "q_proj": linear(heads * (Q_DIM + ROPE_DIM), Q_RANK),
        "kv_b_proj": linear(heads * (Q_DIM + Q_DIM), KV_RANK),
        "q_a_layernorm": SimpleNamespace(
            weight=torch.ones(Q_RANK, dtype=torch.bfloat16, device="npu"), variance_epsilon=1e-5
        ),
        "kv_a_layernorm": SimpleNamespace(
            weight=torch.ones(KV_RANK, dtype=torch.bfloat16, device="npu"), variance_epsilon=1e-4
        ),
        "_flashmla_writer_buffers": (
            torch.empty(1, BLOCK, 1, KV_RANK, dtype=torch.bfloat16, device="npu"),
            torch.empty(1, BLOCK, 1, ROPE_DIM, dtype=torch.bfloat16, device="npu"),
            torch.arange(tokens, dtype=torch.int64, device="npu"),
        ),
    }.items():
        setattr(impl, name, value)
    # A5 retains prefill weights; postload does not read config fields here.
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.get_ascend_config", lambda: SimpleNamespace())
    impl.process_weights_after_loading(torch.bfloat16)
    pack_names = ("weight_dq", "weight_dkv_kr", "weight_uq_qr", "mlapo_W_UK_T", "W_UK_T")
    pointers = {name: getattr(impl, name).data_ptr() for name in pack_names}
    scratch_pointers = tuple(value.data_ptr() for value in impl._flashmla_writer_buffers)
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="npu") * 0.03
    slots = torch.arange(tokens, dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(
        num_decode_tokens=tokens, slot_mapping=slots, external_flashmla=SimpleNamespace(slots=slots)
    )
    fused, raw = _cache(1)
    cache = fused[..., :KV_RANK], fused[..., KV_RANK:]
    for _ in range(2):
        impl.mla_preprocess_only_decode(x, cache, metadata)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode="thread_local", auto_dispatch_capture=True):
        captured = impl.mla_preprocess_only_decode(x, cache, metadata)[0]
    graph.replay()
    old_q = captured.ql_nope.clone()
    for projection in (impl.fused_qkv_a_proj, impl.q_proj, impl.kv_b_proj):
        projection.weight.normal_(std=0.02)
    impl.process_weights_after_loading(torch.bfloat16)
    assert {name: getattr(impl, name).data_ptr() for name in pack_names} == pointers
    assert tuple(value.data_ptr() for value in impl._flashmla_writer_buffers) == scratch_pointers
    eager_fused, eager_raw = _cache(1)
    eager_cache = eager_fused[..., :KV_RANK], eager_fused[..., KV_RANK:]
    eager = impl.mla_preprocess_only_decode(x, eager_cache, metadata)[0]
    graph.replay()
    torch.npu.synchronize()
    for actual, reference in ((captured.ql_nope, eager.ql_nope), (captured.q_pe, eager.q_pe), (raw, eager_raw)):
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0.01, atol=0.003)
    assert not torch.equal(captured.ql_nope, old_q), "replay must observe the reloaded source weights"
