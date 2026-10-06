# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_v41_cp
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod


def test_dsa_v41_custom_op_forwards_its_output_buffer(monkeypatch):
    hidden = torch.zeros(1, 8)
    output = torch.empty_like(hidden)
    impl = Mock()
    attn = SimpleNamespace(v41_impl=impl)
    monkeypatch.setattr(
        dsa_v41,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"layer": attn}),
    )
    dsa_v41.dsa_v41_forward(hidden, output, "layer")
    impl.forward.assert_called_once_with(attn, None, hidden, output)


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize(
    "layout,num_tokens,quantized_kv,a5_batch,communication",
    [
        ("replicated", 5, True, None, False),
        ("replicated", 1, True, None, False),
        ("replicated", 5, False, None, False),
        ("replicated", 5, True, None, True),
        ("replicated", 5, True, "prefill", False),
        ("replicated", 1, True, "decode", False),
        ("replicated", 5, True, "mixed", False),
        ("replicated", 1, True, "dummy", False),
        ("legacy", 5, True, None, False),
        ("legacy", 5, True, "prefill", False),
        ("pcp", 5, True, None, False),
        ("pcp", 5, True, "prefill", False),
        ("empty", 5, True, None, False),
        ("empty", 5, True, "prefill", False),
    ],
)
@torch.inference_mode()
def test_preprocess_equivalence_and_stream_dependencies(
    monkeypatch, layout, num_tokens, quantized_kv, a5_batch, communication, overlap
):
    """Exercise real projections with shared, sliced and PCP-local Query inputs."""
    trace: list[tuple[str, str, str]] = []
    quant_inputs = []
    active = "main"

    class Stream:
        def __init__(self, name):
            self.name = name

        def record_event(self):
            event = f"event{len(trace)}"
            trace.append((self.name, "record", event))
            return event

        def wait_event(self, event):
            trace.append((self.name, "wait", event))

        def wait_stream(self, stream):
            trace.append((self.name, "join", stream.name))

    main, aux = Stream("main"), Stream("aux")

    @contextmanager
    def switch(stream):
        nonlocal active
        previous, active = active, stream.name
        try:
            yield
        finally:
            active = previous

    class Linear(torch.nn.Module):
        def __init__(self, inputs, outputs, *, quantized=True, communicates=False):
            super().__init__()
            shape = (inputs, outputs) if quantized else (outputs, inputs)
            self.weight = torch.nn.Parameter(torch.randn(shape))
            self.bias = torch.nn.Parameter(torch.randn(outputs))
            self.weight_scale = torch.ones(outputs)
            self.quant_method = AscendW8A8DynamicLinearMethod() if quantized else AscendUnquantizedLinearMethod()
            self.gather_output = communicates

        def forward(self, value):
            return self.quant_method.apply(self, value, self.bias)

    def quantize(value, **kwargs):
        quant_inputs.append((active, value.clone()))
        return value, torch.ones(value.shape[0])

    def matmul(value, weight, scale=None, *, bias=None, **kwargs):
        name, quantized = weights[weight.data_ptr()]
        trace.append((active, name, "matmul"))
        return value @ (weight if quantized else weight.T) + bias

    def norm(name):
        def apply(value):
            trace.append((active, name, "norm"))
            return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)

        return apply

    def rope(value, cos, sin, **kwargs):
        trace.append((active, "rope", "apply"))
        start, end = kwargs["partial_slice"]
        assert value.shape[0] == cos.shape[0] == sin.shape[0]
        value[..., start:end].mul_(cos).add_(sin)

    def scatter(cache, slots, values):
        trace.append((active, "cache", "scatter"))
        for slot, value in zip(slots, values):
            if slot[0] >= 0:
                cache[slot[0], slot[1]].copy_(value)

    class A5Backend:
        @staticmethod
        def write_attention_cache(cache, slots, values, *, kind):
            assert kind == "win"
            scatter(cache, slots, values)

    monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(torch.npu, "stream", switch)
    monkeypatch.setattr(dsa_v41, "dsv4_dsa_overlap_stream", lambda: aux)
    monkeypatch.setattr(dsa_v41, "scatter_cache_sk", scatter)
    monkeypatch.setattr(torch_npu, "npu_dynamic_quant", quantize, raising=False)
    monkeypatch.setattr(torch_npu, "npu_quant_matmul", matmul, raising=False)
    monkeypatch.setattr(
        torch.ops.vllm,
        "unquantized_gemm",
        lambda value, weight, bias=None: matmul(value, weight, bias=bias),
        raising=False,
    )
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rope, raising=False)
    torch.manual_seed(7)
    qa, qb = Linear(8, 6), Linear(6, 8)
    kv = Linear(8, 4, quantized=quantized_kv, communicates=communication)
    weights = {
        layer.weight.data_ptr(): (name, isinstance(layer.quant_method, AscendW8A8DynamicLinearMethod))
        for name, layer in (("qa", qa), ("qb", qb), ("kv", kv))
    }
    cache = torch.zeros(2, num_tokens, 4)
    attn = SimpleNamespace(
        wq_a=qa,
        wq_b=qb,
        wkv=kv,
        q_norm=norm("q"),
        kv_norm=norm("kv"),
        n_heads=2,
        head_dim=4,
        nope_head_dim=2,
        packed_cache_ops=A5Backend() if a5_batch else None,
        rotary_emb=SimpleNamespace(layername="layer"),
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=[cache]),
            dsa_attn=SimpleNamespace(
                impl=SimpleNamespace(
                    cv_wq_a=CVLinearWrapper(qa), cv_wq_b=CVLinearWrapper(qb), cv_wkv=CVLinearWrapper(kv)
                )
            ),
        ),
    )
    hidden = torch.randn(num_tokens, 8)
    padded_hidden = torch.cat((hidden, torch.full((3, 8), 9999.0)))
    cos, sin = torch.randn(num_tokens, 1, 1, 2), torch.randn(num_tokens, 1, 1, 2)
    slots = torch.tensor([[1, i] for i in range(num_tokens)])
    slots[-1] = -1
    if a5_batch == "dummy":
        slots.fill_(-1)
    cache_metadata = SimpleNamespace(
        positions=torch.arange(num_tokens),
        swa=SimpleNamespace(
            num_actual_tokens=num_tokens,
            slot_mapping=slots,
            flat_slot_mapping=slots,
            num_prefills=int(a5_batch == "prefill"),
            num_decodes=int(a5_batch == "decode"),
            num_reqs=1,
        ),
        rope=lambda *args: (cos, sin),
    )
    ids = list(range(num_tokens)) if layout == "replicated" else [2, 3, 4] if layout == "legacy" else [0, 4]
    if layout == "empty":
        ids = []
    local_hidden, local_cos, local_sin = hidden[ids], cos[ids], sin[ids]
    query_metadata = (
        cache_metadata
        if layout == "replicated"
        else SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=len(ids), cp_token_range=(2, 5, 3, 5)),
            rope=lambda *args: (local_cos, local_sin),
        )
    )
    impl_class = dsa_v41_cp.AscendDSAV41CPImpl if layout == "legacy" else dsa_v41.AscendDSAV41Impl
    impl = impl_class.__new__(impl_class)
    impl.multistream_dsv4_dsa_overlap = overlap
    impl.role = SimpleNamespace(is_kv_source=False, has_long_context=False)

    expected_qr = attn.q_norm(qa(local_hidden))
    expected_q = qb(expected_qr).unflatten(-1, (2, 4))
    rope(expected_q.unsqueeze(1), local_cos, local_sin, partial_slice=[2, 4])
    expected_kv = attn.kv_norm(kv(hidden)).view(-1, 1, 4)
    rope(expected_kv.unsqueeze(1), cos, sin, partial_slice=[2, 4])
    scatter(cache, slots, expected_kv.squeeze(1))
    expected_cache = cache.clone()
    cache.zero_()
    trace.clear()
    quant_inputs.clear()

    if layout == "legacy":
        impl._global_layer_metadata = lambda by_prefix: cache_metadata
        q, qr, prepared = impl._prepare_inputs_and_caches(attn, padded_hidden, query_metadata, {})
    elif layout == "replicated":
        q, qr, prepared = impl._prepare_inputs_and_caches(attn, padded_hidden, query_metadata, {})
    else:
        local_padded = torch.cat((local_hidden, torch.full((3, 8), 9999.0)))
        q, qr, prepared = impl._preprocess(attn, local_padded, padded_hidden, query_metadata, cache_metadata)
    assert prepared is None
    torch.testing.assert_close(cache, expected_cache)
    assert sum(item[1:] == ("cache", "scatter") for item in trace) == 1
    if not ids:
        assert q is qr is None
        assert not any(name in {"qa", "qb"} for _, name, _ in trace)
        return
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(qr, expected_qr)
    assert qr.is_floating_point()
    share_quant = overlap and layout == "replicated" and quantized_kv and not communication
    assert len(quant_inputs) == 2 + int(quantized_kv and not share_quant)
    torch.testing.assert_close(quant_inputs[0][1], local_hidden)
    if quantized_kv and not share_quant:
        cache_quant = next(value for _, value in quant_inputs[1:] if value.shape[-1] == 8)
        torch.testing.assert_close(cache_quant, hidden)
    if not overlap:
        assert all(stream == "main" for stream, _, _ in trace)
        return
    kv_done = trace[trace.index(("aux", "kv", "matmul")) + 1]
    assert kv_done[:2] == ("aux", "record")
    handoff = trace.index(("main", "wait", kv_done[2]))
    assert handoff < trace.index(("main", "qb", "matmul"))
    query_done = trace[handoff - 1]
    assert query_done[:2] == ("main", "record")
    assert trace.index(("aux", "wait", query_done[2])) < trace.index(("aux", "kv", "norm"))
    join = trace.index(("main", "join", "aux"))
    cache_on_main = a5_batch in {"prefill", "mixed", "dummy"}
    store = trace.index(("main" if cache_on_main else "aux", "cache", "scatter"))
    assert (join < store) == cache_on_main
    assert join < trace.index(("main", "rope", "apply"))
