# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_v41_cp


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


@pytest.mark.parametrize("share_quant", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 5])
@pytest.mark.parametrize("cp", [False, True])
@torch.inference_mode()
def test_preprocess_equivalence_and_stream_dependencies(monkeypatch, share_quant, num_tokens, cp):
    """Check Q/qr/cache parity and the cross-stream producer/consumer ordering."""
    trace = []
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
    def switch(stream, *, enabled):
        nonlocal active
        previous = active
        active = stream.name
        try:
            yield
        finally:
            active = previous

    class Wrapper:
        _has_communication = False

        def __init__(self, name, linear, quant_method):
            self.name, self.linear = name, linear
            self._quant_method = quant_method

        def quantize(self, x):
            trace.append((active, self.name, "quantize"))
            return x, None

        def matmul(self, x, scale, bias=None):
            trace.append((active, self.name, "matmul"))
            assert bias is self.linear.bias
            return self.linear(x)

    def norm(name):
        def apply(x):
            trace.append((active, name, "norm"))
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)

        return apply

    def rope(x, cos, sin, **kwargs):
        trace.append((active, "rope", "apply"))
        # Exercise the in-place write on a view, including Q/KV partial slices.
        start, end = kwargs["partial_slice"]
        assert x.shape[0] == cos.shape[0] == sin.shape[0]
        x[..., start:end].mul_(cos).add_(sin)

    def scatter(cache, slots, values):
        trace.append((active, "cache", "scatter"))
        for slot, value in zip(slots, values):
            if slot[0] >= 0:
                cache[slot[0], slot[1]].copy_(value)

    monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
    monkeypatch.setattr(dsa_v41, "dsv4_dsa_overlap_stream", lambda: aux)
    monkeypatch.setattr(dsa_v41, "npu_stream_switch", switch)
    monkeypatch.setattr(dsa_v41, "scatter_cache_sk", scatter)
    monkeypatch.setattr(dsa_v41_cp, "dsv4_dsa_overlap_stream", lambda: aux)
    monkeypatch.setattr(dsa_v41_cp, "npu_stream_switch", switch)
    monkeypatch.setattr(dsa_v41_cp, "scatter_cache_sk", scatter)
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rope, raising=False)
    torch.manual_seed(7)
    cache = torch.zeros(2, num_tokens, 4)
    q_a, q_b, kv = torch.nn.Linear(8, 6), torch.nn.Linear(6, 8), torch.nn.Linear(8, 4)
    wrappers = SimpleNamespace(
        cv_wq_a=Wrapper("qa", q_a, object()),
        cv_wq_b=Wrapper("qb", q_b, object()),
        cv_wkv=Wrapper("kv", kv, object() if share_quant else SimpleNamespace()),
    )
    attn = SimpleNamespace(
        wq_a=q_a,
        wq_b=q_b,
        wkv=kv,
        q_norm=norm("q"),
        kv_norm=norm("kv"),
        n_local_heads=1 if cp else 2,
        n_heads=2,
        head_dim=4,
        nope_head_dim=2,
        dsa_attn=SimpleNamespace(
            swa_cache_layer=SimpleNamespace(kv_cache=[cache]),
            dsa_attn=SimpleNamespace(impl=wrappers),
        ),
    )
    slots = torch.tensor([[1, i] for i in range(num_tokens)])
    if num_tokens > 1:
        slots[-1] = -1
    metadata = SimpleNamespace(slot_mapping=slots)
    hidden = torch.randn(num_tokens, 8)
    cos, sin = torch.randn(num_tokens, 1, 1, 2), torch.randn(num_tokens, 1, 1, 2)
    start = num_tokens // 2 if cp else 0
    local_hidden, local_cos, local_sin = hidden[start:], cos[start:], sin[start:]
    impl = object.__new__(dsa_v41_cp.DeepseekV41CPImpl if cp else dsa_v41.DeepseekV41EagerAttentionImpl)
    if cp:
        expected_q, expected_qr = impl._project_q(attn, local_hidden, local_cos, local_sin)
        scatter(cache, slots, impl._project_kv(attn, hidden, cos, sin))
    else:
        expected_q, expected_qr = impl.preprocess(attn, hidden, cos, sin, metadata)
    expected_cache = cache.clone()
    cache.zero_()
    trace.clear()
    kwargs = {}
    if cp:
        impl.role = SimpleNamespace(is_kv_source=False)
        attn.rotary_emb = SimpleNamespace(layername="layer")
        metadata.num_actual_tokens = num_tokens
        global_metadata = SimpleNamespace(swa=metadata, rope=lambda *args: (cos, sin))
        metadata = SimpleNamespace(swa=SimpleNamespace(
            num_actual_tokens=num_tokens - start, cp_token_range=(start, num_tokens, num_tokens - start, num_tokens)
        ))
        impl._global_layer_metadata = Mock(return_value=global_metadata)
        monkeypatch.setattr(dsa_v41_cp, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
        metadata = metadata.swa
    q, qr = impl.multistream_preprocess(attn, hidden, local_cos, local_sin, metadata, **kwargs)
    torch.testing.assert_close(q, expected_q)
    torch.testing.assert_close(qr, expected_qr)
    torch.testing.assert_close(cache, expected_cache)
    assert qr.is_floating_point()
    assert (("aux", "kv", "quantize") in trace) == (cp or not share_quant)
    assert trace.count(("aux", "cache", "scatter")) == 1
    kv_mm = trace.index(("aux", "kv", "matmul"))
    kv_done = trace[kv_mm + 1]
    assert kv_done[:2] == ("aux", "record")
    assert trace.index(("main", "wait", kv_done[2])) < trace.index(("main", "qb", "matmul"))
    part3 = trace[trace.index(("main", "wait", kv_done[2])) - 1]
    assert part3[:2] == ("main", "record")
    assert trace.index(("aux", "wait", part3[2])) < trace.index(("aux", "kv", "norm"))
    assert trace.index(("aux", "cache", "scatter")) < trace.index(("main", "join", "aux"))
    assert trace.index(("main", "join", "aux")) < trace.index(("main", "rope", "apply"))


@pytest.mark.parametrize("enabled", [False, True])
def test_forward_selects_preprocess_from_v1_switch(monkeypatch, enabled):
    impl = object.__new__(dsa_v41.DeepseekV41EagerAttentionImpl)
    impl.role = SimpleNamespace(is_kv_source=False)
    hidden = torch.zeros(1, 8)
    q, qr = torch.zeros(1, 2, 4), torch.zeros(1, 6)
    metadata = SimpleNamespace(
        positions=torch.zeros(1), swa=SimpleNamespace(num_actual_tokens=1), rope=lambda *args: (None, None)
    )
    impl._get_layer_metadata = Mock(return_value=metadata)
    impl.preprocess = Mock(return_value=(q, qr))
    impl.multistream_preprocess = Mock(return_value=(q, qr))
    impl._select_sparse_indices = Mock(return_value=None)
    impl._attention = Mock(return_value=q)
    v1_impl = SimpleNamespace(
        multistream_dsv4_dsa_overlap=enabled,
        _forward_o_proj=lambda q, output: output.zero_(),
    )
    attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(layername="layer"),
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=v1_impl)),
        nope_head_dim=2,
        head_dim=4,
    )
    metadata.rope = lambda *args: (torch.zeros(1), torch.zeros(1))
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    output = torch.full_like(hidden, 1)
    result = impl.forward(attn, None, hidden, output)
    selected = impl.multistream_preprocess if enabled else impl.preprocess
    unused = impl.preprocess if enabled else impl.multistream_preprocess
    selected.assert_called_once()
    unused.assert_not_called()
    assert result is output
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("local_tokens", [0, 2])
@pytest.mark.parametrize("is_source", [False, True])
def test_cp_multistream_forward_preserves_full_cache_updates(monkeypatch, local_tokens, is_source):
    impl = object.__new__(dsa_v41_cp.DeepseekV41CPImpl)
    impl.role = SimpleNamespace(is_kv_source=is_source)
    hidden = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    global_cos, global_sin = torch.ones(4), torch.ones(4)
    local_cos, local_sin = global_cos[2 : 2 + local_tokens], global_sin[2 : 2 + local_tokens]
    metadata = SimpleNamespace(
        swa=SimpleNamespace(num_actual_tokens=local_tokens, cp_token_range=(2, 4, 2, 4)),
        positions=torch.arange(2, 2 + local_tokens),
        rope=lambda *args: (local_cos, local_sin),
    )
    global_metadata = SimpleNamespace(
        swa=SimpleNamespace(num_actual_tokens=4),
        positions=torch.arange(4),
        rope=lambda *args: (global_cos, global_sin),
    )
    impl._get_layer_metadata = Mock(return_value=metadata)
    impl._global_layer_metadata = Mock(return_value=global_metadata)
    q, qr = torch.zeros(local_tokens, 2, 4), torch.zeros(local_tokens, 6)
    impl.multistream_preprocess = Mock(return_value=(q, qr))
    impl._update_caches = Mock()
    impl._write_compressed_source = Mock()
    impl._select_sparse_indices = Mock(return_value=None)
    impl._attention = Mock(return_value=q)
    impl._project_output = Mock(side_effect=lambda *args, projected: projected.zero_())
    attn = SimpleNamespace(
        rotary_emb=SimpleNamespace(layername="layer"),
        n_heads=2,
        enable_dsa_cp=True,
        head_dim=4,
        nope_head_dim=2,
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=True))),
    )
    monkeypatch.setattr(dsa_v41, "get_forward_context", lambda: SimpleNamespace(attn_metadata={}))
    monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False)
    output = torch.ones_like(hidden)
    assert impl.forward(attn, None, hidden, output) is output
    impl._project_output.assert_called_once()
    assert torch.count_nonzero(output) == 0
    if local_tokens:
        impl._update_caches.assert_not_called()
        impl.multistream_preprocess.assert_called_once()
        args, kwargs = impl.multistream_preprocess.call_args
        torch.testing.assert_close(args[1], hidden)
        assert args[2] is local_cos and args[3] is local_sin
        assert args[4] is metadata.swa
        assert not kwargs
        # The mocked preprocessor owns compressed-cache writes.
        impl._write_compressed_source.assert_not_called()
        assert impl._select_sparse_indices.call_args.args[-1] is metadata
    else:
        impl.multistream_preprocess.assert_not_called()
        impl._update_caches.assert_called_once()
        args = impl._update_caches.call_args.args
        torch.testing.assert_close(args[1], hidden[:4])
        assert args[2] is global_metadata
        impl._write_compressed_source.assert_not_called()
        impl._attention.assert_not_called()


def test_cp_disabled_overlap_delegates_to_serial_forward(monkeypatch):
    impl = object.__new__(dsa_v41_cp.DeepseekV41CPImpl)
    serial = Mock(return_value=object())
    monkeypatch.setattr(dsa_v41.DeepseekV41EagerAttentionImpl, "forward", serial)
    attn = SimpleNamespace(
        dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(multistream_dsv4_dsa_overlap=False)))
    )
    hidden, output = torch.zeros(2, 8), torch.empty(2, 8)
    assert impl.forward(attn, None, hidden, output) is serial.return_value
    serial.assert_called_once_with(attn, None, hidden, output)
