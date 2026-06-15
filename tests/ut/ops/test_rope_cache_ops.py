from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops import rope_cache_ops


class DummyRotaryEmbedding:
    def __init__(self):
        self.rotary_dim = 4
        self.is_neox_style = True
        self.cos_sin_cache = torch.randn(16, 4)


class DummyDSARopeCache:
    def __init__(self, rotary_dim: int = 4):
        self.positions = torch.tensor([1, 3])
        self.rotary_dim = rotary_dim
        self.cos_cache = torch.randn(16, 1, 1, rotary_dim)
        self.sin_cache = torch.randn(16, 1, 1, rotary_dim)
        self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)
        self.cos = torch.randn(2, 1, 1, rotary_dim)
        self.sin = torch.randn(2, 1, 1, rotary_dim)
        self.materialize = Mock(return_value=(self.cos, self.sin))

    def backend_cos_sin_cache(self):
        half_dim = self.cos_sin_cache.shape[-1] // 2
        return self.cos_sin_cache.narrow(-1, 0, half_dim), self.cos_sin_cache.narrow(-1, half_dim, half_dim)


class FakeOverload:
    def __init__(self, dispatch_keys):
        self._dispatch_keys = set(dispatch_keys)

    def has_kernel_for_dispatch_key(self, dispatch_key):
        return dispatch_key in self._dispatch_keys


class FakeOp:
    def __init__(self, dispatch_keys):
        self.default = FakeOverload(dispatch_keys)

    def __call__(self, *args, **kwargs):
        return None


def teardown_function():
    rope_cache_ops.clear_rope_cache_op_capability_cache()
    rope_cache_ops._UNSUPPORTED_C_ASCEND_OPS.clear()


def test_dispatch_kernel_probe_requires_privateuse1_kernel():
    assert rope_cache_ops._has_dispatch_kernel(FakeOp({"PrivateUse1"}), "PrivateUse1")
    assert not rope_cache_ops._has_dispatch_kernel(FakeOp({"Meta"}), "PrivateUse1")


def test_c_ascend_probe_ignores_schema_without_privateuse1_kernel():
    c_ascend = SimpleNamespace(mla_preprocess_by_cache=FakeOp({"Meta"}))

    with patch.object(rope_cache_ops.torch.ops, "_C_ascend", c_ascend, create=True):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        assert rope_cache_ops._get_c_ascend_op("mla_preprocess_by_cache") is None


def test_vllm_op_probe_skips_dispatch_introspection_during_compile():
    native_op = Mock(return_value=None)
    native_op.default.has_kernel_for_dispatch_key.side_effect = AssertionError("should not probe during compile")
    vllm_namespace = SimpleNamespace(fused_by_cache=native_op)

    with (
        patch.object(rope_cache_ops.torch.ops, "vllm", vllm_namespace, create=True),
        patch.object(rope_cache_ops.torch.compiler, "is_compiling", return_value=True),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        assert rope_cache_ops._get_vllm_op("fused_by_cache") is native_op

    native_op.default.has_kernel_for_dispatch_key.assert_not_called()


def test_c_ascend_mla_preprocess_by_cache_privateuse1_is_true_kernel_by_cache():
    c_ascend = SimpleNamespace(mla_preprocess_by_cache=FakeOp({"PrivateUse1"}))

    with patch.object(rope_cache_ops.torch.ops, "_C_ascend", c_ascend, create=True):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        assert rope_cache_ops._get_c_ascend_op("mla_preprocess_by_cache") is not None
        assert rope_cache_ops.has_mla_preprocess_by_cache_kernel()
        assert rope_cache_ops.has_mla_preprocess_by_cache_backend()


def test_mla_preprocess_by_cache_backend_rejects_legacy_mla_preprocess_fallback():
    c_ascend = SimpleNamespace(mla_preprocess=FakeOp({"PrivateUse1"}))

    with patch.object(rope_cache_ops.torch.ops, "_C_ascend", c_ascend, create=True):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        assert not rope_cache_ops.has_mla_preprocess_by_cache_backend()


def test_c_ascend_probe_lazy_loads_custom_ops_when_namespace_is_missing():
    native_op = FakeOp({"PrivateUse1"})
    loaded_namespace = SimpleNamespace(inplace_partial_rotary_mul_dsa_by_cache=native_op)

    def fake_load_custom_ops():
        rope_cache_ops.torch.ops._C_ascend = loaded_namespace
        return True

    rope_cache_ops.clear_rope_cache_op_capability_cache()
    with (
        patch.object(rope_cache_ops.torch.ops, "_C_ascend", None, create=True),
        patch(
            "vllm_ascend.ops.rope_cache_ops._ensure_c_ascend_custom_ops_loaded",
            side_effect=fake_load_custom_ops,
        ) as lazy_loader,
    ):
        assert rope_cache_ops._get_c_ascend_op("inplace_partial_rotary_mul_dsa_by_cache") is native_op

    lazy_loader.assert_called_once()


def test_aclnn_by_cache_capability_uses_composite_probe():
    capability_op = Mock(side_effect=lambda name: name == "aclnnCompressorByCache")
    c_ascend = SimpleNamespace(aclnn_api_available=capability_op)

    with patch.object(rope_cache_ops.torch.ops, "_C_ascend", c_ascend, create=True):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        assert rope_cache_ops.has_compressor_by_cache_kernel()
        assert not rope_cache_ops.has_compressor_dsa_by_cache_kernel()

    assert [call.args[0] for call in capability_op.call_args_list] == [
        "aclnnCompressorByCache",
        "aclnnCompressorDsaByCache",
    ]


def test_split_qkv_rmsnorm_mrope_by_cache_passes_cache_and_positions_to_fused_op():
    rotary_emb = DummyRotaryEmbedding()
    qkv = torch.randn(2, 16)
    positions = torch.tensor([1, 3])
    expected = (
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.randn(2, 4),
        torch.empty(2, 0),
    )
    fused_op = Mock(return_value=expected)

    with patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=fused_op):
        output = rope_cache_ops.split_qkv_rmsnorm_mrope_by_cache(
            qkv=qkv,
            q_weight=torch.randn(4),
            k_weight=torch.randn(4),
            positions=positions,
            rotary_emb=rotary_emb,
            num_q_heads=1,
            num_kv_heads=1,
            head_size=4,
            eps=1e-6,
            mrope_section=[1, 1, 0],
            is_interleaved=True,
            rope_dim=4,
        )

    assert output is expected
    fused_op.assert_called_once()
    kwargs = fused_op.call_args.kwargs
    assert kwargs["positions"] is positions
    assert kwargs["cos_sin_cache"] is rotary_emb.cos_sin_cache
    assert kwargs["qkv"] is qkv
    assert kwargs["num_q_heads"] == 1
    assert kwargs["num_kv_heads"] == 1
    assert kwargs["rope_dim"] == 4
    assert kwargs["has_gate"] is False


def test_split_qkv_rmsnorm_mrope_by_cache_output_depends_on_layer_owned_cache_rows():
    qkv = torch.zeros(2, 16)
    positions = torch.tensor([1, 3])
    rotary_a = DummyRotaryEmbedding()
    rotary_b = DummyRotaryEmbedding()
    rotary_a.cos_sin_cache = torch.arange(16 * 4, dtype=qkv.dtype).view(16, 4)
    rotary_b.cos_sin_cache = rotary_a.cos_sin_cache + 100

    def fake_fused_op(**kwargs):
        cos, sin = kwargs["cos_sin_cache"].index_select(0, kwargs["positions"]).chunk(2, dim=-1)
        selected = torch.cat((cos, sin), dim=-1)
        return selected, selected + 1, selected + 2, torch.empty(selected.shape[0], 0)

    with patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=Mock(side_effect=fake_fused_op)):
        out_a = rope_cache_ops.split_qkv_rmsnorm_mrope_by_cache(
            qkv=qkv,
            q_weight=torch.ones(4),
            k_weight=torch.ones(4),
            positions=positions,
            rotary_emb=rotary_a,
            num_q_heads=1,
            num_kv_heads=1,
            head_size=4,
            eps=1e-6,
            mrope_section=[1, 1, 0],
            is_interleaved=True,
            rope_dim=4,
        )
        out_b = rope_cache_ops.split_qkv_rmsnorm_mrope_by_cache(
            qkv=qkv,
            q_weight=torch.ones(4),
            k_weight=torch.ones(4),
            positions=positions,
            rotary_emb=rotary_b,
            num_q_heads=1,
            num_kv_heads=1,
            head_size=4,
            eps=1e-6,
            mrope_section=[1, 1, 0],
            is_interleaved=True,
            rope_dim=4,
        )

    expected_a = rotary_a.cos_sin_cache.index_select(0, positions)
    expected_b = rotary_b.cos_sin_cache.index_select(0, positions)
    torch.testing.assert_close(out_a[0], expected_a)
    torch.testing.assert_close(out_b[0], expected_b)
    assert not torch.equal(out_a[0], out_b[0])


def test_split_qkv_rmsnorm_mrope_by_cache_requires_registered_fused_op():
    rotary_emb = DummyRotaryEmbedding()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=None),
        pytest.raises(RuntimeError, match="triton_split_qkv_rmsnorm_mrope_by_cache"),
    ):
        rope_cache_ops.split_qkv_rmsnorm_mrope_by_cache(
            qkv=torch.randn(2, 16),
            q_weight=torch.randn(4),
            k_weight=torch.randn(4),
            positions=torch.tensor([1, 3]),
            rotary_emb=rotary_emb,
            num_q_heads=1,
            num_kv_heads=1,
            head_size=4,
            eps=1e-6,
            mrope_section=[1, 1, 0],
            is_interleaved=True,
        )


def test_mla_preprocess_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    hidden_states = torch.randn(2, 8)
    positions = torch.tensor([1, 3])
    native_op = Mock(return_value=None)
    native_op.default = FakeOverload({"PrivateUse1"})
    c_ascend = SimpleNamespace(mla_preprocess_by_cache=native_op)

    wd_qkv = torch.randn(8, 8)
    gamma1 = torch.randn(8)
    wu_q = torch.randn(8, 8)
    gamma2 = torch.randn(8)
    w_uk_t = torch.randn(8, 8)
    k_nope = torch.randn(2, 8)
    k_pe = torch.randn(2, 8)
    slot_mapping = torch.tensor([0, 1])
    q_out0 = torch.empty(2, 8)
    kv_cache_out0 = torch.empty(2, 8)
    q_out1 = torch.empty(2, 8)
    kv_cache_out1 = torch.empty(2, 8)
    inner_out = torch.empty(2, 8)

    with (
        patch.object(rope_cache_ops.torch.ops, "_C_ascend", c_ascend, create=True),
        patch("vllm_ascend.ops.rope_cache_ops.has_mla_preprocess_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        rope_cache_ops.mla_preprocess_by_cache(
            hidden_states,
            wd_qkv,
            None,
            gamma1,
            None,
            wu_q,
            None,
            gamma2,
            positions,
            rotary_emb,
            w_uk_t,
            k_nope,
            k_pe,
            slot_mapping,
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=True,
            q_out0=q_out0,
            kv_cache_out0=kv_cache_out0,
            q_out1=q_out1,
            kv_cache_out1=kv_cache_out1,
            inner_out=inner_out,
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is hidden_states
    assert native_args[8] is positions
    assert native_args[9] is rotary_emb.cos_sin_cache
    assert native_args[10] is w_uk_t
    assert native_args[11] is k_nope
    assert native_args[12] is k_pe
    assert native_args[13] is slot_mapping
    assert native_op.call_args.kwargs["cache_mode"] == "krope_ctkv"
    assert native_op.call_args.kwargs["enable_inner_out"] is True
    assert native_op.call_args.kwargs["enable_raw_q_out"] is False
    assert native_op.call_args.kwargs["raw_q_out"].numel() == 0


def test_mla_preprocess_by_cache_output_depends_on_layer_owned_cache_rows():
    positions = torch.tensor([1, 3])
    hidden_states = torch.zeros(2, 4)
    rotary_a = DummyRotaryEmbedding()
    rotary_b = DummyRotaryEmbedding()
    rotary_a.cos_sin_cache = torch.arange(16 * 4, dtype=hidden_states.dtype).view(16, 4)
    rotary_b.cos_sin_cache = rotary_a.cos_sin_cache + 100

    def fake_native(*args, **kwargs):
        cos, sin = args[9].index_select(0, args[8]).chunk(2, dim=-1)
        kwargs["q_out1"].copy_(torch.cat((cos, sin), dim=-1))

    native_op = Mock(side_effect=fake_native)

    def run(rotary_emb):
        q_out1 = torch.empty(2, 4)
        with (
            patch("vllm_ascend.ops.rope_cache_ops.has_mla_preprocess_by_cache_kernel", return_value=True),
            patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        ):
            rope_cache_ops.mla_preprocess_by_cache(
                hidden_states,
                torch.empty(4, 4),
                None,
                torch.empty(4),
                None,
                torch.empty(4, 4),
                None,
                torch.empty(4),
                positions,
                rotary_emb,
                torch.empty(4, 4),
                torch.empty(2, 4),
                torch.empty(2, 4),
                torch.tensor([0, 1]),
                q_out0=torch.empty(2, 4),
                kv_cache_out0=torch.empty(2, 4),
                q_out1=q_out1,
                kv_cache_out1=torch.empty(2, 4),
                inner_out=torch.empty(2, 4),
            )
        return q_out1

    out_a = run(rotary_a)
    out_b = run(rotary_b)

    expected_a = rotary_a.cos_sin_cache.index_select(0, positions)
    expected_b = rotary_b.cos_sin_cache.index_select(0, positions)
    torch.testing.assert_close(out_a, expected_a)
    torch.testing.assert_close(out_b, expected_b)
    assert not torch.equal(out_a, out_b)


def test_mla_preprocess_by_cache_fails_closed_without_native_backend():
    rotary_emb = DummyRotaryEmbedding()
    hidden_states = torch.randn(2, 8)
    positions = torch.tensor([1, 3])
    legacy_op = Mock(return_value=None)

    wd_qkv = torch.randn(8, 8)
    gamma1 = torch.randn(8)
    wu_q = torch.randn(8, 8)
    gamma2 = torch.randn(8)
    w_uk_t = torch.randn(8, 8)
    k_nope = torch.randn(2, 8)
    k_pe = torch.randn(2, 8)
    slot_mapping = torch.tensor([0, 1])
    q_out0 = torch.empty(2, 8)
    kv_cache_out0 = torch.empty(2, 8)
    q_out1 = torch.empty(2, 8)
    kv_cache_out1 = torch.empty(2, 8)
    inner_out = torch.empty(2, 8)

    with (
        patch("vllm_ascend.ops.rope_cache_ops.has_mla_preprocess_by_cache_kernel", return_value=False),
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=legacy_op),
        pytest.raises(RuntimeError, match="requires a true by-cache backend"),
    ):
        rope_cache_ops.mla_preprocess_by_cache(
            hidden_states,
            wd_qkv,
            None,
            gamma1,
            None,
            wu_q,
            None,
            gamma2,
            positions,
            rotary_emb,
            w_uk_t,
            k_nope,
            k_pe,
            slot_mapping,
            layout="TD",
            cache_mode="krope_ctkv",
            quant_mode="no_quant",
            enable_inner_out=True,
            q_out0=q_out0,
            kv_cache_out0=kv_cache_out0,
            q_out1=q_out1,
            kv_cache_out1=kv_cache_out1,
            inner_out=inner_out,
        )

    legacy_op.assert_not_called()


def test_mla_preprocess_by_cache_raw_q_requires_native_backend():
    rotary_emb = DummyRotaryEmbedding()

    with (
        patch("vllm_ascend.ops.rope_cache_ops.has_mla_preprocess_by_cache_kernel", return_value=False),
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=Mock()),
        pytest.raises(RuntimeError, match="requires a true by-cache backend"),
    ):
        rope_cache_ops.mla_preprocess_by_cache(
            torch.randn(2, 8),
            torch.randn(8, 8),
            None,
            torch.randn(8),
            None,
            torch.randn(8, 8),
            None,
            torch.randn(8),
            torch.tensor([1, 3]),
            rotary_emb,
            torch.randn(8, 8),
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            q_out0=torch.empty(2, 8),
            kv_cache_out0=torch.empty(2, 8),
            q_out1=torch.empty(2, 8),
            kv_cache_out1=torch.empty(2, 8),
            inner_out=torch.empty(2, 8),
            enable_raw_q_out=True,
            raw_q_out=torch.empty(2, 8),
        )


def test_mla_preprocess_by_cache_requires_native_backend():
    rotary_emb = DummyRotaryEmbedding()

    with (
        patch("vllm_ascend.ops.rope_cache_ops.has_mla_preprocess_by_cache_kernel", return_value=False),
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        pytest.raises(RuntimeError, match="requires a true by-cache backend"),
    ):
        rope_cache_ops.mla_preprocess_by_cache(
            torch.randn(2, 8),
            torch.randn(8, 8),
            None,
            torch.randn(8),
            None,
            torch.randn(8, 8),
            None,
            torch.randn(8),
            torch.tensor([1, 3]),
            rotary_emb,
            torch.randn(8, 8),
            torch.randn(2, 8),
            torch.randn(2, 8),
            torch.tensor([0, 1]),
            q_out0=torch.empty(2, 8),
            kv_cache_out0=torch.empty(2, 8),
            q_out1=torch.empty(2, 8),
            kv_cache_out1=torch.empty(2, 8),
            inner_out=torch.empty(2, 8),
        )


def test_mla_prolog_v2_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    token_x = torch.randn(2, 8)
    positions = torch.tensor([1, 3])
    cache_index = torch.tensor([0, 1])
    kv_cache = torch.randn(2, 8)
    kr_cache = torch.randn(2, 8)
    expected = (torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1))
    native_op = Mock(return_value=expected)

    weights = [torch.randn(8, 8) for _ in range(4)]
    gammas = [torch.randn(8) for _ in range(2)]

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=native_op),
    ):
        output = rope_cache_ops.mla_prolog_v2_by_cache(
            token_x,
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            gammas[0],
            gammas[1],
            positions,
            rotary_emb,
            cache_index,
            kv_cache,
            kr_cache,
            ref_tensor=token_x,
            cache_mode="PA_NZ",
        )

    assert output is expected
    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is token_x
    assert native_args[7] is positions
    assert native_args[8] is rotary_emb.cos_sin_cache
    assert native_args[9] is cache_index
    assert native_args[10] is kv_cache
    assert native_args[11] is kr_cache
    assert native_op.call_args.kwargs["cache_mode"] == "PA_NZ"


def test_mla_prolog_v2_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    token_x = torch.randn(2, 8)
    positions = torch.tensor([1, 3])
    cache_index = torch.tensor([0, 1])
    kv_cache = torch.randn(2, 8)
    kr_cache = torch.randn(2, 8)

    weights = [torch.randn(8, 8) for _ in range(4)]
    gammas = [torch.randn(8) for _ in range(2)]

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=None),
        patch.object(
            rope_cache_ops.torch_npu,
            "npu_mla_prolog_v2",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.mla_prolog_v2_by_cache(
            token_x,
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            gammas[0],
            gammas[1],
            positions,
            rotary_emb,
            cache_index,
            kv_cache,
            kr_cache,
            ref_tensor=token_x,
            layout="TD",
            cache_mode="PA_NZ",
        )

    legacy_op.assert_not_called()


def test_mla_prolog_v3_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    token_x = torch.randn(2, 1, 8)
    positions = torch.tensor([1, 3])
    cache_index = torch.tensor([[0], [1]])
    kv_cache = torch.randn(2, 8)
    kr_cache = torch.randn(2, 8)
    expected = (torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1), torch.randn(1))
    native_op = Mock(return_value=expected)

    weights = [torch.randn(8, 8) for _ in range(4)]
    gammas = [torch.randn(8) for _ in range(2)]

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=native_op),
    ):
        output = rope_cache_ops.mla_prolog_v3_by_cache(
            token_x=token_x,
            weight_dq=weights[0],
            weight_uq_qr=weights[1],
            weight_uk=weights[2],
            weight_dkv_kr=weights[3],
            rmsnorm_gamma_cq=gammas[0],
            rmsnorm_gamma_ckv=gammas[1],
            positions=positions,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            cache_index=cache_index,
            ref_tensor=token_x,
            cache_mode="PA_BSND",
        )

    assert output is expected
    native_op.assert_called_once()
    assert native_op.call_args.kwargs["token_x"] is token_x
    assert native_op.call_args.kwargs["positions"] is positions
    assert native_op.call_args.kwargs["cos_sin_cache"] is rotary_emb.cos_sin_cache
    assert native_op.call_args.kwargs["cache_index"] is cache_index
    assert native_op.call_args.kwargs["cache_mode"] == "PA_BSND"


def test_mla_prolog_v3_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    token_x = torch.randn(2, 1, 8)
    positions = torch.tensor([1, 3])
    cache_index = torch.tensor([[0], [1]])
    kv_cache = torch.randn(2, 8)
    kr_cache = torch.randn(2, 8)

    weights = [torch.randn(8, 8) for _ in range(4)]
    gammas = [torch.randn(8) for _ in range(2)]

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=None),
        patch.object(
            rope_cache_ops.torch_npu,
            "npu_mla_prolog_v3",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.mla_prolog_v3_by_cache(
            token_x=token_x,
            weight_dq=weights[0],
            weight_uq_qr=weights[1],
            weight_uk=weights[2],
            weight_dkv_kr=weights[3],
            rmsnorm_gamma_cq=gammas[0],
            rmsnorm_gamma_ckv=gammas[1],
            positions=positions,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            cache_index=cache_index,
            ref_tensor=token_x,
            layout="TD",
            cache_mode="PA_BSND",
        )

    legacy_op.assert_not_called()


def test_interleave_rope_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    expected = torch.randn_like(x)

    with (
        patch.object(rope_cache_ops.torch_npu, "npu_interleave_rope_by_cache", return_value=expected, create=True)
        as native_op,
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        output = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary_emb)

    assert output is expected
    native_op.assert_called_once()
    call_args = native_op.call_args
    assert call_args.args[0] is x
    assert call_args.args[1] is positions
    assert call_args.args[2] is rotary_emb.cos_sin_cache
    assert call_args.kwargs["rope_dim"] == rotary_emb.rotary_dim
    assert call_args.kwargs["is_neox_style"] is True


def test_interleave_rope_by_cache_uses_triton_cache_fallback_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])

    def fake_rope(qk, **kwargs):
        assert kwargs["cos_sin_cache"] is rotary_emb.cos_sin_cache
        assert kwargs["positions"] is positions
        assert kwargs["rope_dim"] == rotary_emb.rotary_dim
        assert kwargs["is_neox_style"] is True
        return qk + 1

    triton_op = Mock(side_effect=fake_rope)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=None),
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_interleave_rope_by_cache", return_value=triton_op),
        patch.object(rope_cache_ops.torch_npu, "npu_interleave_rope", create=True) as legacy_op,
    ):
        output = rope_cache_ops.interleave_rope_by_cache(x, positions, rotary_emb)

    triton_op.assert_called_once()
    legacy_op.assert_not_called()
    torch.testing.assert_close(output, x + 1)


def test_interleave_rope_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])

    with (
        patch.object(rope_cache_ops.torch_npu, "npu_interleave_rope_by_cache", None, create=True),
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_interleave_rope_by_cache", return_value=None),
        patch.object(rope_cache_ops.torch_npu, "npu_interleave_rope", create=True) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        rope_cache_ops.interleave_rope_by_cache(x, positions, rotary_emb)

    legacy_op.assert_not_called()


def test_rotary_mul_by_cache_native_keeps_fp32_compute_and_casts_back():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4, dtype=torch.float16)
    positions = torch.tensor([1, 3])
    expected = torch.randn(2, 1, 1, 4, dtype=torch.float32)
    native_op = Mock(return_value=expected)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=native_op),
    ):
        output = rope_cache_ops.rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="interleave",
            inverse=True,
            fp32_compute=True,
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0].dtype == torch.float32
    assert native_args[1] is positions
    assert native_args[2].dtype == torch.float32
    assert native_op.call_args.kwargs["rotary_mode"] == "interleave"
    assert native_op.call_args.kwargs["inverse"] is True
    assert output.dtype == x.dtype


def test_rotary_mul_by_cache_uses_triton_cache_fallback_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4, dtype=torch.float16)
    positions = torch.tensor([1, 3])

    def fake_rope(qk, **kwargs):
        assert qk.dtype == torch.float32
        assert kwargs["cos_sin_cache"] is rotary_emb.cos_sin_cache
        assert kwargs["positions"] is positions
        assert kwargs["rope_dim"] == rotary_emb.rotary_dim
        assert kwargs["is_neox_style"] is True
        return qk + 1

    triton_op = Mock(side_effect=fake_rope)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=None),
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_rope_forward_siso", return_value=triton_op),
        patch("vllm_ascend.ops.rope_cache_ops.rotary_mul_materialized") as materialized_op,
    ):
        output = rope_cache_ops.rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="half",
            fp32_compute=True,
        )

    triton_op.assert_called_once()
    materialized_op.assert_not_called()
    assert output.dtype == x.dtype


def test_rotary_mul_by_cache_native_output_depends_on_layer_owned_cache_rows():
    positions = torch.tensor([1, 3])
    x = torch.zeros(2, 1, 1, 4)
    rotary_a = DummyRotaryEmbedding()
    rotary_b = DummyRotaryEmbedding()
    rotary_a.cos_sin_cache = torch.arange(16 * 4, dtype=x.dtype).view(16, 4)
    rotary_b.cos_sin_cache = rotary_a.cos_sin_cache + 100

    def fake_native(x_arg, positions_arg, cos_sin_cache_arg, **kwargs):
        cos, sin = cos_sin_cache_arg.index_select(0, positions_arg).chunk(2, dim=-1)
        return torch.cat((cos, sin), dim=-1).view_as(x_arg)

    with patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=Mock(side_effect=fake_native)):
        out_a = rope_cache_ops.rotary_mul_by_cache(x.clone(), positions, rotary_a)
        out_b = rope_cache_ops.rotary_mul_by_cache(x.clone(), positions, rotary_b)

    expected_a = rotary_a.cos_sin_cache.index_select(0, positions).view_as(x)
    expected_b = rotary_b.cos_sin_cache.index_select(0, positions).view_as(x)
    torch.testing.assert_close(out_a, expected_a)
    torch.testing.assert_close(out_b, expected_b)
    assert not torch.equal(out_a, out_b)


def test_rotary_mul_by_cache_inverse_uses_triton_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    triton_op = Mock(return_value=x.reshape(2, 1, 4) + 1)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=None),
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_rope_forward_siso", return_value=triton_op),
        patch("vllm_ascend.ops.rope_cache_ops.rotary_mul_materialized") as materialized_op,
    ):
        output = rope_cache_ops.rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="interleave",
            inverse=True,
        )

    triton_op.assert_called_once()
    assert triton_op.call_args.kwargs["inverse"] is True
    materialized_op.assert_not_called()
    torch.testing.assert_close(output, x + 1)


def test_rotary_mul_by_cache_native_inplace_copies_returned_tensor():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.zeros(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    expected = torch.ones_like(x)
    native_op = Mock(return_value=expected)

    with patch("vllm_ascend.ops.rope_cache_ops._get_torch_npu_op", return_value=native_op):
        output = rope_cache_ops.rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            inplace=True,
        )

    assert output is None
    assert torch.equal(x, expected)
    native_op.assert_called_once()


def test_kv_rmsnorm_rope_cache_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    kv_no_split = torch.randn(2, 1, 1, 8)
    positions = torch.tensor([0, 1])
    slots = torch.tensor([4, 5], dtype=torch.int32)
    weight = torch.randn(8)
    kv_cache_rope = torch.randn(8, 1, 4)
    kv_cache_nope = torch.randn(8, 1, 4)

    with (
        patch.object(rope_cache_ops.torch_npu, "npu_kv_rmsnorm_rope_cache_by_cache", None, create=True),
        patch.object(
            rope_cache_ops.torch_npu,
            "npu_kv_rmsnorm_rope_cache",
            create=True,
        ) as legacy_op,
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_kv_rmsnorm_rope_cache_by_cache", return_value=None),
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
            kv_no_split,
            weight,
            positions,
            rotary_emb,
            slots,
            kv_cache_rope,
            kv_cache_nope,
            epsilon=1e-6,
            cache_mode="PA_NZ",
            is_output_kv=True,
        )

    legacy_op.assert_not_called()


@pytest.mark.parametrize(("cache_mode", "cache_mode_is_nz"), [("PA", False), ("PA_NZ", True)])
def test_kv_rmsnorm_rope_cache_by_cache_uses_triton_without_materializing(cache_mode, cache_mode_is_nz):
    rotary_emb = DummyRotaryEmbedding()
    kv_no_split = torch.randn(2, 1, 1, 8)
    positions = torch.tensor([0, 1])
    slots = torch.tensor([4, 5], dtype=torch.long)
    weight = torch.randn(4)
    kv_cache_rope = torch.randn(2, 4, 1, 4)
    kv_cache_nope = torch.randn(2, 4, 1, 4)
    expected = (kv_cache_rope, kv_cache_nope, torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4))
    triton_op = Mock(return_value=expected)

    with (
        patch.object(rope_cache_ops.torch_npu, "npu_kv_rmsnorm_rope_cache_by_cache", None, create=True),
        patch.object(rope_cache_ops.torch_npu, "npu_kv_rmsnorm_rope_cache", create=True) as legacy_op,
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_kv_rmsnorm_rope_cache_by_cache", return_value=triton_op),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        output = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
            kv_no_split,
            weight,
            positions,
            rotary_emb,
            slots,
            kv_cache_rope,
            kv_cache_nope,
            epsilon=1e-6,
            cache_mode=cache_mode,
            is_output_kv=True,
        )

    assert output is expected
    legacy_op.assert_not_called()
    triton_op.assert_called_once()
    args = triton_op.call_args.args
    assert args[0] is kv_no_split
    assert args[1] is weight
    assert args[2] is positions
    assert args[3] is rotary_emb.cos_sin_cache
    assert args[4] is slots
    assert args[5] is kv_cache_rope
    assert args[6] is kv_cache_nope
    assert triton_op.call_args.kwargs["epsilon"] == 1e-6
    assert triton_op.call_args.kwargs["rope_dim"] == 4
    assert triton_op.call_args.kwargs["is_neox_style"] is True
    assert triton_op.call_args.kwargs["is_output_kv"] is True
    assert triton_op.call_args.kwargs["cache_mode_is_nz"] is cache_mode_is_nz


def test_kv_rmsnorm_rope_cache_by_cache_negative_slots_skip_native():
    rotary_emb = DummyRotaryEmbedding()
    kv_no_split = torch.randn(2, 1, 1, 8)
    positions = torch.tensor([0, 1])
    slots = torch.tensor([-1, -1], dtype=torch.long)
    weight = torch.randn(4)
    kv_cache_rope = torch.randn(2, 4, 1, 4)
    kv_cache_nope = torch.randn(2, 4, 1, 4)
    expected = (kv_cache_rope, kv_cache_nope, torch.randn(2, 1, 1, 4), torch.randn(2, 1, 1, 4))
    native_op = Mock(return_value=expected)
    triton_op = Mock(return_value=expected)

    with (
        patch.object(
            rope_cache_ops.torch_npu,
            "npu_kv_rmsnorm_rope_cache_by_cache",
            native_op,
            create=True,
        ),
        patch("vllm_ascend.ops.rope_cache_ops._get_triton_kv_rmsnorm_rope_cache_by_cache", return_value=triton_op),
    ):
        rope_cache_ops.clear_rope_cache_op_capability_cache()
        output = rope_cache_ops.kv_rmsnorm_rope_cache_by_cache(
            kv_no_split,
            weight,
            positions,
            rotary_emb,
            slots,
            kv_cache_rope,
            kv_cache_nope,
            epsilon=1e-6,
            cache_mode="PA",
            is_output_kv=True,
            allow_negative_slots=True,
        )

    assert output is expected
    native_op.assert_not_called()
    triton_op.assert_called_once()


def test_rotary_mul_materialized_keeps_inverse_and_fp32_compute():
    x = torch.randn(2, 1, 1, 4, dtype=torch.float16)
    cos = torch.randn(2, 1, 1, 4)
    sin = torch.randn(2, 1, 1, 4)
    expected = torch.randn(2, 1, 1, 4, dtype=torch.float32)

    with patch.object(
        rope_cache_ops.torch_npu,
        "npu_rotary_mul",
        return_value=expected,
        create=True,
    ) as legacy_op:
        output = rope_cache_ops.rotary_mul_materialized(
            x,
            cos,
            sin,
            rotary_mode="interleave",
            inverse=True,
            fp32_compute=True,
        )

    legacy_op.assert_called_once()
    legacy_args = legacy_op.call_args.args
    assert legacy_args[0].dtype == torch.float32
    assert legacy_args[1] is cos
    assert torch.equal(legacy_args[2], -sin)
    assert legacy_op.call_args.kwargs["rotary_mode"] == "interleave"
    assert output.dtype == x.dtype


def test_split_qkv_tp_rmsnorm_rope_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    qkv = torch.randn(2, 16)
    positions = torch.tensor([0, 1])
    q_weight = torch.randn(8)
    k_weight = torch.randn(8)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=None),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops.vllm.split_qkv_tp_rmsnorm_rope",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.split_qkv_tp_rmsnorm_rope_by_cache(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=8,
            kv_hidden_size=4,
            head_dim=4,
            rotary_dim=4,
            eps=1e-6,
            tp_world=2,
            positions=positions,
            rotary_emb=rotary_emb,
        )

    legacy_op.assert_not_called()


def test_split_qkv_tp_rmsnorm_rope_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    qkv = torch.randn(2, 16)
    positions = torch.tensor([0, 1])
    expected = (torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
    native_op = Mock(return_value=expected)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=native_op),
    ):
        output = rope_cache_ops.split_qkv_tp_rmsnorm_rope_by_cache(
            input=qkv,
            q_weight=torch.randn(8),
            k_weight=torch.randn(8),
            q_hidden_size=8,
            kv_hidden_size=4,
            head_dim=4,
            rotary_dim=4,
            eps=1e-6,
            tp_world=2,
            positions=positions,
            rotary_emb=rotary_emb,
        )

    assert output is expected
    native_op.assert_called_once()
    assert native_op.call_args.kwargs["input"] is qkv
    assert native_op.call_args.kwargs["positions"] is positions
    assert native_op.call_args.kwargs["cos_sin_cache"] is rotary_emb.cos_sin_cache


def test_split_qkv_tp_rmsnorm_rope_by_cache_non_neox_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    rotary_emb.is_neox_style = False
    qkv = torch.randn(2, 16)
    positions = torch.tensor([0, 1])
    q_weight = torch.randn(8)
    k_weight = torch.randn(8)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_vllm_op", return_value=native_op),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops.vllm.split_qkv_tp_rmsnorm_rope",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.split_qkv_tp_rmsnorm_rope_by_cache(
            input=qkv,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=8,
            kv_hidden_size=4,
            head_dim=4,
            rotary_dim=4,
            eps=1e-6,
            tp_world=2,
            positions=positions,
            rotary_emb=rotary_emb,
        )

    native_op.assert_not_called()
    legacy_op.assert_not_called()


def test_inplace_partial_rotary_mul_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="interleave",
            partial_slice=[0, 4],
            rope_dim_offset=2,
            inverse=True,
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is x
    assert native_args[1] is positions
    assert native_args[2] is rotary_emb.cos_sin_cache
    assert native_op.call_args.kwargs["partial_slice"] == [0, 4]
    assert native_op.call_args.kwargs["rope_dim_offset"] == 2
    assert native_op.call_args.kwargs["inverse"] is True


def test_inplace_partial_rotary_mul_by_cache_uses_triton_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    triton_output = x + 1

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        patch("vllm_ascend.ops.rope_cache_ops._try_triton_rotary_siso_by_cache", return_value=triton_output)
        as triton_op,
    ):
        rope_cache_ops.inplace_partial_rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="interleave",
            partial_slice=[0, 4],
            inverse=True,
    )

    triton_op.assert_called_once()
    torch.testing.assert_close(x, triton_output)


def test_inplace_partial_rotary_mul_by_cache_refuses_materialized_fallback_when_native_op_is_unsupported():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 1, 1, 4)
    positions = torch.tensor([1, 3])
    native_op = Mock(
        side_effect=RuntimeError(
            "aclnnInplacePartialRotaryMul or aclnnInplacePartialRotaryMulGetWorkspaceSize not in libopapi.so, "
            "or libopapi.sonot found."
        )
    )

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_by_cache_kernel", return_value=True),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.inplace_partial_rotary_mul",
            create=True,
        ) as legacy_op,
        patch("vllm_ascend.ops.rope_cache_ops._try_triton_rotary_siso_by_cache", return_value=None),
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_by_cache(
            x,
            positions,
            rotary_emb,
            rotary_mode="interleave",
            partial_slice=[0, 4],
        )

    native_op.assert_called_once()
    legacy_op.assert_not_called()
    assert "inplace_partial_rotary_mul_by_cache" in rope_cache_ops._UNSUPPORTED_C_ASCEND_OPS


def test_inplace_partial_rotary_mul_dsa_by_cache_uses_native_without_materializing():
    x = torch.randn(2, 1, 1, 16)
    dsa_rope_cache = DummyDSARopeCache(rotary_dim=16)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 16],
            inverse=True,
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is x
    assert native_args[1] is dsa_rope_cache.positions
    assert torch.equal(native_args[2], dsa_rope_cache.cos_sin_cache)
    assert native_op.call_args.kwargs["partial_slice"] == [0, 16]
    assert native_op.call_args.kwargs["inverse"] is True
    dsa_rope_cache.materialize.assert_not_called()


def test_inplace_partial_rotary_mul_dsa_by_cache_casts_cache_to_input_dtype():
    x = torch.randn(2, 1, 1, 16, dtype=torch.float16)
    dsa_rope_cache = DummyDSARopeCache(rotary_dim=16)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 16],
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[2].dtype == x.dtype
    assert torch.equal(native_args[2], dsa_rope_cache.cos_sin_cache.to(dtype=x.dtype))
    dsa_rope_cache.materialize.assert_not_called()


def test_inplace_partial_rotary_mul_dsa_by_cache_rejects_tuple_cos_sin_cache():
    x = torch.randn(2, 1, 1, 16)
    dsa_rope_cache = DummyDSARopeCache(rotary_dim=16)
    dsa_rope_cache.cos_sin_cache = (dsa_rope_cache.cos_cache, dsa_rope_cache.sin_cache)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 16],
        )

    native_op.assert_not_called()
    dsa_rope_cache.materialize.assert_not_called()


def test_inplace_partial_rotary_mul_dsa_by_cache_chunks_native_calls_without_materializing():
    x = torch.randn(2, 1, 1, 128, dtype=torch.float16)
    dsa_rope_cache = DummyDSARopeCache(rotary_dim=128)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 128],
        )

    assert native_op.call_count == 2
    assert native_op.call_args_list[0].kwargs["partial_slice"] == [0, 64]
    assert native_op.call_args_list[0].kwargs["rope_dim"] == 64
    assert native_op.call_args_list[0].args[2].shape[-1] == 128
    assert native_op.call_args_list[0].args[2].dtype == x.dtype
    assert native_op.call_args_list[1].kwargs["partial_slice"] == [64, 128]
    assert native_op.call_args_list[1].kwargs["rope_dim"] == 64
    assert native_op.call_args_list[1].args[2].shape[-1] == 128
    assert native_op.call_args_list[1].args[2].dtype == x.dtype
    dsa_rope_cache.materialize.assert_not_called()


def test_inplace_partial_rotary_mul_dsa_by_cache_refuses_materialized_fallback_when_native_op_is_unsupported():
    x = torch.randn(2, 1, 1, 16)
    dsa_rope_cache = DummyDSARopeCache(rotary_dim=16)
    native_op = Mock(
        side_effect=RuntimeError(
            "aclnnInplacePartialRotaryMul or aclnnInplacePartialRotaryMulGetWorkspaceSize not in libopapi.so, "
            "or libopapi.sonot found."
        )
    )

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.inplace_partial_rotary_mul",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 16],
        )

    native_op.assert_called_once()
    dsa_rope_cache.materialize.assert_not_called()
    legacy_op.assert_not_called()
    assert "inplace_partial_rotary_mul_dsa_by_cache" in rope_cache_ops._UNSUPPORTED_C_ASCEND_OPS


def test_inplace_partial_rotary_mul_dsa_by_cache_refuses_unsupported_shape_without_disabling():
    x = torch.randn(2, 1, 1, 4)
    dsa_rope_cache = DummyDSARopeCache()
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_inplace_partial_rotary_mul_dsa_by_cache_kernel", return_value=True),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.inplace_partial_rotary_mul",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 4],
        )

    native_op.assert_not_called()
    dsa_rope_cache.materialize.assert_not_called()
    legacy_op.assert_not_called()
    assert "inplace_partial_rotary_mul_dsa_by_cache" not in rope_cache_ops._UNSUPPORTED_C_ASCEND_OPS


def test_inplace_partial_rotary_mul_dsa_by_cache_refuses_materialized_fallback():
    x = torch.randn(2, 1, 1, 4)
    dsa_rope_cache = DummyDSARopeCache()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.inplace_partial_rotary_mul",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[0, 4],
            inverse=True,
        )

    dsa_rope_cache.materialize.assert_not_called()
    legacy_op.assert_not_called()


@pytest.mark.parametrize(
    "message",
    [
        "The binary_info_config.json of socVersion [ascend910b] does not support opType [InplacePartialRotaryMul].",
        "aclnnInplacePartialRotaryMul or aclnnInplacePartialRotaryMulGetWorkspaceSize not in libopapi.so, "
        "or libopapi.sonot found.",
    ],
)
def test_inplace_partial_rotary_mul_dsa_by_cache_does_not_call_legacy_fallback(message):
    x = torch.zeros(2, 1, 1, 6)
    expected_rotary = torch.ones(2, 1, 1, 4)
    dsa_rope_cache = DummyDSARopeCache()

    unsupported = RuntimeError(message)
    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.inplace_partial_rotary_mul",
            side_effect=unsupported,
            create=True,
        ) as legacy_op,
        patch.object(
            rope_cache_ops.torch_npu,
            "npu_rotary_mul",
            return_value=expected_rotary,
            create=True,
        ) as torch_npu_rotary,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.inplace_partial_rotary_mul_dsa_by_cache(
            x,
            dsa_rope_cache,
            rotary_mode="interleave",
            partial_slice=[2, 6],
        )

    dsa_rope_cache.materialize.assert_not_called()
    legacy_op.assert_not_called()
    torch_npu_rotary.assert_not_called()


def test_compressor_by_cache_uses_native_without_materializing():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 8)
    positions = torch.tensor([1, 3])
    expected = torch.randn(2, 8)
    native_op = Mock(return_value=expected)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_compressor_by_cache_kernel", return_value=True),
    ):
        output = rope_cache_ops.compressor_by_cache(
            x,
            torch.randn(8, 8),
            torch.randn(8, 8),
            torch.randn(2, 8),
            torch.randn(8),
            torch.randn(8),
            positions,
            rotary_emb,
            ref_tensor=x,
            rope_head_dim=4,
        )

    assert output is expected
    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is x
    assert native_args[6] is positions
    assert native_args[7] is rotary_emb.cos_sin_cache
    assert native_op.call_args.kwargs["rope_head_dim"] == 4


def test_compressor_by_cache_refuses_materialized_fallback():
    rotary_emb = DummyRotaryEmbedding()
    x = torch.randn(2, 8)
    positions = torch.tensor([1, 3])

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.compressor",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.compressor_by_cache(
            x,
            torch.randn(8, 8),
            torch.randn(8, 8),
            torch.randn(2, 8),
            torch.randn(8),
            torch.randn(8),
            positions,
            rotary_emb,
            ref_tensor=x,
            rope_head_dim=4,
        )

    legacy_op.assert_not_called()


def test_compressor_dsa_by_cache_uses_native_without_materializing():
    x = torch.randn(2, 8, dtype=torch.float16)
    wkv = torch.randn(8, 8, dtype=torch.float16)
    wgate = torch.randn(8, 8, dtype=torch.float16)
    state_cache = torch.randn(2, 8)
    ape = torch.randn(8)
    norm_weight = torch.randn(8, dtype=torch.float16)
    dsa_rope_cache = DummyDSARopeCache()
    dsa_rope_cache.cos_cache = dsa_rope_cache.cos_cache.to(dtype=x.dtype)
    dsa_rope_cache.sin_cache = dsa_rope_cache.sin_cache.to(dtype=x.dtype)
    dsa_rope_cache.cos_sin_cache = torch.cat((dsa_rope_cache.cos_cache, dsa_rope_cache.sin_cache), dim=-1)
    expected = torch.randn(2, 8, dtype=torch.float16)
    native_op = Mock(return_value=expected)

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_compressor_dsa_by_cache_kernel", return_value=True),
    ):
        output = rope_cache_ops.compressor_dsa_by_cache(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            dsa_rope_cache,
            rope_head_dim=4,
        )

    assert output is expected
    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[0] is x
    assert native_args[6] is dsa_rope_cache.positions
    assert torch.equal(native_args[7], dsa_rope_cache.cos_sin_cache)
    dsa_rope_cache.materialize.assert_not_called()


def test_compressor_dsa_by_cache_rejects_tuple_cos_sin_cache():
    x = torch.randn(2, 8, dtype=torch.float16)
    wkv = torch.randn(8, 8, dtype=torch.float16)
    wgate = torch.randn(8, 8, dtype=torch.float16)
    state_cache = torch.randn(2, 8)
    ape = torch.randn(8)
    norm_weight = torch.randn(8, dtype=torch.float16)
    dsa_rope_cache = DummyDSARopeCache()
    dsa_rope_cache.cos_sin_cache = (dsa_rope_cache.cos_cache, dsa_rope_cache.sin_cache)
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_compressor_dsa_by_cache_kernel", return_value=True),
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.compressor_dsa_by_cache(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            dsa_rope_cache,
            rope_head_dim=4,
        )

    native_op.assert_not_called()
    dsa_rope_cache.materialize.assert_not_called()


def test_compressor_dsa_by_cache_casts_cache_to_input_dtype_for_native_kernel():
    x = torch.randn(2, 8, dtype=torch.float16)
    wkv = torch.randn(8, 8, dtype=torch.float16)
    wgate = torch.randn(8, 8, dtype=torch.float16)
    state_cache = torch.randn(2, 8)
    ape = torch.randn(8)
    norm_weight = torch.randn(8, dtype=torch.float16)
    dsa_rope_cache = DummyDSARopeCache()
    native_op = Mock()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=native_op),
        patch("vllm_ascend.ops.rope_cache_ops.has_compressor_dsa_by_cache_kernel", return_value=True),
    ):
        rope_cache_ops.compressor_dsa_by_cache(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            dsa_rope_cache,
            rope_head_dim=4,
        )

    native_op.assert_called_once()
    native_args = native_op.call_args.args
    assert native_args[7].dtype == x.dtype
    assert torch.equal(native_args[7], dsa_rope_cache.cos_sin_cache.to(dtype=x.dtype))
    dsa_rope_cache.materialize.assert_not_called()


def test_compressor_dsa_by_cache_refuses_materialized_fallback():
    x = torch.randn(2, 8)
    wkv = torch.randn(8, 8)
    wgate = torch.randn(8, 8)
    state_cache = torch.randn(2, 8)
    ape = torch.randn(8)
    norm_weight = torch.randn(8)
    dsa_rope_cache = DummyDSARopeCache()

    with (
        patch("vllm_ascend.ops.rope_cache_ops._get_c_ascend_op", return_value=None),
        patch(
            "vllm_ascend.ops.rope_cache_ops.torch.ops._C_ascend.compressor",
            create=True,
        ) as legacy_op,
        pytest.raises(RuntimeError, match="refusing to materialize sin/cos"),
    ):
        rope_cache_ops.compressor_dsa_by_cache(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            dsa_rope_cache,
            rope_head_dim=4,
        )

    dsa_rope_cache.materialize.assert_not_called()
    legacy_op.assert_not_called()
