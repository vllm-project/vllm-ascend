from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.mla_v1 import AscendMLAImpl


@pytest.mark.parametrize("is_prefill", [True, False])
@pytest.mark.parametrize("preprocess_path", ["plain", "strided", "prolog"])
@pytest.mark.parametrize("gate_layout", ["standalone", "fused", "none"])
def test_mla_forward_preserves_output_gate(is_prefill, preprocess_path, gate_layout):
    if preprocess_path == "prolog" and gate_layout == "fused":
        pytest.skip("The prolog path does not support a QKV-A-fused gate")

    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    num_tokens, hidden_size = 3, 4
    hidden = torch.ones(num_tokens, hidden_size)
    attention = torch.arange(1, num_tokens * hidden_size + 1, dtype=torch.float32).reshape_as(hidden)
    gate = torch.linspace(-3, 3, num_tokens * hidden_size).reshape_as(hidden)
    impl.num_heads = 2
    impl.v_head_dim = 2
    impl.kv_lora_rank = 2
    impl.use_output_gate = gate_layout != "none"
    impl.qkv_a_gate_fused = gate_layout == "fused"
    impl.g_proj = MagicMock(return_value=(gate, None)) if gate_layout == "standalone" else None
    impl.use_pcp = False
    impl.use_dcp = False
    impl.is_pcp_decode_sharded = False
    impl.enable_mlapo = False
    impl.fa_quant_layer = False
    impl.use_mla_rope = False
    impl.enable_kv_nz = False
    impl._decode_requires_current_kv = lambda _: False
    impl._prefill_requires_current_kv = lambda _: False
    impl._cache_requires_prolog_v3_writer = lambda _: preprocess_path == "prolog"
    impl._uses_strided_kv_adapter = lambda: preprocess_path == "strided"

    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        num_decodes=0 if is_prefill else 1,
        num_prefills=1 if is_prefill else 0,
        num_decode_tokens=0 if is_prefill else num_tokens,
        external_flashmla=None,
    )
    prefill = SimpleNamespace(q_nope=None, q_pe=None, k_nope=None, k_pe=None, value=None)
    prepared = (None, prefill) if is_prefill else (object(), None)
    fused_gate = gate if gate_layout == "fused" else None
    impl._mla_preprocess = MagicMock(return_value=(*prepared, fused_gate))
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=prepared)
    impl._forward_prefill = MagicMock(return_value=attention)
    impl._forward_decode = MagicMock(return_value=attention)
    impl.o_proj = MagicMock(side_effect=lambda x, **_: (x, None))
    cache = torch.empty(2, 2, 4)
    if preprocess_path == "strided":
        cache = cache.transpose(0, 1)
    output = torch.empty_like(hidden)
    with (
        patch("vllm_ascend.attention.mla_v1._EXTRA_CTX", SimpleNamespace(num_tokens=num_tokens)),
        patch("vllm_ascend.attention.mla_v1.maybe_save_kv_layer_to_connector"),
    ):
        result = impl.forward("layer", hidden, (cache, cache), metadata, output)

    expected = attention if gate_layout == "none" else attention * torch.sigmoid(gate)
    assert result is output
    torch.testing.assert_close(result, expected)
    if impl.g_proj is not None:
        impl.g_proj.assert_called_once()
