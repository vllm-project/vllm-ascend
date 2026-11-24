# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_config import KDABackendConfig
from vllm_ascend.ops import kimi_kda
from vllm_ascend.ops import mega_kda_localtp8 as adapter


def metadata():
    slots = torch.tensor([[2, 0], [3, 0], [1, 0], [0, 0]], dtype=torch.int32)
    desc = SimpleNamespace(cache_indices=slots[:, 1], num_accepted_tokens=torch.ones(4, dtype=torch.int32))
    return SimpleNamespace(
        num_spec_decodes=3,
        num_prefills=0,
        num_decodes=0,
        spec_query_start_loc=torch.tensor([0, 8, 9, 15, 15], dtype=torch.int32),
        spec_state_indices_tensor=slots,
        spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=desc),
        recoverssm_context=SimpleNamespace(normalized_key_base_addrs=object(), decay_base_addrs=object()),
    )


def verifier():
    op = object.__new__(adapter.LocalTP8Verify)
    op.workspace = SimpleNamespace(cpu_group=object(), rank=5, addresses=torch.arange(8))
    op.launch = Mock()
    return op


def test_selector_and_group_order(monkeypatch, tmp_path):
    config = KDABackendConfig(backend="mega_kda_localtp8", operator_package_dir=str(tmp_path))
    assert config.backend == "mega_kda_localtp8"
    group = SimpleNamespace(world_size=8, ranks=list(range(8)))
    monkeypatch.setattr(adapter, "get_qkv_iproj_tp_group", lambda: group)
    monkeypatch.setattr(adapter, "get_gproj_tp_group", lambda: group)
    monkeypatch.setattr(adapter, "get_otp_group", lambda: SimpleNamespace(ranks=list(reversed(group.ranks))))
    with pytest.raises(ValueError, match="identical eight-rank"):
        adapter.LocalTP8Verify(tmp_path)


@pytest.mark.parametrize(
    "rows,prefills,peer,expected", [(32, 0, 1, True), (32, 0, 0, False), (32, 1, 1, False), (8, 0, 1, False)]
)
def test_subgroup_selects_one_communication_path(monkeypatch, rows, prefills, peer, expected):
    op = verifier()
    meta = metadata()
    meta.num_prefills = prefills

    def reduce(vote, *, op, group):
        assert group is not None and op == torch.distributed.ReduceOp.MIN
        vote.fill_(min(vote.item(), peer))

    monkeypatch.setattr(adapter.dist, "all_reduce", reduce)
    assert op.supports_batch(torch.empty(rows, 7168, dtype=torch.bfloat16), meta) is expected


def test_serving_abi_preserves_independent_slot_vectors_and_ragged_offsets(monkeypatch):
    op = verifier()
    meta = metadata()
    weight = torch.empty(1)
    linear = SimpleNamespace(weight=weight)

    def record(width, dtype):
        return torch.empty(4, 96, 8, width, dtype=dtype)

    layer = SimpleNamespace(
        in_proj_qkvgfab=SimpleNamespace(_localtp8_qkv=weight, _localtp8_scale=weight),
        kv_cache=(torch.empty(4, 10, 36864, dtype=torch.bfloat16), torch.empty(4, 96, 128, 128)),
        recoverssm_records=SimpleNamespace(correction=record(128, torch.float32), key_gate=record(256, torch.bfloat16)),
        recoverssm_normalized_key_cache=record(128, torch.float32),
        recoverssm_decay_cache=record(128, torch.float32),
        _localtp8_bg_weight=weight,
        _localtp8_gamma=weight,
        ascend_conv1d_weight=weight,
        f_a_proj=linear,
        f_b_proj=linear,
        o_proj=linear,
        A_log=weight,
        dt_bias=weight,
        gate_lower_bound=-5.0,
        o_norm=SimpleNamespace(eps=1e-6),
    )
    monkeypatch.setattr(torch.npu, "current_stream", lambda: SimpleNamespace(npu_stream=7))
    hidden = torch.empty(32, 7168, dtype=torch.bfloat16)
    op(layer, hidden, meta, torch.empty_like(hidden))
    args = op.launch.call_args.args
    desc = meta.spec_decode_metadata.spec_causal_conv1d
    assert args[18:21] == (
        meta.spec_state_indices_tensor.data_ptr(),
        desc.cache_indices.data_ptr(),
        desc.num_accepted_tokens.data_ptr(),
    )
    assert args[23:29] == (op.workspace.addresses.data_ptr(), meta.spec_query_start_loc.data_ptr(), 5, 2, 2, 1)
    assert len(args) == 33
    layer.recoverssm_records.correction = layer.recoverssm_records.correction.transpose(1, 2)
    with pytest.raises(ValueError, match="record layout"):
        op(layer, hidden, meta, torch.empty_like(hidden))


@pytest.mark.parametrize("selector", ["mega_recurrent_kda", "mega_recurrent_kda_mx", "mega_kda", "mega_kda_localtp8"])
def test_model_selects_localtp8_backend(monkeypatch, tmp_path, selector):
    def base_init(layer, config, vc, prefix):
        torch.nn.Module.__init__(layer)
        layer.prefix = prefix
        layer.tp_size = 1
        layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
        layer.num_heads, layer.head_dim, layer.hidden_size = 96, 128, 7168
        layer.projection_size = layer.local_projection_size = 96 * 128
        layer.conv_size, layer.gate_lower_bound = 4, -5.0
        layer.recoverssm_records, layer.use_recoverssm = None, True
        layer.o_norm = torch.nn.Module()
        layer.o_proj = torch.nn.Module()
        layer.conv1d = torch.nn.Module()
        layer.conv1d.weight = torch.empty(0, device="meta")
        layer.conv1d.quant_method = Mock()

    group = SimpleNamespace(world_size=8)
    quant = Mock(spec=kimi_kda.AscendW8A8MXFP8DynamicLinearMethod)
    quant.group_size, quant.dynamic_mx_quant_scale_alg = 32, 0
    qkv = torch.nn.Module()
    qkv.quant_method = SimpleNamespace(quant_method=quant)
    factory = Mock()
    monkeypatch.setattr(kimi_kda.KimiK3DeltaAttention, "__init__", base_init)
    monkeypatch.setattr(kimi_kda, "get_qkv_iproj_tp_group", lambda: group)
    monkeypatch.setattr(kimi_kda, "get_gproj_tp_group", lambda: group)
    monkeypatch.setattr(kimi_kda, "HeadParallelMergedColumnLinear", lambda *a, **kw: qkv)
    monkeypatch.setattr(kimi_kda, "HeadParallelColumnLinear", lambda *a, **kw: torch.nn.Module())
    monkeypatch.setattr(adapter, "LocalTP8Verify", factory)
    backend = KDABackendConfig(backend=selector, operator_package_dir=str(tmp_path))
    monkeypatch.setattr(kimi_kda, "get_ascend_config", lambda: SimpleNamespace(kda_backend_config=backend))
    sizes = {f"{name}_tensor_parallel_size": 8 for name in ("qkv_iproj", "gproj", "oproj")}
    vc = SimpleNamespace(
        additional_config={"finegrained_tp_config": sizes},
        quant_config=SimpleNamespace(uses_kimi_k3_mixed_kda_projection=lambda prefix: True),
        num_speculative_tokens=7,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    config = SimpleNamespace(linear_attn_config={"use_full_rank_gate": True}, rms_norm_eps=1e-6)
    layer = kimi_kda.AscendKimiK3DeltaAttention(config, vc, "layer")
    factory.assert_called_once_with(str(tmp_path))
    assert layer._localtp8_verify is factory.return_value
    assert layer.qkv_layout.heads == layer.gate_layout.heads == 12
    assert qkv._prepare_localtp8_kda
    dispatch = Mock()
    monkeypatch.setattr(torch.ops.vllm, "kimi_localtp8_kda_forward", dispatch)
    hidden = torch.empty(32, 7168, dtype=torch.bfloat16)
    output = layer.forward(hidden, torch.arange(32))
    dispatch.assert_called_once_with(hidden, output, "layer")
    sizes["oproj_tensor_parallel_size"] = 1
    with pytest.raises(ValueError, match="QKV/G/O localTP8"):
        kimi_kda.AscendKimiK3DeltaAttention(config, vc, "layer")
