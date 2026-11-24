# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.attention import mega_dsa


def test_disabled_does_not_import_operator(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MEGA_DSA", "0")
    assert mega_dsa.prepare_mega_dsa(None) is None


@pytest.mark.parametrize("sharded", [(), ("qkv_iproj",), ("oproj",), ("gproj",), ("oproj", "gproj")])
@pytest.mark.parametrize("has_sharded_package", [False, True])
def test_localtp_projection_selection(monkeypatch, sharded, has_sharded_package):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MEGA_DSA", "1")
    pc = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    fine = SimpleNamespace(
        shared_expert_tensor_parallel_size=8,
        mlp_tensor_parallel_size=8,
        lmhead_tensor_parallel_size=8,
        embedding_tensor_parallel_size=8,
    )
    for name in sharded:
        setattr(fine, name + "_tensor_parallel_size", 8)
    monkeypatch.setattr(mega_dsa, "get_ascend_config", lambda: SimpleNamespace(finegrained_tp_config=fine))
    monkeypatch.setattr(
        mega_dsa,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=mega_dsa.DeviceAdaptorFamily.FP8_OPTIMIZED),
    )
    prepare = Mock()
    gate_group, output_group = SimpleNamespace(world_size=8), SimpleNamespace(world_size=8)
    from vllm_ascend.distributed import parallel_state

    monkeypatch.setattr(parallel_state, "get_otp_group", lambda: output_group)
    package = SimpleNamespace(prepare=prepare)
    prepare_sharded = Mock()
    if has_sharded_package:
        package.prepare_localtp = prepare_sharded
    monkeypatch.setitem(sys.modules, "megadsa", package)
    layer = SimpleNamespace(
        is_kimi_k31=True,
        _is_mtp_layer=False,
        tp_size=1,
        vllm_config=SimpleNamespace(parallel_config=pc, model_config=SimpleNamespace()),
        use_rope=False,
        use_index_cache=False,
        rl_weight_update_enabled=False,
        enable_sparse_sfa_c8=True,
        runtime_has_indexer=True,
        g_proj=SimpleNamespace(tp_group=gate_group),
        o_proj=SimpleNamespace(weight=SimpleNamespace(shape=(7168, 1536))),
        begin_gate=Mock(),
        weight_dq_scale=object(),
        local_num_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        indexer=SimpleNamespace(
            enable_sparse_li_c8=True, qk_rope_head_dim=0, topk_tokens=2048, n_head=32, head_dim=128
        ),
    )
    result = mega_dsa.prepare_mega_dsa(layer)
    if not sharded:
        prepare.assert_called_once_with(layer)
        assert result is prepare.return_value
    elif has_sharded_package and "qkv_iproj" not in sharded:
        assert result is prepare_sharded.return_value
        kwargs = prepare_sharded.call_args.kwargs
        assert kwargs["gate_group"] is (gate_group if "gproj" in sharded else None)
        assert kwargs["output_group"] is (output_group if "oproj" in sharded else None)
        assert callable(kwargs["capacity"])
        prepare.assert_not_called()
    else:
        prepare_sharded.assert_not_called()
        prepare.assert_not_called()
        assert result is None
