# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.attention import mega_dsa


def test_disabled_does_not_import_operator(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MEGA_DSA", "0")
    assert mega_dsa.prepare_mega_dsa(None) is None


@pytest.mark.parametrize("sharded", [None, "qkv_iproj", "oproj", "gproj"])
def test_localtp_projection_selection(monkeypatch, sharded):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_MEGA_DSA", "1")
    pc = SimpleNamespace(decode_context_parallel_size=1, prefill_context_parallel_size=1)
    fine = SimpleNamespace(
        shared_expert_tensor_parallel_size=8,
        mlp_tensor_parallel_size=8,
        lmhead_tensor_parallel_size=8,
        embedding_tensor_parallel_size=8,
    )
    if sharded is not None:
        setattr(fine, sharded + "_tensor_parallel_size", 8)
    monkeypatch.setattr(mega_dsa, "get_ascend_config", lambda: SimpleNamespace(finegrained_tp_config=fine))
    monkeypatch.setattr(
        mega_dsa,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=mega_dsa.DeviceAdaptorFamily.FP8_OPTIMIZED),
    )
    prepare = Mock()
    monkeypatch.setitem(sys.modules, "megadsa", SimpleNamespace(prepare=prepare))
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
        g_proj=object(),
        weight_dq_scale=object(),
        local_num_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        indexer=SimpleNamespace(
            enable_sparse_li_c8=True, qk_rope_head_dim=0, topk_tokens=2048, n_head=32, head_dim=128
        ),
    )
    result = mega_dsa.prepare_mega_dsa(layer)
    if sharded is None:
        prepare.assert_called_once_with(layer)
        assert result is prepare.return_value
    else:
        prepare.assert_not_called()
        assert result is None
