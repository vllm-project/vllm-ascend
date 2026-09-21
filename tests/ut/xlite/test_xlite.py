# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.utils import enable_dsa_cp

xlite_adapter = pytest.importorskip("vllm_ascend.xlite.xlite")


@pytest.mark.parametrize("enabled", [False, True])
def test_xlite_uses_native_dsa_weight_layout(enabled):
    config = SimpleNamespace(enable_dsa_cp=enabled, xlite_graph_config=SimpleNamespace(enabled=True))
    enable_dsa_cp.cache_clear()
    try:
        with patch("vllm_ascend.ascend_config.get_ascend_config", return_value=config):
            assert enable_dsa_cp() is enabled
    finally:
        enable_dsa_cp.cache_clear()


def build_dsa_config(enabled, *, supports_dsa_cp=True, pcp=1, dcp=1):
    model = object.__new__(xlite_adapter.DeepseekV32XliteModel)
    model.xlite_config = SimpleNamespace(**({"enable_dsa_cp": False} if supports_dsa_cp else {}))
    hf_config = SimpleNamespace(index_head_dim=128, index_n_heads=32, index_topk=2048, indexer_rope_interleave=True)
    model.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp, decode_context_parallel_size=dcp),
        model_config=SimpleNamespace(hf_text_config=hf_config),
    )
    with (
        patch.object(xlite_adapter.DeepseekV3XliteModel, "_build_model_config"),
        patch(
            "vllm_ascend.xlite.xlite.get_ascend_config",
            return_value=SimpleNamespace(enable_dsa_cp=enabled),
        ),
    ):
        model._build_model_config()
    return model.xlite_config


@pytest.mark.parametrize("enabled", [False, True])
def test_dsa_config_mapping(enabled):
    assert build_dsa_config(enabled).enable_dsa_cp is enabled


def test_missing_dsa_cp_capability():
    build_dsa_config(False, supports_dsa_cp=False)
    with pytest.raises(RuntimeError, match="DSACP configuration"):
        build_dsa_config(True, supports_dsa_cp=False)


@pytest.mark.parametrize("pcp,dcp", [(2, 1), (1, 2)])
def test_dsa_cp_rejects_sharded_cache_topologies(pcp, dcp):
    with pytest.raises(ValueError, match="PCP or DCP"):
        build_dsa_config(True, pcp=pcp, dcp=dcp)
