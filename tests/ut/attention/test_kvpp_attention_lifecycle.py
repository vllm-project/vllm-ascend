import inspect

from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def _ordered(source: str, *markers: str) -> None:
    positions = [source.index(marker) for marker in markers]
    assert positions == sorted(positions), dict(zip(markers, positions))


def test_attn_01_mla_kvpp_lifecycle_wraps_cache_consumption():
    forward = inspect.getsource(AscendMLAImpl.forward)
    preprocess = inspect.getsource(AscendMLAImpl._mla_preprocess)

    assert ".enter_layer(" not in forward
    assert ".leave_layer(" not in forward
    _ordered(
        forward,
        "self._mla_preprocess(",
        "self._forward_decode(",
    )
    _ordered(
        preprocess,
        "self.fused_qkv_a_proj(hidden_states)",
        ".wait_for_layer(layer_name)",
        "self.mla_preprocess_decode(",
    )


def test_sfa_01_main_and_indexer_share_one_kvpp_lifecycle():
    source = inspect.getsource(AscendSFAImpl.forward)

    assert ".enter_layer(" not in source
    assert ".leave_layer(" not in source
    assert source.count(".wait_for_layer(layer_name)") == 2
    _ordered(
        source,
        "self.indexer_select_pre_process(",
        ".wait_for_layer(layer_name)",
        "self.exec_kv(",
        "self._execute_sparse_flash_attention_process(",
    )
