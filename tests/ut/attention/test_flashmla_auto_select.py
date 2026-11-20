# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.flashmla import prepare_flashmla_batch_req_state, split_flashmla_requests
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware_profile import HardwareCapability


def test_flashmla_selection_requires_operator_and_supported_mla_config(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(block_size=128),
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1, cp_kv_cache_interleave_size=128),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="deepseek_v3")),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
    )
    monkeypatch.setattr(
        mla_v1,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: capability is HardwareCapability.MLA_FLASH),
    )
    monkeypatch.setattr(mla_v1, "supports_component_major_mla_pd", lambda _config: True)
    monkeypatch.setattr(mla_v1, "enable_sfa", lambda _config: False)
    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))

    assert AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: None)
    assert not AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))
    config.use_v2_model_runner = False
    assert not AscendMLAImpl._can_use_flashmla(impl)

    config.use_v2_model_runner = True
    impl.num_heads = 48
    assert not AscendMLAImpl._can_use_flashmla(impl)

    impl.num_heads = 8
    config.parallel_config.decode_context_parallel_size = 2
    assert not AscendMLAImpl._can_use_flashmla(impl)

    config.parallel_config.decode_context_parallel_size = 8
    assert AscendMLAImpl._can_use_flashmla(impl)
    impl.num_heads = 12
    assert AscendMLAImpl._can_use_flashmla(impl)
    config.parallel_config.decode_context_parallel_size = 4
    assert not AscendMLAImpl._can_use_flashmla(impl)


@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize(
    "model_type, expected_flashmla",
    [
        ("kimi_k31", False),
        ("glm4_moe", True),
        ("glm4_moe_lite", True),
        ("deepseek_v3", True),
        ("kimi_k3", True),
        ("kimi_linear", True),
    ],
)
def test_flashmla_dspark_target_model_gate(monkeypatch, dcp_size, model_type, expected_flashmla):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=SimpleNamespace(method="dspark"),
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size, cp_kv_cache_interleave_size=128),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            hf_text_config=SimpleNamespace(
                model_type="kimi_linear", **({"index_topk": 2048} if model_type == "kimi_k31" else {})
            ),
        ),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        pcp_enabled=False,
    )
    monkeypatch.setattr(mla_v1, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    monkeypatch.setattr(mla_v1, "supports_component_major_mla_pd", lambda _: True)
    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (object(), object()))
    assert AscendMLAImpl._can_use_flashmla(impl) is expected_flashmla
    config.speculative_config.method = "eagle"
    assert not AscendMLAImpl._can_use_flashmla(impl)


@pytest.mark.parametrize("prefilling", [[True, False, True], [False, False, False]])
def test_request_order_preserves_fields_and_real_prefill_graph_gate(prefilling):
    BatchReqState = namedtuple(
        "BatchReqState",
        [
            "req_ids",
            "num_scheduled_tokens",
            "idx_mapping_np",
            "prefill_len_np",
            "num_computed_prefill_tokens_np",
            "is_prefilling_np",
            "has_prefill",
            "num_tokens",
        ],
    )
    state = BatchReqState(
        ["a", "b", "c"],
        np.array([1, 1, 1]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300]),
        np.array([11, 22, 33]),
        np.array(prefilling),
        any(prefilling),
        3,
    )
    result, uniform = prepare_flashmla_batch_req_state(state, 1)
    order = [1, 0, 2] if state.has_prefill else [0, 1, 2]
    assert result.req_ids == [state.req_ids[index] for index in order]
    for field in state._fields:
        expected = getattr(state, field)
        actual = getattr(result, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected[order])
        elif field != "req_ids":
            assert actual == expected
    assert uniform == (None if state.has_prefill else 1)
    common = SimpleNamespace(
        is_prefilling=torch.from_numpy(result.is_prefilling_np),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=3,
    )
    # A one-token prompt stays FIA prefill; trailing empty graph rows are ignored.
    assert split_flashmla_requests(common) == ((1, 2, 1, 2) if state.has_prefill else (3, 0, 3, 0))


def test_request_split_rejects_decode_after_real_prefill():
    common = SimpleNamespace(
        is_prefilling=torch.tensor([True, False]),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
    )
    with pytest.raises(RuntimeError, match="decode requests before prefill"):
        split_flashmla_requests(common)
