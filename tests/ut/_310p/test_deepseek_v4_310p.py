# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_ascend._310p.deepseek_v4 import (
    DSA_BACKEND_310P,
    get_dsv4_310p_backend,
    is_deepseek_v4_model,
    validate_dsv4_310p_topology,
)
from vllm_ascend.models.deepseek_v4_dspark import DSparkDeepseekV4ForCausalLM


def _model_config(model_type: str, o_groups: int = 8):
    return SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type, o_groups=o_groups))


def test_dsv4_310p_backend_is_selected_automatically() -> None:
    assert (
        get_dsv4_310p_backend(
            model_config=_model_config("deepseek_v4"),
            tensor_parallel_size=8,
            use_mla=True,
            use_sparse=False,
            use_compress=True,
        )
        == DSA_BACKEND_310P
    )
    assert (
        get_dsv4_310p_backend(
            model_config=_model_config("deepseek_v4"),
            tensor_parallel_size=8,
            use_mla=True,
            use_sparse=False,
            use_compress=False,
        )
        == DSA_BACKEND_310P
    )


def test_dsv4_310p_backend_does_not_override_other_attention_or_models() -> None:
    assert (
        get_dsv4_310p_backend(
            model_config=_model_config("deepseek_v4"),
            tensor_parallel_size=8,
            use_mla=True,
            use_sparse=True,
            use_compress=False,
        )
        is None
    )
    assert (
        get_dsv4_310p_backend(
            model_config=_model_config("deepseek_v4"),
            tensor_parallel_size=8,
            use_mla=False,
            use_sparse=False,
            use_compress=False,
        )
        is None
    )
    assert (
        get_dsv4_310p_backend(
            model_config=_model_config("deepseek_v3"),
            tensor_parallel_size=8,
            use_mla=True,
            use_sparse=False,
            use_compress=False,
        )
        is None
    )


def test_dsv4_310p_topology_fails_before_weight_loading() -> None:
    validate_dsv4_310p_topology(_model_config("deepseek_v4", o_groups=8), tensor_parallel_size=8)

    with pytest.raises(ValueError, match=r"tensor_parallel_size.*o_groups \(8\).*got 4"):
        validate_dsv4_310p_topology(_model_config("deepseek_v4", o_groups=8), tensor_parallel_size=4)

    validate_dsv4_310p_topology(_model_config("qwen3", o_groups=8), tensor_parallel_size=4)


def test_deepseek_v4_model_detection_is_version_tolerant() -> None:
    assert is_deepseek_v4_model(_model_config("deepseek_v4"))
    assert not is_deepseek_v4_model(_model_config("qwen3"))
    assert not is_deepseek_v4_model(SimpleNamespace())


def test_deepseek_v4_dspark_reports_non_causal_attention() -> None:
    model = SimpleNamespace(model=SimpleNamespace(layers={"0": object(), "1": object()}))
    assert DSparkDeepseekV4ForCausalLM.get_draft_attn_causal(model) == [False, False]
