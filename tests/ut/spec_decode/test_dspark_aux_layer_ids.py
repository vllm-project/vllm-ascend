# SPDX-License-Identifier: Apache-2.0
"""Shared DSpark auxiliary boundaries preserve upstream checkpoint semantics."""

from types import SimpleNamespace

import pytest

from vllm_ascend.spec_decode.dspark_utils import get_dspark_aux_layer_ids


def _config(hf_config, *, explicit_ids=None):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="dspark", draft_model_config=SimpleNamespace(hf_config=hf_config)),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=78)),
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"dspark_aux_hidden_state_layer_ids": explicit_ids}
        ),
    )


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 22, 38, 58, 74]),
        SimpleNamespace(dspark_target_layer_ids=[1, 21, 37, 57, 73], model_type="glm5"),
        SimpleNamespace(dspark_target_layer_ids=[2, 22, 38, 58, 74], model_type="deepseek_v41"),
        SimpleNamespace(target_layer_ids=[1, 21, 37, 57, 73]),
        SimpleNamespace(dflash_config={"target_layer_ids": [1, 21, 37, 57, 73]}),
        SimpleNamespace(eagle_config={"layer_ids": [2, 22, 38, 58, 74]}),
    ],
)
def test_checkpoint_resolution_preserves_upstream_capture_boundary_semantics(hf_config):
    assert get_dspark_aux_layer_ids(_config(hf_config)) == (2, 22, 38, 58, 74)


def test_checkpoint_schema_takes_precedence_over_connector_capture_option():
    config = _config(SimpleNamespace(dspark_target_layer_ids=[1, 21], model_type="glm5"), explicit_ids=[0, 78])
    assert get_dspark_aux_layer_ids(config) == (2, 22)


def test_missing_checkpoint_schema_does_not_fall_back_to_p_capture_option():
    assert get_dspark_aux_layer_ids(_config(SimpleNamespace(), explicit_ids=[0, 78])) == ()


@pytest.mark.parametrize("ids", [[22, 2], [2, 2], [79], [-1], [True]])
def test_checkpoint_boundaries_are_validated_against_target_model(ids):
    with pytest.raises(ValueError, match="ordered unique target-layer boundaries"):
        get_dspark_aux_layer_ids(_config(SimpleNamespace(eagle_aux_hidden_state_layer_ids=ids)))


def test_explicit_p_capture_boundaries_are_not_incremented_again():
    config = _config(SimpleNamespace(), explicit_ids=[2, 22])
    config.speculative_config = None
    assert get_dspark_aux_layer_ids(config) == (2, 22)


def test_non_dspark_draft_does_not_activate_checkpoint_resolution():
    config = _config(SimpleNamespace(eagle_aux_hidden_state_layer_ids=[2, 22]))
    config.speculative_config.method = "mtp"
    assert get_dspark_aux_layer_ids(config) == ()


def test_shared_layer_lookup_is_independent_of_pd_and_execution_constraints():
    config = _config(SimpleNamespace(), explicit_ids=[2, 22])
    config.speculative_config = None
    config.kv_transfer_config.is_kv_producer = False
    config.kv_transfer_config.is_kv_consumer = True
    config.model_config.enforce_eager = False
    config.parallel_config = SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=2)
    config.cache_config = SimpleNamespace(enable_prefix_caching=True)
    assert get_dspark_aux_layer_ids(config) == (2, 22)
