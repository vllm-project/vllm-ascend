from types import SimpleNamespace

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _runner_with_draft_config(**hf_config_fields):
    runner = object.__new__(NPUModelRunner)
    runner.speculative_config = SimpleNamespace(
        use_dspark=lambda: True, draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(**hf_config_fields))
    )
    return runner


def test_dspark_aux_layers_prefer_capture_layer_ids():
    runner = _runner_with_draft_config(
        eagle_aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
        target_layer_ids=[7, 22, 38, 54, 69],
    )

    assert runner._get_eagle3_aux_layers_from_config() == (8, 23, 39, 55, 70)


def test_dspark_aux_layers_accept_speculators_field():
    runner = _runner_with_draft_config(
        aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
    )

    assert runner._get_eagle3_aux_layers_from_config() == (8, 23, 39, 55, 70)


def test_dspark_aux_layers_convert_legacy_target_layer_ids():
    runner = _runner_with_draft_config(
        target_layer_ids=[7, 22, 38, 54, 69],
    )

    assert runner._get_eagle3_aux_layers_from_config() == (8, 23, 39, 55, 70)


def test_dspark_aux_layers_missing_config_returns_none():
    runner = _runner_with_draft_config()

    assert runner._get_eagle3_aux_layers_from_config() is None
