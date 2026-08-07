from types import SimpleNamespace

from vllm_ascend.compilation.acl_graph import get_graph_param_key
from vllm_ascend.compilation.compiler_interface import _compute_decode_cudagraph_batch_sizes


def _config(*, use_v2_model_runner: bool, has_full_cudagraphs: bool):
    speculative_config = SimpleNamespace(
        num_speculative_tokens=3,
        num_speculative_tokens_per_batch_size=[
            (1, 2, 1),
            (3, 3, 2),
            (4, 4, 3),
        ],
        uses_dynamic_speculative_decoding=lambda: True,
    )
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 5, 8],
            max_cudagraph_capture_size=8,
            cudagraph_mode=SimpleNamespace(
                has_full_cudagraphs=lambda: has_full_cudagraphs,
            ),
        ),
    )


def test_static_kernel_shapes_cover_dynamic_sd_full_graph_sizes():
    config = _config(use_v2_model_runner=True, has_full_cudagraphs=True)

    assert _compute_decode_cudagraph_batch_sizes(config) == [2, 3, 4, 6, 8]


def test_dynamic_sd_static_kernel_shapes_unchanged_without_full_graph():
    config = _config(use_v2_model_runner=True, has_full_cudagraphs=False)

    assert _compute_decode_cudagraph_batch_sizes(config) == [4, 5, 8]


def test_dynamic_sd_static_kernel_shapes_unchanged_for_model_runner_v1():
    config = _config(use_v2_model_runner=False, has_full_cudagraphs=True)

    assert _compute_decode_cudagraph_batch_sizes(config) == [4, 5, 8]


def test_dynamic_sd_graph_params_distinguish_same_token_count():
    speculative_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: True)
    short_query_metadata = {
        "layer": SimpleNamespace(actual_seq_lengths_q=[2, 4, 6, 8]),
    }
    long_query_metadata = {
        "layer": SimpleNamespace(actual_seq_lengths_q=[4, 8]),
    }

    assert get_graph_param_key(8, speculative_config, short_query_metadata, is_draft_model=False) == (8, 2)
    assert get_graph_param_key(8, speculative_config, long_query_metadata, is_draft_model=False) == (8, 4)


def test_graph_param_key_unchanged_outside_dynamic_target_graph():
    static_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: False)
    dynamic_config = SimpleNamespace(uses_dynamic_speculative_decoding=lambda: True)
    metadata = {"layer": SimpleNamespace(actual_seq_lengths_q=[2, 4])}

    assert get_graph_param_key(4, static_config, metadata, is_draft_model=False) == 4
    assert get_graph_param_key(4, dynamic_config, metadata, is_draft_model=True) == 4
