from unittest.mock import patch

from vllm_ascend.ops.triton.v2 import apply_grammar_bitmask as grammar


def test_empty_grammar_bitmask_grid_skips_kernel_launch():
    with (
        patch.object(grammar, "init_device_properties_triton"),
        patch.object(grammar, "get_vectorcore_num", return_value=48),
    ):
        launch = grammar._apply_grammar_bitmask_kernel[(0, 1)]
        assert launch(None, 0, None, None, None, 0, 0, 1, 8192) is None


def test_nonempty_grammar_bitmask_grid_returns_launcher():
    with (
        patch.object(grammar, "init_device_properties_triton"),
        patch.object(grammar, "get_vectorcore_num", return_value=48),
    ):
        launch = grammar._apply_grammar_bitmask_kernel[(4, 19)]
        assert callable(launch)
