from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vllm_ascend.utils import get_compressed_pos_and_indices

REPO_ROOT = Path(__file__).resolve().parents[3]


def _kv_cache_group(compress_ratio: int):
    return SimpleNamespace(kv_cache_spec=SimpleNamespace(compress_ratio=compress_ratio))


def test_compressed_positions_depend_on_corrected_num_computed_tokens():
    scheduled_tokens = np.array([1], dtype=np.int32)
    request_indices = np.arange(1, dtype=np.int32)
    kv_cache_groups = [_kv_cache_group(compress_ratio=4)]

    optimistic_positions, _, optimistic_lengths = get_compressed_pos_and_indices(
        np.array([4], dtype=np.int32),
        scheduled_tokens,
        request_indices,
        use_compress=True,
        kv_cache_groups=kv_cache_groups,
    )
    corrected_positions, corrected_req_indices, corrected_lengths = get_compressed_pos_and_indices(
        np.array([3], dtype=np.int32),
        scheduled_tokens,
        request_indices,
        use_compress=True,
        kv_cache_groups=kv_cache_groups,
    )

    np.testing.assert_array_equal(optimistic_positions[0], np.array([], dtype=np.int64))
    np.testing.assert_array_equal(optimistic_lengths[0], np.array([0]))
    np.testing.assert_array_equal(corrected_positions[0], np.array([0]))
    np.testing.assert_array_equal(corrected_req_indices[0], np.array([0]))
    np.testing.assert_array_equal(corrected_lengths[0], np.array([1]))


def test_model_runner_builds_compressed_positions_after_prepare_inputs():
    model_runner = (REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text()
    prepare_inputs = model_runner.split("    def _prepare_inputs(", 1)[1].split(
        "    def _rebuild_input_ids_with_corrected_positions(", 1
    )[0]

    # The compressed helper needs corrected async-spec CPU counters. Keep it
    # out of _prepare_inputs so that path does not force an early NPU wait.
    assert "get_compressed_pos_and_indices" not in prepare_inputs
    assert "deferred async-spec corrections" in model_runner
    assert "get_compressed_pos_and_indices" in model_runner
