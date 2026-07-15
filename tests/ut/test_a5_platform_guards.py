#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Guards for A5 (Ascend 950) platform-level safety behavior.

A5 HDK has an ACL graph capture-size incompatibility (see the TODO in
``vllm_ascend/platform.py``) and does not support SFA DCP with sparse C8 yet.
Both guards live in ``vllm_ascend/platform.py`` and are A5-only; because there
is no A5 CI runner yet, removing them during refactoring would let A5
regressions slip through silently.

The capture-size guard (``prune_capture_sizes_for_950``) is a standalone
function and is verified behaviorally here, including the empty / at-limit /
over-limit boundaries. The SFA-DCP guard is an inline ``raise`` inside
``NPUPlatform.check_and_update_config``; rather than grepping source text (which
would pass even for dead code / comments), we drive the real code path with a
mocked config so the ``NotImplementedError`` must actually fire on A5.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_ascend.platform import (
    MAX_CAPTURE_SIZES_FOR_950,
    NPUPlatform,
    prune_capture_sizes_for_950,
)
from vllm_ascend.utils import AscendDeviceType

# ---------------------------------------------------------------------------
# prune_capture_sizes_for_950 — behavior + boundary coverage
# ---------------------------------------------------------------------------


def _cfg(capture_sizes):
    """Build a minimal config object accepted by ``prune_capture_sizes_for_950``."""
    return SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(capture_sizes)))


def test_prune_is_noop_when_capture_sizes_empty():
    """An empty capture-size list must short-circuit without touching config."""
    cfg = _cfg([])
    with mock.patch("vllm_ascend.platform.update_cudagraph_capture_sizes") as mocked_update:
        prune_capture_sizes_for_950(cfg)
    mocked_update.assert_not_called()


def test_prune_is_noop_at_exact_limit():
    """``len == MAX_CAPTURE_SIZES_FOR_950`` hits the ``<=`` boundary -> no-op.

    This is the most fragile off-by-one edge: one element more would prune.
    """
    sizes = list(range(1, MAX_CAPTURE_SIZES_FOR_950 + 1))  # len == limit
    cfg = _cfg(sizes)
    with mock.patch("vllm_ascend.platform.update_cudagraph_capture_sizes") as mocked_update:
        prune_capture_sizes_for_950(cfg)
    mocked_update.assert_not_called()
    # The original list is left untouched.
    assert cfg.compilation_config.cudagraph_capture_sizes == sizes


def test_prune_is_noop_below_limit():
    """``len < limit`` is a no-op."""
    cfg = _cfg([1, 2, 4])  # len 3 < 4
    with mock.patch("vllm_ascend.platform.update_cudagraph_capture_sizes") as mocked_update:
        prune_capture_sizes_for_950(cfg)
    mocked_update.assert_not_called()


def test_prune_samples_down_to_limit_preserving_endpoints():
    """Over the limit, output keeps endpoints, stays ascending, all from source."""
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]  # len 8 > 4
    cfg = _cfg(sizes)
    captured = {}

    def fake_update(vllm_config, new_sizes):
        captured["sizes"] = list(new_sizes)

    with mock.patch(
        "vllm_ascend.platform.update_cudagraph_capture_sizes",
        side_effect=fake_update,
    ) as mocked_update:
        prune_capture_sizes_for_950(cfg)

    mocked_update.assert_called_once()
    pruned = captured["sizes"]
    # Cardinality contract.
    assert len(pruned) == MAX_CAPTURE_SIZES_FOR_950
    # Endpoints preserved so the smallest / largest batch sizes stay captured.
    assert pruned[0] == sizes[0]
    assert pruned[-1] == sizes[-1]
    # Sampling without replacement, preserving ascending order.
    assert pruned == sorted(pruned)
    assert len(set(pruned)) == len(pruned)
    assert set(pruned).issubset(set(sizes))


# ---------------------------------------------------------------------------
# SFA-DCP sparse-C8 guard — behavior-level (replaces source string grep)
# ---------------------------------------------------------------------------


def _sfa_dcp_vllm_config():
    """A config that routes ``check_and_update_config`` to the SFA-DCP block.

    Concrete integers are used wherever the platform code does arithmetic or
    comparisons on the parallel config (``> 1``, ``*``, ``!=``), so the heavy
    branches are skipped deterministically and execution reaches the A5 raise.
    """
    vllm_config = mock.MagicMock()
    # Skip the device-type early return.
    vllm_config.device_config = None
    vllm_config.model_config = mock.MagicMock()
    vllm_config.model_config.enforce_eager = True
    vllm_config.model_config.is_encoder_decoder = False
    vllm_config.model_config.enable_sleep_mode = True
    vllm_config.model_config.hf_text_config = SimpleNamespace()  # no index_topk
    vllm_config.kv_transfer_config = None
    vllm_config.speculative_config = None
    vllm_config.additional_config = {"enable_sparse_c8": True}

    pc = vllm_config.parallel_config
    pc.tensor_parallel_size = 1
    pc.decode_context_parallel_size = 1
    pc.prefill_context_parallel_size = 1
    pc.worker_cls = "NPUWorker"

    from vllm.config.compilation import CUDAGraphMode

    cc = vllm_config.compilation_config
    cc.cudagraph_mode = CUDAGraphMode.NONE
    return vllm_config


def _sfa_dcp_ascend_config():
    """A mock AscendConfig whose optional branches all stay disabled."""
    ascend_config = mock.MagicMock()
    # Skip the additional_config setdefault/update branches.
    ascend_config.ascend_fusion_config = None
    # xlite_graph_config.enabled must be falsy to avoid overwriting cudagraph_mode.
    ascend_config.xlite_graph_config.enabled = False
    ascend_config.enable_balance_scheduling = False
    ascend_config.short_request_first_config.enabled = False
    ascend_config.recompute_scheduler_enable = False
    ascend_config.SLO_limits_for_dynamic_batch = -1  # concrete int (compared != -1)
    ascend_config.profiling_chunk_config.enabled = False
    return ascend_config


def test_sfa_dcp_sparse_c8_raises_not_implemented_on_a5():
    """A5 + SFA-DCP replicated indexer + sparse C8 must raise NotImplementedError.

    This drives the real ``NPUPlatform.check_and_update_config`` path (heavy
    dependencies mocked) so the guard must actually fire on A5. Deleting or
    bypassing the inline ``raise`` makes this test fail (no NotImplementedError),
    which is the regression signal a bare string-grep could not give.
    """
    vllm_config = _sfa_dcp_vllm_config()

    with (
        mock.patch(
            "vllm_ascend.platform.init_ascend_config",
            return_value=_sfa_dcp_ascend_config(),
        ),
        mock.patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization"),
        mock.patch.object(NPUPlatform, "_validate_layer_sharding_config"),
        mock.patch.object(NPUPlatform, "_validate_draft_decode_context_parallel_config"),
        mock.patch.object(NPUPlatform, "_validate_parallel_config"),
        mock.patch.object(NPUPlatform, "_fix_incompatible_config"),
        mock.patch.object(NPUPlatform, "_validate_kv_load_failure_policy"),
        mock.patch("vllm_ascend.logger.configure_ascend_file_logging"),
        mock.patch("vllm_ascend.logger.configure_ascend_logging"),
        mock.patch("vllm_ascend.platform.refresh_block_size"),
        mock.patch("vllm_ascend.platform.model_uses_sfa_sparse", return_value=True),
        mock.patch(
            "vllm_ascend.platform.enable_sfa_dcp_replicated_indexer",
            return_value=True,
        ),
        mock.patch(
            "vllm_ascend.platform.get_ascend_device_type",
            return_value=AscendDeviceType.A5,
        ),
        mock.patch("vllm_ascend.platform.enable_sp", return_value=False),
        pytest.raises(NotImplementedError, match="not supported on A5"),
    ):
        NPUPlatform.check_and_update_config(vllm_config)
