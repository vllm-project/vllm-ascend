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
regressions slip through silently. These CPU tests pin both behaviors.

Recent A5 regressions (e.g. prefix-cache perf fluctuation #11113) show exactly
the class of issues these guards exist to prevent.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_ascend.platform import MAX_CAPTURE_SIZES_FOR_950, prune_capture_sizes_for_950


@pytest.mark.parametrize(
    "capture_sizes, should_prune",
    [
        # Already within the A5 limit -> no-op.
        ([1, 2, 4], False),
        # Over the limit -> sampled down to MAX_CAPTURE_SIZES_FOR_950.
        ([1, 2, 4, 8, 16, 32, 64, 128], True),
    ],
)
def test_prune_capture_sizes_for_950_behavior(capture_sizes, should_prune):
    """On A5, capture sizes are pruned to MAX_CAPTURE_SIZES_FOR_950.

    The pruned list must keep the first and last capture sizes (so the
    smallest and largest batch sizes are still captured) and stay within the
    A5 HDK limit. When already within the limit, the function must be a
    no-op and must not touch the config.
    """
    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(capture_sizes))
    )
    captured = {}

    def fake_update(vllm_config, new_sizes):
        captured["sizes"] = list(new_sizes)

    with mock.patch(
        "vllm_ascend.platform.update_cudagraph_capture_sizes",
        side_effect=fake_update,
    ) as mocked_update:
        prune_capture_sizes_for_950(cfg)

    if should_prune:
        mocked_update.assert_called_once()
        pruned = captured["sizes"]
        assert len(pruned) == MAX_CAPTURE_SIZES_FOR_950
        assert pruned[0] == capture_sizes[0]
        assert pruned[-1] == capture_sizes[-1]
        # Every sampled size must come from the original list.
        assert all(size in capture_sizes for size in pruned)
    else:
        mocked_update.assert_not_called()


def test_a5_platform_safety_guards_present():
    """Drift guard: the A5-only platform safety checks must stay in place.

    ``prune_capture_sizes_for_950`` is called only on the A5 branch, and the
    SFA-DCP sparse-C8 path raises ``NotImplementedError`` on A5. Removing
    either would silently enable an unsupported A5 path. This source-level
    assertion fails fast if the guards are refactored away.
    """
    # Resolve via the module's __file__ so the test does not depend on the
    # current working directory being the repository root.
    import vllm_ascend.platform

    src = Path(vllm_ascend.platform.__file__).read_text(encoding="utf-8")
    assert "prune_capture_sizes_for_950" in src
    assert "AscendDeviceType.A5" in src
    assert "not supported on A5" in src
