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
"""Unit tests for the A5 (Ascend 950) platform guards.

This UT layer keeps only what is genuinely unit-testable on CPU:

* ``prune_capture_sizes_for_950`` -- a pure function with no A5-runtime
  dependency inside; its pruning contract (empty / at-limit / over-limit) is
  verified behaviorally here.
* An AST structural drift guard for the inline SFA-DCP ``NotImplementedError``
  that lives inside ``NPUPlatform.check_and_update_config``. That raise is an
  A5-runtime behavior, so its *firing* is validated by the e2e single-card A5
  suite, not mocked here. This UT only asserts the raise stays *wired to the
  A5 branch* in source -- a text grep cannot prove that (it would pass for a
  comment or dead code), but an AST walk can.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from vllm_ascend.platform import (
    MAX_CAPTURE_SIZES_FOR_950,
    prune_capture_sizes_for_950,
)

# ---------------------------------------------------------------------------
# prune_capture_sizes_for_950 -- pure-function behavior + boundaries
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
    assert len(pruned) == MAX_CAPTURE_SIZES_FOR_950
    assert pruned[0] == sizes[0]
    assert pruned[-1] == sizes[-1]
    assert pruned == sorted(pruned)
    assert len(set(pruned)) == len(pruned)
    assert set(pruned).issubset(set(sizes))


# ---------------------------------------------------------------------------
# SFA-DCP A5 raise -- AST structural drift guard
# ---------------------------------------------------------------------------


def _raised_exception_name(exc) -> str | None:
    """Return the exception class name raised by an ``ast.Raise`` exc node."""
    node = exc
    # raise X(...) -> exc is a Call; unwrap to the callable
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    return None


def _references_a5(node) -> bool:
    """True if an AST node's text references ``AscendDeviceType.A5``."""
    return any(isinstance(n, ast.Attribute) and n.attr == "A5" for n in ast.walk(node))


def test_sfa_dcp_a5_raise_guard_is_wired_in_check_and_update_config():
    """AST guard: the SFA-DCP ``NotImplementedError`` must stay wired to the
    A5 branch of ``NPUPlatform.check_and_update_config``.

    The raise firing on real A5 is validated by the e2e single-card A5 suite.
    Here we only assert, via AST (not text grep), that the raise still exists
    *inside* an A5-gated ``if``. Removing the guard or unwiring it from the A5
    condition fails this test; a comment/dead-code string would not satisfy it.
    """
    import vllm_ascend.platform as platform_mod

    tree = ast.parse(Path(platform_mod.__file__).read_text(encoding="utf-8"))

    method = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "check_and_update_config"),
        None,
    )
    assert method is not None, "check_and_update_config not found in platform.py"

    found = False
    for node in ast.walk(method):
        if not (isinstance(node, ast.If) and _references_a5(node.test)):
            continue
        # Is there a `raise NotImplementedError` within this A5-gated branch?
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise) and _raised_exception_name(sub.exc) == "NotImplementedError":
                found = True
                break
        if found:
            break

    assert found, (
        "SFA-DCP NotImplementedError guard is missing or no longer wired to the "
        "A5 branch in NPUPlatform.check_and_update_config"
    )
