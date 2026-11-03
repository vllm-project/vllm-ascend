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
"""Tests for the host-sync guard in ``BreakableACLGraphWrapper._replay``.

The guard skips the per-replay ``torch.npu.current_stream().synchronize()``
only for an ``UpdatableGraph`` that registered attention tasks: such a graph
already orders the async parameter update against the previous replay
(``wait_stream`` / update-stream barrier) and the replayed attention node
against its own update (captured per-task ``ExternalEvent``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_ascend.compilation import breakable_aclgraph as breakable_module
from vllm_ascend.compilation.breakable_aclgraph import BreakableACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import UpdatableGraph


def _make_wrapper(use_eagle: bool = False, enable_enpu: bool = False) -> BreakableACLGraphWrapper:
    """Build the wrapper without running the vLLM base ``__init__``.

    ``_replay`` only reads ``use_eagle``/``enable_enpu`` from ``self`` and
    delegates the replay itself to ``BreakableCUDAGraphWrapper._replay``,
    which every test patches out.
    """
    wrapper = object.__new__(BreakableACLGraphWrapper)
    wrapper.use_eagle = use_eagle
    wrapper.enable_enpu = enable_enpu
    return wrapper


def _updatable_graph(tasks: list) -> UpdatableGraph:
    graph = UpdatableGraph()
    graph.tasks = tasks
    return graph


def _replay(
    wrapper: BreakableACLGraphWrapper,
    runtime_mode: CUDAGraphMode,
    graph: object,
    *,
    is_draft_model: bool = False,
):
    """Run ``_replay`` and report whether the host synchronize happened.

    Returns ``(output, synchronize_mock, super_replay_mock)``.
    """
    entry = SimpleNamespace(aclgraph=graph, output="replayed-output")
    stream = MagicMock()
    fake_torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream))

    with (
        patch.object(
            breakable_module,
            "get_forward_context",
            return_value=SimpleNamespace(cudagraph_runtime_mode=runtime_mode),
        ),
        patch.object(
            breakable_module,
            "_EXTRA_CTX",
            SimpleNamespace(is_draft_model=is_draft_model),
        ),
        patch.object(breakable_module, "torch", fake_torch),
        patch.object(
            breakable_module.BreakableCUDAGraphWrapper,
            "_replay",
        ) as super_replay,
    ):
        output = wrapper._replay(entry, (), {})

    return output, stream.synchronize, super_replay


def test_full_replay_of_updatable_graph_with_tasks_skips_host_sync():
    """The registered per-task events already order update vs replay."""
    wrapper = _make_wrapper()
    graph = _updatable_graph(tasks=[object(), object()])

    output, synchronize, super_replay = _replay(wrapper, CUDAGraphMode.FULL, graph)

    synchronize.assert_not_called()
    super_replay.assert_called_once()
    assert output == "replayed-output"


def test_full_replay_of_updatable_graph_without_tasks_keeps_host_sync():
    """No per-task events means nothing orders update vs replay but the sync."""
    wrapper = _make_wrapper()
    graph = _updatable_graph(tasks=[])

    output, synchronize, super_replay = _replay(wrapper, CUDAGraphMode.FULL, graph)

    synchronize.assert_called_once()
    super_replay.assert_called_once()
    assert output == "replayed-output"


def test_full_replay_of_plain_graph_keeps_host_sync():
    """A graph that is not an ``UpdatableGraph`` keeps the original barrier."""
    wrapper = _make_wrapper()

    output, synchronize, super_replay = _replay(wrapper, CUDAGraphMode.FULL, MagicMock())

    synchronize.assert_called_once()
    super_replay.assert_called_once()
    assert output == "replayed-output"


def test_full_replay_of_mock_with_tasks_attribute_keeps_host_sync():
    """``isinstance`` is what matters, not a ``tasks`` attribute on any object."""
    wrapper = _make_wrapper()
    lookalike = MagicMock()
    lookalike.tasks = [object()]

    _, synchronize, _ = _replay(wrapper, CUDAGraphMode.FULL, lookalike)

    synchronize.assert_called_once()


def test_entry_without_aclgraph_keeps_host_sync():
    """An entry that carries no graph must not raise; the barrier stays."""
    wrapper = _make_wrapper()
    entry = SimpleNamespace(output="replayed-output")
    stream = MagicMock()
    fake_torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream))

    with (
        patch.object(
            breakable_module,
            "get_forward_context",
            return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
        ),
        patch.object(
            breakable_module,
            "_EXTRA_CTX",
            SimpleNamespace(is_draft_model=False),
        ),
        patch.object(breakable_module, "torch", fake_torch),
        patch.object(
            breakable_module.BreakableCUDAGraphWrapper,
            "_replay",
        ) as super_replay,
    ):
        output = wrapper._replay(entry, (), {})

    stream.synchronize.assert_called_once()
    super_replay.assert_called_once()
    assert output == "replayed-output"


@pytest.mark.parametrize("runtime_mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.NONE])
def test_non_full_replay_never_synchronizes(runtime_mode):
    """Only FULL replays had the barrier."""
    wrapper = _make_wrapper()

    _, synchronize, super_replay = _replay(wrapper, runtime_mode, _updatable_graph(tasks=[]))

    synchronize.assert_not_called()
    super_replay.assert_called_once()


def test_enpu_mode_skips_host_sync():
    """With NPU graph-ex the model runner already orders update vs replay."""
    wrapper = _make_wrapper(enable_enpu=True)

    _, synchronize, _ = _replay(wrapper, CUDAGraphMode.FULL, MagicMock())

    synchronize.assert_not_called()


def test_draft_eagle_skips_host_sync():
    """The merged EAGLE draft graph does not need the barrier."""
    wrapper = _make_wrapper(use_eagle=True)

    _, synchronize, _ = _replay(wrapper, CUDAGraphMode.FULL, MagicMock(), is_draft_model=True)

    synchronize.assert_not_called()


def test_draft_model_without_eagle_still_synchronizes():
    """Only the EAGLE draft is exempt; a plain draft keeps the barrier."""
    wrapper = _make_wrapper(use_eagle=False)

    _, synchronize, _ = _replay(wrapper, CUDAGraphMode.FULL, MagicMock(), is_draft_model=True)

    synchronize.assert_called_once()
