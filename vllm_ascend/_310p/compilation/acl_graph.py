#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""310P-only ACLGraphWrapper for MTP FULL-replay sync overhead.

MRV1 wraps the model with shared ``ACLGraphWrapper``. To keep A2/910 mainline
untouched, 310P loads this subclass via ``apply_310p_aclgraph_patches()`` from
``NPUWorker310`` and rebinds imports used by MRV1 / MTP proposer.

Replay-only changes vs mainline:
- ``synchronize()`` → ``wait_stream(update_stream)``; skip when UpdatableGraph
  has no tasks (typical 310P SpecDecoding).
- Skip empty ``graph.update``.
Capture path still delegates to the parent implementation.
"""

from __future__ import annotations

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.logger import logger

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.compilation.updatable_graph import ContextSource, SharedSource, UpdatableGraph
from vllm_ascend.utils import use_updatable_graph

_PATCHED = False


class ACLGraphWrapper310(ACLGraphWrapper):
    """310P MRV1 ACLGraph wrapper with weaker FULL-replay barriers."""

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        # Capture (missing entry / missing graph) stays on the mainline path.
        entry = self.concrete_aclgraph_entries.get(batch_descriptor)
        if entry is None or entry.aclgraph is None:
            return super().__call__(*args, **kwargs)

        if self.is_debugging_mode:
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            assert new_input_addresses == entry.input_addresses, (
                f"Input addresses for aclgraphs are different "
                f"during replay. Expected {entry.input_addresses}, "
                f"got {new_input_addresses}"
            )

        logger.info_once("Replaying aclgraph")
        # Prior host-side graph-param updates run on ``update_stream``. Wait only
        # for that stream before FULL replay. A full ``synchronize()`` was a major
        # MTP decode tax on small hybrid models (e.g. Qwen3.5-2B) where kernels
        # are short. When UpdatableGraph has no tasks, skip the wait entirely.
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
        updatable = self.runtime_mode == CUDAGraphMode.FULL and use_updatable_graph(self.attn_backend)
        graph_has_tasks = bool(getattr(entry.aclgraph, "tasks", None))
        if not self.enable_enpu and need_sync and not (updatable and not graph_has_tasks):
            if self.update_stream is not None:
                torch.npu.current_stream().wait_stream(self.update_stream)
            else:
                torch.npu.current_stream().synchronize()
        if updatable:
            self._updatable_graph_replay(forward_context, entry.aclgraph)
        else:
            entry.aclgraph.replay()
        return entry.output

    def _updatable_graph_replay(self, forward_context, graph: UpdatableGraph):
        assert self.update_stream is not None
        if _EXTRA_CTX.is_draft_model:
            resolved_tasks = graph.resolve_tasks(SharedSource(self.draft_model_metadata))
        else:
            resolved_tasks = graph.resolve_tasks(ContextSource(forward_context.attn_metadata))
        if self.enable_enpu:
            if resolved_tasks:
                graph.update(self.update_stream, resolved_tasks)
            graph.replay()
        else:
            graph.replay()
            # 310P SpecDecoding registers no graph_task — skip empty update.
            if resolved_tasks:
                graph.update(self.update_stream, resolved_tasks)


def apply_310p_aclgraph_patches() -> None:
    """Install 310P FULL-replay barriers onto the shared ACLGraphWrapper class.

    Patches methods in-place so every MRV1 import site sees the 310P behavior
    without rebinding the class name (mypy forbids ``Module.Class = SubClass``).
    Only loaded from ``NPUWorker310`` (310P processes).
    """
    global _PATCHED
    if _PATCHED:
        return

    # method-assign: intentional monkeypatch for 310P-only worker process.
    ACLGraphWrapper.__call__ = ACLGraphWrapper310.__call__  # type: ignore[method-assign,assignment]
    ACLGraphWrapper._updatable_graph_replay = ACLGraphWrapper310._updatable_graph_replay  # type: ignore[method-assign,assignment]
    _PATCHED = True
