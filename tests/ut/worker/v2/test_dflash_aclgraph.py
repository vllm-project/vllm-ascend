# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import DFlashCudaGraphManager

from vllm_ascend.attention.fa_v1 import AscendFlashAttentionBackend
from vllm_ascend.worker.v2.spec_decode.dflash.aclgraph import DFlashAclGraphManager


def test_flash_attn_replay_does_not_build_or_update_host_metadata(monkeypatch):
    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    manager.speculator = SimpleNamespace(
        attn_backends={"draft.0": AscendFlashAttentionBackend, "draft.1": AscendFlashAttentionBackend},
        build_draft_attn_metadatas=Mock(side_effect=AssertionError("unexpected host metadata build")),
    )
    manager._updatable_graph_replay = Mock(side_effect=AssertionError("unexpected task update"))
    replay = Mock(return_value=object())
    monkeypatch.setattr(DFlashCudaGraphManager, "run_fullgraph", replay)
    desc = SimpleNamespace(num_tokens=18, num_reqs=3)
    assert manager.run_fullgraph(desc) is replay.return_value
    replay.assert_called_once_with(desc)


@pytest.mark.parametrize("mixed_backends", [False, True])
def test_other_backends_keep_metadata_update_path(monkeypatch, mixed_backends):
    from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as module

    manager = DFlashAclGraphManager.__new__(DFlashAclGraphManager)
    backends = {"draft.0": object()}
    if mixed_backends:
        backends["draft.1"] = AscendFlashAttentionBackend
    metadata = [{"draft.0": object()}]
    upper_bound = object()
    manager.speculator = SimpleNamespace(
        attn_backends=backends,
        input_batch=SimpleNamespace(seq_lens_cpu_upper_bound=upper_bound),
        build_draft_attn_metadatas=Mock(return_value=metadata),
    )
    manager._updatable_graph_replay = Mock(return_value=object())
    monkeypatch.setattr(module, "use_updatable_graph", lambda _: True)
    desc = SimpleNamespace(num_tokens=18, num_reqs=3)
    assert manager.run_fullgraph(desc) is manager._updatable_graph_replay.return_value
    manager.speculator.build_draft_attn_metadatas.assert_called_once_with(3, upper_bound)
    manager._updatable_graph_replay.assert_called_once_with(desc, metadata)
