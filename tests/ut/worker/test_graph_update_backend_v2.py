# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionBackend
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend


@pytest.mark.parametrize("gdn_first", [False, True])
def test_graph_update_backend_skips_gdn(gdn_first):
    attention = Mock()
    attention.get_impl_cls.return_value = object()
    gdn_group = SimpleNamespace(backend=AscendGDNAttentionBackend)
    attention_group = SimpleNamespace(backend=attention)
    groups = [[gdn_group], [attention_group]] if gdn_first else [[attention_group], [gdn_group]]

    assert _get_graph_update_backend(groups) is attention


def test_graph_update_backend_skips_none_impl():
    metadata = Mock()
    metadata.get_impl_cls.return_value = None
    attention = Mock()
    attention.get_impl_cls.return_value = object()

    groups = [[SimpleNamespace(backend=metadata), SimpleNamespace(backend=attention)]]
    assert _get_graph_update_backend(groups) is attention


@pytest.mark.parametrize("groups", [[], [[SimpleNamespace(backend=AscendGDNAttentionBackend)]]])
def test_graph_update_backend_requires_executable_backend(groups):
    with pytest.raises(RuntimeError, match="No executable attention backend"):
        _get_graph_update_backend(groups)


def test_graph_update_backend_preserves_unexpected_errors():
    backend = Mock()
    backend.get_impl_cls.side_effect = ValueError("invalid backend config")

    with pytest.raises(ValueError, match="invalid backend config"):
        _get_graph_update_backend([[SimpleNamespace(backend=backend)]])
