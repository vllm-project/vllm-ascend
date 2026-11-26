# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from vllm_ascend.target_prep_masks import _enabled, mask_write_host_scope


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("graph_manager", [False, True])
def test_mask_fusion_is_scoped_to_hy4_decode(enabled, graph_manager):
    runner = SimpleNamespace(model_state=SimpleNamespace(_decode_optimizations=enabled))
    owner = SimpleNamespace(model_runner=runner) if graph_manager else runner

    @mask_write_host_scope
    def call(owner):
        assert _enabled() is enabled
        assert not _enabled(capturing=True)

    assert not _enabled()
    call(owner)
    assert not _enabled()


def test_mask_scope_restores_after_failure_and_excludes_other_models():
    @mask_write_host_scope
    def unrelated(owner):
        assert not _enabled()

    @mask_write_host_scope
    def call(owner):
        assert _enabled()
        unrelated(SimpleNamespace(model_state=object()))
        assert _enabled()
        raise RuntimeError("scope failure")

    owner = SimpleNamespace(model_state=SimpleNamespace(_decode_optimizations=True))
    with pytest.raises(RuntimeError, match="scope failure"):
        call(owner)
    assert not _enabled()
