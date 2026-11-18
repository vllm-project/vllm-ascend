# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from types import SimpleNamespace

import torch

from vllm_ascend.hisparse.host_mirror import _DirectMirrorSources, _source_layout_valid


def make_worker():
    backing = torch.zeros(16, dtype=torch.uint8)
    handles = []
    for offset in (0, 4):
        cache = torch.as_strided(backing, (2, 2, 2), (8, 2, 1), offset)
        runtime = SimpleNamespace(hot_backing=backing, hot=SimpleNamespace(cache=cache), resident_source_index=0)
        runtime.index_group = SimpleNamespace(leader=runtime, followers=[])
        handles.append(SimpleNamespace(runtime=runtime))
    # Core only retains handles/config; deliberately omit the removed worker fields.
    return SimpleNamespace(
        cache_handles=handles,
        resident_caches=[h.runtime.hot.cache for h in handles],
        host_caches=[torch.zeros((2, 4), dtype=torch.uint8) for _ in handles],
        kernel_block_size=2,
        _draft_layers=(1,),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "target": SimpleNamespace(hisparse_cache=handles[0]),
                    "draft": SimpleNamespace(hisparse_cache=handles[1]),
                },
            ),
        ),
    )


def test_current_core_worker_contract_binds_runtime_backing_and_draft_handle():
    worker = make_worker()
    assert _source_layout_valid(worker)
    sources = _DirectMirrorSources.bind(worker, SimpleNamespace(draft_attn_layer_names=["draft"]))
    assert sources is not None
    assert sources._targets == (0,)


def test_reject_unrecognized_draft_binding():
    assert _DirectMirrorSources.bind(make_worker(), SimpleNamespace(draft_attn_layer_names=["other"])) is None


def test_reject_overlapping_target_and_draft_page_ranges():
    worker = make_worker()
    worker.resident_caches[1] = worker.resident_caches[0]
    worker.cache_handles[1].runtime.hot.cache = worker.resident_caches[0]
    assert not _source_layout_valid(worker)


def test_reject_draft_resolver_that_can_overwrite_target_group():
    worker = make_worker()
    worker.cache_handles[1].runtime.index_group.followers = [worker.cache_handles[0].runtime]
    assert _DirectMirrorSources.bind(worker, SimpleNamespace(draft_attn_layer_names=["draft"])) is None
