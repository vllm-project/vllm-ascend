from types import SimpleNamespace

import pytest

from vllm_ascend.patch.platform.patch_dspark_mamba_phase_fence import (
    _DEFERRED_REQUEST_IDS_ATTR,
    _is_padded_transition_window,
    _make_cached_request_data_wrapper,
    _make_mamba_split_wrapper,
)


def _make_scheduler(*, method="dspark", role="kv_consumer", num_spec_tokens=7):
    kv_transfer_config = SimpleNamespace(
        kv_role=role,
        is_kv_consumer=role == "kv_consumer",
        is_kv_producer=role == "kv_producer",
    )
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(method=method),
            kv_transfer_config=kv_transfer_config,
        ),
        num_spec_tokens=num_spec_tokens,
    )


def _make_request(*, num_computed_tokens=0, num_tokens=380):
    return SimpleNamespace(
        request_id="request-0",
        num_computed_tokens=num_computed_tokens,
        num_tokens=num_tokens,
    )


def test_partial_padded_window_runs_target_only():
    calls = []

    def original(_scheduler, _request, num_new_tokens, *_args):
        calls.append(num_new_tokens)
        return 4 if num_new_tokens == 8 else num_new_tokens

    scheduler = _make_scheduler()
    request = _make_request()
    wrapped = _make_mamba_split_wrapper(original)

    assert wrapped(
        scheduler,
        request,
        8,
        128,
        251,
    ) == 1
    assert calls == [8, 1]
    assert getattr(scheduler, _DEFERRED_REQUEST_IDS_ATTR) == {request.request_id}


@pytest.mark.parametrize(
    ("scheduler", "request", "num_new_tokens"),
    [
        (_make_scheduler(method="eagle"), _make_request(num_computed_tokens=379), 8),
        (_make_scheduler(role="kv_producer"), _make_request(num_computed_tokens=379), 8),
        (_make_scheduler(), _make_request(num_computed_tokens=379, num_tokens=381), 8),
        (_make_scheduler(), _make_request(num_computed_tokens=379), 7),
    ],
)
def test_transition_detection_is_strictly_scoped(
    scheduler,
    request,
    num_new_tokens,
):
    assert not _is_padded_transition_window(
        scheduler,
        request,
        num_new_tokens,
        0,
        0,
    )


def test_complete_padded_window_is_unchanged():
    def original(_scheduler, _request, num_new_tokens, *_args):
        return num_new_tokens

    scheduler = _make_scheduler()
    request = _make_request(num_computed_tokens=379)
    wrapped = _make_mamba_split_wrapper(original)

    assert wrapped(scheduler, request, 8) == 8
    assert not hasattr(scheduler, _DEFERRED_REQUEST_IDS_ATTR)


def test_cached_request_data_drops_only_synthetic_drafts():
    captured = {}

    def original(
        _scheduler,
        running_reqs,
        resumed_reqs,
        num_scheduled_tokens,
        spec_decode_tokens,
        req_to_new_blocks,
    ):
        captured["spec_decode_tokens"] = dict(spec_decode_tokens)
        return "cached-request-data"

    scheduler = _make_scheduler()
    setattr(
        scheduler,
        _DEFERRED_REQUEST_IDS_ATTR,
        {"request-0", "request-with-real-drafts"},
    )
    spec_decode_tokens = {
        "request-0": [-1] * 7,
        "request-with-real-drafts": [11, 12],
        "other-request": [-1] * 7,
    }
    wrapped = _make_cached_request_data_wrapper(original)

    result = wrapped(
        scheduler,
        [],
        [],
        {
            "request-0": 1,
            "request-with-real-drafts": 1,
            "other-request": 8,
        },
        spec_decode_tokens,
        {},
    )

    assert result == "cached-request-data"
    assert captured["spec_decode_tokens"] == {
        "request-with-real-drafts": [11, 12],
        "other-request": [-1] * 7,
    }
    assert getattr(scheduler, _DEFERRED_REQUEST_IDS_ATTR) == set()
